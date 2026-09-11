from __future__ import annotations

import unittest
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd

from test_core import FakeProvider, FakeSession
from tpr_terminal.analytics import ExposureAnalyticsEngine
from tpr_terminal.contract_analytics import ContractAnalyticsEngine
from tpr_terminal.math_engine import implied_volatility_solve
from tpr_terminal.models import ExpirationChain, RateCurve
from tpr_terminal.pricing import black_scholes_price, greek_stack
from tpr_terminal.quant_accuracy import (
    adaptive_greek_validation,
    gamma_exposure_1pct,
    sanitize_quote,
)
from tpr_terminal.regression import build_shadow_comparison, compare_gex_frames
from tpr_terminal.renderer import HeatmapRenderer


class TwoExpirySession(FakeSession):
    def expirations(self) -> list[date]:
        return [date(2026, 7, 17), date(2026, 7, 24)]

    def option_chain(self, expiration: date) -> ExpirationChain:
        original = super().option_chain(self.expiry)
        return ExpirationChain(
            expiration,
            original.calls.copy(),
            original.puts.copy(),
        )


class TwoExpiryProvider(FakeProvider):
    def open_session(self, ticker: str) -> TwoExpirySession:
        return TwoExpirySession(self.zero_oi)


class Alpha41AccuracyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.curve = RateCurve((0.0, 3650.0), (0.045, 0.045), "test", date(2026, 7, 13))
        self.now = datetime(2026, 7, 13, 14, tzinfo=UTC)

    def test_quote_sanitizer_rejects_crossed_market_and_bad_last(self) -> None:
        quote = sanitize_quote(
            option_side="call",
            spot=100.0,
            strike=100.0,
            time_years=0.25,
            risk_free_rate=0.04,
            dividend_yield=0.01,
            bid=5.0,
            ask=4.0,
            last=150.0,
        )
        self.assertIsNone(quote.mark)
        self.assertEqual(quote.mark_source, "unusable")
        self.assertEqual(quote.quote_quality, 0.0)
        self.assertTrue(any("crossed" in warning for warning in quote.warnings))

    def test_diagnostic_iv_solver_recovers_known_volatility(self) -> None:
        market_price = float(
            black_scholes_price(
                "call",
                100.0,
                np.asarray([105.0]),
                np.asarray([45.0 / 365.25]),
                np.asarray([0.27]),
                np.asarray([0.04]),
                np.asarray([0.01]),
            )[0]
        )
        solved = implied_volatility_solve(
            option_kind="call",
            market_price=market_price,
            spot=100.0,
            strike=105.0,
            time_years=45.0 / 365.25,
            risk_free_rate=0.04,
            dividend_yield=0.01,
        )
        self.assertTrue(solved.converged)
        self.assertAlmostEqual(float(solved.volatility), 0.27, places=8)
        self.assertLess(float(solved.residual), 1e-8)

    def test_canonical_gex_formula_applies_multiplier_once(self) -> None:
        calculated = float(
            gamma_exposure_1pct(
                0.02,
                position_contracts=1_000,
                multiplier=100,
                spot=100.0,
                polarity=-1.0,
            )
        )
        self.assertEqual(calculated, -200_000.0)

    def test_adaptive_validation_covers_core_and_higher_greeks(self) -> None:
        stack = greek_stack(
            "call",
            spot=100.0,
            strikes=np.asarray([100.0]),
            time_years=np.asarray([30.0 / 365.25]),
            volatility=np.asarray([0.25]),
            risk_free_rate=np.asarray([0.04]),
            dividend_yield=np.asarray([0.01]),
        )
        analytic = {
            name: float(getattr(stack, name)[0])
            for name in ("delta", "gamma", "theta", "vega", "rho")
        }
        validation = adaptive_greek_validation(
            option_side="call",
            spot=100.0,
            strike=100.0,
            time_years=30.0 / 365.25,
            volatility=0.25,
            risk_free_rate=0.04,
            dividend_yield=0.01,
            analytic=analytic,
        )
        self.assertEqual(set(validation.core_errors), {"delta", "gamma", "theta", "vega", "rho"})
        self.assertEqual(
            set(validation.higher_order_spreads),
            {"vanna", "charm", "vomma", "speed", "color", "zomma", "ultima"},
        )
        self.assertLess(validation.maximum_core_error, 0.001)

    def test_shadow_mode_preserves_legacy_heatmap_exactly(self) -> None:
        common = dict(
            min_contracts_to_render=1,
            min_usable_ratio=0.0,
            min_confidence_to_render=0.0,
        )
        legacy = ExposureAnalyticsEngine(FakeProvider(), quant_engine_mode="legacy", **common)
        shadow = ExposureAnalyticsEngine(FakeProvider(), quant_engine_mode="shadow", **common)
        legacy_result = legacy.analyze(
            "TEST",
            max_expiries=1,
            liquidity_mode="open_interest",
            exposure_mode="net_proxy",
            rate_curve=self.curve,
            now=self.now,
        )
        shadow_result = shadow.analyze(
            "TEST",
            max_expiries=1,
            liquidity_mode="open_interest",
            exposure_mode="net_proxy",
            rate_curve=self.curve,
            now=self.now,
        )
        pd.testing.assert_series_equal(
            legacy_result.matrix_records["gex"],
            shadow_result.matrix_records["gex"],
            check_names=False,
            rtol=0.0,
            atol=0.0,
        )
        self.assertIn("candidate_gex", shadow_result.contract_records.columns)
        self.assertIn("candidate_surface_iv", shadow_result.contract_records.columns)
        self.assertIn("candidate_surface_source", shadow_result.contract_records.columns)
        self.assertEqual(shadow_result.quant_engine_mode, "shadow")
        comparison = build_shadow_comparison(shadow_result.contract_records)
        self.assertEqual(
            set(comparison.columns),
            {"expiry", "strike", "legacy_gex", "candidate_gex"},
        )

    def test_candidate_mode_selects_candidate_surface_values(self) -> None:
        engine = ExposureAnalyticsEngine(
            FakeProvider(),
            quant_engine_mode="candidate",
            min_contracts_to_render=1,
            min_usable_ratio=0.0,
            min_confidence_to_render=0.0,
        )
        result = engine.analyze(
            "TEST",
            max_expiries=1,
            liquidity_mode="open_interest",
            exposure_mode="net_proxy",
            rate_curve=self.curve,
            now=self.now,
        )
        records = result.contract_records
        np.testing.assert_allclose(records["iv"], records["candidate_iv"])
        self.assertTrue(records["iv_blend_weight"].between(0.0, 1.0).all())
        self.assertTrue(np.isfinite(records["candidate_gex"]).all())
        self.assertIn("legacy_surface_iv", records.columns)

    def test_candidate_contract_report_has_canonical_state_and_validation(self) -> None:
        engine = ContractAnalyticsEngine(
            FakeProvider(),
            contract_multiplier=100,
            quant_engine_mode="candidate",
        )
        report = engine.analyze(
            "TEST",
            expiration=date(2026, 7, 17),
            strike=100.0,
            option_side="call",
            quantity=1,
            position="long",
            scope="per_contract",
            rate_curve=self.curve,
            now=self.now,
        )
        self.assertEqual(report.quant_engine_mode, "candidate")
        self.assertIsNotNone(report.forward_price)
        self.assertIsNotNone(report.settlement_time)
        expected_years = (report.settlement_time - self.now).total_seconds() / 86_400.0 / 365.25
        self.assertAlmostEqual(report.time_years, expected_years, places=12)
        self.assertIn(report.mark_source, {"midpoint", "last_fallback", "unusable"})
        self.assertEqual(set(report.core_validation_errors), {"delta", "gamma", "theta", "vega", "rho"})
        self.assertTrue(all(np.isfinite(value.relative_spread) for value in report.values.values()))

    def test_matrix_aggregation_sums_contract_gex_without_rescaling(self) -> None:
        contracts = pd.DataFrame(
            {
                "expiration": [date(2026, 7, 17)] * 3,
                "strike": [100.0, 100.0, 105.0],
                "gex": [250_000.0, -100_000.0, 80_000.0],
                "abs_gex": [250_000.0, 100_000.0, 80_000.0],
                "contract_quality": [1.0, 1.0, 1.0],
                "liquidity_source": ["open_interest"] * 3,
                "is_iv_repaired": [False] * 3,
                "days_to_expiry": [4.0] * 3,
                "rate": [0.04] * 3,
                "dividend_yield": [0.01] * 3,
            }
        )
        matrix = ExposureAnalyticsEngine._build_matrix_records(contracts)
        grouped = matrix.set_index("strike")["gex"]
        self.assertEqual(float(grouped.loc[100.0]), 150_000.0)
        self.assertEqual(float(grouped.loc[105.0]), 80_000.0)

    def test_canonical_snapshot_views_keep_overlapping_cells_identical(self) -> None:
        engine = ExposureAnalyticsEngine(
            TwoExpiryProvider(),
            min_contracts_to_render=1,
            min_usable_ratio=0.0,
            min_confidence_to_render=0.0,
        )
        canonical = engine.analyze(
            "TEST",
            max_expiries=2,
            liquidity_mode="open_interest",
            exposure_mode="net_proxy",
            rate_curve=self.curve,
            now=self.now,
        )
        intraday = engine.slice_result(canonical, max_expiries=1)
        swing = engine.slice_result(canonical, max_expiries=2)

        overlap = intraday.matrix_records.merge(
            swing.matrix_records,
            on=["expiration", "strike"],
            suffixes=("_intraday", "_swing"),
            validate="one_to_one",
        )
        self.assertFalse(overlap.empty)
        np.testing.assert_array_equal(
            overlap["gex_intraday"].to_numpy(),
            overlap["gex_swing"].to_numpy(),
        )
        np.testing.assert_array_equal(
            overlap["abs_gex_intraday"].to_numpy(),
            overlap["abs_gex_swing"].to_numpy(),
        )
        self.assertEqual(intraday.snapshot_id, canonical.snapshot_id)
        self.assertEqual(swing.snapshot_id, canonical.snapshot_id)
        self.assertEqual(intraday.quality.expirations_processed, 1)
        self.assertEqual(swing.quality.expirations_processed, 2)

    def test_regression_harness_scores_exact_match(self) -> None:
        frame = pd.DataFrame(
            {
                "expiry": ["07/17/2026"] * 4,
                "strike": [95.0, 100.0, 105.0, 110.0],
                "gex": [-3_000_000.0, -1_000_000.0, 2_000_000.0, 5_000_000.0],
            }
        )
        metrics = compare_gex_frames(frame, frame, top_k=2)
        self.assertEqual(metrics.sign_accuracy, 1.0)
        self.assertEqual(metrics.weighted_absolute_error, 0.0)
        self.assertEqual(metrics.rank_correlation, 1.0)
        self.assertTrue(metrics.positive_max_match)
        self.assertTrue(metrics.negative_max_match)

    def test_renderer_extrema_locates_positive_and_negative_nodes(self) -> None:
        matrix = pd.DataFrame(
            [[1.0, -7.0], [9.0, -2.0]],
            index=[100.0, 105.0],
            columns=["07/17/2026", "07/24/2026"],
        )
        positive, negative = HeatmapRenderer._global_extrema_positions(matrix)
        self.assertEqual(positive, (105.0, "07/17/2026"))
        self.assertEqual(negative, (100.0, "07/24/2026"))

    def test_renderer_global_node_ranks_are_unique_and_sign_separated(self) -> None:
        matrix = pd.DataFrame(
            [
                [12.0, -4.0, 7.0],
                [-20.0, 30.0, -9.0],
                [18.0, -15.0, 2.0],
            ],
            index=[100.0, 105.0, 110.0],
            columns=["07/17/2026", "07/24/2026", "07/31/2026"],
        )
        positive, negative = HeatmapRenderer._ranked_node_positions(matrix, limit=3)
        self.assertEqual(
            positive,
            [
                (105.0, "07/24/2026"),
                (110.0, "07/17/2026"),
                (100.0, "07/17/2026"),
            ],
        )
        self.assertEqual(
            negative,
            [
                (105.0, "07/17/2026"),
                (110.0, "07/24/2026"),
                (105.0, "07/31/2026"),
            ],
        )
        self.assertEqual(len(set(positive)), 3)
        self.assertEqual(len(set(negative)), 3)
        self.assertTrue(set(positive).isdisjoint(negative))

        lookup = HeatmapRenderer._node_rank_lookup(matrix, limit=3)
        self.assertEqual(
            sum(value == ("positive", 1) for value in lookup.values()), 1
        )
        self.assertEqual(
            sum(value == ("negative", 1) for value in lookup.values()), 1
        )
        self.assertEqual(
            sum(value[0] == "positive" for value in lookup.values()), 3
        )
        self.assertEqual(
            sum(value[0] == "negative" for value in lookup.values()), 3
        )

        positive_styles = [
            HeatmapRenderer._node_highlight_style("positive", rank)
            for rank in (1, 2, 3)
        ]
        negative_styles = [
            HeatmapRenderer._node_highlight_style("negative", rank)
            for rank in (1, 2, 3)
        ]
        self.assertEqual(positive_styles[0][3], "P1")
        self.assertEqual(negative_styles[0][3], "N1")
        self.assertGreater(positive_styles[0][1], positive_styles[1][1])
        self.assertGreater(positive_styles[1][1], positive_styles[2][1])
        self.assertGreater(negative_styles[0][2], negative_styles[1][2])
        self.assertGreater(negative_styles[1][2], negative_styles[2][2])

    def test_renderer_node_ranks_handle_one_sided_matrices(self) -> None:
        positive_only = pd.DataFrame(
            [[1.0, 4.0], [3.0, 2.0]],
            index=[100.0, 105.0],
            columns=["07/17/2026", "07/24/2026"],
        )
        positive, negative = HeatmapRenderer._ranked_node_positions(
            positive_only, limit=3
        )
        self.assertEqual(len(positive), 3)
        self.assertEqual(negative, [])
        self.assertEqual(positive[0], (100.0, "07/24/2026"))

        negative_only = -positive_only
        positive, negative = HeatmapRenderer._ranked_node_positions(
            negative_only, limit=3
        )
        self.assertEqual(positive, [])
        self.assertEqual(len(negative), 3)
        self.assertEqual(negative[0], (100.0, "07/24/2026"))


if __name__ == "__main__":
    unittest.main()
