from __future__ import annotations

import unittest
from datetime import UTC, date, datetime

import numpy as np
from test_core import FakeProvider

from tpr_terminal.analytics import ExposureAnalyticsEngine
from tpr_terminal.contract_analytics import ContractAnalyticsEngine
from tpr_terminal.models import RateCurve


class Alpha42CanonicalStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.curve = RateCurve(
            (0.0, 3650.0),
            (0.045, 0.045),
            "test",
            date(2026, 7, 13),
        )
        self.now = datetime(2026, 7, 13, 14, tzinfo=UTC)
        self.engine = ExposureAnalyticsEngine(
            FakeProvider(),
            min_contracts_to_render=1,
            min_usable_ratio=0.0,
            min_confidence_to_render=0.0,
            quant_engine_mode="legacy",
        )

    def analyze(self):
        return self.engine.analyze(
            "TEST",
            max_expiries=1,
            liquidity_mode="open_interest",
            exposure_mode="net_proxy",
            rate_curve=self.curve,
            now=self.now,
        )

    def test_legacy_heatmap_calculation_remains_locked(self) -> None:
        result = self.analyze()
        expected_contract_gex = np.asarray(
            [
                40475.432547128759,
                154012.72932067909,
                64071.106747398939,
                -38309.054345961966,
                -154012.72932067895,
                -64801.140454871718,
            ]
        )
        expected_matrix_gex = np.asarray(
            [
                2166.3782011667936,
                1.4551915228366852e-10,
                -730.0337074727795,
            ]
        )
        np.testing.assert_array_equal(
            result.contract_records["gex"].to_numpy(),
            expected_contract_gex,
        )
        np.testing.assert_array_equal(
            result.matrix_records["gex"].to_numpy(),
            expected_matrix_gex,
        )

    def test_analysis_materializes_one_snapshot_bound_contract_ledger(self) -> None:
        result = self.analyze()
        ledger = result.contract_ledger
        self.assertIsNotNone(ledger)
        assert ledger is not None
        self.assertEqual(ledger.snapshot_id, result.snapshot_id)
        self.assertEqual(ledger.ticker, result.ticker)
        self.assertEqual(len(ledger), len(result.contract_records))
        self.assertEqual(len({state.contract_state_id for state in ledger.states}), len(ledger))
        self.assertTrue(
            all(state.snapshot_id == result.snapshot_id for state in ledger.states)
        )

    def test_contract_state_preserves_heatmap_inputs_and_greeks_exactly(self) -> None:
        result = self.analyze()
        assert result.contract_ledger is not None
        state = result.contract_ledger.resolve(
            expiration=date(2026, 7, 17),
            strike=100.0,
            option_side="call",
        )
        row = result.contract_records.loc[
            (result.contract_records["expiration"] == date(2026, 7, 17))
            & (result.contract_records["side"] == "call")
            & np.isclose(result.contract_records["strike"], 100.0)
        ].iloc[0]
        self.assertEqual(state.selected_implied_volatility, float(row["iv"]))
        self.assertEqual(state.greek("gamma"), float(row["gamma"]))
        self.assertEqual(state.gamma_exposure_1pct, float(row["gex"]))
        self.assertEqual(state.bid, float(row["bid"]))
        self.assertEqual(state.ask, float(row["ask"]))
        self.assertEqual(state.last_price, float(row["lastPrice"]))

    def test_greek_report_gamma_matches_corresponding_heatmap_contract(self) -> None:
        result = self.analyze()
        assert result.contract_ledger is not None
        report_engine = ContractAnalyticsEngine(
            FakeProvider(),
            contract_multiplier=100,
            quant_engine_mode="legacy",
        )
        for side in ("call", "put"):
            with self.subTest(side=side):
                state = result.contract_ledger.resolve(
                    expiration=date(2026, 7, 17),
                    strike=100.0,
                    option_side=side,
                )
                report = report_engine.analyze_canonical_state(
                    state,
                    quantity=1,
                    position="long",
                    scope="market_open_interest",
                )
                self.assertEqual(report.snapshot_id, result.snapshot_id)
                self.assertEqual(report.contract_state_id, state.contract_state_id)
                self.assertEqual(report.contract_state_source, "canonical_ledger")
                self.assertEqual(report.heatmap_gamma_per_share, state.greek("gamma"))
                self.assertEqual(report.heatmap_gex_1pct, state.gamma_exposure_1pct)
                self.assertEqual(
                    report.values["gamma"].mid,
                    state.greek("gamma")
                    * state.polarity
                    * state.contract_multiplier
                    * state.open_interest,
                )
                self.assertEqual(
                    report.dollar_gamma_1pct.mid,
                    state.gamma_exposure_1pct,
                )

    def test_expiry_views_share_the_canonical_ledger_without_rebuilding(self) -> None:
        canonical = self.analyze()
        view = self.engine.slice_result(canonical, max_expiries=1)
        self.assertIs(view.contract_ledger, canonical.contract_ledger)
        self.assertEqual(view.snapshot_id, canonical.snapshot_id)

    def test_contract_state_ids_are_deterministic_for_same_snapshot(self) -> None:
        first = self.analyze()
        second = self.analyze()
        assert first.contract_ledger is not None
        assert second.contract_ledger is not None
        self.assertEqual(first.snapshot_id, second.snapshot_id)
        self.assertEqual(
            [state.contract_state_id for state in first.contract_ledger.states],
            [state.contract_state_id for state in second.contract_ledger.states],
        )


if __name__ == "__main__":
    unittest.main()
