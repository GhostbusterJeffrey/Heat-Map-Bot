from __future__ import annotations

import unittest
from datetime import date

import numpy as np
import pandas as pd

from tpr_terminal.analytics import ExposureAnalyticsEngine
from tpr_terminal.models import DataQualityReport, ForwardSummary
from tpr_terminal.renderer import HeatmapRenderer
from tpr_terminal.scenario import build_gamma_flips, build_pin_levels
from tpr_terminal.surface_health import build_surface_health_report


class DummyProvider:
    pass


class Alpha2UnifiedTests(unittest.TestCase):
    def test_structural_wall_can_differ_from_raw_peak(self) -> None:
        engine = ExposureAnalyticsEngine(
            DummyProvider(),
            min_contracts_to_render=0,
            min_usable_ratio=0,
            min_confidence_to_render=0,
            min_wall_percentile=0.40,
        )
        rows = []
        expiries = ["07/17/2026", "07/24/2026", "07/31/2026"]
        for expiry in expiries:
            rows.append({"strike": 100.0, "expiry": expiry, "gex": 100.0, "abs_gex": 100.0, "cell_quality": 0.95})
            rows.append({"strike": 95.0, "expiry": expiry, "gex": -80.0, "abs_gex": 80.0, "cell_quality": 0.90})
        rows.append({"strike": 110.0, "expiry": expiries[0], "gex": 500.0, "abs_gex": 500.0, "cell_quality": 0.39})
        rows.append({"strike": 105.0, "expiry": expiries[0], "gex": -350.0, "abs_gex": 350.0, "cell_quality": 0.35})
        records = pd.DataFrame(rows)
        totals, raw_pos, _, structural_pos, _ = engine._build_structural_walls(records, 100.0)
        self.assertFalse(totals.empty)
        self.assertIsNotNone(raw_pos)
        self.assertIsNotNone(structural_pos)
        assert raw_pos is not None and structural_pos is not None
        self.assertEqual(raw_pos.strike, 110.0)
        self.assertEqual(structural_pos.strike, 100.0)

    def test_dynamic_engine_detects_multiple_gamma_flips(self) -> None:
        contracts = pd.DataFrame(
            {
                "strike": [90.0, 100.0, 110.0],
                "time_years": [0.10, 0.10, 0.10],
                "iv": [0.20, 0.20, 0.20],
                "rate": [0.04, 0.04, 0.04],
                "dividend_yield": [0.0, 0.0, 0.0],
                "liquidity": [3000.0, 5000.0, 3000.0],
                "polarity": [1.0, -1.0, 1.0],
                "contract_quality": [1.0, 1.0, 1.0],
                "side": ["call", "call", "call"],
                "gex": [0.0, 0.0, 0.0],
                "days_to_expiry": [36.5, 36.5, 36.5],
                "expiration": [date(2026, 8, 21)] * 3,
            }
        )
        _, flips = build_gamma_flips(
            contracts,
            spot=100.0,
            multiplier=100,
            min_confidence=0.0,
            grid_low_pct=0.70,
            grid_high_pct=1.30,
            grid_points=801,
        )
        self.assertGreaterEqual(len(flips), 2)
        self.assertLess(flips[0].distance.__abs__(), flips[1].distance.__abs__() + 20)
        self.assertTrue(all(0 <= flip.shock_stability <= 1 for flip in flips))

    def test_pin_distance_penalty_prefers_nearby_strike(self) -> None:
        contracts = pd.DataFrame(
            {
                "strike": [95.0, 100.0, 105.0],
                "time_years": [0.02] * 3,
                "iv": [0.25] * 3,
                "rate": [0.04] * 3,
                "dividend_yield": [0.0] * 3,
                "liquidity": [5000.0, 20000.0, 5000.0],
                "polarity": [1.0, -1.0, 1.0],
                "contract_quality": [1.0] * 3,
                "side": ["call", "put", "call"],
                "gex": [1e6, -5e6, 1e6],
                "days_to_expiry": [2.0] * 3,
                "expiration": [date(2026, 7, 17)] * 3,
            }
        )
        totals = pd.DataFrame(
            {
                "strike": [95.0, 100.0, 105.0],
                "gex": [1e6, -5e6, 1e6],
                "abs_gex": [1e6, 5e6, 1e6],
                "structural_score": [0.5, 0.9, 0.5],
            }
        )
        pins = build_pin_levels(
            contracts,
            totals,
            spot=100.0,
            expected_move=3.0,
            multiplier=100,
            min_score=0.0,
        )
        self.assertTrue(pins)
        self.assertEqual(pins[0].strike, 100.0)

    def test_surface_health_penalizes_repairs(self) -> None:
        contracts = pd.DataFrame(
            {
                "expiration": [date(2026, 8, 21)] * 5,
                "strike": [90, 95, 100, 105, 110],
                "iv": [0.30, 0.27, 0.25, 0.26, 0.29],
                "time_years": [0.1] * 5,
            }
        )
        clean_quality = DataQualityReport(
            expirations_requested=1,
            expirations_processed=1,
            contracts_seen=5,
            contracts_used=5,
            iv_market_solved=5,
            parity_expirations=1,
        )
        repaired_quality = DataQualityReport(
            expirations_requested=1,
            expirations_processed=1,
            contracts_seen=5,
            contracts_used=5,
            iv_values_interpolated=5,
            parity_expirations=0,
        )
        forwards = [ForwardSummary(date(2026, 8, 21), 100.2, 0.01, 5, 5.0, True, "test")]
        clean = build_surface_health_report(contracts, clean_quality, forwards, spot=100.0)
        repaired = build_surface_health_report(contracts, repaired_quality, [], spot=100.0)
        self.assertGreater(clean.score, repaired.score)

    def test_color_scale_handles_all_zero_matrix(self) -> None:
        scale = HeatmapRenderer._color_scale(np.zeros((4, 4)))
        self.assertGreaterEqual(scale["floor"], 1.0)
        background, text, _ = HeatmapRenderer._cell_colors(0.0, scale)
        self.assertIsInstance(background, str)
        self.assertIsInstance(text, str)

    def test_color_scale_preserves_sign_borders_at_extremes(self) -> None:
        values = np.array([[-1e9, -1e6, 1e6, 1e9]], dtype=float)
        scale = HeatmapRenderer._color_scale(values)
        _, _, positive_border = HeatmapRenderer._cell_colors(1e9, scale)
        _, _, negative_border = HeatmapRenderer._cell_colors(-1e9, scale)
        self.assertEqual(positive_border, "#ffd700")
        self.assertEqual(negative_border, "#ff003c")


if __name__ == "__main__":
    unittest.main()
