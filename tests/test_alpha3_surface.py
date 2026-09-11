from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from tpr_terminal.analytics import ExposureAnalyticsEngine
from tpr_terminal.branding import BrandingManager
from tpr_terminal.models import DataQualityReport, ForwardSummary
from tpr_terminal.pricing import (
    american_binomial_price,
    black_scholes_delta,
    black_scholes_price,
    greek_stack,
)
from tpr_terminal.volatility_surface import fit_volatility_surface, raw_ssvi


class DummyProvider:
    pass


class Alpha3Tests(unittest.TestCase):
    def _surface_contracts(self) -> tuple[pd.DataFrame, list[ForwardSummary], DataQualityReport]:
        rows: list[dict[str, object]] = []
        expiries = [date(2026, 8, 21), date(2026, 9, 18), date(2026, 12, 18)]
        times = [0.10, 0.18, 0.43]
        thetas = [0.008, 0.014, 0.030]
        strikes = np.linspace(80, 120, 21)
        rho, eta, gamma = -0.35, 0.65, 0.50
        summaries: list[ForwardSummary] = []
        for expiry, time_years, theta in zip(expiries, times, thetas, strict=True):
            forward = 100.0 * np.exp(0.03 * time_years)
            summaries.append(ForwardSummary(expiry, forward, 0.01, 12, 2.0, True, "test"))
            k = np.log(strikes / forward)
            w = raw_ssvi(k, np.full_like(k, theta), rho, eta, gamma)
            iv = np.sqrt(w / time_years)
            for strike, volatility in zip(strikes, iv, strict=True):
                rows.append(
                    {
                        "expiration": expiry,
                        "strike": strike,
                        "side": "call",
                        "openInterest": 100.0,
                        "volume": 10.0,
                        "liquidity": 100.0,
                        "liquidity_source": "open_interest",
                        "iv": float(volatility),
                        "iv_source": "market_solved",
                        "is_iv_repaired": False,
                        "polarity": 1.0,
                        "mid_price": 1.0,
                        "quote_quality": 0.95,
                        "contract_quality": 0.95,
                        "days_to_expiry": time_years * 365.25,
                        "time_years": time_years,
                        "rate": 0.04,
                        "dividend_yield": 0.01,
                    }
                )
        quality = DataQualityReport(
            expirations_requested=3,
            expirations_processed=3,
            contracts_seen=len(rows),
            contracts_used=len(rows),
            valid_market_quotes=len(rows),
            iv_market_solved=len(rows),
            reported_oi_contracts=len(rows),
            valid_contracts=len(rows),
            parity_expirations=3,
        )
        return pd.DataFrame(rows), summaries, quality

    def test_ssvi_surface_calibrates_synthetic_data(self) -> None:
        contracts, summaries, quality = self._surface_contracts()
        fitted, surface = fit_volatility_surface(contracts, summaries, quality, spot=100.0)
        self.assertGreater(surface.fitted_contract_ratio, 0.90)
        self.assertIn(surface.mode, {"ssvi", "svi"})
        self.assertTrue(np.isfinite(fitted["surface_iv"]).all())
        self.assertLess(surface.weighted_fit_rmse, 0.02)

    def test_greek_vanna_matches_delta_finite_difference(self) -> None:
        strike = np.array([100.0])
        time = np.array([30 / 365.25])
        vol = np.array([0.25])
        rate = np.array([0.04])
        dividend = np.array([0.01])
        stack = greek_stack(
            "call",
            spot=100.0,
            strikes=strike,
            time_years=time,
            volatility=vol,
            risk_free_rate=rate,
            dividend_yield=dividend,
        )
        step = 1e-4
        plus = black_scholes_delta("call", 100.0, strike, time, vol + step, rate, dividend)
        minus = black_scholes_delta("call", 100.0, strike, time, vol - step, rate, dividend)
        numerical = (plus - minus) / (2 * step)
        self.assertAlmostEqual(float(stack.vanna[0]), float(numerical[0]), places=5)

    def test_american_put_not_below_european_put(self) -> None:
        european = float(
            black_scholes_price(
                "put",
                100.0,
                np.array([110.0]),
                np.array([0.5]),
                np.array([0.25]),
                np.array([0.05]),
                np.array([0.0]),
            )[0]
        )
        american = american_binomial_price(
            "put",
            spot=100.0,
            strike=110.0,
            time_years=0.5,
            volatility=0.25,
            risk_free_rate=0.05,
            dividend_yield=0.0,
            steps=300,
        )
        self.assertGreaterEqual(american + 1e-8, european)

    def test_wall_breadth_ignores_tiny_expiry_contributions(self) -> None:
        engine = ExposureAnalyticsEngine(
            DummyProvider(),
            min_expiry_share_for_breadth=0.05,
            min_contracts_to_render=0,
            min_usable_ratio=0.0,
            min_confidence_to_render=0.0,
        )
        records = pd.DataFrame(
            {
                "strike": [100.0, 100.0, 100.0, 105.0, 105.0, 105.0],
                "gex": [1_000_000.0, 10.0, 5.0, -500_000.0, -5.0, -2.0],
                "abs_gex": [1_000_000.0, 10.0, 5.0, 500_000.0, 5.0, 2.0],
                "cell_quality": [0.9] * 6,
                "expiry": ["08/21/2026", "09/18/2026", "12/18/2026"] * 2,
            }
        )
        totals, *_ = engine._build_structural_walls(records, 102.0)
        breadth = float(totals.loc[totals["strike"] == 100.0, "expiry_breadth"].iloc[0])
        self.assertLess(breadth, 1.0)
        self.assertAlmostEqual(breadth, 1 / 3)

    def test_branding_manager_includes_all_member_fallbacks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = BrandingManager(Path(directory))
            names = [asset.member.display_name for asset in manager.assets]
            self.assertEqual(
                names,
                [
                    "VIOLA TRADES",
                    "JP THE DEVELOPER",
                    "SIBEI SHIOK",
                    "JEFF THE SCRIPTER",
                ],
            )
            self.assertTrue(all(asset.is_placeholder for asset in manager.assets))


if __name__ == "__main__":
    unittest.main()
