from __future__ import annotations

import unittest
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd

from tpr_terminal.analytics import ExposureAnalyticsEngine
from tpr_terminal.confidence import build_confidence_report
from tpr_terminal.forward_engine import estimate_forward_from_put_call_parity
from tpr_terminal.market_data import MarketDataProvider, MarketDataSession
from tpr_terminal.math_engine import black_scholes_price, implied_volatility_brent
from tpr_terminal.models import DataQualityReport, ExpirationChain, RateCurve, UnderlyingSnapshot


class AutoSession(MarketDataSession):
    provider_name = "Synthetic"
    provider_note = "Synthetic"

    def underlying(self) -> UnderlyingSnapshot:
        return UnderlyingSnapshot("AUTO", 100.0, datetime(2026, 7, 13, 14, tzinfo=UTC), 0.01)

    def expirations(self) -> list[date]:
        return [date(2026, 7, 17)]

    def option_chain(self, expiration: date) -> ExpirationChain:
        base = {
            "strike": [95.0, 100.0, 105.0],
            "openInterest": [0, 200, 0],
            "volume": [30, 15, 25],
            "impliedVolatility": [0.30, 0.28, 0.31],
            "bid": [6.0, 3.0, 1.2],
            "ask": [6.4, 3.4, 1.5],
            "lastPrice": [6.2, 3.2, 1.35],
        }
        calls = pd.DataFrame(base)
        puts = pd.DataFrame(base)
        puts["bid"] = [1.1, 2.8, 6.0]
        puts["ask"] = [1.4, 3.2, 6.4]
        return ExpirationChain(expiration, calls, puts)


class AutoProvider(MarketDataProvider):
    def open_session(self, ticker: str) -> MarketDataSession:
        return AutoSession()


class V4FoundationTests(unittest.TestCase):
    def test_implied_vol_solver_recovers_known_volatility(self) -> None:
        price = float(
            black_scholes_price(
                option_kind="call",
                spot=100.0,
                strikes=100.0,
                time_years=30 / 365.25,
                volatility=0.27,
                risk_free_rate=0.04,
                dividend_yield=0.01,
            )
        )
        solved = implied_volatility_brent(
            option_kind="call",
            market_price=price,
            spot=100.0,
            strike=100.0,
            time_years=30 / 365.25,
            risk_free_rate=0.04,
            dividend_yield=0.01,
        )
        self.assertIsNotNone(solved)
        self.assertAlmostEqual(float(solved), 0.27, places=7)

    def test_put_call_parity_forward(self) -> None:
        spot = 100.0
        time_years = 0.25
        rate = 0.04
        dividend = 0.015
        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        calls = black_scholes_price(
            option_kind="call",
            spot=spot,
            strikes=strikes,
            time_years=time_years,
            volatility=0.25,
            risk_free_rate=rate,
            dividend_yield=dividend,
        )
        puts = black_scholes_price(
            option_kind="put",
            spot=spot,
            strikes=strikes,
            time_years=time_years,
            volatility=0.25,
            risk_free_rate=rate,
            dividend_yield=dividend,
        )
        call_frame = pd.DataFrame({"strike": strikes, "bid": calls - 0.02, "ask": calls + 0.02})
        put_frame = pd.DataFrame({"strike": strikes, "bid": puts - 0.02, "ask": puts + 0.02})
        estimate = estimate_forward_from_put_call_parity(
            call_frame,
            put_frame,
            spot=spot,
            time_years=time_years,
            risk_free_rate=rate,
        )
        expected_forward = spot * np.exp((rate - dividend) * time_years)
        self.assertIsNotNone(estimate)
        assert estimate is not None
        self.assertAlmostEqual(estimate.forward, expected_forward, places=5)
        self.assertAlmostEqual(estimate.dividend_yield, dividend, places=5)
        self.assertTrue(estimate.reliable)

    def test_auto_liquidity_uses_fallback_and_reports_it(self) -> None:
        curve = RateCurve((0.0, 3650.0), (0.04, 0.04), "test", date(2026, 7, 13))
        engine = ExposureAnalyticsEngine(AutoProvider(), min_contracts_to_render=1, min_usable_ratio=0.0, min_confidence_to_render=0.0)
        result = engine.analyze(
            "AUTO",
            max_expiries=1,
            liquidity_mode="auto",
            exposure_mode="net_proxy",
            rate_curve=curve,
            now=datetime(2026, 7, 13, 14, tzinfo=UTC),
        )
        self.assertGreater(result.quality.volume_fallback_contracts, 0)
        self.assertEqual(result.effective_liquidity_mode, "OI + volume fallback")
        self.assertGreater(float(result.matrix_records["gex"].abs().sum()), 0.0)

    def test_confidence_marks_sparse_data_low(self) -> None:
        quality = DataQualityReport(
            expirations_requested=6,
            expirations_processed=2,
            contracts_seen=1200,
            contracts_used=18,
            valid_market_quotes=100,
            iv_market_solved=10,
            iv_vendor_used=8,
            reported_oi_contracts=18,
            parity_expirations=0,
        )
        report = build_confidence_report(quality)
        self.assertLess(report.score, 40)
        self.assertIn(report.label, {"Low", "Very low"})


if __name__ == "__main__":
    unittest.main()
