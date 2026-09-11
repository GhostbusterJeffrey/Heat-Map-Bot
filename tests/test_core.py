from __future__ import annotations

import unittest
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd

from tpr_terminal.analytics import ExposureAnalyticsEngine
from tpr_terminal.market_calendar import USMarketCalendar
from tpr_terminal.market_data import MarketDataProvider, MarketDataSession
from tpr_terminal.math_engine import black_scholes_gamma, repair_implied_volatility
from tpr_terminal.models import ExpirationChain, RateCurve, UnderlyingSnapshot
from tpr_terminal.rates import TreasuryCurveService
from tpr_terminal.renderer import HeatmapRenderer


class FakeSession(MarketDataSession):
    provider_name = "Synthetic Test Feed"
    provider_note = "Synthetic test data"

    def __init__(self, zero_oi: bool = False) -> None:
        self.zero_oi = zero_oi
        self.expiry = date(2026, 7, 17)

    def underlying(self) -> UnderlyingSnapshot:
        return UnderlyingSnapshot("TEST", 100.0, datetime(2026, 7, 13, 14, tzinfo=UTC), 0.01)

    def expirations(self) -> list[date]:
        return [self.expiry]

    def option_chain(self, expiration: date) -> ExpirationChain:
        oi = [0, 0, 0] if self.zero_oi else [100, 300, 150]
        base = {
            "strike": [95.0, 100.0, 105.0],
            "openInterest": oi,
            "volume": [10, 20, 15],
            "impliedVolatility": [0.28, np.nan, 0.31],
            "bid": [6.0, 3.0, 1.3],
            "ask": [6.4, 3.4, 1.6],
            "lastPrice": [6.2, 3.2, 1.45],
        }
        calls = pd.DataFrame(base)
        puts = pd.DataFrame(base)
        puts["bid"] = [1.1, 2.8, 6.0]
        puts["ask"] = [1.4, 3.2, 6.4]
        puts["lastPrice"] = [1.25, 3.0, 6.2]
        return ExpirationChain(expiration, calls, puts)


class FakeProvider(MarketDataProvider):
    def __init__(self, zero_oi: bool = False) -> None:
        self.zero_oi = zero_oi

    def open_session(self, ticker: str) -> MarketDataSession:
        return FakeSession(self.zero_oi)


class CoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.curve = RateCurve((0.0, 3650.0), (0.045, 0.045), "test", date(2026, 7, 13))
        self.now = datetime(2026, 7, 13, 14, tzinfo=UTC)

    def test_black_scholes_gamma_is_finite_positive(self) -> None:
        gamma = black_scholes_gamma(
            spot=100.0,
            strikes=np.array([100.0]),
            time_years=30 / 365.25,
            volatility=np.array([0.25]),
            risk_free_rate=0.045,
            dividend_yield=0.01,
        )
        self.assertTrue(np.isfinite(gamma[0]))
        self.assertGreater(gamma[0], 0.0)

    def test_iv_repair(self) -> None:
        strikes = pd.Series([90.0, 100.0, 110.0])
        iv, repaired = repair_implied_volatility(strikes, pd.Series([0.3, np.nan, 0.4]), 100.0)
        self.assertEqual(repaired, 1)
        self.assertTrue(np.isfinite(iv).all())
        self.assertGreater(iv.iloc[1], 0.3)
        self.assertLess(iv.iloc[1], 0.4)

    def test_2026_day_after_thanksgiving_early_close(self) -> None:
        calendar = USMarketCalendar()
        close = calendar.session_close(date(2026, 11, 27))
        self.assertTrue(close.early_close)
        self.assertEqual(close.close_utc.hour, 18)  # 1pm EST

    def test_analysis_and_expected_move(self) -> None:
        engine = ExposureAnalyticsEngine(FakeProvider(), min_contracts_to_render=1, min_usable_ratio=0.0, min_confidence_to_render=0.0)
        result = engine.analyze(
            "TEST",
            max_expiries=1,
            liquidity_mode="open_interest",
            exposure_mode="net_proxy",
            rate_curve=self.curve,
            now=self.now,
        )
        self.assertEqual(result.quality.expirations_processed, 1)
        self.assertFalse(result.matrix_records.empty)
        self.assertEqual(len(result.expected_moves), 1)
        self.assertGreater(result.expected_moves[0].dollars, 0)
        self.assertIn("expiry", result.matrix_records.columns)

    def test_zero_exposure_expiry_is_retained(self) -> None:
        engine = ExposureAnalyticsEngine(FakeProvider(zero_oi=True), min_contracts_to_render=0, min_usable_ratio=0.0, min_confidence_to_render=0.0)
        result = engine.analyze(
            "TEST",
            max_expiries=1,
            liquidity_mode="open_interest",
            exposure_mode="net_proxy",
            rate_curve=self.curve,
            now=self.now,
        )
        self.assertEqual(result.quality.expirations_processed, 1)
        self.assertEqual(float(result.matrix_records["gex"].abs().sum()), 0.0)

    def test_renderer_outputs_png(self) -> None:
        engine = ExposureAnalyticsEngine(FakeProvider(), min_contracts_to_render=1, min_usable_ratio=0.0, min_confidence_to_render=0.0)
        result = engine.analyze(
            "TEST",
            max_expiries=1,
            liquidity_mode="open_interest",
            exposure_mode="net_proxy",
            rate_curve=self.curve,
            now=self.now,
        )
        renderer = HeatmapRenderer(
            main_logo_path=None,
            developer_logo_path=None,
            trader_logo_path=None,
        )
        payload, displayed = renderer.render(result, strikes_count=5)
        self.assertTrue(payload.startswith(b"\x89PNG"))
        self.assertGreaterEqual(displayed, 3)

    def test_treasury_xml_parser(self) -> None:
        xml = b"""<?xml version='1.0'?>
        <feed xmlns:m='x'><entry><content><m:properties>
        <m:NEW_DATE>2026-07-10T00:00:00</m:NEW_DATE>
        <m:BC_1MONTH>4.50</m:BC_1MONTH><m:BC_3MONTH>4.40</m:BC_3MONTH>
        </m:properties></content></entry></feed>"""
        service = TreasuryCurveService(enabled=True, static_fallback_rate=0.04, ttl_seconds=60)
        curve = service.parse_xml(xml)
        self.assertEqual(curve.as_of, date(2026, 7, 10))
        self.assertEqual(len(curve.tenor_days), 2)
        self.assertFalse(curve.used_fallback)


if __name__ == "__main__":
    unittest.main()
