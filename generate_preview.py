from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from tpr_terminal.analytics import ExposureAnalyticsEngine
from tpr_terminal.market_data import MarketDataProvider, MarketDataSession
from tpr_terminal.models import ExpirationChain, RateCurve, UnderlyingSnapshot
from tpr_terminal.pricing import black_scholes_price
from tpr_terminal.renderer import HeatmapRenderer

SPOT = 749.17
NOW = datetime(2026, 7, 13, 15, 30, tzinfo=UTC)


class PreviewSession(MarketDataSession):
    provider_name = "Deterministic Alpha 4.0 preview feed"
    provider_note = "Synthetic data used only for build validation."

    def __init__(self) -> None:
        self._expiries = [date(2026, 7, 17) + timedelta(days=7 * index) for index in range(12)]

    def underlying(self) -> UnderlyingSnapshot:
        return UnderlyingSnapshot("SPY", SPOT, NOW, 0.011)

    def expirations(self) -> list[date]:
        return self._expiries

    def option_chain(self, expiration: date) -> ExpirationChain:
        days = max((expiration - NOW.date()).days + 0.25, 0.5)
        t = days / 365.25
        strikes = np.arange(700.0, 801.0, 2.0)
        k = np.log(strikes / (SPOT * np.exp((0.04 - 0.011) * t)))
        total_var = (0.18 + 0.06 * np.sqrt(t)) ** 2 * t
        smile = total_var * (1.0 + 0.70 * np.maximum(-k, 0) + 0.30 * k * k / 0.02)
        iv = np.sqrt(np.maximum(smile, 1e-8) / t)
        calls_price = black_scholes_price("call", SPOT, strikes, t, iv, 0.04, 0.011)
        puts_price = black_scholes_price("put", SPOT, strikes, t, iv, 0.04, 0.011)
        expiry_index = self._expiries.index(expiration)
        decay = np.exp(-expiry_index / 8.0)
        call_oi = (
            250
            + 9_000 * decay * np.exp(-0.5 * ((strikes - 760) / 3.2) ** 2)
            + 2_000 * np.exp(-0.5 * ((strikes - 775) / 5.0) ** 2)
        )
        put_oi = (
            300
            + 10_500 * decay * np.exp(-0.5 * ((strikes - 740) / 3.5) ** 2)
            + 3_200 * np.exp(-0.5 * ((strikes - 725) / 5.5) ** 2)
        )
        rng = np.random.default_rng(42 + expiry_index)
        call_oi *= rng.uniform(0.85, 1.15, size=len(strikes))
        put_oi *= rng.uniform(0.85, 1.15, size=len(strikes))

        def frame(prices: np.ndarray, oi: np.ndarray) -> pd.DataFrame:
            spread = np.maximum(0.03, 0.025 * np.maximum(prices, 0.5))
            return pd.DataFrame(
                {
                    "strike": strikes,
                    "openInterest": np.round(oi),
                    "volume": np.round(np.maximum(5, oi * rng.uniform(0.02, 0.08, size=len(oi)))),
                    "impliedVolatility": iv,
                    "bid": np.maximum(prices - spread / 2, 0.01),
                    "ask": prices + spread / 2,
                    "lastPrice": prices,
                }
            )

        return ExpirationChain(expiration, frame(calls_price, call_oi), frame(puts_price, put_oi))


class PreviewProvider(MarketDataProvider):
    def open_session(self, ticker: str) -> MarketDataSession:
        return PreviewSession()


if __name__ == "__main__":
    curve = RateCurve((0.0, 3650.0), (0.04, 0.04), "synthetic", NOW.date())
    engine = ExposureAnalyticsEngine(
        PreviewProvider(),
        min_contracts_to_render=1,
        min_usable_ratio=0.0,
        min_confidence_to_render=0.0,
    )
    result = engine.analyze(
        "SPY",
        max_expiries=12,
        liquidity_mode="auto",
        exposure_mode="net_proxy",
        rate_curve=curve,
        now=NOW,
    )
    renderer = HeatmapRenderer(assets_dir=Path(__file__).resolve().parent / "assets")
    payload, _ = renderer.render(result, 51)
    output = Path(__file__).resolve().parent / "preview_v4_alpha4.png"
    output.write_bytes(payload)
    print(output)
    print(result.surface.mode, result.surface.score, result.confidence.score)
