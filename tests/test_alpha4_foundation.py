from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import pandas as pd

from tpr_terminal.contract_analytics import ContractAnalyticsEngine
from tpr_terminal.market_data import MarketDataProvider, MarketDataSession
from tpr_terminal.models import ExpirationChain, RateCurve, UnderlyingSnapshot
from tpr_terminal.pricing import black_scholes_price, greek_stack
from tpr_terminal.renderer import HeatmapRenderer
from tpr_terminal.snapshot_store import OptionSnapshotStore


class _FakeSession(MarketDataSession):
    provider_name = "Synthetic test feed"
    provider_note = "Synthetic"

    def __init__(self) -> None:
        self.expiry = date(2026, 8, 21)
        self.spot = 102.0
        self.rate = 0.04
        self.dividend = 0.01
        now = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
        self._underlying = UnderlyingSnapshot("TEST", self.spot, now, self.dividend)
        t = 38.0 / 365.25
        strikes = np.asarray([95.0, 100.0, 105.0])
        vol = np.asarray([0.24, 0.22, 0.23])
        common = {
            "strikes": strikes,
            "time_years": np.full(3, t),
            "volatility": vol,
            "risk_free_rate": np.full(3, self.rate),
            "dividend_yield": np.full(3, self.dividend),
        }
        calls = black_scholes_price("call", self.spot, **common)
        puts = black_scholes_price("put", self.spot, **common)
        self._chain = ExpirationChain(
            expiration=self.expiry,
            calls=self._frame(strikes, calls, vol, "C"),
            puts=self._frame(strikes, puts, vol, "P"),
        )

    @staticmethod
    def _frame(strikes, prices, vol, code):
        return pd.DataFrame(
            {
                "contractSymbol": [f"TEST260821{code}{int(k * 1000):08d}" for k in strikes],
                "strike": strikes,
                "bid": np.maximum(prices - 0.03, 0.01),
                "ask": prices + 0.03,
                "lastPrice": prices,
                "openInterest": [1000, 2500, 1300],
                "volume": [100, 200, 150],
                "impliedVolatility": vol,
            }
        )

    def underlying(self) -> UnderlyingSnapshot:
        return self._underlying

    def expirations(self) -> list[date]:
        return [self.expiry]

    def option_chain(self, expiration: date) -> ExpirationChain:
        assert expiration == self.expiry
        return self._chain


class _FakeProvider(MarketDataProvider):
    def open_session(self, ticker: str) -> MarketDataSession:
        return _FakeSession()


def test_alpha4_greek_stack_includes_all_requested_orders() -> None:
    stack = greek_stack(
        "call",
        spot=100.0,
        strikes=np.asarray([100.0]),
        time_years=np.asarray([30.0 / 365.25]),
        volatility=np.asarray([0.20]),
        risk_free_rate=np.asarray([0.04]),
        dividend_yield=np.asarray([0.01]),
    )
    for name in (
        "delta",
        "gamma",
        "theta",
        "vega",
        "rho",
        "vanna",
        "charm",
        "vomma",
        "speed",
        "color",
        "zomma",
        "ultima",
    ):
        value = float(getattr(stack, name)[0])
        assert np.isfinite(value), name


def test_contract_report_has_quote_interval_and_units() -> None:
    engine = ContractAnalyticsEngine(_FakeProvider(), contract_multiplier=100)
    curve = RateCurve((1.0, 365.0), (0.04, 0.04), "test", date(2026, 7, 14))
    report = engine.analyze(
        "TEST",
        expiration=date(2026, 8, 21),
        strike=100.0,
        option_side="call",
        quantity=2,
        position="long",
        scope="custom_position",
        rate_curve=curve,
        now=datetime(2026, 7, 14, 12, 0, tzinfo=UTC),
    )
    assert report.scale_factor == 200.0
    assert report.implied_volatility_bid is not None
    assert report.implied_volatility_ask is not None
    assert report.values["gamma"].low is not None
    assert "requested position" in report.values["gamma"].unit
    assert np.isfinite(report.values["ultima"].mid)
    assert report.validation_error < 0.03


def test_snapshot_store_round_trip(tmp_path) -> None:
    store = OptionSnapshotStore(tmp_path / "history.sqlite3", retention_days=30)
    run = store.begin_run(ticker="TEST", provider="Synthetic", model_version="4.0-test")
    store.update_underlying(
        run.run_id,
        source_as_of=datetime(2026, 7, 14, tzinfo=UTC),
        spot=100.0,
        trailing_dividend_yield=0.01,
    )
    calls = pd.DataFrame({"strike": [100.0], "openInterest": [123]})
    puts = pd.DataFrame({"strike": [100.0], "openInterest": [456]})
    expiry = date(2026, 8, 21)
    store.record_chain(run.run_id, expiration=expiry, calls=calls, puts=puts)
    loaded_calls, loaded_puts = store.load_chain(run.run_id, expiry)
    assert int(loaded_calls.loc[0, "openInterest"]) == 123
    assert int(loaded_puts.loc[0, "openInterest"]) == 456
    stats = store.stats()
    assert stats.runs == 1
    assert stats.chains == 1
    assert stats.contracts == 2


def test_micro_exposure_opacity_is_monotonic_and_absolute() -> None:
    values = [0.0, 5_000.0, 50_000.0, 250_000.0, 500_000.0, 5_000_000.0]
    alpha = [HeatmapRenderer._magnitude_alpha(value) for value in values]
    assert alpha == sorted(alpha)
    assert alpha[0] >= 0.08
    assert alpha[-1] == 1.0
    assert HeatmapRenderer._color_scale(np.asarray([10_000.0, 1e10]))["floor"] == 5_000.0


def test_snapshot_store_compares_open_interest_changes(tmp_path) -> None:
    store = OptionSnapshotStore(tmp_path / "history.sqlite3", retention_days=30)
    expiry = date(2026, 8, 21)
    old = store.begin_run(
        ticker="TEST",
        provider="Synthetic",
        model_version="4.0-test",
        captured_at=datetime(2026, 7, 14, 12, 0, tzinfo=UTC),
    )
    new = store.begin_run(
        ticker="TEST",
        provider="Synthetic",
        model_version="4.0-test",
        captured_at=datetime(2026, 7, 15, 12, 0, tzinfo=UTC),
    )
    old_calls = pd.DataFrame(
        {"contractSymbol": ["TESTC100"], "strike": [100.0], "openInterest": [100]}
    )
    new_calls = pd.DataFrame(
        {"contractSymbol": ["TESTC100"], "strike": [100.0], "openInterest": [150]}
    )
    old_puts = pd.DataFrame(
        {"contractSymbol": ["TESTP100"], "strike": [100.0], "openInterest": [200]}
    )
    new_puts = pd.DataFrame(
        {"contractSymbol": ["TESTP100"], "strike": [100.0], "openInterest": [175]}
    )
    store.record_chain(old.run_id, expiration=expiry, calls=old_calls, puts=old_puts)
    store.record_chain(new.run_id, expiration=expiry, calls=new_calls, puts=new_puts)
    comparison = store.compare_latest("TEST", expiration=expiry)
    assert comparison.call_oi_change == 50
    assert comparison.put_oi_change == -25
    assert comparison.total_oi_change == 25
    assert comparison.top_increases[0].strike == 100.0
    assert comparison.top_decreases[0].side == "put"


def test_alpha4_discord_commands_register(monkeypatch) -> None:
    import asyncio
    import importlib.util

    if importlib.util.find_spec("discord") is None:
        return
    monkeypatch.setenv("SNAPSHOT_ENABLED", "false")
    from tpr_terminal.bot import TPRTerminalBot, TerminalCog
    from tpr_terminal.settings import Settings

    async def inspect() -> set[str]:
        bot = TPRTerminalBot(Settings.from_env())
        await bot.add_cog(TerminalCog(bot))
        names = {command.name for command in bot.tree.get_commands()}
        await bot.close()
        return names

    names = asyncio.run(inspect())
    assert {"heatmap", "greek", "greeks", "oichange", "health"}.issubset(names)
