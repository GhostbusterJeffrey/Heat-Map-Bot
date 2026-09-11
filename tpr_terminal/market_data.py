from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd

from .models import ExpirationChain, UnderlyingSnapshot

logger = logging.getLogger(__name__)


class MarketDataSession(ABC):
    provider_name: str
    provider_note: str

    @abstractmethod
    def underlying(self) -> UnderlyingSnapshot:
        raise NotImplementedError

    @abstractmethod
    def expirations(self) -> list[date]:
        raise NotImplementedError

    @abstractmethod
    def option_chain(self, expiration: date) -> ExpirationChain:
        raise NotImplementedError


class MarketDataProvider(ABC):
    @abstractmethod
    def open_session(self, ticker: str) -> MarketDataSession:
        raise NotImplementedError


class YahooFinanceProvider(MarketDataProvider):
    """Public-data fallback provider.

    The engine is intentionally provider-agnostic. A licensed OPRA/vendor adapter
    can replace this class without changing analytics, rendering, or Discord code.
    """

    def open_session(self, ticker: str) -> MarketDataSession:
        return _YahooSession(ticker)


class _YahooSession(MarketDataSession):
    provider_name = "Yahoo Finance via yfinance"
    provider_note = (
        "Public research/education fallback. Verify vendor terms and obtain a licensed "
        "feed before commercial redistribution or institutional positioning claims."
    )

    def __init__(self, ticker: str) -> None:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError(
                "yfinance is required for the Yahoo provider. Install requirements.txt."
            ) from exc
        self._symbol = ticker.strip().upper()
        self._ticker = yf.Ticker(self._symbol)
        self._underlying: UnderlyingSnapshot | None = None
        self._expirations: list[date] | None = None
        self._chains: dict[date, ExpirationChain] = {}

    def underlying(self) -> UnderlyingSnapshot:
        if self._underlying is not None:
            return self._underlying
        spot, observed_at = self._resolve_spot()
        self._underlying = UnderlyingSnapshot(
            ticker=self._symbol,
            spot=spot,
            as_of=observed_at,
            trailing_dividend_yield=self._trailing_dividend_yield(spot),
        )
        return self._underlying

    def expirations(self) -> list[date]:
        if self._expirations is not None:
            return list(self._expirations)
        parsed: list[date] = []
        for raw in self._ticker.options or ():
            try:
                parsed.append(datetime.strptime(str(raw), "%Y-%m-%d").date())
            except ValueError:
                logger.warning("Ignoring malformed expiry %r for %s", raw, self._symbol)
        self._expirations = sorted(set(parsed))
        return list(self._expirations)

    def option_chain(self, expiration: date) -> ExpirationChain:
        cached = self._chains.get(expiration)
        if cached is not None:
            return cached
        chain = self._ticker.option_chain(expiration.isoformat())
        result = ExpirationChain(
            expiration=expiration,
            calls=chain.calls.copy(),
            puts=chain.puts.copy(),
        )
        self._chains[expiration] = result
        return result

    def _resolve_spot(self) -> tuple[float, datetime]:
        try:
            info: Any = self._ticker.fast_info
            for key in ("lastPrice", "last_price", "regularMarketPrice"):
                try:
                    raw = info[key]
                except Exception:
                    raw = getattr(info, key, None)
                if raw is not None and float(raw) > 0:
                    return float(raw), datetime.now(UTC)
        except Exception as exc:
            logger.debug("fast_info failed for %s: %s", self._symbol, exc)

        attempts = (("5d", "1m"), ("1mo", "1d"))
        for period, interval in attempts:
            try:
                history = self._ticker.history(
                    period=period,
                    interval=interval,
                    auto_adjust=False,
                    actions=False,
                    repair=True,
                )
                close = pd.to_numeric(history.get("Close"), errors="coerce").dropna()
                if close.empty or float(close.iloc[-1]) <= 0:
                    continue
                timestamp = close.index[-1]
                if hasattr(timestamp, "to_pydatetime"):
                    observed = timestamp.to_pydatetime()
                    observed = (
                        observed.replace(tzinfo=UTC)
                        if observed.tzinfo is None
                        else observed.astimezone(UTC)
                    )
                else:
                    observed = datetime.now(UTC)
                return float(close.iloc[-1]), observed
            except Exception as exc:
                logger.debug(
                    "History lookup failed for %s (%s/%s): %s",
                    self._symbol,
                    period,
                    interval,
                    exc,
                )
        raise ValueError(f"Unable to resolve a valid spot price for {self._symbol}.")

    def _trailing_dividend_yield(self, spot: float) -> float:
        try:
            dividends = self._ticker.dividends
            if dividends is None or dividends.empty:
                return 0.0
            values = pd.to_numeric(dividends, errors="coerce").dropna()
            if values.empty:
                return 0.0
            index = values.index
            if getattr(index, "tz", None) is None:
                index = index.tz_localize(UTC)
            else:
                index = index.tz_convert(UTC)
            cutoff = datetime.now(UTC) - timedelta(days=365)
            trailing = values[index >= cutoff]
            annual_cash = float(trailing.sum())
            if annual_cash <= 0 or spot <= 0:
                return 0.0
            return min(max(annual_cash / spot, 0.0), 0.25)
        except Exception as exc:
            logger.debug("Dividend estimate failed for %s: %s", self._symbol, exc)
            return 0.0


class SnapshottingMarketDataProvider(MarketDataProvider):
    """Decorator that records every fetched option chain to an Alpha 4 store."""

    def __init__(self, provider: MarketDataProvider, store, *, model_version: str) -> None:
        self._provider = provider
        self._store = store
        self._model_version = model_version

    def open_session(self, ticker: str) -> MarketDataSession:
        inner = self._provider.open_session(ticker)
        return _SnapshottingSession(
            inner,
            self._store,
            ticker=ticker,
            model_version=self._model_version,
        )


class _SnapshottingSession(MarketDataSession):
    def __init__(self, inner: MarketDataSession, store, *, ticker: str, model_version: str) -> None:
        self._inner = inner
        self._store = store
        self.provider_name = inner.provider_name
        self.provider_note = inner.provider_note
        self._ticker = ticker
        self._model_version = model_version
        self._run = None
        self._underlying_written = False

    def _ensure_run(self):
        if self._run is None:
            self._run = self._store.begin_run(
                ticker=self._ticker,
                provider=self.provider_name,
                model_version=self._model_version,
            )
        return self._run

    def underlying(self) -> UnderlyingSnapshot:
        value = self._inner.underlying()
        if not self._underlying_written:
            run = self._ensure_run()
            self._store.update_underlying(
                run.run_id,
                source_as_of=value.as_of,
                spot=value.spot,
                trailing_dividend_yield=value.trailing_dividend_yield,
            )
            self._underlying_written = True
        return value

    def expirations(self) -> list[date]:
        return self._inner.expirations()

    def option_chain(self, expiration: date) -> ExpirationChain:
        chain = self._inner.option_chain(expiration)
        run = self._ensure_run()
        self._store.record_chain(
            run.run_id,
            expiration=expiration,
            calls=chain.calls,
            puts=chain.puts,
        )
        return chain
