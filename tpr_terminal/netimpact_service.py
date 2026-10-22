"""Shared application path used by the Discord command and local preview CLI."""
from __future__ import annotations

import math
import threading
import time
from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from .market_calendar import USMarketCalendar
from .netimpact_config import NetImpactSettings

ET = ZoneInfo("America/New_York")
SCOPE = "20-name maximum watchlist | nearest available expiry within 7 days | 11 strikes | calls + puts"


def _number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def _stamp(value: Any) -> datetime | None:
    number = _number(value)
    if number is None or number <= 0:
        return None
    try:
        return datetime.fromtimestamp(number / 1000 if number > 100_000_000_000 else number, UTC)
    except (ValueError, OSError, OverflowError):
        return None


def close_estimate_report(universe: dict, session_date: date, now: datetime | None = None) -> dict:
    """Unsigned daily volume valued at final observed mark; never execution flow."""
    now = now or datetime.now(UTC)
    metadata = universe.get("metadata", {})
    contracts = universe.get("contracts", [])
    tickers = metadata.get("tickers") or list(dict.fromkeys(c["ticker"] for c in contracts))
    marks: list[datetime] = []
    rows = []
    session_close = USMarketCalendar().session_close(session_date).close_utc
    mark_start, mark_end = session_close - timedelta(minutes=15), session_close + timedelta(minutes=15)
    for ticker in tickers:
        selected = [c for c in contracts if c["ticker"] == ticker]
        calls = puts = 0.0
        missing = 0
        valued = 0
        volume_total = 0
        for contract in selected:
            volume = _number(contract.get("volume"))
            multiplier = _number(contract.get("contract_size"))
            bid, ask = _number(contract.get("bid")), _number(contract.get("ask"))
            bid_time, ask_time = _stamp(contract.get("bid_date")), _stamp(contract.get("ask_date"))
            valid_quote = (bid is not None and ask is not None and 0 <= bid <= ask and ask > 0
                           and bid_time is not None and ask_time is not None
                           and bid_time.astimezone(ET).date() == session_date
                           and ask_time.astimezone(ET).date() == session_date
                           and mark_start <= bid_time <= mark_end
                           and mark_start <= ask_time <= mark_end)
            if volume is None or volume < 0 or volume != int(volume) or multiplier is None or multiplier <= 0:
                missing += 1
                continue
            if volume == 0:
                valued += 1
                continue
            if not valid_quote:
                missing += 1
                continue
            premium = volume * ((bid + ask) / 2) * multiplier
            if contract.get("option_type") == "call":
                calls += premium
            elif contract.get("option_type") == "put":
                puts += premium
            else:
                missing += 1
                continue
            marks.extend((bid_time, ask_time))
            volume_total += int(volume)
            valued += 1
        # A partly priced ticker cannot silently compete with fully priced names.
        available = bool(selected) and missing == 0 and valued == len(selected)
        rows.append({"ticker": ticker, "net_call": calls if available else None,
                     "net_put": puts if available else None, "net": calls - puts if available else None,
                     "contracts": len(selected), "trades": None, "volume": volume_total,
                     "excluded": missing, "available": available,
                     "status": "estimated" if available else "missing marks or contracts"})
    source_at = max(marks).isoformat() if marks else None
    source_from = min(marks).isoformat() if marks else None
    return {
        "session_date": session_date.isoformat(), "window": "session", "mode": "close_estimate",
        "start_at": datetime.combine(session_date, datetime.min.time(), ET).replace(hour=9, minute=30).astimezone(UTC).isoformat(),
        "as_of": source_at, "source_as_of": source_at, "source_from": source_from,
        "fetched_at": now.isoformat(), "status": "estimate — trade direction unknown",
        "scope": metadata.get("scope", SCOPE), "subscribed_contracts": len(contracts), "rows": rows,
        "notes": [
            "Daily contract volume × final observed bid/ask midpoint; not executed premium or directional net flow.",
            "Only quotes within 15 minutes of regular close are valued; older marks are unavailable.",
            "Marks are asynchronous and may include later-closing ETF options. Expired contracts may be unavailable.",
            "After-close selection is not a reconstruction of a universe recorded at the open.",
            "Tickers with missing marks or contracts are unavailable, never zero.",
        ],
        "universe_metadata": metadata,
        "source_contracts": contracts,
    }


class NetImpactService:
    def __init__(self, settings: NetImpactSettings | None = None):
        self.settings = settings or NetImpactSettings.from_env()
        self._lock = threading.RLock()
        self._close_cache: tuple[float, dict] | None = None

    def flow_report(self, window: str = "session", session_date: date | None = None, now: datetime | None = None) -> dict:
        from .netimpact_core import FlowStore
        if window not in {"session", "15m", "60m"}:
            raise ValueError("Choose session, 15m, or 60m.")
        now = now or datetime.now(UTC)
        local = now.astimezone(ET)
        calendar = USMarketCalendar()
        target = session_date or local.date()
        if session_date is None:
            if calendar.is_session(target) and (local.hour < 9 or (local.hour == 9 and local.minute < 30)):
                target -= timedelta(days=1)
            target = calendar.previous_session(target)
        store = FlowStore(self.settings.db_path)
        try:
            report = store.snapshot(target.isoformat(), window=window, as_of=now)
        finally:
            store.close()
        if not report.get("rows"):
            report["rows"] = [{"ticker": name, "net": None, "net_call": None, "net_put": None, "available": False, "status": "not collected", "contracts": 0, "trades": 0, "excluded": 0} for name in self.settings.tickers]
        report.setdefault("scope", SCOPE)
        return report

    def close_report(self, now: datetime | None = None) -> dict:
        from .tradier_flow import TradierClient, select_universe
        now = now or datetime.now(UTC)
        with self._lock:
            local = now.astimezone(ET)
            calendar = USMarketCalendar()
            session_date = local.date()
            if not calendar.is_session(session_date):
                raise ValueError("The close preview requires same-session quotes. Use a saved report on non-trading days.")
            close = calendar.session_close(session_date).close_utc
            if now < close:
                raise ValueError("Close preview is available after the session closes. Live mode uses recorded executions.")
            if self._close_cache and self._close_cache[1]["session_date"] == session_date.isoformat() and time.monotonic() - self._close_cache[0] < 300:
                return self._close_cache[1]
            client = TradierClient(self.settings.read_token())
            universe = select_universe(client, list(self.settings.tickers), session_date, spot_mode="close")
            report = close_estimate_report(universe, session_date, now)
            self._close_cache = (time.monotonic(), report)
            return report

    def report(self, window: str = "session", mode: str = "flow") -> dict:
        if mode == "close_estimate":
            if window != "session":
                raise ValueError("Close activity estimates only support the session window.")
            return self.close_report()
        if mode != "flow":
            raise ValueError("Choose flow or close_estimate.")
        return self.flow_report(window)

    def render(self, report: dict) -> bytes:
        from .netimpact_renderer import render_netimpact
        return render_netimpact(report)
