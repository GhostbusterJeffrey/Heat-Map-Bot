"""Market-data-only Tradier transport and a bounded near-term collector.

No account or trading endpoints are exposed. Credentials and session identifiers
are never included in raised errors, persisted collector state, or log output.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime
from datetime import time as day_time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .market_calendar import USMarketCalendar

NY = ZoneInfo("America/New_York")
API = "https://api.tradier.com/v1"
STREAM = "https://stream.tradier.com/v1/markets/events"
SYMBOL = re.compile(r"^[A-Za-z0-9.^_\-/]{1,64}$")
OPTION = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")


class TradierDataError(RuntimeError):
    """Sanitized provider failure; safe to show in the bot's health output."""

    def __init__(self, category: str, status: int | None = None) -> None:
        self.category = category
        self.status = status
        super().__init__(f"Tradier market data: {category}" + (f" (HTTP {status})" if status else ""))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _many(value: Any) -> list[dict]:
    if isinstance(value, dict):
        return [value]
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _symbols(values: list[str]) -> str:
    if not values or any(not isinstance(value, str) or not SYMBOL.fullmatch(value) for value in values):
        raise ValueError("Invalid market-data symbol list")
    return ",".join(values)


def _date_time_parameter(value: datetime | str) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("Historical bounds must be timezone-aware")
        return value.astimezone(NY).strftime("%Y-%m-%d %H:%M")
    datetime.strptime(value, "%Y-%m-%d %H:%M")
    return value


class TradierClient:
    """Production market data with fixed hosts, verified TLS and finite waits."""

    _PATHS = frozenset({"/markets/clock", "/markets/quotes", "/markets/options/expirations", "/markets/options/chains", "/markets/timesales", "/markets/history", "/markets/events/session"})

    def __init__(self, token: str, *, timeout: float = 10.0, opener=None, min_interval: float = 0.55) -> None:
        if not token or any(char.isspace() for char in token):
            raise ValueError("A production market-data token is required")
        self._headers = {"Accept": "application/json", "Authorization": "Bearer " + token}
        self.timeout = max(1.0, min(float(timeout), 30.0))
        self._opener = opener or urllib.request.build_opener(_NoRedirect())
        self._min_interval = max(0.0, float(min_interval))
        self._request_lock = threading.Lock()
        self._last_request = 0.0

    def _pace(self) -> None:
        with self._request_lock:
            wait = self._last_request + self._min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()

    def _json(self, path: str, params: dict | None = None, *, method: str = "GET") -> dict:
        if path not in self._PATHS or (method != "GET" and (path != "/markets/events/session" or method != "POST")):
            raise ValueError("Only approved market-data endpoints are supported")
        url = API + path + ("?" + urllib.parse.urlencode(params) if params else "")
        request = urllib.request.Request(url, headers=self._headers, method=method, data=b"" if method == "POST" else None)
        for attempt in range(3):
            self._pace()
            try:
                with self._opener.open(request, timeout=self.timeout) as response:
                    data = json.load(response)
                if not isinstance(data, dict) or "errors" in data or "fault" in data or "error" in data:
                    raise TradierDataError("invalid_response")
                return data
            except TradierDataError:
                raise
            except urllib.error.HTTPError as exc:
                status = exc.code
                exc.close()
                if (status == 429 or status >= 500) and attempt < 2:
                    time.sleep(2 ** (attempt + 1))
                    continue
                raise TradierDataError("request_rejected", status) from None
            except TimeoutError:
                raise TradierDataError("timeout") from None
            except (urllib.error.URLError, OSError):
                raise TradierDataError("connection_failed") from None
            except (ValueError, UnicodeError):
                raise TradierDataError("invalid_response") from None
        raise TradierDataError("retry_limit")

    def clock(self) -> dict:
        value = self._json("/markets/clock").get("clock")
        if not isinstance(value, dict):
            raise TradierDataError("missing_clock")
        return value

    def quotes(self, symbols: list[str]) -> list[dict]:
        return _many((self._json("/markets/quotes", {"symbols": _symbols(symbols)}).get("quotes") or {}).get("quote"))

    def expirations(self, ticker: str) -> list[str]:
        result = (self._json("/markets/options/expirations", {"symbol": _symbols([ticker]), "includeAllRoots": "false"}).get("expirations") or {}).get("date", [])
        if isinstance(result, str):
            result = [result]
        return sorted({value for value in result if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)}) if isinstance(result, list) else []

    def chain(self, ticker: str, expiry: str | date) -> list[dict]:
        value = str(expiry)
        date.fromisoformat(value)
        result = self._json("/markets/options/chains", {"symbol": _symbols([ticker]), "expiration": value, "greeks": "false"})
        return _many((result.get("options") or {}).get("option"))

    def timesales(self, symbol: str, start: datetime | str, end: datetime | str, interval: str = "tick") -> list[dict]:
        if interval not in {"tick", "1min", "5min", "15min"}:
            raise ValueError("Unsupported time-and-sales interval")
        params = {"symbol": _symbols([symbol]), "start": _date_time_parameter(start), "end": _date_time_parameter(end), "interval": interval, "session_filter": "open"}
        # Tick history may lack contemporaneous quotes. Callers must inspect the
        # actual schema instead of assuming these records are classifiable flow.
        return _many((self._json("/markets/timesales", params).get("series") or {}).get("data"))

    def daily_history(self, symbol: str, session_date: date | str) -> dict | None:
        session = str(session_date)
        date.fromisoformat(session)
        payload = self._json("/markets/history", {"symbol": _symbols([symbol]), "interval": "daily", "start": session, "end": session})
        rows = _many((payload.get("history") or {}).get("day"))
        return next((row for row in rows if row.get("date") == session), None)

    def iter_events(self, symbols: list[str], stop_event: threading.Event, on_connected: Callable[[], None] | None = None) -> Iterator[dict]:
        symbol_list = _symbols(symbols)
        if stop_event.is_set():
            return
        session = self._json("/markets/events/session", method="POST")
        session_id = (session.get("stream") or {}).get("sessionid")
        if not isinstance(session_id, str) or not session_id:
            raise TradierDataError("missing_stream_session")
        payload = urllib.parse.urlencode({"sessionid": session_id, "symbols": symbol_list, "filter": "timesale,quote,summary", "linebreak": "true", "validOnly": "false", "advancedDetails": "true"}).encode("ascii")
        headers = dict(self._headers, **{"Content-Type": "application/x-www-form-urlencoded"})
        request = urllib.request.Request(STREAM, data=payload, headers=headers, method="POST")
        try:
            self._pace()
            with self._opener.open(request, timeout=self.timeout) as response:
                if on_connected:
                    on_connected()
                while not stop_event.is_set():
                    line = response.readline(1_048_577)
                    if not line:
                        raise TradierDataError("stream_closed")
                    if len(line) > 1_048_576:
                        raise TradierDataError("oversized_stream_event")
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except (ValueError, UnicodeError):
                        raise TradierDataError("invalid_stream_event") from None
                    if not isinstance(event, dict):
                        raise TradierDataError("invalid_stream_event")
                    if "error" in event or "errors" in event:
                        raise TradierDataError("stream_rejected")
                    if event.get("type") in {"quote", "timesale", "summary"}:
                        yield event
        except TradierDataError:
            raise
        except urllib.error.HTTPError as exc:
            raise TradierDataError("stream_rejected", exc.code) from None
        except TimeoutError:
            raise TradierDataError("stream_idle_timeout") from None
        except (urllib.error.URLError, OSError):
            raise TradierDataError("stream_connection_failed") from None


def _standard_contract(row: dict, ticker: str, expiry: date) -> bool:
    match = OPTION.fullmatch(str(row.get("symbol", "")))
    normalized_root = re.sub(r"[.\-]", "", ticker)
    if not match or match.group(1) != normalized_root:
        return False
    if match.group(2) != expiry.strftime("%y%m%d") or row.get("expiration_date") != expiry.isoformat():
        return False
    expected_side = "call" if match.group(3) == "C" else "put"
    if row.get("option_type") != expected_side or _number(row.get("contract_size")) != 100:
        return False
    strike = _number(row.get("strike"))
    if strike is None or strike <= 0 or abs(int(match.group(4)) / 1000 - strike) > 0.00001:
        return False
    return not row.get("adjusted", False)


def select_universe(client: TradierClient, tickers: list[str], session_date: date, spot_mode: str = "current") -> dict:
    """Select one <=7-DTE expiry and up to 11 paired strikes per ticker.

    A date-matched daily close is preferred for an after-close preview. If absent,
    last is accepted only when its timestamp belongs to that regular session.
    This is still a post-close selection, not a reconstructed opening universe.
    """
    if spot_mode not in {"current", "close"}:
        raise ValueError("spot_mode must be current or close")
    normalized = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
    _symbols(normalized)
    contracts: list[dict] = []
    details: dict[str, dict] = {}
    metadata = {"tickers": normalized, "scope": "Nearest available expiry within 7 calendar days; up to 11 paired strikes per ticker", "collection_start": datetime.now(UTC).isoformat(), "selection_mode": spot_mode, "selection_as_of": datetime.now(UTC).isoformat(), "ticker_details": details, "max_dte": 7, "strikes_per_ticker": 11}
    try:
        quote_map = {row.get("symbol"): row for row in client.quotes(normalized)}
    except TradierDataError as exc:
        for ticker in normalized:
            details[ticker] = {"status": "unavailable", "reason": exc.category}
        return {"contracts": contracts, "metadata": metadata}
    calendar = USMarketCalendar()
    regular_close = calendar.session_close(session_date).close_utc.astimezone(NY)
    for ticker in normalized:
        detail = details[ticker] = {"status": "unavailable"}
        quote = quote_map.get(ticker, {})
        spot = _number(quote.get("last")) if spot_mode == "current" else None
        detail["spot_basis"] = "last"
        detail["quote_trade_date"] = quote.get("trade_date")
        if spot_mode == "close":
            try:
                history = client.daily_history(ticker, session_date)
            except TradierDataError as exc:
                history = None
                detail["history_error"] = exc.category
            spot = _number((history or {}).get("close"))
            detail["spot_basis"] = "dated_daily_close"
            detail["spot_session_date"] = session_date.isoformat() if spot and spot > 0 else None
        if spot_mode == "close" and (spot is None or spot <= 0):
            timestamp = _number(quote.get("trade_date"))
            trade_time = datetime.fromtimestamp(timestamp / 1000, UTC).astimezone(NY) if timestamp and timestamp > 0 else None
            if trade_time and trade_time.date() == session_date and day_time(9, 30) <= trade_time.time() <= regular_close.time():
                spot = _number(quote.get("last"))
                detail["spot_basis"] = "last_regular_session_trade"
        if spot is None or spot <= 0:
            detail["reason"] = "no_usable_underlying_price"
            continue
        detail["spot"] = spot
        try:
            expiries = [date.fromisoformat(value) for value in client.expirations(ticker)]
            eligible = sorted(value for value in expiries if 0 <= (value - session_date).days <= 7)
            if not eligible:
                detail["reason"] = "no_expiry_within_7_days"
                continue
            expiry = eligible[0]
            chain = [row for row in client.chain(ticker, expiry.isoformat()) if _standard_contract(row, ticker, expiry)]
        except TradierDataError as exc:
            detail["reason"] = exc.category
            continue
        except ValueError:
            detail["reason"] = "invalid_expiry_response"
            continue
        side_map: dict[float, dict[str, dict]] = {}
        for row in chain:
            side_map.setdefault(float(row["strike"]), {})[row["option_type"]] = row
        strikes = sorted(strike for strike, sides in side_map.items() if set(sides) == {"call", "put"})
        if not strikes:
            detail["reason"] = "no_standard_paired_contracts"
            continue
        center = min(range(len(strikes)), key=lambda index: abs(strikes[index] - spot))
        lower = max(0, min(center - 5, len(strikes) - 11))
        selected = strikes[lower : lower + 11]
        for strike in selected:
            for side in ("call", "put"):
                row = side_map[strike][side]
                contract = {key: row[key] for key in ("symbol", "option_type", "expiration_date", "strike", "contract_size")}
                contract["ticker"] = ticker
                # Keep the fetched quote/volume only as explicitly named preview
                # evidence. It is never ingested into the directional ledger.
                contract["snapshot"] = {key: row[key] for key in ("last", "bid", "ask", "volume", "trade_date", "bid_date", "ask_date") if key in row}
                contract.update(contract["snapshot"])
                contracts.append(contract)
        detail.update(status="selected", expiration=expiry.isoformat(), dte=(expiry - session_date).days, strike_low=selected[0], strike_high=selected[-1], strike_count=len(selected), contract_count=2 * len(selected))
    metadata["subscribed_contracts"] = len(contracts)
    return {"contracts": contracts, "metadata": metadata}


class _CollectorLock:
    """OS-held lock; a crash releases ownership without stale PID guessing."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.file = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a+b")
        self.file.seek(0, 2)
        if self.file.tell() == 0:
            self.file.write(b"0")
            self.file.flush()
        self.file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            self.file = None
            raise RuntimeError("Another net-impact collector owns this database") from None
        return self

    def __exit__(self, *args):
        if self.file:
            self.file.close()
            self.file = None


class _DeadlineStop:
    """Event-like stop condition without a timer thread or mutating its parent."""

    def __init__(self, parent: threading.Event, seconds: float) -> None:
        self.parent = parent
        self.deadline = time.monotonic() + max(0.0, seconds)

    def is_set(self) -> bool:
        return self.parent.is_set() or time.monotonic() >= self.deadline


class FlowCollector:
    """One stream per database; frozen daily cohort and explicit reconnect gaps.

    run() is intended for a separate always-on process. run_once() opens at most
    one stream and is bounded by market close, stop_event and socket timeouts.
    """

    def __init__(self, client: TradierClient, store, tickers: list[str], *, now: Callable[[], datetime] | None = None, calendar: USMarketCalendar | None = None) -> None:
        self.client, self.store, self.tickers = client, store, list(tickers)
        self._now = now or (lambda: datetime.now(UTC))
        self.calendar = calendar or USMarketCalendar()
        self._db_path = Path(store.path)
        self._lock_path = self._db_path.with_suffix(self._db_path.suffix + ".collector.lock")
        self._state_path = self._db_path.with_suffix(self._db_path.suffix + ".collector-state.json")
        self.last_status: dict = {"status": "not_started"}

    def _state(self, session: str) -> dict:
        try:
            value = json.loads(self._state_path.read_text(encoding="utf-8"))
            return value if value.get("session_date") == session else {}
        except (OSError, ValueError, AttributeError):
            return {}

    def _checkpoint(self, session: str, when: datetime) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"session_date": session, "last_activity": when.astimezone(UTC).isoformat()}), encoding="utf-8")
        temporary.replace(self._state_path)

    def run_once(self, stop_event: threading.Event, now: datetime | None = None, *, max_seconds: float | None = None) -> dict:
        with _CollectorLock(self._lock_path):
            return self._run_once(stop_event, now=now, max_seconds=max_seconds)

    def _run_once(self, stop_event: threading.Event, now: datetime | None = None, *, max_seconds: float | None = None) -> dict:
        current = now or self._now()
        if current.tzinfo is None:
            raise ValueError("Collector time must be timezone-aware")
        current = current.astimezone(UTC)
        local = current.astimezone(NY)
        session = local.date().isoformat()
        if stop_event.is_set():
            return {"status": "stopped"}
        clock = self.client.clock()
        if clock.get("date") != session or clock.get("state") != "open" or not self.calendar.is_session(local.date()):
            self.last_status = {"status": "market_closed", "session_date": session}
            return self.last_status
        opened = datetime.combine(local.date(), day_time(9, 30), NY).astimezone(UTC)
        closed = self.calendar.session_close(local.date()).close_utc
        # Market clock can shorten the calendar's standard session for ad-hoc
        # closures; never extend this v1 collector beyond the equity close.
        try:
            transition = datetime.combine(local.date(), datetime.strptime(str(clock.get("next_change")), "%H:%M").time(), NY).astimezone(UTC)
            if opened < transition <= closed:
                closed = transition
        except ValueError:
            pass
        if not opened <= current < closed:
            return {"status": "outside_collection_window", "session_date": session}
        universe = self.store.load_universe(session)
        if universe is None:
            universe = select_universe(self.client, self.tickers, local.date())
            if not universe["contracts"]:
                self.last_status = {"status": "no_eligible_contracts", "session_date": session}
                return self.last_status
            universe["metadata"]["collection_start"] = current.isoformat()
            universe["metadata"]["session_close"] = closed.isoformat()
            self.store.save_universe(session, universe["contracts"], universe["metadata"])
        symbols = [row["symbol"] for row in universe["contracts"]]
        if not symbols:
            self.last_status = {"status": "no_eligible_contracts", "session_date": session}
            return self.last_status
        state = self._state(session)
        try:
            gap_start = datetime.fromisoformat(state.get("last_activity", "")).astimezone(UTC)
        except ValueError:
            gap_start = opened
        gap_start = max(opened, min(gap_start, current))
        last_activity = current
        last_checkpoint = 0.0
        counts = {"timesale": 0, "quote": 0, "summary": 0}

        def connected() -> None:
            connected_at = self._now().astimezone(UTC)
            if connected_at > gap_start:
                self.store.record_gap(session, gap_start, min(connected_at, closed), "startup_or_reconnect")
            self._checkpoint(session, connected_at)

        stream_stop = stop_event if max_seconds is None else _DeadlineStop(stop_event, max_seconds)
        events = self.client.iter_events(symbols, stream_stop, on_connected=connected)
        status = "stopped"
        try:
            for event in events:
                observed = self._now().astimezone(UTC)
                if stream_stop.is_set() or observed >= closed:
                    status = "session_complete" if observed >= closed else "stopped"
                    break
                kind = event.get("type")
                if kind in counts:
                    counts[kind] += 1
                # The ledger uses quote/summary receipt only for freshness;
                # only its timesale branch is allowed to add premium.
                if kind in counts:
                    self.store.ingest(session, event, observed)
                last_activity = observed
                if time.monotonic() - last_checkpoint >= 5:
                    self._checkpoint(session, last_activity)
                    last_checkpoint = time.monotonic()
            else:
                status = "stopped" if stream_stop.is_set() else "stream_ended"
        except TradierDataError as exc:
            status = exc.category
        finally:
            if hasattr(events, "close"):
                events.close()
            self._checkpoint(session, min(last_activity, closed))
        self.last_status = {"status": status, "session_date": session, "contracts": len(symbols), "event_counts": counts}
        return self.last_status

    def run(self, stop_event: threading.Event) -> None:
        delay = 2.0
        with _CollectorLock(self._lock_path):
            while not stop_event.is_set():
                try:
                    status = self._run_once(stop_event)
                except TradierDataError as exc:
                    status = self.last_status = {"status": exc.category}
                if status.get("status") in {"market_closed", "outside_collection_window", "session_complete", "no_eligible_contracts"}:
                    delay = 30.0
                else:
                    delay = min(30.0, max(2.0, delay * 2))
                stop_event.wait(delay)
