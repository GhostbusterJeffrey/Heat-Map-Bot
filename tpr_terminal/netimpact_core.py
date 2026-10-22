"""Execution-based premium ledger for the selected /netimpact universe.

Quotes never add premium. Unresolved amendments are retained, but invalidate the
affected ticker's result rather than pretending that a cancellation was matched.
Money is accumulated as Decimal from the stored execution values, not floats.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections import Counter
from contextlib import contextmanager
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .market_calendar import USMarketCalendar

NEW_YORK = ZoneInfo("America/New_York")
ZERO = Decimal("0")
FUTURE_TOLERANCE = timedelta(seconds=5)


def _aware(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("An explicitly timezone-aware timestamp is required")
    return value.astimezone(UTC)


def _timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return _aware(value)
    if isinstance(value, bool) or value is None:
        raise ValueError("Missing event timestamp")
    text = str(value).strip()
    try:
        number = Decimal(text)
    except InvalidOperation:
        return _aware(datetime.fromisoformat(text.replace("Z", "+00:00")))
    if not number.is_finite():
        raise ValueError("Invalid event timestamp")
    if abs(number) >= Decimal("100000000000"):
        number /= 1000
    return datetime.fromtimestamp(float(number), tz=UTC)


def _iso(value: datetime) -> str:
    return _aware(value).isoformat(timespec="microseconds")


def _number(value: Any) -> Decimal:
    if value is None or isinstance(value, bool):
        raise ValueError("Missing numeric value")
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("Invalid numeric value") from exc
    if not number.is_finite():
        raise ValueError("Invalid numeric value")
    return number


def _marker(value: Any) -> bool:
    if value in (False, None, 0, "false", "False", "0", ""):
        return False
    if value in (True, 1, "true", "True", "1"):
        return True
    raise ValueError("Unrecognized amendment flag")


def session_bounds(session_date: str, metadata: dict | None = None) -> tuple[datetime, datetime]:
    """Common equity-session bounds, including early closes; ET is DST-aware."""
    day = date.fromisoformat(session_date)
    calendar = USMarketCalendar()
    if not calendar.is_session(day):
        raise ValueError("The requested date is not a market session")
    metadata = metadata or {}
    opening = datetime.combine(day, time(9, 30), tzinfo=NEW_YORK).astimezone(UTC)
    closing = calendar.session_close(day).close_utc
    if metadata.get("session_open"):
        opening = _timestamp(metadata["session_open"])
    if metadata.get("session_close"):
        closing = _timestamp(metadata["session_close"])
    if opening >= closing or any(t.astimezone(NEW_YORK).date() != day for t in (opening, closing)):
        raise ValueError("Invalid session bounds")
    return opening, closing


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _contract(row: dict) -> dict:
    required = ("symbol", "ticker", "option_type", "expiration_date", "strike", "contract_size")
    if any(key not in row for key in required):
        raise ValueError("Contract metadata is incomplete")
    result = {key: row[key] for key in required}
    result["symbol"] = str(result["symbol"]).strip().upper()
    result["ticker"] = str(result["ticker"]).strip().upper()
    result["option_type"] = str(result["option_type"]).lower()
    if not result["symbol"] or not result["ticker"] or result["option_type"] not in ("call", "put"):
        raise ValueError("Invalid contract identity")
    date.fromisoformat(str(result["expiration_date"]))
    strike = _number(result["strike"])
    multiplier = _number(result["contract_size"])
    if strike < 0 or multiplier <= 0 or multiplier != multiplier.to_integral_value():
        raise ValueError("Invalid contract strike or multiplier")
    result["strike"] = float(strike)
    result["contract_size"] = int(multiplier)
    result["expiration_date"] = str(result["expiration_date"])
    return result


class FlowStore:
    """SQLite-backed, restart-safe record of a fixed session's selected contracts.

    ``snapshot`` values are intentionally absent when no signed executions were
    observed, or a cancellation/correction could have changed the result. A row
    with a genuine zero requires at least one eligible classified execution.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute("PRAGMA busy_timeout=30000")
        self._universe_cache: dict[str, dict] = {}
        self._contract_cache: dict[str, dict] = {}
        self._activity_cache: dict[str, str | None] = {}
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS flow_universes (
                    session_date TEXT PRIMARY KEY,
                    contracts_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS flow_events (
                    id INTEGER PRIMARY KEY,
                    session_date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    identity_key TEXT NOT NULL,
                    event_key TEXT NOT NULL UNIQUE,
                    payload_hash TEXT NOT NULL,
                    event_at TEXT,
                    received_at TEXT NOT NULL,
                    raw_json TEXT NOT NULL,
                    side TEXT,
                    premium TEXT,
                    call_premium TEXT NOT NULL,
                    put_premium TEXT NOT NULL,
                    excluded_reason TEXT,
                    unresolved INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS flow_events_window
                    ON flow_events(session_date, event_at, ticker);
                CREATE INDEX IF NOT EXISTS flow_events_identity
                    ON flow_events(identity_key);
                CREATE TABLE IF NOT EXISTS flow_activity (
                    session_date TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    event_at TEXT
                );
                CREATE INDEX IF NOT EXISTS flow_activity_window
                    ON flow_activity(session_date, received_at);
                CREATE TABLE IF NOT EXISTS flow_gaps (
                    session_date TEXT NOT NULL,
                    start_at TEXT NOT NULL,
                    end_at TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    UNIQUE(session_date, start_at, end_at, reason)
                );
                """
            )

    @contextmanager
    def _connect(self):
        # One serialized connection avoids opening a SQLite handle per print.
        with self._lock, self._connection:
            yield self._connection

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def save_universe(self, session_date: str, contracts: list[dict], metadata: dict) -> None:
        opening, closing = session_bounds(session_date, metadata)
        selected = sorted((_contract(row) for row in contracts), key=lambda row: row["symbol"])
        if len({row["symbol"] for row in selected}) != len(selected):
            raise ValueError("Duplicate contract symbols in universe")
        if any(date.fromisoformat(row["expiration_date"]) < date.fromisoformat(session_date) for row in selected):
            raise ValueError("Expired contracts cannot be subscribed for this session")
        saved = dict(metadata)
        saved["tickers"] = list(dict.fromkeys(
            [str(t).strip().upper() for t in saved.get("tickers", []) if str(t).strip()]
            + [row["ticker"] for row in selected]
        ))
        saved["session_open"] = _iso(opening)
        saved["session_close"] = _iso(closing)
        saved["collection_start"] = _iso(_timestamp(saved.get("collection_start") or datetime.now(UTC)))
        saved.setdefault("scope", "Selected near-term contracts only")
        with self._lock, self._connect() as connection:
            previous = connection.execute(
                "SELECT contracts_json, metadata_json FROM flow_universes WHERE session_date=?",
                (session_date,),
            ).fetchone()
            if previous is not None:
                old_metadata = json.loads(previous["metadata_json"])
                changed = previous["contracts_json"] != _json(selected) or any(
                    old_metadata.get(key) != saved.get(key)
                    for key in ("tickers", "scope", "session_open", "session_close")
                )
                if changed:
                    raise ValueError("This session's contract universe is locked; use the stored selection")
                return  # Preserve the actual original collection start across restart.
            connection.execute(
                "INSERT INTO flow_universes VALUES (?, ?, ?)",
                (session_date, _json(selected), _json(saved)),
            )
            self._universe_cache[session_date] = {"contracts": selected, "metadata": saved}
            self._contract_cache[session_date] = {row["symbol"]: row for row in selected}

    def load_universe(self, session_date: str) -> dict | None:
        result = self._get_universe(session_date)
        # Callers cannot mutate the cohort cached by the ingestion path.
        return json.loads(_json(result)) if result is not None else None

    def _get_universe(self, session_date: str) -> dict | None:
        with self._lock, self._connect() as connection:
            if session_date in self._universe_cache:
                return self._universe_cache[session_date]
            row = connection.execute(
                "SELECT contracts_json, metadata_json FROM flow_universes WHERE session_date=?",
                (session_date,),
            ).fetchone()
            if row is None:
                return None
            result = {"contracts": json.loads(row["contracts_json"]), "metadata": json.loads(row["metadata_json"])}
            self._universe_cache[session_date] = result
            self._contract_cache[session_date] = {contract["symbol"]: contract for contract in result["contracts"]}
            return result

    def ingest(self, session_date: str, event: dict, received_at: datetime) -> dict:
        received = _aware(received_at)
        universe = self._get_universe(session_date)
        if universe is None:
            raise ValueError("Save the session's universe before ingesting events")
        contracts = self._contract_cache[session_date]
        symbol = str(event.get("symbol", "")).upper()
        contract = contracts.get(symbol)
        if contract is None:
            return {"status": "ignored", "reason": "unsubscribed_contract"}
        kind = event.get("type")
        if kind not in ("timesale", "quote", "summary", "trade", "tradex"):
            return {"status": "ignored", "reason": "unsupported_event"}
        event_at = None
        try:
            event_at = _timestamp(event.get("date", event.get("timestamp")))
        except (ValueError, OverflowError, OSError):
            pass
        opening, closing = session_bounds(session_date, universe["metadata"])
        event_iso = _iso(event_at) if event_at else None
        with self._lock, self._connect() as connection:
            # Only one freshness row per second: quote traffic can be very high.
            activity_second = received.replace(microsecond=0)
            if session_date not in self._activity_cache:
                self._activity_cache[session_date] = connection.execute(
                    "SELECT MAX(received_at) AS value FROM flow_activity WHERE session_date=?", (session_date,)
                ).fetchone()["value"]
            latest = self._activity_cache[session_date]
            if latest is None or latest < _iso(activity_second):
                connection.execute("INSERT INTO flow_activity VALUES (?, ?, ?)", (session_date, _iso(received), event_iso))
                self._activity_cache[session_date] = _iso(received)
            if kind != "timesale":
                return {"status": "ignored", "reason": "not_execution"}

            raw = _json(event)
            payload_hash = hashlib.sha256(raw.encode()).hexdigest()
            seq = event.get("seq")
            exch = str(event.get("exch", "")).strip()
            amendment = False
            marker_error = False
            try:
                amendment = _marker(event.get("cancel")) or _marker(event.get("correction"))
            except ValueError:
                marker_error = True
            sequence_valid = seq is not None and not isinstance(seq, bool) and bool(str(seq).strip()) and bool(exch)
            identity = _json([session_date, symbol, exch, str(seq), "amendment" if amendment or marker_error else "normal"])
            identity_key = hashlib.sha256(identity.encode()).hexdigest()
            event_key = identity_key if sequence_valid else identity_key + ":" + payload_hash
            previous = connection.execute(
                "SELECT payload_hash FROM flow_events WHERE event_key=?", (event_key,)
            ).fetchone()
            if previous is not None and previous["payload_hash"] == payload_hash:
                return {"status": "duplicate", "reason": "already_recorded"}
            conflict = previous is not None
            if conflict:
                # A changed print sharing an identifier cannot safely be added again.
                event_key = identity_key + ":" + payload_hash
                if connection.execute("SELECT 1 FROM flow_events WHERE event_key=?", (event_key,)).fetchone():
                    return {"status": "duplicate", "reason": "already_recorded"}

            reason = None
            unresolved = False
            side = None
            premium = None
            call_premium = ZERO
            put_premium = ZERO
            collection_start = _timestamp(universe["metadata"]["collection_start"])
            if amendment or marker_error or conflict:
                reason = "unresolved_amendment" if amendment else "sequence_conflict" if conflict else "invalid_amendment_flag"
                unresolved = True
            elif event_at is None:
                reason = "missing_or_invalid_timestamp"
            elif event_at < opening or event_at > closing:
                reason = "outside_session"
            elif event_at > received + FUTURE_TOLERANCE:
                reason = "future_timestamp"
            elif event_at < collection_start:
                reason = "before_collection"
            elif not sequence_valid:
                reason = "missing_execution_identity"
            elif str(event.get("session", "")).lower() != "normal":
                reason = "non_normal_session"
            elif event.get("flag") not in (None, ""):
                reason = "unverified_trade_condition"
            else:
                try:
                    price = _number(event.get("last", event.get("price")))
                    size = _number(event.get("size"))
                    bid = _number(event.get("bid"))
                    ask = _number(event.get("ask"))
                    if price <= 0 or size <= 0 or size != size.to_integral_value():
                        reason = "invalid_execution"
                    elif bid < 0 or ask <= bid:
                        reason = "invalid_quote"
                    else:
                        premium = price * size * Decimal(contract["contract_size"])
                        difference = price * 2 - bid - ask
                        if difference == 0:
                            side, reason = "mid", "midpoint"
                        else:
                            side = "ask" if difference > 0 else "bid"
                            signed = premium if side == "ask" else -premium
                            if contract["option_type"] == "call":
                                call_premium = signed
                            else:
                                put_premium = signed
                except (ValueError, InvalidOperation):
                    reason = "missing_or_invalid_price_size_quote"
            connection.execute(
                """INSERT INTO flow_events (
                    session_date,ticker,symbol,identity_key,event_key,payload_hash,event_at,received_at,
                    raw_json,side,premium,call_premium,put_premium,excluded_reason,unresolved
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (session_date, contract["ticker"], symbol, identity_key, event_key, payload_hash,
                 event_iso, _iso(received), raw, side, str(premium) if premium is not None else None,
                 str(call_premium), str(put_premium), reason, int(unresolved)),
            )
        return {
            "status": "excluded" if reason else "counted", "reason": reason, "side": side,
            "ticker": contract["ticker"], "net": float(call_premium - put_premium),
            "unresolved": unresolved,
        }

    def record_gap(self, session_date: str, start_at: datetime, end_at: datetime, reason: str) -> None:
        start, end = _aware(start_at), _aware(end_at)
        if end < start:
            raise ValueError("Gap end must not precede its start")
        date.fromisoformat(session_date)
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO flow_gaps VALUES (?,?,?,?)",
                (session_date, _iso(start), _iso(end), str(reason)[:512]),
            )

    def snapshot(self, session_date: str, window: str = "session", as_of: datetime | None = None) -> dict:
        if window not in ("session", "15m", "60m"):
            raise ValueError("Window must be session, 15m, or 60m")
        requested = _aware(as_of or datetime.now(UTC))
        universe = self.load_universe(session_date)
        metadata = universe["metadata"] if universe else {}
        opening, closing = session_bounds(session_date, metadata)
        cutoff = min(requested, closing)
        # Before the open the query interval is empty; never label a report with
        # a future data cutoff just to align it with the scheduled opening.
        nominal_start = opening if window == "session" else max(
            opening, cutoff - timedelta(minutes=int(window[:-1]))
        )
        collection_start = _timestamp(metadata.get("collection_start") or requested)
        start = min(cutoff, max(nominal_start, collection_start))
        report = {
            "session_date": session_date, "window": window, "mode": "flow",
            "start_at": _iso(start), "as_of": _iso(cutoff), "requested_at": _iso(requested),
            "session_open": _iso(opening), "session_close": _iso(closing),
            "status": "no_data", "scope": metadata.get("scope", "No subscribed universe"),
            "notes": [], "rows": [], "subscribed_contracts": len(universe["contracts"]) if universe else 0,
            "last_received_at": None, "last_event_at": None, "gaps": [],
            "excluded_reasons": {}, "classified_trades": 0, "excluded_trades": 0,
        }
        if universe is None:
            report["notes"].append("No execution data was collected for this session.")
            return report
        with self._lock, self._connect() as connection:
            executions = connection.execute(
                """SELECT * FROM flow_events WHERE session_date=? AND event_at>=? AND event_at<=?
                   AND received_at<=? ORDER BY id""",
                (session_date, _iso(nominal_start), _iso(cutoff), _iso(requested)),
            ).fetchall()
            # A missing timestamp or unknown amendment cannot be located in a rolling
            # window; keep the uncertainty visible for the whole affected session.
            anomalies = connection.execute(
                """SELECT ticker, excluded_reason, unresolved, event_at FROM flow_events WHERE session_date=?
                   AND received_at<=? AND (unresolved=1 OR event_at IS NULL)""",
                (session_date, _iso(requested)),
            ).fetchall()
            latest = connection.execute(
                """SELECT MAX(received_at) AS received FROM flow_activity
                   WHERE session_date=? AND received_at<=?""", (session_date, _iso(requested)),
            ).fetchone()
            latest_trade = connection.execute(
                """SELECT MAX(event_at) AS event_at FROM flow_events WHERE session_date=?
                   AND excluded_reason IS NULL AND event_at<=? AND received_at<=?""",
                (session_date, _iso(cutoff), _iso(requested)),
            ).fetchone()
            gaps = connection.execute(
                "SELECT * FROM flow_gaps WHERE session_date=? AND start_at<? AND end_at>?",
                (session_date, _iso(cutoff), _iso(nominal_start)),
            ).fetchall()
        report["last_received_at"] = latest["received"]
        report["last_event_at"] = latest_trade["event_at"]
        report["gaps"] = [dict(row) for row in gaps]
        excluded = Counter()
        unresolved_tickers = {row["ticker"] for row in anomalies if row["unresolved"]}
        unknown_time_tickers = {row["ticker"] for row in anomalies if not row["unresolved"]}
        contracts_by_ticker = Counter(row["ticker"] for row in universe["contracts"])
        tickers = metadata.get("tickers", list(contracts_by_ticker))
        totals = {ticker: {"call": ZERO, "put": ZERO, "trades": 0, "excluded": 0} for ticker in tickers}
        for event in executions:
            bucket = totals.setdefault(event["ticker"], {"call": ZERO, "put": ZERO, "trades": 0, "excluded": 0})
            if event["excluded_reason"]:
                excluded[event["excluded_reason"]] += 1
                bucket["excluded"] += 1
            else:
                bucket["call"] += Decimal(event["call_premium"])
                bucket["put"] += Decimal(event["put_premium"])
                bucket["trades"] += 1
        # Timestamp-free events were absent from the window query; count once here.
        for event in anomalies:
            if event["event_at"] is None:
                bucket = totals[event["ticker"]]
                bucket["excluded"] += 1
                excluded[event["excluded_reason"]] += 1

        partial = collection_start > nominal_start or bool(gaps)
        stale_limit = max(15, int(metadata.get("stale_after_seconds", 120)))
        last_received = _timestamp(latest["received"]) if latest["received"] else None
        stale = requested >= opening and cutoff - (last_received or collection_start) > timedelta(seconds=stale_limit)
        for ticker in tickers:
            bucket = totals[ticker]
            available = bucket["trades"] > 0 and ticker not in unresolved_tickers
            status = "observed" if available else "no_executions"
            if not contracts_by_ticker[ticker]:
                status = "no_contracts"
            elif ticker in unresolved_tickers:
                status = "unresolved_amendment"
            elif available and (partial or ticker in unknown_time_tickers):
                status = "partial"
            if stale and available:
                status = "stale"
            report["rows"].append({
                "ticker": ticker,
                "net_call": float(bucket["call"]) if available else None,
                "net_put": float(bucket["put"]) if available else None,
                "net": float(bucket["call"] - bucket["put"]) if available else None,
                "contracts": contracts_by_ticker[ticker], "trades": bucket["trades"],
                "excluded": bucket["excluded"], "available": available, "status": status,
            })
        report["rows"].sort(key=lambda row: (not row["available"], -(row["net"] or 0), row["ticker"]))
        report["classified_trades"] = sum(row["trades"] for row in report["rows"])
        report["excluded_trades"] = sum(excluded.values())
        report["excluded_reasons"] = dict(excluded)
        if any(row["available"] for row in report["rows"]):
            report["status"] = "stale" if stale else "partial" if partial or unresolved_tickers or unknown_time_tickers else "observed"
        if collection_start > nominal_start:
            report["notes"].append("Collection started after the requested window began; earlier trades are unavailable.")
        if gaps:
            report["notes"].append("The requested window contains a recorded feed interruption; totals cover observed executions only.")
        if stale:
            report["notes"].append("Recent feed activity is missing; totals may omit executions after the last received event.")
        if unresolved_tickers:
            report["notes"].append("Unresolved cancellations/corrections or conflicting identifiers: " + ", ".join(sorted(unresolved_tickers)) + ". Affected values are withheld.")
        if unknown_time_tickers:
            report["notes"].append("Some executions lacked usable timestamps and were excluded.")
        if not report["classified_trades"]:
            report["notes"].append("No eligible signed executions were observed; missing values do not mean zero flow.")
        if excluded:
            report["notes"].append("Midpoint and unclassifiable executions are excluded from signed premium.")
        report["notes"].append("Bid/ask classification estimates direction; it does not identify opening/closing trades or spread intent.")
        return report
