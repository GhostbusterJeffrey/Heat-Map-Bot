from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
import urllib.error
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import patch

from tpr_terminal.tradier_flow import (
    FlowCollector,
    TradierClient,
    TradierDataError,
    _CollectorLock,
    _NoRedirect,
    select_universe,
)

DAY = date(2026, 10, 21)
NOW = datetime(2026, 10, 21, 14, 0, tzinfo=UTC)


def contract(ticker, strike, side, expiry="2026-10-23", multiplier=100):
    return {"symbol": f"{ticker}{date.fromisoformat(expiry):%y%m%d}{'C' if side == 'call' else 'P'}{int(strike * 1000):08}", "option_type": side, "strike": strike, "expiration_date": expiry, "contract_size": multiplier, "volume": 20, "last": 1.5, "bid": 1.4, "ask": 1.6, "trade_date": 1792605600000}


class FakeClient:
    def __init__(self):
        self.spot = 100
        self.after_hours = False
        self.history_available = True
        self.expiry_values = ["2026-10-23", "2026-10-30"]
        self.clock_state = "open"
        self.quote_calls = 0
        self.chain_values = [contract("TEST", strike, side) for strike in range(90, 111) for side in ("call", "put")]
        self.stream_calls = 0

    def quotes(self, symbols):
        self.quote_calls += 1
        trade_time = datetime(2026, 10, 21, 21 if self.after_hours else 19, 59, tzinfo=UTC)
        return [{"symbol": symbol, "last": 110 if self.after_hours else self.spot, "close": 999, "trade_date": int(trade_time.timestamp() * 1000)} for symbol in symbols if symbol != "MISSING"]

    def daily_history(self, symbol, session_date):
        return {"date": str(session_date), "close": self.spot} if self.history_available else None

    def expirations(self, ticker):
        return self.expiry_values

    def chain(self, ticker, expiry):
        return self.chain_values

    def clock(self):
        return {"date": DAY.isoformat(), "state": self.clock_state, "next_change": "16:00"}

    def iter_events(self, symbols, stop_event, on_connected=None):
        self.stream_calls += 1
        if on_connected:
            on_connected()
        yield {"type": "quote", "symbol": symbols[0], "bid": 1.4, "ask": 1.6}
        yield {"type": "timesale", "symbol": symbols[0], "last": 1.6, "size": 1, "bid": 1.4, "ask": 1.6, "date": int(NOW.timestamp() * 1000), "seq": 1}


class FakeStore:
    def __init__(self, path):
        self.path = path
        self.saved = None
        self.gaps = []
        self.events = []

    def load_universe(self, day):
        return self.saved

    def save_universe(self, day, contracts, metadata):
        self.saved = {"contracts": contracts, "metadata": metadata}

    def record_gap(self, day, start, end, reason):
        self.gaps.append((day, start, end, reason))

    def ingest(self, day, event, received_at):
        self.events.append((day, event, received_at))


class Response(io.BytesIO):
    status = 200


class Opener:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append(request)
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return Response(value if isinstance(value, bytes) else json.dumps(value).encode())


class SelectionTests(unittest.TestCase):
    def test_nearest_expiry_and_eleven_paired_centered_strikes(self):
        result = select_universe(FakeClient(), ["TEST"], DAY)
        rows = result["contracts"]
        self.assertEqual(len(rows), 22)
        self.assertEqual({row["strike"] for row in rows}, set(range(95, 106)))
        self.assertEqual({row["expiration_date"] for row in rows}, {"2026-10-23"})
        self.assertEqual({row["contract_size"] for row in rows}, {100})
        self.assertEqual(rows[0]["volume"], 20)
        self.assertEqual(result["metadata"]["ticker_details"]["TEST"]["status"], "selected")

    def test_after_hours_last_never_overrides_dated_close(self):
        client = FakeClient()
        client.after_hours = True
        result = select_universe(client, ["TEST"], DAY, "close")
        self.assertEqual({row["strike"] for row in result["contracts"]}, set(range(95, 106)))
        self.assertEqual(result["metadata"]["ticker_details"]["TEST"]["spot_basis"], "dated_daily_close")

    def test_missing_close_cannot_fall_back_to_after_hours_or_prev_close(self):
        client = FakeClient()
        client.after_hours = True
        client.history_available = False
        result = select_universe(client, ["TEST"], DAY, "close")
        self.assertEqual(result["contracts"], [])
        self.assertEqual(result["metadata"]["ticker_details"]["TEST"]["reason"], "no_usable_underlying_price")

    def test_coverage_failure_is_not_a_zero_row(self):
        result = select_universe(FakeClient(), ["TEST", "MISSING"], DAY)
        self.assertEqual(result["metadata"]["tickers"], ["TEST", "MISSING"])
        self.assertEqual(result["metadata"]["ticker_details"]["MISSING"]["status"], "unavailable")
        self.assertNotIn("MISSING", {row["ticker"] for row in result["contracts"]})

    def test_expiry_outside_seven_days_and_adjusted_contracts_excluded(self):
        client = FakeClient()
        client.expiry_values = ["2026-10-30"]
        self.assertEqual(select_universe(client, ["TEST"], DAY)["contracts"], [])
        client = FakeClient()
        client.chain_values = [contract("TEST", 100, "call", multiplier=150), contract("TEST", 100, "put")]
        self.assertEqual(select_universe(client, ["TEST"], DAY)["contracts"], [])


class TransportTests(unittest.TestCase):
    def test_http_error_never_exposes_token_url_or_response_body(self):
        error = urllib.error.HTTPError("https://example.invalid/SECRET", 401, "SECRET", {}, io.BytesIO(b"SECRET"))
        client = TradierClient("SECRET", opener=Opener([error]), min_interval=0)
        with self.assertRaises(TradierDataError) as caught:
            client.clock()
        self.assertEqual(str(caught.exception), "Tradier market data: request_rejected (HTTP 401)")
        self.assertIsNone(_NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.invalid"))

    def test_bounded_retry_then_normalized_single_quote(self):
        error = urllib.error.HTTPError("https://api.tradier.com", 429, "rate", {}, io.BytesIO())
        opener = Opener([error, {"quotes": {"quote": {"symbol": "SPY"}}}])
        with patch("tpr_terminal.tradier_flow.time.sleep"):
            rows = TradierClient("secret", opener=opener, min_interval=0).quotes(["SPY"])
        self.assertEqual(rows, [{"symbol": "SPY"}])
        self.assertEqual(len(opener.requests), 2)

    def test_market_only_allowlist(self):
        client = TradierClient("secret", opener=Opener([]), min_interval=0)
        with self.assertRaises(ValueError):
            client._json("/accounts/123/orders", method="POST")

    def test_stream_uses_fixed_host_filters_and_stops_cleanly(self):
        opener = Opener([{"stream": {"sessionid": "private-session"}}, b'{"type":"timesale","symbol":"SPY","size":1}\n'])
        client = TradierClient("private-token", opener=opener, min_interval=0)
        stop = threading.Event()
        connected = []
        events = client.iter_events(["SPY"], stop, lambda: connected.append(True))
        self.assertEqual(next(events)["type"], "timesale")
        stop.set()
        self.assertEqual(list(events), [])
        self.assertEqual(connected, [True])
        request = opener.requests[1]
        self.assertEqual(request.full_url, "https://stream.tradier.com/v1/markets/events")
        self.assertNotIn("private-token", request.full_url)
        self.assertIn(b"validOnly=false", request.data)
        self.assertIn(b"timesale%2Cquote%2Csummary", request.data)

    def test_daily_history_requires_matching_date(self):
        opener = Opener([{"history": {"day": {"date": "2026-10-20", "close": 90}}}])
        self.assertIsNone(TradierClient("secret", opener=opener, min_interval=0).daily_history("SPY", DAY))


class CollectorTests(unittest.TestCase):
    def test_frozen_universe_reused_and_quotes_forwarded_for_freshness(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient()
            store = FakeStore(Path(directory) / "flow.sqlite3")
            collector = FlowCollector(client, store, ["TEST"], now=lambda: NOW)
            first = collector.run_once(threading.Event())
            client.spot = 110
            collector.run_once(threading.Event())
            self.assertEqual(client.quote_calls, 1)
            self.assertEqual(first["event_counts"]["quote"], 1)
            self.assertEqual(len(store.events), 4)
            self.assertEqual([row[1]["type"] for row in store.events], ["quote", "timesale", "quote", "timesale"])
            self.assertEqual(store.gaps[0][1], datetime(2026, 10, 21, 13, 30, tzinfo=UTC))
            self.assertEqual(store.gaps[0][2], NOW)

    def test_closed_market_does_not_select_or_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient()
            client.clock_state = "closed"
            store = FakeStore(Path(directory) / "flow.sqlite3")
            status = FlowCollector(client, store, ["TEST"], now=lambda: NOW).run_once(threading.Event())
            self.assertEqual(status["status"], "market_closed")
            self.assertEqual(client.quote_calls, 0)
            self.assertEqual(client.stream_calls, 0)

    def test_empty_failed_selection_is_retryable(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient()
            client.expiry_values = []
            store = FakeStore(Path(directory) / "flow.sqlite3")
            status = FlowCollector(client, store, ["TEST"], now=lambda: NOW).run_once(threading.Event())
            self.assertEqual(status["status"], "no_eligible_contracts")
            self.assertIsNone(store.saved)

    def test_lock_prevents_second_collector(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "collector.lock"
            with _CollectorLock(path):
                with self.assertRaises(RuntimeError):
                    with _CollectorLock(path):
                        pass
            with _CollectorLock(path):
                pass


if __name__ == "__main__":
    unittest.main()
