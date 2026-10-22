from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tpr_terminal.netimpact_core import FlowStore, session_bounds


class NetImpactCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "flow.sqlite3"
        self.store = FlowStore(self.path)
        self.day = "2026-10-21"
        self.opening = datetime(2026, 10, 21, 13, 30, tzinfo=UTC)
        self.now = self.opening + timedelta(minutes=30)
        self.contracts = [
            {"symbol": "ABC261023C00100000", "ticker": "ABC", "option_type": "call", "expiration_date": "2026-10-23", "strike": 100, "contract_size": 100},
            {"symbol": "ABC261023P00100000", "ticker": "ABC", "option_type": "put", "expiration_date": "2026-10-23", "strike": 100, "contract_size": 100},
            {"symbol": "XYZ261023C00100000", "ticker": "XYZ", "option_type": "call", "expiration_date": "2026-10-23", "strike": 100, "contract_size": 10},
        ]
        self.metadata = {"tickers": ["ABC", "XYZ", "MISSING"], "scope": "Test cohort", "collection_start": self.opening.isoformat()}
        self.store.save_universe(self.day, self.contracts, self.metadata)

    def tearDown(self) -> None:
        self.store.close()
        self.temp.cleanup()

    def event(self, seq=1, *, symbol=None, at=None, **overrides):
        timestamp = at or self.now
        row = {
            "type": "timesale", "symbol": symbol or self.contracts[0]["symbol"],
            "exch": "Q", "last": "2.10", "size": "10", "bid": "2.00", "ask": "2.10",
            "date": int(timestamp.timestamp() * 1000), "seq": seq, "session": "normal",
            "flag": "", "cancel": False, "correction": False,
        }
        row.update(overrides)
        return row

    def ingest(self, event, at=None):
        return self.store.ingest(self.day, event, at or self.now)

    def row(self, ticker="ABC", **kwargs):
        report = self.store.snapshot(self.day, as_of=kwargs.pop("as_of", self.now), **kwargs)
        return next(row for row in report["rows"] if row["ticker"] == ticker)

    def test_four_execution_signs_are_call_net_minus_put_net(self):
        self.ingest(self.event(1, last="2.10"))      # +2100 call
        self.ingest(self.event(2, last="2.00"))      # -2000 call
        self.ingest(self.event(3, symbol=self.contracts[1]["symbol"], last="2.10"))  # +2100 put
        self.ingest(self.event(4, symbol=self.contracts[1]["symbol"], last="2.00", size="20"))  # -4000 put
        row = self.row()
        self.assertEqual(row["net_call"], 100)
        self.assertEqual(row["net_put"], -1900)
        self.assertEqual(row["net"], 2000)
        self.assertEqual(row["trades"], 4)

    def test_multiplier_comes_from_contract_metadata(self):
        self.ingest(self.event(symbol=self.contracts[2]["symbol"], last="2.10", size="3"))
        self.assertEqual(self.row("XYZ")["net"], 63)

    def test_large_premium_is_preserved_without_an_unverified_cap(self):
        self.ingest(self.event(last="10", bid="9.9", ask="10", size="3000"))
        self.assertEqual(self.row()["net"], 3_000_000)

    def test_decimal_midpoint_does_not_become_false_direction(self):
        outcome = self.ingest(self.event(last="0.15", bid="0.10", ask="0.20"))
        self.assertEqual(outcome["reason"], "midpoint")
        self.assertIsNone(self.row()["net"])
        self.assertEqual(self.row()["excluded"], 1)

    def test_duplicate_print_is_not_added_twice_after_restart(self):
        event = self.event()
        self.assertEqual(self.ingest(event)["status"], "counted")
        self.store.close()
        self.store = FlowStore(self.path)
        self.assertEqual(self.ingest(event)["status"], "duplicate")
        self.assertEqual(self.row()["net"], 2100)

    def test_same_sequence_on_other_exchange_is_distinct(self):
        self.ingest(self.event(exch="Q"))
        self.ingest(self.event(exch="Z"))
        self.assertEqual(self.row()["net"], 4200)

    def test_changed_payload_same_identity_invalidates_ticker(self):
        self.ingest(self.event())
        result = self.ingest(self.event(last="2.00"))
        self.assertEqual(result["reason"], "sequence_conflict")
        self.assertEqual(self.row()["status"], "unresolved_amendment")
        self.assertIsNone(self.row()["net"])

    def test_cancel_and_correction_never_silently_leave_a_reliable_bar(self):
        for flag in ("cancel", "correction"):
            with self.subTest(flag=flag):
                self.ingest(self.event())
                outcome = self.ingest(self.event(**{flag: True}))
                self.assertTrue(outcome["unresolved"])
                row = self.row()
                self.assertFalse(row["available"])
                self.assertIsNone(row["net"])

    def test_false_string_markers_do_not_count_as_amendments(self):
        self.assertEqual(self.ingest(self.event(cancel="false", correction="0"))["status"], "counted")

    def test_quote_and_trade_channels_cannot_double_count_timesale(self):
        self.ingest(self.event())
        self.assertEqual(self.ingest(self.event(type="trade"))["reason"], "not_execution")
        self.assertEqual(self.ingest(self.event(type="quote"))["reason"], "not_execution")
        self.assertEqual(self.row()["trades"], 1)

    def test_unusable_quotes_trade_flags_and_sizes_are_excluded(self):
        cases = [
            {"bid": "2.10", "ask": "2.00"},
            {"bid": "2.10", "ask": "2.10"},
            {"ask": None}, {"last": "NaN"}, {"size": "1.5"}, {"size": -1},
            {"flag": "UNKNOWN"}, {"session": "post"}, {"exch": ""}, {"seq": None},
        ]
        for seq, changes in enumerate(cases, 1):
            with self.subTest(changes=changes):
                event = self.event(seq)
                event.update(changes)
                self.assertEqual(self.ingest(event)["status"], "excluded")
        self.assertIsNone(self.row()["net"])

    def test_rolling_window_uses_execution_time_and_common_cutoff(self):
        self.ingest(self.event(1, at=self.now - timedelta(minutes=20)), at=self.now - timedelta(minutes=20))
        self.ingest(self.event(2, at=self.now - timedelta(minutes=5), last="2.00"), at=self.now - timedelta(minutes=5))
        self.assertEqual(self.row(window="session")["net"], 100)
        self.assertEqual(self.row(window="15m")["net"], -2000)

    def test_events_received_after_asof_cannot_leak_into_old_snapshot(self):
        self.ingest(self.event(at=self.now), at=self.now + timedelta(seconds=5))
        self.assertIsNone(self.row(as_of=self.now)["net"])
        self.assertEqual(self.row(as_of=self.now + timedelta(seconds=5))["net"], 2100)

    def test_prior_date_after_close_and_future_timestamps_are_excluded(self):
        old = self.event(1, at=self.now - timedelta(days=1))
        future = self.event(2, at=self.now + timedelta(minutes=1))
        after_close = self.event(3, at=datetime(2026, 10, 21, 20, 1, tzinfo=UTC))
        self.assertEqual(self.ingest(old)["reason"], "outside_session")
        self.assertEqual(self.ingest(future)["reason"], "future_timestamp")
        self.assertEqual(self.ingest(after_close)["reason"], "outside_session")
        self.assertIsNone(self.row()["net"])

    def test_missing_timestamps_remain_visible_as_exclusions(self):
        self.ingest(self.event(date=None))
        self.ingest(self.event(2))
        report = self.store.snapshot(self.day, as_of=self.now)
        self.assertEqual(report["excluded_reasons"]["missing_or_invalid_timestamp"], 1)
        self.assertEqual(self.row()["excluded"], 1)
        self.assertEqual(self.row()["status"], "partial")

    def test_no_data_is_not_a_zero_bar_but_offsetting_trades_can_be_zero(self):
        self.assertIsNone(self.row()["net"])
        self.assertFalse(self.row()["available"])
        self.assertEqual(self.row("MISSING")["status"], "no_contracts")
        self.ingest(self.event(1, last="2", bid="1.9", ask="2"))
        self.ingest(self.event(2, last="2", bid="2", ask="2.1"))
        self.assertEqual(self.row()["net"], 0)
        self.assertTrue(self.row()["available"])

    def test_universe_cannot_be_changed_during_session(self):
        changed = [dict(row) for row in self.contracts]
        changed[0]["strike"] = 101
        with self.assertRaises(ValueError):
            self.store.save_universe(self.day, changed, self.metadata)
        self.store.save_universe(self.day, self.contracts, dict(self.metadata, collection_start=self.now.isoformat()))
        self.assertEqual(self.store.load_universe(self.day)["metadata"]["collection_start"], self.opening.isoformat(timespec="microseconds"))

    def test_loaded_universe_cannot_mutate_cached_cohort(self):
        loaded = self.store.load_universe(self.day)
        loaded["contracts"][0]["ticker"] = "MUTATED"
        self.ingest(self.event())
        self.assertEqual(self.row()["net"], 2100)

    def test_gap_marks_only_overlapping_window_partial(self):
        self.ingest(self.event(1, at=self.now - timedelta(minutes=3)), at=self.now - timedelta(minutes=3))
        self.ingest(self.event(2, type="quote"), at=self.now)
        self.store.record_gap(self.day, self.now - timedelta(minutes=25), self.now - timedelta(minutes=20), "connection lost")
        self.assertEqual(self.row()["status"], "partial")
        self.assertEqual(self.row(window="15m")["status"], "observed")

    def test_quote_traffic_keeps_feed_fresh_without_new_execution(self):
        self.ingest(self.event(at=self.now - timedelta(minutes=10)), at=self.now - timedelta(minutes=10))
        self.assertEqual(self.row()["status"], "stale")
        self.ingest(self.event(type="quote"), at=self.now)
        report = self.store.snapshot(self.day, as_of=self.now)
        self.assertEqual(self.row()["status"], "observed")
        self.assertEqual(report["last_event_at"], (self.now - timedelta(minutes=10)).isoformat(timespec="microseconds"))
        self.assertEqual(report["last_received_at"], self.now.isoformat(timespec="microseconds"))

    def test_new_york_dates_dst_and_early_close_bounds(self):
        summer_open, _ = session_bounds("2026-10-21")
        winter_open, _ = session_bounds("2026-12-01")
        _, early_close = session_bounds("2026-11-27")
        self.assertEqual(summer_open.hour, 13)
        self.assertEqual(winter_open.hour, 14)
        self.assertEqual(early_close.hour, 18)  # 13:00 EST.
        with self.assertRaises(ValueError):
            session_bounds("2026-10-24")
        # ET's date, not UTC midnight, defines the market session.
        report = self.store.snapshot(self.day, as_of=datetime(2026, 10, 22, 1, tzinfo=UTC))
        self.assertEqual(report["session_date"], "2026-10-21")
        self.assertEqual(report["as_of"], "2026-10-21T20:00:00.000000+00:00")

    def test_explicit_early_close_override_is_respected(self):
        opening, closing = session_bounds(self.day, {"session_close": "2026-10-21T13:00:00-04:00"})
        self.assertEqual(opening.hour, 13)
        self.assertEqual(closing.hour, 17)

    def test_premarket_snapshot_does_not_report_a_future_asof(self):
        before_open = self.opening - timedelta(minutes=30)
        report = self.store.snapshot(self.day, as_of=before_open)
        self.assertEqual(report["as_of"], before_open.isoformat(timespec="microseconds"))
        self.assertEqual(report["status"], "no_data")
        self.assertTrue(all(row["net"] is None for row in report["rows"]))

    def test_naive_times_are_rejected(self):
        with self.assertRaises(ValueError):
            self.store.ingest(self.day, self.event(), datetime(2026, 10, 21, 10))
        with self.assertRaises(ValueError):
            self.store.snapshot(self.day, as_of=datetime(2026, 10, 21, 10))


if __name__ == "__main__":
    unittest.main()
