from __future__ import annotations

import copy
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from PIL import Image

from tpr_terminal.netimpact_config import DEFAULT_TICKERS, NetImpactSettings
from tpr_terminal.netimpact_service import NetImpactService, close_estimate_report

DAY = date(2026, 10, 21)
CLOSE = datetime(2026, 10, 21, 20, tzinfo=UTC)


def close_universe():
    contracts = []
    for kind, price, volume in (("call", 2.0, 100), ("put", 3.0, 50)):
        contracts.append({
            "ticker": "TEST", "symbol": "TEST261023" + ("C" if kind == "call" else "P") + "00100000",
            "option_type": kind, "expiration_date": "2026-10-23", "strike": 100.0,
            "contract_size": 100, "volume": volume, "bid": price - 0.10,
            "ask": price + 0.10, "bid_date": int(CLOSE.timestamp() * 1000),
            "ask_date": int(CLOSE.timestamp() * 1000),
        })
    return {"contracts": contracts, "metadata": {"tickers": ["TEST", "MISSING"], "scope": "Test fixture coverage"}}


class ServiceIntegrationTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / ".pytest_cache"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.assertTrue(Path(self.temp.name).resolve().is_relative_to(scratch.resolve()))
        self.addCleanup(self.temp.cleanup)
        self.settings = NetImpactSettings(db_path=Path(self.temp.name) / "flow.sqlite3")

    def test_default_configuration_is_twenty_unique_names(self):
        self.assertEqual(len(DEFAULT_TICKERS), 20)
        self.assertEqual(len(set(DEFAULT_TICKERS)), 20)

    def test_no_collector_yields_unavailable_names_and_does_not_need_token(self):
        service = NetImpactService(self.settings)
        with patch.object(NetImpactSettings, "read_token", side_effect=AssertionError("No network/token needed")):
            report = service.flow_report(now=CLOSE + timedelta(hours=2))
        self.assertEqual(report["mode"], "flow")
        self.assertEqual(report["session_date"], DAY.isoformat())
        self.assertEqual(len(report["rows"]), 20)
        self.assertTrue(all(row["net"] is None and not row["available"] for row in report["rows"]))

    def test_premarket_selects_previous_completed_session(self):
        service = NetImpactService(self.settings)
        # Monday 09:15 EDT should use Friday, not Saturday or future Monday trades.
        report = service.flow_report(now=datetime(2026, 10, 26, 13, 15, tzinfo=UTC))
        self.assertEqual(report["session_date"], "2026-10-23")
        self.assertEqual(datetime.fromisoformat(report["as_of"]).date(), date(2026, 10, 23))

    def test_closing_estimate_uses_marks_but_never_claims_execution_premium(self):
        universe = close_universe()
        before = copy.deepcopy(universe)
        fetched = CLOSE + timedelta(hours=3)
        report = close_estimate_report(universe, DAY, fetched)
        rows = {row["ticker"]: row for row in report["rows"]}
        self.assertEqual(report["mode"], "close_estimate")
        self.assertEqual(rows["TEST"]["net_call"], 20_000)
        self.assertEqual(rows["TEST"]["net_put"], 15_000)
        self.assertEqual(rows["TEST"]["net"], 5_000)
        self.assertIsNone(rows["TEST"]["trades"])
        self.assertFalse(rows["MISSING"]["available"])
        self.assertIsNone(rows["MISSING"]["net"])
        self.assertEqual(datetime.fromisoformat(report["source_as_of"]), CLOSE)
        self.assertEqual(datetime.fromisoformat(report["fetched_at"]), fetched)
        self.assertIn("not executed premium", " ".join(report["notes"]))
        self.assertEqual(universe, before)

    def test_one_missing_mark_withholds_whole_ticker_instead_of_biased_ranking(self):
        universe = close_universe()
        universe["contracts"][1]["ask"] = None
        report = close_estimate_report(universe, DAY, CLOSE + timedelta(hours=2))
        row = next(item for item in report["rows"] if item["ticker"] == "TEST")
        self.assertFalse(row["available"])
        self.assertIsNone(row["net"])
        self.assertEqual(row["excluded"], 1)

    def test_same_day_opening_quote_is_not_a_usable_closing_mark(self):
        universe = close_universe()
        early = datetime(2026, 10, 21, 13, 31, tzinfo=UTC)
        universe["contracts"][1]["bid_date"] = int(early.timestamp() * 1000)
        universe["contracts"][1]["ask_date"] = int(early.timestamp() * 1000)
        report = close_estimate_report(universe, DAY, CLOSE + timedelta(hours=2))
        row = next(item for item in report["rows"] if item["ticker"] == "TEST")
        self.assertFalse(row["available"], "A 09:31 quote must not value the entire session as a closing mark")
        self.assertIsNone(row["net"])

    def test_close_estimate_refuses_rolling_windows_before_any_network_call(self):
        service = NetImpactService(self.settings)
        with patch.object(service, "close_report", side_effect=AssertionError("Must not fetch")):
            with self.assertRaises(ValueError):
                service.report(window="15m", mode="close_estimate")

    def test_close_cache_does_not_bypass_next_day_session_validation(self):
        service = NetImpactService(self.settings)
        # Same process/cache age, but a new premarket date must not serve yesterday silently.
        with patch.object(NetImpactSettings, "read_token", return_value="fixture-token"), \
             patch("tpr_terminal.tradier_flow.TradierClient"), \
             patch("tpr_terminal.tradier_flow.select_universe", return_value=close_universe()):
            service.close_report(now=datetime(2026, 10, 22, 3, 59, tzinfo=UTC))
            with self.assertRaises(ValueError):
                service.close_report(now=datetime(2026, 10, 22, 4, 1, tzinfo=UTC))

    def test_status_cli_reads_shared_store_without_credentials(self):
        from tpr_terminal.netimpact_cli import main
        captured = io.StringIO()
        with patch.object(NetImpactSettings, "from_env", return_value=self.settings), \
             patch.object(NetImpactSettings, "read_token", side_effect=AssertionError("Must not read token")), \
             redirect_stdout(captured):
            code = main(["status"])
        self.assertEqual(code, 0)
        self.assertIn('"mode": "flow"', captured.getvalue())
        self.assertIn('"available": false', captured.getvalue())


class _ImmediateContext:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


class _ImmediateLoop:
    async def run_in_executor(self, _executor, callback):
        return callback()


def _complete_inline(coroutine):
    """Drive mocked async boundaries without creating Windows socketpair loops."""
    try:
        yielded = coroutine.send(None)
    except StopIteration:
        return
    finally:
        coroutine.close()
    raise AssertionError(f"Unexpected external async work: {type(yielded).__name__}")


class DiscordCommandIntegrationTests(unittest.TestCase):
    def run_command(self, fixture, mode, *, permitted=True, provider_error=None):
        from tpr_terminal.bot import TerminalCog
        calls = []

        class Service:
            def report(self, window, requested_mode):
                calls.append((window, requested_mode))
                if provider_error:
                    raise provider_error
                return fixture

            def render(self, result):
                from tpr_terminal.netimpact_renderer import render_netimpact
                self.rendered = result
                return render_netimpact(result)

        bot = SimpleNamespace(
            settings=SimpleNamespace(max_upload_mb=10, elite_role_id=None, elite_role_name="Elite"),
            has_terminal_access=lambda _interaction: permitted,
            netimpact_service=Service(), analysis_executor=None, render_executor=None,
            analysis_semaphore=_ImmediateContext(), render_semaphore=_ImmediateContext(),
        )
        interaction = SimpleNamespace(
            response=SimpleNamespace(defer=AsyncMock(), send_message=AsyncMock()),
            followup=SimpleNamespace(send=AsyncMock()),
        )
        cog = TerminalCog(bot)
        with patch("tpr_terminal.bot.asyncio.get_running_loop", return_value=_ImmediateLoop()):
            _complete_inline(TerminalCog.netimpact.callback(cog, interaction, window="session", mode=mode))
        return interaction, calls

    def test_actual_command_callback_returns_close_estimate_png_and_warning(self):
        fixture = close_estimate_report(close_universe(), DAY, CLOSE + timedelta(hours=2))
        interaction, calls = self.run_command(fixture, "close_estimate")
        self.assertEqual(calls, [("session", "close_estimate")])
        interaction.response.defer.assert_awaited_once()
        interaction.followup.send.assert_awaited_once()
        sent = interaction.followup.send.await_args.kwargs
        self.assertIn("Close Activity Estimate", sent["embed"].title)
        self.assertIn("Trade direction is unknown", sent["embed"].description)
        with Image.open(sent["file"].fp) as image:
            self.assertEqual(image.format, "PNG")
            self.assertIn("Direction of trades unknown", image.info["Description"])
        sent["file"].close()

    def test_command_denies_unauthorized_user_without_fetching_or_rendering(self):
        interaction, calls = self.run_command({}, "flow", permitted=False)
        self.assertEqual(calls, [])
        interaction.response.send_message.assert_awaited_once()
        self.assertTrue(interaction.response.send_message.await_args.kwargs["ephemeral"])
        interaction.followup.send.assert_not_awaited()

    def test_command_does_not_echo_provider_exception_payload(self):
        sentinel = "private-provider-payload-not-for-chat"
        interaction, _ = self.run_command({}, "flow", provider_error=RuntimeError(sentinel))
        args = interaction.followup.send.await_args
        self.assertNotIn(sentinel, str(args))
        self.assertTrue(args.kwargs["ephemeral"])


if __name__ == "__main__":
    unittest.main()
