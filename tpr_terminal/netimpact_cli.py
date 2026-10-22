"""Local equivalents of /netimpact, plus the shared collector process."""
from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from pathlib import Path

from .netimpact_config import NetImpactSettings
from .netimpact_service import NetImpactService


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The Printing Room net-impact collector and local command preview")
    commands = parser.add_subparsers(dest="command", required=True)
    chart = commands.add_parser("chart", help="Render the same report used by /netimpact")
    chart.add_argument("--mode", choices=("flow", "close_estimate"), default="flow")
    chart.add_argument("--window", choices=("session", "15m", "60m"), default="session")
    chart.add_argument("--output", type=Path, required=True)
    chart.add_argument("--report", type=Path, help="Also save the underlying report JSON")
    commands.add_parser("collect", help="Run one market-data collector until stopped")
    status = commands.add_parser("status", help="Read recorded flow without connecting to Tradier")
    status.add_argument("--window", choices=("session", "15m", "60m"), default="session")
    args = parser.parse_args(argv)
    try:
        settings = NetImpactSettings.from_env()
        if args.command == "collect":
            from .netimpact_core import FlowStore
            from .tradier_flow import FlowCollector, TradierClient
            stop = threading.Event()
            signal.signal(signal.SIGINT, lambda *_: stop.set())
            signal.signal(signal.SIGTERM, lambda *_: stop.set())
            collector = FlowCollector(TradierClient(settings.read_token()), FlowStore(settings.db_path), list(settings.tickers))
            print("Net-impact collector started. Market data only; keep this process running during the session.", flush=True)
            collector.run(stop)
            return 0
        service = NetImpactService(settings)
        if args.command == "status":
            print(json.dumps(service.flow_report(args.window), indent=2))
            return 0
        report = service.report(args.window, args.mode)
        png = service.render(report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(png)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"mode": report["mode"], "session_date": report["session_date"],
                          "status": report["status"], "tickers": len(report["rows"]),
                          "available_tickers": sum(bool(r.get("available")) for r in report["rows"]),
                          "source_as_of": report.get("source_as_of", report.get("as_of")),
                          "png": str(args.output.resolve())}, indent=2))
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        # Provider exception payloads or unexpected tracebacks could contain credentials.
        print(f"Net-impact operation failed ({type(exc).__name__}). Check local configuration and data availability.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
