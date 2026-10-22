"""Run from the bot's application directory inside its running container.

Reads market data only. Never prints tokens, provider bodies, or tracebacks.
Does not start the collector or send Discord messages.
"""
import json
import sys
from pathlib import Path


def emit(check, **details):
    print(json.dumps({"check": check, **details}), flush=True)


def main():
    sys.path.insert(0, str(Path.cwd()))
    stage = "imports"
    try:
        from tpr_terminal.netimpact_config import NetImpactSettings
        from tpr_terminal.netimpact_service import NetImpactService
        from tpr_terminal.tradier_flow import TradierClient, TradierDataError
        import tpr_terminal.renderer as renderer

        emit("python", version=sys.version.split()[0])
        emit("renderer", positive_node_color=renderer.HeatmapRenderer._node_highlight_style("positive", 1)[0])
        stage = "environment settings"
        settings = NetImpactSettings.from_env()
        stage = "token configuration"
        token = settings.read_token()
        emit("token", configured=True)
        stage = "Tradier market clock"
        TradierClient(token).clock()
        emit("Tradier market clock", success=True)
        stage = "close estimate (session window)"
        service = NetImpactService(settings)
        report = service.report("session", "close_estimate")
        emit("close estimate", success=True, available_tickers=sum(bool(r.get("available")) for r in report["rows"]))
        stage = "chart rendering"
        png = service.render(report)
        emit("chart rendering", success=True, bytes=len(png))
    except Exception as exc:
        details = {"success": False, "stage": stage, "exception": type(exc).__name__}
        # Only the provider's deliberately sanitized exception fields are exposed.
        if "TradierDataError" in locals() and isinstance(exc, TradierDataError):
            allowed = {"request_rejected", "timeout", "connection_failed", "invalid_response", "retry_limit", "missing_clock"}
            details["category"] = exc.category if exc.category in allowed else "other_provider_error"
            details["http_status"] = exc.status if isinstance(exc.status, int) else None
        if stage == "token configuration":
            details["hint"] = "Token missing/invalid or configured secret file unreadable in this container."
        emit("failure", **details)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
