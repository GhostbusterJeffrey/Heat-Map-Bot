"""Presentation-only rendering of observed flow or explicitly labelled estimates."""
from __future__ import annotations

import io
import math
import textwrap
import threading
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator

BACKGROUND = "#070709"
FOREGROUND = "#f2f4f7"
MUTED = "#929aa8"
CYAN = "#55d4e7"
POSITIVE = "#31b892"
NEGATIVE = "#ef4c68"
GOLD = "#f5ca69"
_RENDER_LOCK = threading.RLock()
_EASTERN = ZoneInfo("America/New_York")


def _money(value: float, *, signed: bool = False) -> str:
    """Dollar labels with preserved sign and no misleading rounded negative zero."""
    number = float(value)
    if not math.isfinite(number):
        return "N/A"
    absolute = abs(number)
    if absolute < 0.005:
        return "$0"
    sign = "−" if number < 0 else "+" if signed else ""
    if absolute >= 1_000_000_000:
        amount = f"{absolute / 1_000_000_000:.2f}".rstrip("0").rstrip(".") + "B"
    elif absolute >= 1_000_000:
        amount = f"{absolute / 1_000_000:.2f}".rstrip("0").rstrip(".") + "M"
    elif absolute >= 1_000:
        amount = f"{absolute / 1_000:.1f}".rstrip("0").rstrip(".") + "K"
    elif absolute >= 1:
        amount = f"{absolute:,.2f}".rstrip("0").rstrip(".")
    else:
        amount = f"{absolute:.2f}"
    return f"{sign}${amount}"


def _local_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(_EASTERN)


def _timestamp(value: Any) -> str:
    local = _local_timestamp(value)
    if local is None:
        return "unavailable"
    hour = local.hour % 12 or 12
    meridiem = "AM" if local.hour < 12 else "PM"
    return f"{local:%b} {local.day}, {local.year} · {hour}:{local:%M:%S} {meridiem} {local:%Z}"


def _source_range(start: Any, end: Any) -> str:
    first, last = _local_timestamp(start), _local_timestamp(end)
    if first is None or last is None or first == last:
        return _timestamp(end)
    if first.date() != last.date() or first.tzname() != last.tzname():
        return f"{_timestamp(start)} – {_timestamp(end)}"
    first_time = f"{first.hour % 12 or 12}:{first:%M:%S} {'AM' if first.hour < 12 else 'PM'}"
    last_time = f"{last.hour % 12 or 12}:{last:%M:%S} {'AM' if last.hour < 12 else 'PM'}"
    return f"{first:%b} {first.day}, {first.year} · {first_time}–{last_time} {last:%Z}"


def _mode_labels(mode: str) -> dict[str, str]:
    if mode == "close_estimate":
        return {
            "badge": "CLOSE ACTIVITY ESTIMATE",
            "subtitle": "Call–put volume × closing mark • Direction of trades unknown",
            "positive": "Call-heavy activity",
            "negative": "Put-heavy activity",
            "axis": "Estimated call premium − estimated put premium ($)",
            "empty": "No usable closing observations",
            "color": GOLD,
        }
    if mode == "demo":
        return {
            "badge": "DEMONSTRATION · SYNTHETIC DATA",
            "subtitle": "Layout demonstration only • These values are not market observations",
            "positive": "Positive sample",
            "negative": "Negative sample",
            "axis": "Demonstration value ($)",
            "empty": "No demonstration rows",
            "color": GOLD,
        }
    if mode != "flow":
        raise ValueError("Unsupported net impact report mode")
    return {
        "badge": "OBSERVED OPTIONS FLOW",
        "subtitle": "Signed execution premium • Trade direction inferred from execution-time bid/ask",
        "positive": "Positive net premium",
        "negative": "Negative net premium",
        "axis": "Net call premium − net put premium ($)",
        "empty": "No classified trades available",
        "color": CYAN,
    }


def _valid_rows(report: dict) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    unavailable: list[str] = []
    for original in report.get("rows", []):
        if not isinstance(original, dict):
            continue
        ticker = str(original.get("ticker", "Unknown")).strip() or "Unknown"
        value = original.get("net")
        # Booleans are not monetary values, even though float(True) succeeds.
        if original.get("available") is False or value is None or isinstance(value, bool):
            unavailable.append(ticker)
            continue
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            unavailable.append(ticker)
            continue
        if not math.isfinite(number):
            unavailable.append(ticker)
            continue
        rows.append(dict(original, ticker=ticker, net=number))
    return sorted(rows, key=lambda row: (-row["net"], row["ticker"])), unavailable


def _build_figure(report: dict) -> Figure:
    mode = str(report.get("mode", "flow"))
    labels = _mode_labels(mode)
    rows, unavailable = _valid_rows(report)
    notes = [str(note).strip() for note in report.get("notes", []) if str(note).strip()]
    priority_notes: list[str] = []
    if mode == "close_estimate":
        priority_notes.append("Daily volume × final observed midpoint; estimated activity, not reconstructed buy/sell flow.")
        cohort_note = next((note for note in notes if "after-close" in note.lower()),
                           "Closing estimates do not reconstruct the opening contract universe.")
        priority_notes.append(cohort_note)
        notes = [note for note in notes if note != cohort_note and not note.startswith("Daily contract volume")]
    if mode == "flow":
        priority_notes.append("Coverage is limited to tracked contracts. Midpoint and unclassifiable executions do not add signed premium.")
    if unavailable:
        priority_notes.append("Unavailable (excluded from ranking): " + ", ".join(unavailable))
    # The JSON report retains full diagnostics; the image prioritizes three
    # readable caveats, including cohort and availability limitations.
    notes = list(dict.fromkeys(priority_notes + notes))[:3]
    note_lines: list[str] = []
    for note in notes:
        note_lines.extend(textwrap.wrap(note, width=145, break_long_words=False) or [""])
    scope_lines = textwrap.wrap(str(report.get("scope") or "Tracked options only"), width=115, break_long_words=False)
    scope_lines = scope_lines or ["Tracked options only"]
    header_height = 2.18 + max(0, len(scope_lines) - 1) * 0.18
    footer_height = 1.10 + len(note_lines) * 0.16
    plot_height = max(2.25, len(rows) * 0.30)
    height = header_height + plot_height + footer_height
    figure = Figure(figsize=(11.5, height), facecolor=BACKGROUND, dpi=160)
    FigureCanvasAgg(figure)

    def text(x: float, inches_from_top: float, message: str, **kwargs):
        kwargs.setdefault("color", FOREGROUND)
        return figure.text(x, 1 - inches_from_top / height, message,
                           fontfamily="DejaVu Sans", va="top", **kwargs)

    text(0.055, 0.24, "THE PRINTING ROOM", fontsize=10, fontweight="bold", color=GOLD)
    text(0.055, 0.52, "NET IMPACT", fontsize=25, fontweight="bold")
    text(0.945, 0.61, labels["badge"], fontsize=9, fontweight="bold", ha="right",
         color=labels["color"], bbox={"boxstyle": "round,pad=0.6", "facecolor": "#11141a", "edgecolor": labels["color"], "linewidth": 0.7})
    text(0.055, 1.04, labels["subtitle"], fontsize=10.3, color=FOREGROUND)
    text(0.055, 1.34, "\n".join(scope_lines), fontsize=8.6, color=MUTED, linespacing=1.6)
    source_stamp = report.get("source_as_of") or report.get("as_of")
    # Deliberately show report time, never wall-clock render time.
    text(0.055, header_height - 0.53,
         f"Session {report.get('session_date', 'unavailable')}  •  {str(report.get('window', 'session')).upper()}",
         fontsize=8.2, color=MUTED)
    time_text = ("Source marks " + _source_range(report.get("source_from"), source_stamp)) if mode == "close_estimate" else "As of " + _timestamp(source_stamp)
    text(0.055, header_height - 0.29, time_text, fontsize=8.2, color=MUTED)
    status = str(report.get("status") or "unverified").replace("_", " ").upper()
    text(0.945, header_height - 0.53, status, fontsize=8, fontweight="bold", ha="right",
         color=GOLD if mode != "flow" or status not in {"LIVE", "COMPLETE", "COLLECTING"} else POSITIVE)

    axes = figure.add_axes([0.14, footer_height / height, 0.79, plot_height / height], facecolor=BACKGROUND)
    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.set_axisbelow(True)
    axes.grid(axis="y", color="#3e4651", linewidth=0.6, linestyle=(0, (1, 4)), alpha=0.75)
    axes.axvline(0, color="#6a7380", linewidth=0.85, zorder=1)
    largest = max((abs(row["net"]) for row in rows), default=0)
    span = largest * 1.28 if largest else 1.0
    axes.set_xlim(-span, span)
    axes.xaxis.set_major_locator(MaxNLocator(nbins=7, symmetric=True, prune="both"))
    axes.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: _money(value)))
    axes.tick_params(axis="x", colors="#cbd0d9", labelsize=8.3, length=0, pad=9)
    axes.tick_params(axis="y", colors=CYAN, labelsize=10, length=0, pad=13)
    axes.set_xlabel(labels["axis"], color="#cbd0d9", fontsize=8.7, labelpad=12)
    if rows:
        offsets = list(range(len(rows)))
        axes.set_yticks(offsets, [row["ticker"] for row in rows])
        axes.set_ylim(len(rows) - 0.45, -0.55)
        for offset, row in enumerate(rows):
            value = row["net"]
            color = POSITIVE if value > 0 else NEGATIVE if value < 0 else MUTED
            axes.barh(offset, value, height=0.53, color=color, zorder=3)
            if value == 0:
                axes.scatter([0], [offset], s=12, color=MUTED, zorder=4)
            padding = span * 0.02
            axes.text(value + (padding if value >= 0 else -padding), offset, _money(value, signed=True),
                      ha="left" if value >= 0 else "right", va="center", color=FOREGROUND,
                      fontsize=9.0, fontfamily="DejaVu Sans", clip_on=False)
    else:
        axes.set_yticks([])
        axes.set_xticks([])
        axes.set_xlabel("")
        axes.text(0.5, 0.57, labels["empty"], transform=axes.transAxes,
                  color=FOREGROUND, ha="center", va="center", fontsize=17, fontweight="bold")
        axes.text(0.5, 0.39, "Missing observations are not zero net impact.", transform=axes.transAxes,
                  color=MUTED, ha="center", va="center", fontsize=10)

    legend_top = header_height + plot_height + 0.62
    text(0.055, legend_top, "●  " + labels["positive"], fontsize=8.5, color=POSITIVE)
    text(0.31, legend_top, "●  " + labels["negative"], fontsize=8.5, color=NEGATIVE)
    text(0.945, legend_top, f"{len(rows)} shown  /  {len(rows) + len(unavailable)} tracked", fontsize=8.5,
         color=MUTED, ha="right")
    if note_lines:
        text(0.055, legend_top + 0.28, "\n".join(note_lines), fontsize=7.4, color=MUTED, linespacing=1.35)
    return figure


def render_netimpact(report: dict) -> bytes:
    """Return a PNG using report provenance and explicit mode semantics.

    Rendering performs no network requests and does not mutate the report.
    """
    with _RENDER_LOCK:
        figure = _build_figure(report)
        try:
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", facecolor=BACKGROUND, dpi=160,
                           metadata={"Title": "The Printing Room — Net Impact",
                                     "Description": _mode_labels(str(report.get("mode", "flow")))["subtitle"]})
            return buffer.getvalue()
        finally:
            figure.clear()
