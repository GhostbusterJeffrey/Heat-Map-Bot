from __future__ import annotations

import io
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from .branding import BrandingManager
from .models import AnalysisResult, WallLevel

logger = logging.getLogger(__name__)
DARK = np.asarray(mcolors.to_rgb("#070709"))
BACKGROUND = "#070709"

# Fixed semantic exposure thresholds. The base renderer stays dark and
# structural while continuous ramps communicate magnitude within each sign.
POSITIVE_GOLD_THRESHOLD = 1_000_000_000.0
POSITIVE_BLUE_THRESHOLD = 10_000_000_000.0

NEGATIVE_RED_PEAK = 300_000_000.0
NEGATIVE_BLACK_START = 500_000_000.0
NEGATIVE_BLACK_THRESHOLD = 1_000_000_000.0
MICRO_EXPOSURE_MAX = 500_000.0
MICRO_DISPLAY_FLOOR = 5_000.0


class HeatmapRenderer:
    def __init__(
        self,
        *,
        assets_dir: Path | None = None,
        main_logo_path: Path | None = None,
        developer_logo_path: Path | None = None,
        trader_logo_path: Path | None = None,
        sibei_logo_path: Path | None = None,
        jeff_logo_path: Path | None = None,
    ) -> None:
        # Individual-path arguments are retained for backward compatibility.
        # Alpha 4.0 discovers supported file extensions by asset stem.
        if assets_dir is None:
            supplied = next(
                (
                    path.parent
                    for path in (
                        main_logo_path,
                        developer_logo_path,
                        trader_logo_path,
                        sibei_logo_path,
                        jeff_logo_path,
                    )
                    if path is not None
                ),
                Path(__file__).resolve().parent.parent / "assets",
            )
            assets_dir = supplied
        self._branding = BrandingManager(assets_dir)

    def render(self, result: AnalysisResult, strikes_count: int) -> tuple[bytes, int]:
        records = result.matrix_records
        matrix = self._pivot(records, "gex")
        quality = self._pivot(records, "cell_quality")
        repairs = self._pivot(records, "iv_repair_ratio")
        fallbacks = self._pivot(records, "volume_fallback_ratio")
        ordered_columns = sorted(
            matrix.columns,
            key=lambda value: pd.to_datetime(value, format="%m/%d/%Y"),
        )
        matrix = matrix[ordered_columns]
        quality = quality.reindex(index=matrix.index, columns=ordered_columns, fill_value=0.0)
        repairs = repairs.reindex(index=matrix.index, columns=ordered_columns, fill_value=0.0)
        fallbacks = fallbacks.reindex(index=matrix.index, columns=ordered_columns, fill_value=0.0)

        selected = self._select_strikes(
            matrix,
            result.spot,
            strikes_count,
            self._important_levels(result),
        )
        matrix = matrix.loc[selected].sort_index(ascending=False)
        quality = quality.loc[matrix.index, matrix.columns]
        repairs = repairs.loc[matrix.index, matrix.columns]
        fallbacks = fallbacks.loc[matrix.index, matrix.columns]

        rows, cols = matrix.shape
        if rows == 0 or cols == 0:
            raise ValueError("The selected heatmap matrix is empty.")

        scale = self._color_scale(matrix.to_numpy(dtype=float))
        matrix_width = max(13.5, cols * 1.82)
        side_width = 6.15
        total_width = matrix_width + side_width
        figure_height = max(9.8, rows * 0.415 + 2.75)
        figure = plt.figure(figsize=(total_width, figure_height))
        figure.patch.set_facecolor(BACKGROUND)
        # Reserve a real header band rather than floating title and branding
        # against the canvas edge. The matrix begins below both the title and
        # the horizontal expiration labels.
        grid = figure.add_gridspec(
            1,
            2,
            width_ratios=[matrix_width, side_width],
            left=0.044,
            right=0.986,
            top=0.785,
            bottom=0.075,
            wspace=0.034,
        )
        axis = figure.add_subplot(grid[0, 0])
        info_axis = figure.add_subplot(grid[0, 1])
        axis.set_facecolor("#070709")
        info_axis.set_facecolor("#0d0d12")

        cell_font = float(np.clip(9.1 - max(rows - 55, 0) * 0.025, 6.8, 9.1))
        marker_font = max(4.8, cell_font * 0.64)
        markers_enabled = rows * cols <= 650
        node_ranks = self._node_rank_lookup(matrix, limit=3)

        for col_idx in range(cols):
            if col_idx % 2:
                axis.add_patch(
                    Rectangle(
                        (col_idx, 0),
                        1,
                        rows,
                        facecolor="#111117",
                        edgecolor="none",
                        alpha=0.12,
                        zorder=0,
                    )
                )

        for row_idx, strike in enumerate(matrix.index):
            y = rows - 1 - row_idx
            for col_idx, expiry in enumerate(matrix.columns):
                value = float(matrix.loc[strike, expiry])
                q = float(np.clip(quality.loc[strike, expiry], 0.0, 1.0))
                background, text, border = self._cell_colors(value, scale)
                magnitude_alpha = self._magnitude_alpha(value)
                quality_alpha = self._quality_alpha(value, q)
                background = self._dim_color(
                    background,
                    magnitude_alpha * quality_alpha,
                )
                axis.add_patch(
                    Rectangle(
                        (col_idx, y),
                        1,
                        1,
                        facecolor=background,
                        edgecolor=border,
                        linewidth=0.70,
                        zorder=2,
                    )
                )
                axis.text(
                    col_idx + 0.5,
                    y + 0.52,
                    self._format_amount(value, floor=scale["floor"]),
                    ha="center",
                    va="center",
                    color=text,
                    fontsize=cell_font,
                    fontweight="bold",
                    zorder=4,
                )
                if markers_enabled:
                    marker = ""
                    if float(fallbacks.loc[strike, expiry]) >= 0.25:
                        marker += "v"
                    if float(repairs.loc[strike, expiry]) >= 0.25:
                        marker += "•"
                    if marker:
                        axis.text(
                            col_idx + 0.92,
                            y + 0.88,
                            marker,
                            ha="right",
                            va="top",
                            color="#d0d0db",
                            fontsize=marker_font,
                            alpha=0.75,
                            zorder=5,
                        )
                ranked_node = node_ranks.get((strike, expiry))
                if ranked_node is not None:
                    node_side, node_rank = ranked_node
                    node_color, node_width, node_alpha, node_marker = (
                        self._node_highlight_style(node_side, node_rank)
                    )
                    axis.add_patch(
                        Rectangle(
                            (col_idx + 0.035, y + 0.035),
                            0.93,
                            0.93,
                            facecolor="none",
                            edgecolor=node_color,
                            linewidth=node_width,
                            alpha=node_alpha,
                            zorder=8,
                        )
                    )
                    if markers_enabled:
                        axis.text(
                            col_idx + 0.08,
                            y + 0.90,
                            node_marker,
                            ha="left",
                            va="top",
                            color=node_color,
                            fontsize=marker_font,
                            fontweight="bold",
                            alpha=node_alpha,
                            zorder=9,
                        )

        for x in range(cols + 1):
            axis.axvline(
                x,
                color="#353540",
                linewidth=1.12 if x else 0.82,
                alpha=0.78,
                zorder=6,
            )

        row_map = {float(strike): rows - 1 - idx for idx, strike in enumerate(matrix.index)}
        self._draw_overlays(axis, result, row_map, cols, matrix.index)

        axis.set_xlim(-0.34, cols)
        axis.set_ylim(0, rows)
        axis.set_xticks(np.arange(cols) + 0.5)
        expiry_labels = [self._format_expiry_label(value) for value in matrix.columns]
        expiry_font = self._expiry_font_size(matrix_width, cols)
        axis.set_xticklabels(
            expiry_labels,
            color="#c4c4d0",
            fontsize=expiry_font,
            fontweight="semibold",
            rotation=0,
            ha="center",
            va="bottom",
        )
        axis.xaxis.tick_top()
        axis.tick_params(axis="x", pad=10)
        expiry_tick_labels = axis.get_xticklabels()
        if expiry_tick_labels:
            expiry_tick_labels[0].set_color("#ffffff")
            expiry_tick_labels[0].set_fontweight("bold")
        axis.set_yticks(np.arange(rows) + 0.5)
        axis.set_yticklabels(
            [self._format_strike(value) for value in reversed(matrix.index)],
            color="#9898aa",
            fontsize=float(np.clip(10.1 - max(rows - 65, 0) * 0.025, 7.8, 10.1)),
            fontweight="bold",
        )
        axis.tick_params(axis="both", length=0)
        nearest = min(matrix.index, key=lambda value: abs(float(value) - result.spot))
        for label in axis.get_yticklabels():
            try:
                if float(label.get_text()) == float(nearest):
                    label.set_color("#00f0ff")
            except ValueError:
                pass
        for spine in axis.spines.values():
            spine.set_visible(False)

        self._draw_info_panel(info_axis, result, figure_height)
        self._draw_header(figure, result, total_width, figure_height)
        footer_font = float(np.clip(total_width * 0.27, 6.8, 8.4))
        figure.text(
            0.044,
            0.012,
            "Cyan = spot  •  Blue P1/P2/P3 = ranked positive nodes  •  Purple N1/N2/N3 = ranked negative nodes  •  Pale-blue dash = gamma flip  •  Lime/magenta dots = structural walls  •  Amber = estimated pin\n"
            "Exposure colors: positive dark→light green, gold ≥+$1B, bright blue ≥+$10B; negative dark→hot red, fading to black by -$1B. "
            "Micro intensity is logarithmic from $0–$500K. Dim cells carry weaker input quality. v = volume fallback, • = IV repair. "
            "Net exposure remains a public-data sign-convention proxy, not observed dealer inventory.",
            color="#77778a",
            fontsize=footer_font,
            ha="left",
            va="bottom",
        )

        output = io.BytesIO()
        figure.savefig(
            output,
            format="png",
            dpi=150,
            bbox_inches="tight",
            pad_inches=0.16,
            facecolor="#070709",
            edgecolor="none",
        )
        plt.close(figure)
        output.seek(0)
        return output.read(), rows

    @staticmethod
    def _pivot(records: pd.DataFrame, value: str) -> pd.DataFrame:
        return records.pivot_table(
            index="strike",
            columns="expiry",
            values=value,
            aggfunc="sum" if value == "gex" else "mean",
            fill_value=0.0,
        )

    @staticmethod
    def _important_levels(result: AnalysisResult) -> list[float]:
        values: list[float] = []
        for item in (
            result.levels.positive_wall,
            result.levels.negative_wall,
            result.levels.primary_pin,
        ):
            if item is not None:
                values.append(float(item.strike))
        if result.levels.primary_gamma_flip is not None:
            values.append(float(result.levels.primary_gamma_flip.level))
        return values

    def _draw_overlays(
        self,
        axis,
        result: AnalysisResult,
        row_map: dict[float, int],
        cols: int,
        displayed_strikes: pd.Index,
    ) -> None:
        available = np.asarray([float(value) for value in displayed_strikes], dtype=float)
        if available.size == 0:
            return

        def nearest_row(level: float | None) -> tuple[float, float] | None:
            if level is None or level < available.min() or level > available.max():
                return None
            strike = float(available[np.argmin(np.abs(available - level))])
            return strike, float(row_map[strike])

        spot_row = nearest_row(result.spot)
        flip_row = nearest_row(result.levels.gamma_flip)
        pos_row = nearest_row(result.levels.positive_wall_strike)
        neg_row = nearest_row(result.levels.negative_wall_strike)
        pin_row = nearest_row(
            None if result.levels.primary_pin is None else result.levels.primary_pin.strike
        )
        used: dict[float, int] = {}

        def draw(row, color: str, style: str, width: float, label: str, offset: float) -> None:
            if row is None:
                return
            strike, y = row
            collision = used.get(strike, 0)
            used[strike] = collision + 1
            y_line = y + offset + collision * 0.055
            axis.hlines(
                y_line,
                0,
                cols,
                colors=color,
                linestyles=style,
                linewidth=width,
                alpha=0.95,
                zorder=9,
            )
            axis.text(
                -0.06,
                y + 0.5,
                label,
                color=color,
                fontsize=7.0,
                fontweight="bold",
                ha="right",
                va="center",
                zorder=10,
            )

        draw(spot_row, "#00f0ff", "solid", 2.0, "SPOT", 0.02)
        draw(flip_row, "#8bdcff", "dashed", 1.7, "FLIP", 0.14)
        draw(pos_row, "#c8ff31", "dotted", 1.6, "+W", 0.27)
        draw(neg_row, "#ff4c88", "dotted", 1.6, "-W", 0.37)
        if pin_row is not None:
            _, y = pin_row
            axis.text(
                -0.20,
                y + 0.5,
                "◆",
                color="#ffb000",
                fontsize=10,
                ha="center",
                va="center",
                zorder=10,
            )

    def _draw_info_panel(self, axis, result: AnalysisResult, figure_height: float) -> None:
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#282835")
            spine.set_linewidth(1.0)
        levels = result.levels
        expected = result.expected_moves[0] if result.expected_moves else None
        scale = float(np.clip(figure_height / 18.0, 0.82, 1.15))

        axis.text(
            0.065,
            0.972,
            "TPR V4 INTELLIGENCE",
            transform=axis.transAxes,
            color="#b000ff",
            fontsize=13.8 * scale,
            fontweight="bold",
            va="top",
        )
        regime_color = (
            "#39ff14"
            if levels.total_net_gex > 0
            else "#ff0055"
            if levels.total_net_gex < 0
            else "#ffd700"
        )
        axis.text(
            0.065,
            0.928,
            levels.regime.upper(),
            transform=axis.transAxes,
            color=regime_color,
            fontsize=11.0 * scale,
            fontweight="bold",
            va="top",
        )

        y = 0.868
        y = self._panel_wall(axis, y, "POSITIVE STRUCTURAL WALL", levels.positive_wall, "#39ff14", scale)
        y = self._panel_wall(axis, y, "NEGATIVE STRUCTURAL WALL", levels.negative_wall, "#ff0055", scale)

        flip = levels.primary_gamma_flip
        axis.text(0.065, y, "PRIMARY GAMMA FLIP", transform=axis.transAxes, color="#77778a", fontsize=8.1 * scale, fontweight="bold")
        if flip is not None:
            axis.text(0.065, y - 0.031, f"${flip.level:,.2f}", transform=axis.transAxes, color="#00f0ff", fontsize=13.4 * scale, fontweight="bold")
            detail = f"{flip.distance:+.2f} ({flip.distance_pct:+.2%}) • Stability {flip.shock_stability:.0%}"
            axis.text(0.065, y - 0.058, detail, transform=axis.transAxes, color="#a8a8b8", fontsize=7.6 * scale)
            if flip.model_spread is not None:
                axis.text(0.065, y - 0.080, f"Sticky-model spread ${flip.model_spread:,.2f}", transform=axis.transAxes, color="#8f8fa2", fontsize=7.2 * scale)
        else:
            axis.text(0.065, y - 0.031, "—", transform=axis.transAxes, color="#77778a", fontsize=13.4 * scale)
        y -= 0.116

        pin = levels.primary_pin
        pin_heading = self._pin_heading(pin.pin_score) if pin is not None else "ESTIMATED PIN LEVEL"
        axis.text(0.065, y, pin_heading, transform=axis.transAxes, color="#77778a", fontsize=8.1 * scale, fontweight="bold")
        if pin is not None:
            axis.text(0.065, y - 0.031, f"${pin.strike:,.2f}", transform=axis.transAxes, color="#ffb000", fontsize=13.2 * scale, fontweight="bold")
            axis.text(0.065, y - 0.058, f"Score {pin.pin_score:.0f}/100 • {pin.label.upper()}", transform=axis.transAxes, color="#a8a8b8", fontsize=7.6 * scale)
        else:
            axis.text(0.065, y - 0.031, "—", transform=axis.transAxes, color="#77778a", fontsize=12.8 * scale)
        y -= 0.096

        axis.text(0.065, y, "TOTAL NET GEX", transform=axis.transAxes, color="#77778a", fontsize=8.1 * scale, fontweight="bold")
        axis.text(0.065, y - 0.033, self._format_amount(levels.total_net_gex), transform=axis.transAxes, color="#ffd700", fontsize=13.6 * scale, fontweight="bold")
        y -= 0.080

        axis.text(0.065, y, "EXPECTED MOVE", transform=axis.transAxes, color="#77778a", fontsize=8.1 * scale, fontweight="bold")
        if expected:
            axis.text(0.065, y - 0.032, f"±${expected.dollars:,.2f} ({expected.percent:.2%})", transform=axis.transAxes, color="#ffffff", fontsize=11.9 * scale, fontweight="bold")
            axis.text(0.065, y - 0.056, f"Through {expected.expiration:%m/%d/%Y} close", transform=axis.transAxes, color="#a0a0b0", fontsize=7.4 * scale)
        else:
            axis.text(0.065, y - 0.032, "—", transform=axis.transAxes, color="#77778a", fontsize=11.9 * scale)
        y -= 0.090

        axis.text(0.065, y, "MODEL / INPUT CONFIDENCE", transform=axis.transAxes, color="#77778a", fontsize=8.1 * scale, fontweight="bold")
        conf_color = "#39ff14" if result.confidence.score >= 80 else "#ffd700" if result.confidence.score >= 65 else "#ff0055"
        axis.text(0.065, y - 0.031, f"{result.confidence.score:.0f}/100 — {result.confidence.grade} / {result.confidence.label.upper()}", transform=axis.transAxes, color=conf_color, fontsize=10.2 * scale, fontweight="bold")
        y -= 0.076

        axis.text(0.065, y, "VOLATILITY SURFACE", transform=axis.transAxes, color="#77778a", fontsize=8.1 * scale, fontweight="bold")
        mode = result.surface.mode.upper().replace("_", "-")
        axis.text(0.065, y - 0.030, f"{result.surface_health.display} • {mode}", transform=axis.transAxes, color="#8bdcff", fontsize=10.0 * scale, fontweight="bold")
        axis.text(0.065, y - 0.054, f"Fitted {result.surface.fitted_contract_ratio:.0%} • Market IV {result.surface.market_solve_rate:.0%}", transform=axis.transAxes, color="#a0a0b0", fontsize=7.3 * scale)
        y -= 0.091

        axis.text(0.065, y, "LIQUIDITY", transform=axis.transAxes, color="#77778a", fontsize=8.1 * scale, fontweight="bold")
        axis.text(0.065, y - 0.029, result.liquidity.mode.upper(), transform=axis.transAxes, color="#d0d0db", fontsize=8.9 * scale, fontweight="bold")
        axis.text(0.065, y - 0.052, f"Exposure fallback {result.liquidity.volume_exposure_share:.0%}", transform=axis.transAxes, color="#a0a0b0", fontsize=7.3 * scale)
        y -= 0.084

        axis.text(0.065, y, "DATA QUALITY", transform=axis.transAxes, color="#77778a", fontsize=8.1 * scale, fontweight="bold")
        axis.text(0.065, y - 0.027, f"{result.quality.expirations_processed}/{result.quality.expirations_requested} expirations", transform=axis.transAxes, color="#bdbdca", fontsize=7.8 * scale)
        axis.text(0.065, y - 0.050, f"{result.quality.contracts_used:,}/{result.quality.contracts_seen:,} contracts", transform=axis.transAxes, color="#bdbdca", fontsize=7.8 * scale)
        axis.text(0.065, y - 0.071, f"Snapshot {result.snapshot_id}", transform=axis.transAxes, color="#77778a", fontsize=6.6 * scale)

    @staticmethod
    def _panel_wall(axis, y: float, label: str, wall: WallLevel | None, color: str, scale: float) -> float:
        axis.text(0.065, y, label, transform=axis.transAxes, color="#77778a", fontsize=8.1 * scale, fontweight="bold")
        if wall is not None:
            axis.text(0.065, y - 0.031, f"${wall.strike:,.2f}  {HeatmapRenderer._format_amount(wall.signed_exposure)}", transform=axis.transAxes, color=color, fontsize=12.1 * scale, fontweight="bold")
            axis.text(0.065, y - 0.056, f"{wall.stability.title()} • Breadth {wall.expiry_breadth:.0%} • Shock {wall.shock_stability:.0%}", transform=axis.transAxes, color="#a8a8b8", fontsize=7.2 * scale)
        else:
            axis.text(0.065, y - 0.031, "—", transform=axis.transAxes, color="#77778a", fontsize=12.1 * scale)
        return y - 0.101

    @staticmethod
    def _pin_heading(score: float) -> str:
        if score >= 80:
            return "VERY HIGH PIN PRESSURE"
        if score >= 65:
            return "HIGH PIN PRESSURE"
        if score >= 45:
            return "MODERATE PIN CANDIDATE"
        return "WEAK PIN CANDIDATE"

    def _draw_header(
        self,
        figure,
        result: AnalysisResult,
        total_width: float,
        figure_height: float,
    ) -> None:
        """Draw a collision-safe title, logo, and three-member branding block."""
        header_bottom = 0.815
        header_top = 0.978
        branding_x = float(
            np.clip(0.725 + max(total_width - 28.0, 0.0) * 0.0017, 0.720, 0.775)
        )
        main_size = float(np.clip((header_top - header_bottom) * 0.58, 0.064, 0.086))
        main_x = 0.018
        main_y = header_bottom + ((header_top - header_bottom) - main_size) / 2
        if self._branding.main_logo is not None:
            logo_axis = figure.add_axes([main_x, main_y, main_size, main_size], zorder=10)
            logo_axis.imshow(self._branding.main_logo)
            logo_axis.axis("off")
            title_x = main_x + main_size + 0.018
        else:
            title_x = 0.044

        title_right = branding_x - 0.026
        available_width_inches = max(total_width * (title_right - title_x), 3.0)
        surface_label = result.surface.mode.upper().replace("_", "-")
        title = f"THE PRINTING ROOM // {result.ticker} V4 ALPHA 4.2 DEV.1 NET GEX PROXY"
        subtitle = (
            f"SPOT ${result.spot:,.2f}  •  {surface_label} SURFACE / "
            "PARITY-CARRY / SCENARIO RISK"
        )
        title_font = self._fit_header_font(
            title,
            available_width_inches,
            preferred=float(np.clip(total_width * 0.68, 17.0, 23.0)),
            minimum=14.0,
            maximum=23.0,
        )
        subtitle_font = self._fit_header_font(
            subtitle,
            available_width_inches,
            preferred=title_font * 0.78,
            minimum=10.0,
            maximum=18.0,
        )
        # Lower the text into the reserved header instead of pressing it
        # against the save boundary.
        figure.text(
            title_x,
            header_top - 0.040,
            title,
            color="#00f0ff",
            fontsize=title_font,
            fontweight="bold",
            ha="left",
            va="top",
        )
        figure.text(
            title_x,
            header_top - 0.071,
            subtitle,
            color="#00f0ff",
            fontsize=subtitle_font,
            fontweight="bold",
            ha="left",
            va="top",
        )
        self._draw_branding(
            figure,
            total_width,
            figure_height,
            header_bottom=header_bottom,
            header_top=header_top,
            branding_x=branding_x,
        )

    def _draw_branding(
        self,
        figure,
        total_width: float,
        figure_height: float,
        *,
        header_bottom: float,
        header_top: float,
        branding_x: float,
    ) -> None:
        assets = self._branding.assets
        if not assets:
            return
        header_height = header_top - header_bottom
        gap = float(np.clip(header_height * 0.055, 0.005, 0.010))
        max_avatar_from_rows = (header_height - gap * (len(assets) - 1)) / len(assets)
        avatar = float(
            np.clip(
                min(0.62 / figure_height, max_avatar_from_rows * 0.88),
                0.025,
                0.046,
            )
        )
        font = float(
            np.clip(
                min(total_width * 0.40, avatar * figure_height * 25.0),
                8.0,
                13.0,
            )
        )
        block_height = len(assets) * avatar + (len(assets) - 1) * gap
        start_y = header_bottom + (header_height + block_height) / 2 - avatar
        for index, asset in enumerate(assets):
            y = start_y - index * (avatar + gap)
            member_axis = figure.add_axes([branding_x, y, avatar, avatar], zorder=10)
            member_axis.imshow(asset.image)
            member_axis.axis("off")
            figure.text(
                branding_x + avatar + 0.008,
                y + avatar / 2,
                asset.member.display_name,
                color=asset.member.accent,
                fontsize=font,
                fontweight="bold",
                va="center",
                ha="left",
            )

    @staticmethod
    def _fit_header_font(
        text: str,
        available_width_inches: float,
        *,
        preferred: float,
        minimum: float,
        maximum: float,
    ) -> float:
        # Approximate bold sans-serif glyph width at 0.60em. This keeps the
        # header deterministic in headless rendering without relying on a
        # GUI canvas measurement pass.
        width_limited = available_width_inches * 72.0 / max(len(text) * 0.60, 1.0)
        return float(np.clip(min(preferred, width_limited), minimum, maximum))

    @staticmethod
    def _format_expiry_label(value: object) -> str:
        stamp = pd.to_datetime(value, format="%m/%d/%Y", errors="coerce")
        if pd.isna(stamp):
            stamp = pd.to_datetime(value, errors="coerce")
        if pd.isna(stamp):
            return str(value)
        return f"{stamp.strftime('%b')} {stamp.day} '{stamp.strftime('%y')}"

    @staticmethod
    def _expiry_font_size(matrix_width: float, columns: int) -> float:
        column_width_inches = matrix_width / max(columns, 1)
        # Horizontal labels such as Jul 17 '26 generally fit comfortably at
        # this scale; shrink progressively for unusually dense term structures.
        return float(np.clip(column_width_inches * 5.7, 6.5, 11.2))

    @staticmethod
    def _select_strikes(matrix: pd.DataFrame, spot: float, count: int, important_levels: list[float]) -> list[float]:
        strikes = np.asarray(sorted(float(value) for value in matrix.index), dtype=float)
        if strikes.size == 0:
            raise ValueError("No strikes are available for rendering.")
        desired = max(5, int(count))
        desired += 1 if desired % 2 == 0 else 0
        desired = min(desired, len(strikes))
        center = int(np.argmin(np.abs(strikes - spot)))
        start = max(0, center - desired // 2)
        stop = min(len(strikes), start + desired)
        start = max(0, stop - desired)
        selected = set(float(value) for value in strikes[start:stop])
        important_nearest: set[float] = set()
        for level in important_levels:
            if spot * 0.80 <= level <= spot * 1.20:
                nearest = float(strikes[np.argmin(np.abs(strikes - level))])
                important_nearest.add(nearest)
                selected.add(nearest)
        while len(selected) > desired:
            removable = [value for value in selected if value not in important_nearest]
            if not removable:
                break
            selected.remove(max(removable, key=lambda value: abs(value - spot)))
        if len(selected) < desired:
            remaining = [value for value in strikes if float(value) not in selected]
            remaining.sort(key=lambda value: abs(float(value) - spot))
            for value in remaining[: desired - len(selected)]:
                selected.add(float(value))
        return sorted(selected)

    @staticmethod
    def _global_extrema_positions(
        matrix: pd.DataFrame,
    ) -> tuple[tuple[object, object] | None, tuple[object, object] | None]:
        positive, negative = HeatmapRenderer._ranked_node_positions(matrix, limit=1)
        return (
            positive[0] if positive else None,
            negative[0] if negative else None,
        )

    @staticmethod
    def _ranked_node_positions(
        matrix: pd.DataFrame,
        *,
        limit: int = 3,
    ) -> tuple[list[tuple[object, object]], list[tuple[object, object]]]:
        """Rank positive and negative cells independently across the whole map.

        Equal values use deterministic matrix order as a tie-breaker. Rankings
        never feed back into GEX, opacity, or the underlying cell color scale.
        """

        values = matrix.to_numpy(dtype=float)
        positive: list[tuple[float, int, tuple[object, object]]] = []
        negative: list[tuple[float, int, tuple[object, object]]] = []
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = float(values[row, column])
                if not np.isfinite(value) or value == 0.0:
                    continue
                position = (matrix.index[row], matrix.columns[column])
                flat_index = row * values.shape[1] + column
                if value > 0.0:
                    positive.append((value, flat_index, position))
                else:
                    negative.append((value, flat_index, position))

        count = max(int(limit), 0)
        positive.sort(key=lambda item: (-item[0], item[1]))
        negative.sort(key=lambda item: (item[0], item[1]))
        return (
            [item[2] for item in positive[:count]],
            [item[2] for item in negative[:count]],
        )

    @staticmethod
    def _node_rank_lookup(
        matrix: pd.DataFrame,
        *,
        limit: int = 3,
    ) -> dict[tuple[object, object], tuple[str, int]]:
        positive, negative = HeatmapRenderer._ranked_node_positions(
            matrix, limit=limit
        )
        lookup = {
            position: ("positive", rank)
            for rank, position in enumerate(positive, start=1)
        }
        lookup.update(
            {
                position: ("negative", rank)
                for rank, position in enumerate(negative, start=1)
            }
        )
        return lookup

    @staticmethod
    def _node_highlight_style(
        side: str,
        rank: int,
    ) -> tuple[str, float, float, str]:
        if side == "positive":
            color = "#ffd700"
            prefix = "P"
        elif side == "negative":
            color = "#b000ff"
            prefix = "N"
        else:
            raise ValueError(f"Unsupported node-highlight side: {side}")
        styles = {
            1: (2.85, 1.00),
            2: (2.10, 0.78),
            3: (1.55, 0.58),
        }
        if rank not in styles:
            raise ValueError(f"Unsupported node-highlight rank: {rank}")
        linewidth, alpha = styles[rank]
        return color, linewidth, alpha, f"{prefix}{rank}"

    @staticmethod
    def _color_scale(values: np.ndarray) -> dict[str, float]:
        flat = values[np.isfinite(values)]
        nonzero = np.abs(flat[flat != 0])
        positives = flat[flat > 0]
        negatives = np.abs(flat[flat < 0])

        def qs(data: np.ndarray) -> tuple[float, float, float, float]:
            if data.size >= 10:
                return tuple(float(value) for value in np.percentile(data, [50, 80, 95, 99]))
            fallback = max(float(np.max(data)) if data.size else 1.0, 1.0)
            return fallback * 0.20, fallback * 0.45, fallback * 0.75, fallback

        p50, p80, p95, p99 = qs(positives)
        n50, n80, n95, n99 = qs(negatives)
        # Alpha 4 uses an absolute micro-exposure display floor so low-GEX
        # tickers remain comparable with SPY-scale charts. Values below $5K
        # are visually present but rendered as an em dash.
        floor = MICRO_DISPLAY_FLOOR
        return {
            "p50": p50,
            "p80": p80,
            "p95": p95,
            "p99": p99,
            "n50": n50,
            "n80": n80,
            "n95": n95,
            "n99": n99,
            "floor": floor,
        }

    @classmethod
    def _cell_colors(cls, value: float, scale: dict[str, float]) -> tuple[str, str, str]:
        if not np.isfinite(value) or value == 0:
            return "#111116", "#4a4a5a", "#202027"
        magnitude = abs(value)
        if value > 0:
            if magnitude >= POSITIVE_BLUE_THRESHOLD:
                return "#00bfff", "#000000", "#00f0ff"
            if magnitude >= POSITIVE_GOLD_THRESHOLD:
                ratio = cls._tier_ratio(
                    magnitude,
                    POSITIVE_GOLD_THRESHOLD,
                    POSITIVE_BLUE_THRESHOLD,
                )
                return cls._blend("#ffd700", "#fff0a0", ratio), "#000000", "#ffd700"
            if magnitude >= 500_000_000.0:
                ratio = cls._tier_ratio(magnitude, 500_000_000.0, POSITIVE_GOLD_THRESHOLD)
                return cls._blend("#b9f8c4", "#f4fff5", ratio), "#000000", "#8dffa0"
            if magnitude >= 150_000_000.0:
                ratio = cls._tier_ratio(magnitude, 150_000_000.0, 500_000_000.0)
                background = cls._blend("#43d565", "#b9f8c4", ratio)
                return background, "#000000", "#5cff76"
            if magnitude >= 50_000_000.0:
                ratio = cls._tier_ratio(magnitude, 50_000_000.0, 150_000_000.0)
                return cls._blend("#1fa542", "#43d565", ratio), "#051208", "#39ff14"
            if magnitude >= 10_000_000.0:
                ratio = cls._tier_ratio(magnitude, 10_000_000.0, 50_000_000.0)
                return cls._blend("#126f2d", "#1fa542", ratio), "#d8ffe0", "#39ff14"
            if magnitude >= 1_000_000.0:
                ratio = cls._tier_ratio(magnitude, 1_000_000.0, 10_000_000.0)
                return cls._blend("#0d4a20", "#126f2d", ratio), "#d8ffe0", "#2ae95b"
            if magnitude >= MICRO_EXPOSURE_MAX:
                ratio = cls._tier_ratio(magnitude, MICRO_EXPOSURE_MAX, 1_000_000.0)
                return cls._blend("#0b3217", "#0d4a20", ratio), "#b9ffc7", "#1b7734"
            return "#0b3217", "#7dff91", "#1b542a"

        if magnitude >= NEGATIVE_BLACK_THRESHOLD:
            return "#000000", "#ff1744", "#ff003c"
        if magnitude >= NEGATIVE_BLACK_START:
            ratio = cls._tier_ratio(
                magnitude,
                NEGATIVE_BLACK_START,
                NEGATIVE_BLACK_THRESHOLD,
            )
            return cls._blend("#7a0014", "#000000", ratio), "#ff365d", "#ff003c"
        if magnitude >= NEGATIVE_RED_PEAK:
            ratio = cls._tier_ratio(magnitude, NEGATIVE_RED_PEAK, NEGATIVE_BLACK_START)
            return cls._blend("#ff1744", "#7a0014", ratio), "#ffffff", "#ff1744"
        if magnitude >= 150_000_000.0:
            ratio = cls._tier_ratio(magnitude, 150_000_000.0, NEGATIVE_RED_PEAK)
            return cls._blend("#e1152f", "#ff1744", ratio), "#ffffff", "#ff1744"
        if magnitude >= 50_000_000.0:
            ratio = cls._tier_ratio(magnitude, 50_000_000.0, 150_000_000.0)
            return cls._blend("#a91022", "#e1152f", ratio), "#ffe2e7", "#ff1744"
        if magnitude >= 10_000_000.0:
            ratio = cls._tier_ratio(magnitude, 10_000_000.0, 50_000_000.0)
            return cls._blend("#6f0d19", "#a91022", ratio), "#ffd6dc", "#e51b32"
        if magnitude >= 1_000_000.0:
            ratio = cls._tier_ratio(magnitude, 1_000_000.0, 10_000_000.0)
            return cls._blend("#4c0c14", "#6f0d19", ratio), "#ffcbd3", "#b51c2f"
        if magnitude >= MICRO_EXPOSURE_MAX:
            ratio = cls._tier_ratio(magnitude, MICRO_EXPOSURE_MAX, 1_000_000.0)
            return cls._blend("#350a10", "#4c0c14", ratio), "#ffb8c4", "#781522"
        return "#350a10", "#ff879b", "#521018"

    @staticmethod
    def _tier_ratio(value: float, lower: float, upper: float) -> float:
        if upper <= lower:
            return 1.0
        return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))

    @staticmethod
    def _blend(start: str, end: str, ratio: float) -> str:
        start_rgb = np.asarray(mcolors.to_rgb(start))
        end_rgb = np.asarray(mcolors.to_rgb(end))
        return mcolors.to_hex(start_rgb + (end_rgb - start_rgb) * float(np.clip(ratio, 0, 1)))

    @staticmethod
    def _magnitude_alpha(value: float) -> float:
        """Absolute, cross-ticker intensity for the $0–$500K micro tier."""
        magnitude = min(abs(float(value)), MICRO_EXPOSURE_MAX)
        normalized = np.log1p(magnitude / MICRO_DISPLAY_FLOOR) / np.log1p(
            MICRO_EXPOSURE_MAX / MICRO_DISPLAY_FLOOR
        )
        return float(np.clip(0.08 + 0.92 * normalized, 0.08, 1.0))

    @staticmethod
    def _quality_alpha(value: float, quality: float) -> float:
        alpha = 0.45 + 0.55 * float(np.clip(quality, 0.0, 1.0))
        magnitude = abs(value)
        if value > 0 and magnitude >= POSITIVE_GOLD_THRESHOLD:
            alpha = max(alpha, 0.78)
        if value < 0 and magnitude >= NEGATIVE_BLACK_START:
            alpha = max(alpha, 0.88)
        return float(np.clip(alpha, 0.0, 1.0))

    @staticmethod
    def _dim_color(color: str, alpha: float) -> str:
        rgb = np.asarray(mcolors.to_rgb(color))
        mixed = DARK + (rgb - DARK) * float(np.clip(alpha, 0.0, 1.0))
        return mcolors.to_hex(mixed)

    @staticmethod
    def _format_amount(value: float | None, floor: float = 1.0) -> str:
        if value is None or not np.isfinite(value) or abs(value) < floor:
            return "—"
        sign = "+" if value > 0 else "-"
        magnitude = abs(value)
        if magnitude >= 1e9:
            return f"{sign}${magnitude / 1e9:.1f}B"
        if magnitude >= 1e6:
            return f"{sign}${magnitude / 1e6:.1f}M"
        if magnitude >= 1e3:
            return f"{sign}${magnitude / 1e3:.1f}K"
        return f"{sign}${magnitude:.0f}"

    @staticmethod
    def _format_strike(value: float) -> str:
        number = float(value)
        return str(int(number)) if number.is_integer() else f"{number:.2f}".rstrip("0").rstrip(".")
