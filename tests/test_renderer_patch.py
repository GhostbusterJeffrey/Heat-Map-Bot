from __future__ import annotations

import unittest

import numpy as np

from tpr_terminal.renderer import HeatmapRenderer


class Alpha3RendererPatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scale = HeatmapRenderer._color_scale(
            np.array(
                [[-1_500_000_000.0, -500_000_000.0, 250_000_000.0, 1_000_000_000.0]],
                dtype=float,
            )
        )

    def test_expiry_label_uses_terminal_format(self) -> None:
        self.assertEqual(HeatmapRenderer._format_expiry_label("07/17/2026"), "Jul 17 '26")
        self.assertEqual(HeatmapRenderer._format_expiry_label("01/05/2027"), "Jan 5 '27")

    def test_expiry_font_shrinks_for_denser_term_structure(self) -> None:
        sparse = HeatmapRenderer._expiry_font_size(18.2, 10)
        dense = HeatmapRenderer._expiry_font_size(18.2, 24)
        self.assertGreater(sparse, dense)
        self.assertGreaterEqual(dense, 6.5)

    def test_positive_ramp_stays_green_then_uses_requested_extremes(self) -> None:
        dark, _, _ = HeatmapRenderer._cell_colors(500_000.0, self.scale)
        medium, _, _ = HeatmapRenderer._cell_colors(50_000_000.0, self.scale)
        light, text_light, _ = HeatmapRenderer._cell_colors(999_999_999.0, self.scale)
        gold, text_gold, border_gold = HeatmapRenderer._cell_colors(1_000_000_000.0, self.scale)
        blue, text_blue, border_blue = HeatmapRenderer._cell_colors(10_000_000_000.0, self.scale)
        self.assertEqual(dark, "#0b3217")
        self.assertEqual(medium, "#1fa542")
        self.assertEqual(light, "#f4fff5")
        self.assertEqual(text_light, "#000000")
        self.assertEqual(gold, "#ffd700")
        self.assertEqual(text_gold, "#000000")
        self.assertEqual(border_gold, "#ffd700")
        self.assertEqual(blue, "#00bfff")
        self.assertEqual(text_blue, "#000000")
        self.assertEqual(border_blue, "#00f0ff")

    def test_negative_ramp_progresses_from_dark_red_to_black(self) -> None:
        dark, _, _ = HeatmapRenderer._cell_colors(-500_000.0, self.scale)
        medium, _, _ = HeatmapRenderer._cell_colors(-50_000_000.0, self.scale)
        peak, _, _ = HeatmapRenderer._cell_colors(-300_000_000.0, self.scale)
        black, text_black, border_black = HeatmapRenderer._cell_colors(-1_000_000_000.0, self.scale)
        self.assertEqual(dark, "#350a10")
        self.assertEqual(medium, "#a91022")
        self.assertEqual(peak, "#ff1744")
        self.assertEqual(black, "#000000")
        self.assertEqual(text_black, "#ff1744")
        self.assertEqual(border_black, "#ff003c")

    def test_micro_opacity_returns_to_original_500k_ceiling(self) -> None:
        self.assertLess(HeatmapRenderer._magnitude_alpha(5_000.0), 1.0)
        self.assertEqual(HeatmapRenderer._magnitude_alpha(500_000.0), 1.0)
        self.assertEqual(HeatmapRenderer._magnitude_alpha(50_000_000.0), 1.0)

    def test_dynamic_floor_keeps_six_figure_values_visible(self) -> None:
        scale = HeatmapRenderer._color_scale(
            np.array([[100_000.0, 628_600.0, 1_000_000_000.0]], dtype=float)
        )
        self.assertLessEqual(scale["floor"], 100_000.0)
        self.assertNotEqual(HeatmapRenderer._format_amount(628_600.0, floor=scale["floor"]), "—")

    def test_header_font_is_bounded_and_width_aware(self) -> None:
        short = HeatmapRenderer._fit_header_font(
            "SPY",
            10.0,
            preferred=23.0,
            minimum=14.0,
            maximum=23.0,
        )
        long = HeatmapRenderer._fit_header_font(
            "THE PRINTING ROOM // VERY LONG TITLE THAT MUST SHRINK",
            5.0,
            preferred=23.0,
            minimum=14.0,
            maximum=23.0,
        )
        self.assertEqual(short, 23.0)
        self.assertLessEqual(long, short)
        self.assertGreaterEqual(long, 14.0)


if __name__ == "__main__":
    unittest.main()
