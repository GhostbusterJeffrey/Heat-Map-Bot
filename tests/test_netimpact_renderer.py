from __future__ import annotations

import copy
import io
import unittest

from PIL import Image

from tpr_terminal.netimpact_renderer import _build_figure, _money, _timestamp, render_netimpact


def report(mode="flow", values=(1_000_000.0, -500_000.0, 0.0)):
    return {
        "session_date": "2026-10-21", "window": "session", "mode": mode,
        "as_of": "2026-10-21T20:00:00+00:00", "status": "partial_session",
        "scope": "20-name watchlist • nearest expiry ≤7 DTE • 11 strikes • calls + puts",
        "notes": ["Test fixture only; these are not actual market observations."],
        "rows": [{"ticker": f"TEST{index}", "net": value, "available": value is not None}
                 for index, value in enumerate(values)],
    }


class NetImpactRendererTests(unittest.TestCase):
    def test_close_estimate_never_claims_inferred_execution_flow(self):
        fixture = report("close_estimate")
        fixture["source_as_of"] = "2026-10-21T19:59:30+00:00"
        figure = _build_figure(fixture)
        texts = "\n".join(item.get_text() for item in figure.texts)
        self.assertIn("CLOSE ACTIVITY ESTIMATE", texts)
        self.assertIn("Direction of trades unknown", texts)
        self.assertIn("3:59:30 PM EDT", texts)
        self.assertNotIn("OBSERVED OPTIONS FLOW", texts)
        self.assertNotIn("4:00:00 PM EDT", texts)
        figure.clear()

    def test_close_estimate_shows_oldest_and_newest_marks_and_cohort_caveat(self):
        fixture = report("close_estimate")
        fixture["source_from"] = "2026-10-21T19:58:00+00:00"
        fixture["source_as_of"] = "2026-10-21T20:14:59+00:00"
        fixture["notes"] = ["First note", "Second note", "Third note", "After-close selection is not a reconstruction of the opening universe."]
        figure = _build_figure(fixture)
        texts = "\n".join(item.get_text() for item in figure.texts)
        self.assertIn("3:58:00 PM–4:14:59 PM EDT", texts)
        self.assertIn("After-close selection", texts)
        self.assertNotIn("Second note", texts)
        figure.clear()

    def test_unknown_and_invalid_values_never_become_zero_bars(self):
        fixture = report(values=(None, float("nan"), float("inf"), 0.0))
        fixture["rows"][1]["available"] = True
        fixture["rows"][2]["available"] = True
        figure = _build_figure(fixture)
        axes = figure.axes[0]
        self.assertEqual([tick.get_text() for tick in axes.get_yticklabels()], ["TEST3"])
        self.assertEqual(len(axes.patches), 1)
        self.assertEqual(axes.patches[0].get_width(), 0)
        texts = "\n".join(item.get_text() for item in figure.texts)
        self.assertIn("Unavailable (excluded from ranking): TEST0, TEST1, TEST2", texts)
        figure.clear()

    def test_bar_widths_preserve_actual_money_and_symmetric_scale(self):
        fixture = report(values=(-3_000.0, 12_000_000.0, 0.0))
        figure = _build_figure(fixture)
        axes = figure.axes[0]
        self.assertEqual([bar.get_width() for bar in axes.patches], [12_000_000.0, 0.0, -3_000.0])
        low, high = axes.get_xlim()
        self.assertEqual(low, -high)
        figure.clear()

    def test_png_handles_one_sided_zero_and_empty_reports_without_mutation(self):
        for values in ((10.0, 2.0), (-10.0, -2.0), (0.0, 0.0), (None,), ()):
            with self.subTest(values=values):
                fixture = report("demo", values)
                original = copy.deepcopy(fixture)
                data = render_netimpact(fixture)
                with Image.open(io.BytesIO(data)) as image:
                    self.assertEqual(image.format, "PNG")
                    self.assertGreater(image.width, 1000)
                    self.assertIn("not market observations", image.info["Description"])
                self.assertEqual(fixture, original)

    def test_time_uses_eastern_dst_and_refuses_ambiguous_naive_time(self):
        self.assertIn("4:00:00 PM EDT", _timestamp("2026-10-21T20:00:00Z"))
        self.assertIn("4:00:00 PM EST", _timestamp("2026-12-21T21:00:00Z"))
        self.assertEqual(_timestamp("2026-10-21T20:00:00"), "unavailable")
        self.assertEqual(_timestamp("bad"), "unavailable")

    def test_money_signs_and_units(self):
        self.assertEqual(_money(-2_500_000), "−$2.5M")
        self.assertEqual(_money(1_000_000_000, signed=True), "+$1B")
        self.assertEqual(_money(0.0, signed=True), "$0")
        self.assertEqual(_money(-0.001), "$0")
        self.assertEqual(_money(0.25), "$0.25")


if __name__ == "__main__":
    unittest.main()
