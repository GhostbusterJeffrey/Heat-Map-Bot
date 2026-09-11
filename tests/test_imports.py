from __future__ import annotations

import importlib.util
import unittest


class ImportTests(unittest.TestCase):
    def test_service_imports(self) -> None:
        import tpr_terminal.service  # noqa: F401

    def test_bot_imports_when_discord_is_installed(self) -> None:
        if importlib.util.find_spec("discord") is None:
            self.skipTest("discord.py is not installed in the isolated build environment")
        import tpr_terminal.bot  # noqa: F401


if __name__ == "__main__":
    unittest.main()
