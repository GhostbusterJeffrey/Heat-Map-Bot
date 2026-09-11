from __future__ import annotations

from .bot import TPRTerminalBot
from .logging_setup import configure_logging
from .settings import PROJECT_ROOT, Settings


def main() -> None:
    settings = Settings.from_env()
    configure_logging(settings.log_level, PROJECT_ROOT)
    settings.validate()
    bot = TPRTerminalBot(settings)
    bot.run(settings.discord_token, log_handler=None)
