"""Configuration for the independent, shared market-data collector."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TICKERS = (
    "SPY", "QQQ", "IWM", "TSLA", "NVDA", "AMD", "AAPL", "MSFT", "META", "AMZN",
    "GOOGL", "AVGO", "ORCL", "PLTR", "MU", "NFLX", "COIN", "MSTR", "INTC", "CRM",
)
ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class NetImpactSettings:
    tickers: tuple[str, ...] = DEFAULT_TICKERS
    db_path: Path = ROOT / "data" / "netimpact.sqlite3"
    token_file: Path | None = None

    @classmethod
    def from_env(cls) -> NetImpactSettings:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        names = tuple(dict.fromkeys(x.strip().upper() for x in os.getenv("NETIMPACT_TICKERS", ",".join(DEFAULT_TICKERS)).split(",") if x.strip()))
        if not names or len(names) > 20 or any(not re.fullmatch(r"[A-Z][A-Z0-9./-]{0,9}", x) for x in names):
            raise ValueError("NETIMPACT_TICKERS must contain 1–20 unique equity/ETF symbols.")
        path = Path(os.getenv("NETIMPACT_DB_PATH", "data/netimpact.sqlite3")).expanduser()
        path = path if path.is_absolute() else ROOT / path
        token_path = os.getenv("TRADIER_ENV_FILE", "").strip()
        token_file = Path(token_path).expanduser() if token_path else None
        if token_file and not token_file.is_absolute():
            token_file = ROOT / token_file
        return cls(names, path.resolve(), token_file)

    def read_token(self) -> str:
        value = os.getenv("TRADIER_ACCESS_TOKEN", "").strip()
        if not value and self.token_file:
            try:
                for line in self.token_file.read_text(encoding="utf-8-sig").splitlines():
                    key, sep, candidate = line.partition("=")
                    if sep and key.strip() == "TRADIER_ACCESS_TOKEN":
                        value = candidate.strip().strip('"').strip("'")
                        break
            except OSError:
                raise ValueError("The local Tradier token file could not be read.") from None
        if not value or any(char.isspace() for char in value):
            raise ValueError("Set a valid TRADIER_ACCESS_TOKEN locally, or configure TRADIER_ENV_FILE.")
        return value
