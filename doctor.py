from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
SUPPORTED = (".jpg", ".jpeg", ".png", ".webp")


def check(label: str, ok: bool, detail: str = "") -> bool:
    symbol = "PASS" if ok else "FAIL"
    print(f"[{symbol}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def _asset(stem: str) -> Path | None:
    for extension in SUPPORTED:
        candidate = ROOT / "assets" / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def _check_asset(stem: str, label: str) -> None:
    path = _asset(stem)
    if path is None:
        check(f"Optional branding: {label}", False, "not installed; monogram fallback will be used")
        return
    try:
        with Image.open(path) as image:
            width, height = image.size
            warning = ""
            if width < 500 or height < 500:
                warning = " (below recommended 500px minimum)"
            check(f"Branding: {label}", True, f"{path.name} — {width}×{height}{warning}")
    except Exception as exc:
        check(f"Branding: {label}", False, f"unreadable: {exc}; monogram fallback will be used")


def main() -> int:
    print("TPR Terminal V4 Alpha 4.2 Dev.1 Doctor\n")
    all_ok = True
    all_ok &= check("Python version", sys.version_info >= (3, 12), platform.python_version())
    for module in (
        "discord",
        "yfinance",
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",
        "dotenv",
        "PIL",
    ):
        try:
            loaded = importlib.import_module(module)
            version = getattr(loaded, "__version__", "installed")
            all_ok &= check(f"Dependency: {module}", True, str(version))
        except Exception as exc:
            all_ok &= check(f"Dependency: {module}", False, str(exc))

    env_file = ROOT / ".env"
    all_ok &= check(".env file", env_file.exists(), str(env_file))
    if env_file.exists():
        from dotenv import load_dotenv

        load_dotenv(env_file)
        token = os.getenv("DISCORD_TOKEN", "").strip()
        guild = os.getenv("DEV_GUILD_ID", "").strip()
        role_id = os.getenv("ELITE_ROLE_ID", "").strip()
        role_name = os.getenv("ELITE_ROLE_NAME", "").strip()
        all_ok &= check("Discord token configured", len(token) > 30)
        check("Development guild configured", bool(guild), guild or "global sync")
        all_ok &= check("Access role configured", bool(role_id or role_name), role_id or role_name)

    _check_asset("main", "The Printing Room")
    _check_asset("trader", "Viola Trades")
    _check_asset("developer", "JP The Developer")
    _check_asset("sibei", "Sibei Shiok")
    _check_asset("jeff", "Jeff The Scripter")

    try:
        from tpr_terminal.settings import Settings
        from tpr_terminal.snapshot_store import OptionSnapshotStore

        settings = Settings.from_env()
        if settings.snapshot_enabled:
            store = OptionSnapshotStore(
                settings.snapshot_db_path,
                retention_days=settings.snapshot_retention_days,
            )
            stats = store.stats()
            all_ok &= check(
                "Historical snapshot store",
                True,
                f"{settings.snapshot_db_path} — {stats.chains} chains saved",
            )
        else:
            check("Historical snapshot store", True, "disabled by configuration")
    except Exception as exc:
        all_ok &= check("Historical snapshot store", False, str(exc))

    try:
        from tpr_terminal.volatility_surface import raw_ssvi, raw_svi

        _ = raw_svi
        _ = raw_ssvi
        all_ok &= check("SVI/SSVI calibration layer", True, "available")
    except Exception as exc:
        all_ok &= check("SVI/SSVI calibration layer", False, str(exc))

    print("\n" + ("System is ready." if all_ok else "Fix the FAIL items above."))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
