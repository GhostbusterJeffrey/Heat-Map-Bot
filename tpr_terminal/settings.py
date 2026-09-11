from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None or not raw.strip() else int(raw)


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None or not raw.strip() else float(raw)


def _path(name: str, default: str | None = None) -> Path | None:
    raw = os.getenv(name, default or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


@dataclass(frozen=True, slots=True)
class Settings:
    discord_token: str
    dev_guild_id: int | None
    elite_role_id: int | None
    elite_role_name: str
    owner_user_id: int | None

    default_preset: str
    default_expiries: int
    default_strikes: int
    max_expiries: int
    max_strikes: int
    cooldown_seconds: float
    max_concurrent_analyses: int
    analysis_workers: int
    render_workers: int
    max_upload_mb: float

    intraday_expiries: int
    intraday_strikes: int
    intraday_window_pct: float
    swing_expiries: int
    swing_strikes: int
    swing_window_pct: float
    macro_expiries: int
    macro_strikes: int
    macro_window_pct: float

    min_contracts_to_render: int
    min_usable_ratio: float
    min_confidence_to_render: float
    max_wall_distance_pct: float
    min_wall_quality: float
    min_wall_percentile: float
    min_expiry_share_for_breadth: float
    min_flip_confidence: float
    flip_grid_low_pct: float
    flip_grid_high_pct: float
    flip_grid_points: int
    pin_decay_days: float
    min_pin_score_to_display: float
    pin_shock_pct: float
    volume_liquidity_weight: float

    result_cache_ttl_seconds: int
    rate_cache_ttl_seconds: int
    provider_cache_ttl_seconds: int

    snapshot_enabled: bool
    snapshot_db_path: Path
    snapshot_retention_days: int

    use_treasury_curve: bool
    static_risk_free_rate: float
    contract_multiplier: int
    default_liquidity_mode: str
    default_exposure_mode: str
    quant_engine_mode: str

    assets_dir: Path
    main_logo_path: Path | None
    developer_logo_path: Path | None
    trader_logo_path: Path | None
    sibei_logo_path: Path | None
    jeff_logo_path: Path | None
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        def optional_int(name: str) -> int | None:
            raw = os.getenv(name, "").strip()
            return int(raw) if raw else None

        assets_dir = _path("ASSETS_DIR", "assets") or (PROJECT_ROOT / "assets")
        return cls(
            discord_token=os.getenv("DISCORD_TOKEN", "").strip(),
            dev_guild_id=optional_int("DEV_GUILD_ID"),
            elite_role_id=optional_int("ELITE_ROLE_ID"),
            elite_role_name=os.getenv("ELITE_ROLE_NAME", "Elite").strip(),
            owner_user_id=optional_int("OWNER_USER_ID"),
            default_preset=os.getenv("DEFAULT_PRESET", "swing").strip(),
            default_expiries=_int("DEFAULT_EXPIRIES", 12),
            default_strikes=_int("DEFAULT_STRIKES", 61),
            max_expiries=_int("MAX_EXPIRIES", 24),
            max_strikes=_int("MAX_STRIKES", 101),
            cooldown_seconds=_float("COOLDOWN_SECONDS", 15.0),
            max_concurrent_analyses=_int("MAX_CONCURRENT_ANALYSES", 2),
            analysis_workers=_int("ANALYSIS_WORKERS", 2),
            render_workers=_int("RENDER_WORKERS", 1),
            max_upload_mb=_float("MAX_UPLOAD_MB", 9.5),
            intraday_expiries=_int("INTRADAY_EXPIRIES", 5),
            intraday_strikes=_int("INTRADAY_STRIKES", 41),
            intraday_window_pct=_float("INTRADAY_WINDOW_PCT", 0.05),
            swing_expiries=_int("SWING_EXPIRIES", 12),
            swing_strikes=_int("SWING_STRIKES", 61),
            swing_window_pct=_float("SWING_WINDOW_PCT", 0.10),
            macro_expiries=_int("MACRO_EXPIRIES", 20),
            macro_strikes=_int("MACRO_STRIKES", 81),
            macro_window_pct=_float("MACRO_WINDOW_PCT", 0.18),
            min_contracts_to_render=_int("MIN_CONTRACTS_TO_RENDER", 50),
            min_usable_ratio=_float("MIN_USABLE_RATIO", 0.05),
            min_confidence_to_render=_float("MIN_CONFIDENCE_TO_RENDER", 35.0),
            max_wall_distance_pct=_float("MAX_WALL_DISTANCE_PCT", 0.12),
            min_wall_quality=_float("MIN_WALL_QUALITY", 0.40),
            min_wall_percentile=_float("MIN_WALL_PERCENTILE", 0.70),
            min_expiry_share_for_breadth=_float("MIN_EXPIRY_SHARE_FOR_BREADTH", 0.025),
            min_flip_confidence=_float("MIN_FLIP_CONFIDENCE", 0.45),
            flip_grid_low_pct=_float("FLIP_GRID_LOW_PCT", 0.85),
            flip_grid_high_pct=_float("FLIP_GRID_HIGH_PCT", 1.15),
            flip_grid_points=_int("FLIP_GRID_POINTS", 601),
            pin_decay_days=_float("PIN_DECAY_DAYS", 7.0),
            min_pin_score_to_display=_float("MIN_PIN_SCORE_TO_DISPLAY", 35.0),
            pin_shock_pct=_float("PIN_SHOCK_PCT", 0.0025),
            volume_liquidity_weight=_float("VOLUME_LIQUIDITY_WEIGHT", 0.50),
            result_cache_ttl_seconds=_int("RESULT_CACHE_TTL_SECONDS", 75),
            rate_cache_ttl_seconds=_int("RATE_CACHE_TTL_SECONDS", 21_600),
            provider_cache_ttl_seconds=_int("PROVIDER_CACHE_TTL_SECONDS", 30),
            snapshot_enabled=_bool("SNAPSHOT_ENABLED", True),
            snapshot_db_path=(
                _path("SNAPSHOT_DB_PATH", "data/tpr_options_history.sqlite3")
                or (PROJECT_ROOT / "data" / "tpr_options_history.sqlite3")
            ),
            snapshot_retention_days=_int("SNAPSHOT_RETENTION_DAYS", 730),
            use_treasury_curve=_bool("USE_TREASURY_CURVE", True),
            static_risk_free_rate=_float("STATIC_RISK_FREE_RATE", 0.045),
            contract_multiplier=_int("CONTRACT_MULTIPLIER", 100),
            default_liquidity_mode=os.getenv("DEFAULT_LIQUIDITY_MODE", "auto").strip(),
            default_exposure_mode=os.getenv("DEFAULT_EXPOSURE_MODE", "net_proxy").strip(),
            quant_engine_mode=os.getenv("QUANT_ENGINE_MODE", "legacy").strip().lower(),
            assets_dir=assets_dir,
            main_logo_path=_path("MAIN_LOGO_PATH", "assets/main.jpg"),
            developer_logo_path=_path("DEVELOPER_LOGO_PATH", "assets/developer.jpg"),
            trader_logo_path=_path("TRADER_LOGO_PATH", "assets/trader.jpg"),
            sibei_logo_path=_path("SIBEI_LOGO_PATH", "assets/sibei.jpg"),
            jeff_logo_path=_path("JEFF_LOGO_PATH", "assets/jeff.jpg"),
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
        )

    def validate(self) -> None:
        if not self.discord_token:
            raise RuntimeError("DISCORD_TOKEN is missing. Run `python configure.py` or edit .env.")
        if self.dev_guild_id is not None and self.dev_guild_id <= 0:
            raise RuntimeError("DEV_GUILD_ID must be a positive Discord server ID.")
        if self.elite_role_id is None and not self.elite_role_name:
            raise RuntimeError("Set ELITE_ROLE_ID or ELITE_ROLE_NAME.")
        if self.default_preset not in {"intraday", "swing", "macro"}:
            raise RuntimeError("DEFAULT_PRESET must be intraday, swing, or macro.")
        if not (1 <= self.default_expiries <= self.max_expiries):
            raise RuntimeError("DEFAULT_EXPIRIES must be between 1 and MAX_EXPIRIES.")
        if not (5 <= self.default_strikes <= self.max_strikes):
            raise RuntimeError("DEFAULT_STRIKES must be between 5 and MAX_STRIKES.")
        if self.max_concurrent_analyses < 1 or self.analysis_workers < 1:
            raise RuntimeError("Concurrency settings must be at least 1.")
        if self.render_workers != 1:
            raise RuntimeError("RENDER_WORKERS must remain 1 for Matplotlib safety.")
        if self.default_liquidity_mode not in {
            "auto",
            "open_interest",
            "oi_volume_fallback",
            "volume_dominant",
        }:
            raise RuntimeError("Unsupported DEFAULT_LIQUIDITY_MODE.")
        if self.default_exposure_mode not in {"net_proxy", "gross"}:
            raise RuntimeError("Unsupported DEFAULT_EXPOSURE_MODE.")
        if self.quant_engine_mode not in {"legacy", "shadow", "candidate"}:
            raise RuntimeError(
                "QUANT_ENGINE_MODE must be legacy, shadow, or candidate."
            )
        if self.snapshot_retention_days < 1:
            raise RuntimeError("SNAPSHOT_RETENTION_DAYS must be at least 1.")
        if not (0.0 <= self.min_expiry_share_for_breadth <= 1.0):
            raise RuntimeError("MIN_EXPIRY_SHARE_FOR_BREADTH must be between 0 and 1.")
