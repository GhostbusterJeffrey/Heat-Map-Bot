from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Literal

import pandas as pd

if TYPE_CHECKING:
    from .canonical_state import CanonicalContractLedger

LiquidityMode = Literal["auto", "open_interest", "oi_volume_fallback", "volume_dominant"]
ExposureMode = Literal["net_proxy", "gross"]
OptionSide = Literal["call", "put"]
SurfaceMode = Literal["ssvi", "svi", "calendar_regularized_svi", "raw_iv"]
InstrumentType = Literal[
    "european_index",
    "american_equity",
    "american_etf",
    "unknown",
]


@dataclass(frozen=True, slots=True)
class UnderlyingSnapshot:
    ticker: str
    spot: float
    as_of: datetime
    trailing_dividend_yield: float


@dataclass(slots=True)
class ExpirationChain:
    expiration: date
    calls: pd.DataFrame
    puts: pd.DataFrame


@dataclass(frozen=True, slots=True)
class RateCurve:
    tenor_days: tuple[float, ...]
    rates_cc: tuple[float, ...]
    source: str
    as_of: date | None
    used_fallback: bool = False

    def rate_for_days(self, days: float) -> float:
        import numpy as np

        if not self.tenor_days:
            raise ValueError("Rate curve is empty.")
        return float(
            np.interp(
                max(float(days), 0.0),
                np.asarray(self.tenor_days, dtype=float),
                np.asarray(self.rates_cc, dtype=float),
            )
        )


@dataclass(slots=True)
class DataQualityReport:
    expirations_requested: int = 0
    expirations_processed: int = 0
    expirations_skipped: list[str] = field(default_factory=list)
    contracts_seen: int = 0
    contracts_used: int = 0
    valid_market_quotes: int = 0
    iv_market_solved: int = 0
    iv_vendor_used: int = 0
    iv_values_interpolated: int = 0
    iv_flat_fallback: int = 0
    iv_surface_fitted: int = 0
    volume_fallback_contracts: int = 0
    reported_oi_contracts: int = 0
    zero_oi_contracts: int = 0
    contracts_with_volume: int = 0
    valid_contracts: int = 0
    parity_expirations: int = 0
    parity_pairs_used: int = 0
    oi_abs_exposure: float = 0.0
    volume_abs_exposure: float = 0.0
    total_abs_exposure: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def compact_summary(self) -> str:
        parts = [
            f"{self.expirations_processed}/{self.expirations_requested} expirations",
            f"{self.contracts_used:,}/{self.contracts_seen:,} contracts",
            f"{self.iv_market_solved:,} market-IV solves",
        ]
        if self.iv_surface_fitted:
            parts.append(f"{self.iv_surface_fitted:,} surface fits")
        if self.iv_values_interpolated:
            parts.append(f"{self.iv_values_interpolated:,} IV interpolations")
        if self.volume_fallback_contracts:
            parts.append(f"{self.volume_fallback_contracts:,} volume fallbacks")
        if self.parity_expirations:
            parts.append(f"{self.parity_expirations} parity forwards")
        if self.expirations_skipped:
            parts.append(f"{len(self.expirations_skipped)} skipped")
        return " | ".join(parts)


@dataclass(frozen=True, slots=True)
class LiquidityReport:
    mode: str
    oi_contracts: int
    volume_fallback_contracts: int
    fallback_ratio: float
    usable_ratio: float
    oi_coverage: float
    volume_coverage: float
    oi_exposure_share: float
    volume_exposure_share: float


@dataclass(frozen=True, slots=True)
class SVIParameters:
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    weighted_rmse: float
    points_used: int
    success: bool


@dataclass(frozen=True, slots=True)
class SSVIParameters:
    rho: float
    eta: float
    gamma: float
    weighted_rmse: float
    points_used: int
    success: bool


@dataclass(frozen=True, slots=True)
class VolatilitySurfaceResult:
    mode: SurfaceMode
    svi_parameters: dict[date, SVIParameters]
    ssvi_parameters: SSVIParameters | None
    forwards: dict[date, float]
    discount_factors: dict[date, float]
    atm_total_variance: dict[date, float]
    market_solve_rate: float
    fitted_contract_ratio: float
    parity_rmse: float
    butterfly_violation_rate: float
    calendar_violation_rate: float
    weighted_fit_rmse: float
    score: float
    grade: str
    reasons: tuple[str, ...]

    @property
    def display(self) -> str:
        return f"{self.score:.0f}/100 — {self.grade}"


@dataclass(frozen=True, slots=True)
class SurfaceHealthReport:
    score: float
    grade: str
    mode: str
    market_iv_solve_rate: float
    fitted_contract_ratio: float
    vendor_iv_fallback_rate: float
    repaired_iv_rate: float
    parity_rmse: float
    weighted_fit_rmse: float
    smile_roughness: float
    calendar_violation_rate: float
    butterfly_warning_rate: float
    reasons: tuple[str, ...]

    @property
    def display(self) -> str:
        return f"{self.score:.0f}/100 — {self.grade}"


@dataclass(frozen=True, slots=True)
class ConfidenceReport:
    score: float
    grade: str
    label: str
    reasons: tuple[str, ...]
    expiry_coverage: float
    contract_coverage: float
    quote_coverage: float
    reported_oi_share: float
    non_interpolated_iv_share: float
    parity_coverage: float

    @property
    def display(self) -> str:
        return f"{self.label} ({self.score:.0f}/100, {self.grade})"


@dataclass(frozen=True, slots=True)
class ForwardSummary:
    expiration: date
    forward: float
    dividend_yield: float
    pairs_used: int
    dispersion_bps: float
    reliable: bool
    method: str


@dataclass(frozen=True, slots=True)
class ExpectedMove:
    expiration: date
    days_to_expiry: float
    dollars: float
    percent: float
    lower: float
    upper: float
    method: str
    horizon_utc: datetime | None = None


@dataclass(frozen=True, slots=True)
class GreekExposureSummary:
    gamma_exposure: float
    vanna_exposure_1vol: float
    charm_exposure_1day: float
    speed_exposure_1pct: float
    color_exposure_1day: float
    zomma_exposure_1vol: float


@dataclass(frozen=True, slots=True)
class WallLevel:
    strike: float
    signed_exposure: float
    absolute_exposure: float
    expiry_breadth: float
    contract_quality: float
    local_dominance: float
    structural_score: float
    stability: str
    shock_stability: float = 1.0


@dataclass(frozen=True, slots=True)
class GammaFlip:
    level: float
    distance: float
    distance_pct: float
    slope_at_root: float
    shock_stability: float
    confidence: float
    sticky_strike_level: float | None = None
    sticky_delta_level: float | None = None
    model_spread: float | None = None


@dataclass(frozen=True, slots=True)
class PinLevel:
    strike: float
    distance: float
    distance_pct: float
    restoring_flow_score: float
    near_term_concentration: float
    expiry_breadth: float
    quality: float
    pin_score: float
    label: str


@dataclass(frozen=True, slots=True)
class LevelSummary:
    positive_wall: WallLevel | None
    negative_wall: WallLevel | None
    raw_positive_peak: WallLevel | None
    raw_negative_peak: WallLevel | None
    primary_gamma_flip: GammaFlip | None
    secondary_gamma_flips: tuple[GammaFlip, ...]
    primary_pin: PinLevel | None
    secondary_pins: tuple[PinLevel, ...]
    total_net_gex: float
    total_absolute_gex: float
    regime: str

    @property
    def positive_wall_strike(self) -> float | None:
        return None if self.positive_wall is None else self.positive_wall.strike

    @property
    def positive_wall_value(self) -> float | None:
        return None if self.positive_wall is None else self.positive_wall.signed_exposure

    @property
    def negative_wall_strike(self) -> float | None:
        return None if self.negative_wall is None else self.negative_wall.strike

    @property
    def negative_wall_value(self) -> float | None:
        return None if self.negative_wall is None else self.negative_wall.signed_exposure

    @property
    def gamma_flip(self) -> float | None:
        return None if self.primary_gamma_flip is None else self.primary_gamma_flip.level

    @property
    def gamma_flips(self) -> tuple[float, ...]:
        values: list[float] = []
        if self.primary_gamma_flip is not None:
            values.append(self.primary_gamma_flip.level)
        values.extend(item.level for item in self.secondary_gamma_flips)
        return tuple(values)

    @property
    def gamma_flip_distance_percent(self) -> float | None:
        return None if self.primary_gamma_flip is None else self.primary_gamma_flip.distance_pct


@dataclass(frozen=True, slots=True)
class BrandMember:
    key: str
    display_name: str
    asset_stem: str
    accent: str
    priority: int


@dataclass(slots=True)
class AnalysisResult:
    ticker: str
    spot: float
    as_of: datetime
    snapshot_id: str
    trailing_dividend_yield: float
    exposure_mode: ExposureMode
    requested_liquidity_mode: LiquidityMode
    effective_liquidity_mode: str
    provider_name: str
    provider_note: str
    rate_curve_source: str
    rate_curve_as_of: date | None
    matrix_records: pd.DataFrame
    contract_records: pd.DataFrame
    strike_totals: pd.DataFrame
    profile: pd.DataFrame
    expected_moves: list[ExpectedMove]
    forward_summaries: list[ForwardSummary]
    levels: LevelSummary
    quality: DataQualityReport
    liquidity: LiquidityReport
    surface: VolatilitySurfaceResult
    surface_health: SurfaceHealthReport
    confidence: ConfidenceReport
    greek_exposure: GreekExposureSummary
    interpretation: tuple[str, ...]
    quant_engine_mode: str = "legacy"
    contract_ledger: CanonicalContractLedger | None = None

    @property
    def liquidity_mode(self) -> str:
        return self.effective_liquidity_mode


@dataclass(frozen=True, slots=True)
class RenderedHeatmap:
    png_bytes: bytes
    result: AnalysisResult
    displayed_strikes: int


GreekName = Literal[
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "vanna",
    "charm",
    "vomma",
    "speed",
    "color",
    "zomma",
    "ultima",
]
GreekScope = Literal["per_share", "per_contract", "custom_position", "market_open_interest"]
PositionSide = Literal["long", "short"]


@dataclass(frozen=True, slots=True)
class GreekValueBand:
    low: float | None
    mid: float
    high: float | None
    unit: str
    stable: bool = True
    relative_spread: float = 0.0


@dataclass(frozen=True, slots=True)
class ContractGreekReport:
    ticker: str
    expiration: date
    strike: float
    option_side: OptionSide
    spot: float
    as_of: datetime
    instrument_type: InstrumentType
    model_label: str
    provider_name: str
    contract_symbol: str | None
    bid: float | None
    ask: float | None
    last_price: float | None
    open_interest: int
    volume: int
    quote_quality: float
    implied_volatility_bid: float | None
    implied_volatility_mid: float
    implied_volatility_ask: float | None
    implied_volatility_source: str
    time_years: float
    days_to_expiry: float
    risk_free_rate: float
    dividend_yield: float
    contract_multiplier: int
    quantity: int
    position: PositionSide
    scope: GreekScope
    scale_factor: float
    values: dict[str, GreekValueBand]
    dollar_gamma_1pct: GreekValueBand
    validation_error: float
    validation_label: str
    warnings: tuple[str, ...]
    quant_engine_mode: str = "legacy"
    core_validation_errors: dict[str, float] = field(default_factory=dict)
    unstable_greeks: tuple[str, ...] = ()
    forward_price: float | None = None
    selected_mark: float | None = None
    mark_source: str = "unknown"
    settlement_time: datetime | None = None
    legacy_implied_volatility: float | None = None
    candidate_implied_volatility: float | None = None
    snapshot_id: str | None = None
    contract_state_id: str | None = None
    contract_state_source: str = "standalone"
    heatmap_gamma_per_share: float | None = None
    heatmap_gex_1pct: float | None = None
