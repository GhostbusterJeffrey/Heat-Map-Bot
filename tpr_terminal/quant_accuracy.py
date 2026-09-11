from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal, Mapping

import numpy as np

from .models import OptionSide
from .pricing import black_scholes_price, greek_stack

QuantEngineMode = Literal["legacy", "shadow", "candidate"]


@dataclass(frozen=True, slots=True)
class SanitizedQuote:
    """One auditable option-price mark and the diagnostics behind it."""

    bid: float | None
    ask: float | None
    last: float | None
    mark: float | None
    mark_source: str
    quote_quality: float
    spread_ratio: float | None
    lower_bound: float
    upper_bound: float
    valid_market: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ContractState:
    """Canonical scalar state shared by contract analytics and audit tooling.

    Heatmap DataFrames use the same field names.  Keeping the scalar contract
    representation explicit prevents `/greek`, `/greeks`, and matrix cells from
    silently drifting onto different clocks, carry inputs, marks, or units.
    """

    ticker: str
    expiration: date
    valuation_time: datetime
    settlement_time: datetime
    option_side: OptionSide
    spot: float
    strike: float
    time_years: float
    risk_free_rate: float
    dividend_yield: float
    forward: float
    quote: SanitizedQuote
    vendor_iv: float | None
    selected_iv: float
    iv_source: str
    open_interest: int
    volume: int
    multiplier: int


@dataclass(frozen=True, slots=True)
class GreekValidationReport:
    core_errors: Mapping[str, float]
    higher_order_spreads: Mapping[str, float]
    unstable_greeks: tuple[str, ...]
    spot_bumps: tuple[float, ...]
    volatility_bump_scales: tuple[float, ...]

    @property
    def maximum_core_error(self) -> float:
        return max(self.core_errors.values(), default=0.0)


GREEK_DEFINITIONS: dict[str, str] = {
    "delta": "option-value change per $1 underlying move",
    "gamma": "delta change per $1 underlying move",
    "theta": "option-value change per calendar day as expiration approaches",
    "vega": "option-value change per +1 volatility-point move",
    "rho": "option-value change per +1 percentage-point rate move",
    "vanna": "delta change per +1 volatility-point move",
    "charm": "delta change per calendar day as expiration approaches",
    "vomma": "vega change per +1 volatility-point move",
    "speed": "gamma change per $1 underlying move",
    "color": "gamma change per calendar day as expiration approaches",
    "zomma": "gamma change per +1 volatility-point move",
    "ultima": "vomma change per +1 volatility-point move",
}


def option_price_bounds(
    *,
    option_side: OptionSide,
    spot: float,
    strike: float,
    time_years: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> tuple[float, float]:
    t = max(float(time_years), 0.0)
    discounted_spot = float(spot) * np.exp(-float(dividend_yield) * t)
    discounted_strike = float(strike) * np.exp(-float(risk_free_rate) * t)
    if option_side == "call":
        return max(discounted_spot - discounted_strike, 0.0), discounted_spot
    return max(discounted_strike - discounted_spot, 0.0), discounted_strike


def sanitize_quote(
    *,
    option_side: OptionSide,
    spot: float,
    strike: float,
    time_years: float,
    risk_free_rate: float,
    dividend_yield: float,
    bid: object,
    ask: object,
    last: object,
) -> SanitizedQuote:
    """Select a no-arbitrage-consistent mark without hiding quote defects."""

    parsed_bid = _finite_nonnegative(bid)
    parsed_ask = _finite_nonnegative(ask)
    parsed_last = _finite_nonnegative(last)
    lower, upper = option_price_bounds(
        option_side=option_side,
        spot=spot,
        strike=strike,
        time_years=time_years,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )
    tolerance = max(float(spot) * 1e-8, 1e-7)
    warnings: list[str] = []

    valid_market = bool(
        parsed_bid is not None
        and parsed_ask is not None
        and parsed_ask > 0.0
        and parsed_ask >= parsed_bid
    )
    spread_ratio: float | None = None
    midpoint: float | None = None
    if valid_market:
        midpoint = (parsed_bid + parsed_ask) / 2.0
        spread_ratio = (parsed_ask - parsed_bid) / max(midpoint, 0.05)
        if parsed_bid == 0.0:
            warnings.append("The quoted market has a zero bid.")
        if not (lower - tolerance <= midpoint <= upper + tolerance):
            warnings.append("The bid/ask midpoint violates model no-arbitrage bounds.")
            valid_market = False
    elif parsed_bid is not None and parsed_ask is not None and parsed_ask < parsed_bid:
        warnings.append("The quoted market is crossed.")

    last_inside_bounds = bool(
        parsed_last is not None and lower - tolerance <= parsed_last <= upper + tolerance
    )
    last_inside_market = bool(
        valid_market
        and parsed_last is not None
        and parsed_bid is not None
        and parsed_ask is not None
        and parsed_bid - tolerance <= parsed_last <= parsed_ask + tolerance
    )

    if valid_market and midpoint is not None:
        mark = float(np.clip(midpoint, lower, upper))
        mark_source = "midpoint"
        quality = 1.0 / (1.0 + 2.0 * max(spread_ratio or 0.0, 0.0))
        if parsed_bid == 0.0:
            quality *= 0.35
        if last_inside_market:
            quality = min(1.0, quality + 0.05)
    elif last_inside_bounds:
        mark = parsed_last
        mark_source = "last_fallback"
        quality = 0.20
        warnings.append("The last trade was used because no valid two-sided mark was available.")
    else:
        mark = None
        mark_source = "unusable"
        quality = 0.0
        if parsed_last is not None and not last_inside_bounds:
            warnings.append("The last trade violates model no-arbitrage bounds.")

    return SanitizedQuote(
        bid=parsed_bid,
        ask=parsed_ask,
        last=parsed_last,
        mark=mark,
        mark_source=mark_source,
        quote_quality=float(np.clip(quality, 0.0, 1.0)),
        spread_ratio=spread_ratio,
        lower_bound=float(lower),
        upper_bound=float(upper),
        valid_market=valid_market,
        warnings=tuple(warnings),
    )


def gamma_exposure_1pct(
    gamma: np.ndarray | float,
    *,
    position_contracts: np.ndarray | float,
    multiplier: np.ndarray | float,
    spot: float,
    polarity: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Canonical dollar-gamma proxy for a one-percent underlying move."""

    return (
        np.asarray(gamma, dtype=float)
        * np.asarray(position_contracts, dtype=float)
        * np.asarray(multiplier, dtype=float)
        * np.asarray(polarity, dtype=float)
        * float(spot) ** 2
        * 0.01
    )


def adaptive_greek_validation(
    *,
    option_side: OptionSide,
    spot: float,
    strike: float,
    time_years: float,
    volatility: float,
    risk_free_rate: float,
    dividend_yield: float,
    analytic: Mapping[str, float],
) -> GreekValidationReport:
    """Cross-check core Greeks and detect unstable finite-difference higher Greeks."""

    spot_fractions = (5e-5, 1e-4, 2e-4, 5e-4)
    price_args = {
        "option_kind": option_side,
        "strikes": np.asarray([strike], dtype=float),
        "time_years": np.asarray([time_years], dtype=float),
        "volatility": np.asarray([volatility], dtype=float),
        "risk_free_rate": np.asarray([risk_free_rate], dtype=float),
        "dividend_yield": np.asarray([dividend_yield], dtype=float),
    }

    def price(
        *,
        test_spot: float = spot,
        test_time: float = time_years,
        test_vol: float = volatility,
        test_rate: float = risk_free_rate,
    ) -> float:
        return float(
            black_scholes_price(
                option_side,
                test_spot,
                price_args["strikes"],
                np.asarray([max(test_time, 1e-10)]),
                np.asarray([max(test_vol, 1e-8)]),
                np.asarray([test_rate]),
                price_args["dividend_yield"],
            )[0]
        )

    base = price()
    delta_estimates: list[float] = []
    gamma_estimates: list[float] = []
    spot_bumps: list[float] = []
    for fraction in spot_fractions:
        bump = max(float(spot) * fraction, 1e-5)
        plus = price(test_spot=spot + bump)
        minus = price(test_spot=max(spot - bump, 1e-8))
        delta_estimates.append((plus - minus) / (2.0 * bump))
        gamma_estimates.append((plus - 2.0 * base + minus) / bump**2)
        spot_bumps.append(bump)

    time_bumps = tuple(
        min(max(time_years * fraction, 1.0 / (365.25 * 24.0)), time_years * 0.45)
        for fraction in (5e-4, 1e-3, 2e-3)
    )
    vol_bumps = tuple(max(volatility * fraction, minimum) for fraction, minimum in (
        (5e-4, 5e-5),
        (1e-3, 1e-4),
        (2e-3, 2e-4),
    ))
    rate_bumps = (1e-5, 2e-5, 5e-5)

    theta_estimates = [
        -(price(test_time=time_years + bump) - price(test_time=max(time_years - bump, 1e-10)))
        / (2.0 * bump)
        for bump in time_bumps
    ]
    vega_estimates = [
        (price(test_vol=volatility + bump) - price(test_vol=max(volatility - bump, 1e-8)))
        / (2.0 * bump)
        for bump in vol_bumps
    ]
    rho_estimates = [
        (price(test_rate=risk_free_rate + bump) - price(test_rate=risk_free_rate - bump))
        / (2.0 * bump)
        for bump in rate_bumps
    ]

    numerical = {
        "delta": float(np.median(delta_estimates)),
        "gamma": float(np.median(gamma_estimates)),
        "theta": float(np.median(theta_estimates)),
        "vega": float(np.median(vega_estimates)),
        "rho": float(np.median(rho_estimates)),
    }
    floors = {"delta": 1e-7, "gamma": 1e-9, "theta": 1e-6, "vega": 1e-6, "rho": 1e-6}
    core_errors = {
        name: _relative_error(float(analytic[name]), value, floors[name])
        for name, value in numerical.items()
    }

    bump_scales = (0.5, 1.0, 2.0)
    higher_names = ("vanna", "charm", "vomma", "speed", "color", "zomma", "ultima")
    estimates = {name: [] for name in higher_names}
    for scale in bump_scales:
        stack = greek_stack(
            option_side,
            spot=spot,
            strikes=np.asarray([strike], dtype=float),
            time_years=np.asarray([time_years], dtype=float),
            volatility=np.asarray([volatility], dtype=float),
            risk_free_rate=np.asarray([risk_free_rate], dtype=float),
            dividend_yield=np.asarray([dividend_yield], dtype=float),
            bump_scale=scale,
        )
        for name in higher_names:
            estimates[name].append(float(getattr(stack, name)[0]))

    spreads = {name: _relative_spread(values) for name, values in estimates.items()}
    thresholds = {
        "vanna": 0.06,
        "charm": 0.08,
        "vomma": 0.08,
        "speed": 0.12,
        "color": 0.15,
        "zomma": 0.15,
        "ultima": 0.30,
    }
    unstable = tuple(name for name in higher_names if spreads[name] > thresholds[name])
    return GreekValidationReport(
        core_errors=core_errors,
        higher_order_spreads=spreads,
        unstable_greeks=unstable,
        spot_bumps=tuple(spot_bumps),
        volatility_bump_scales=bump_scales,
    )


def _relative_error(left: float, right: float, floor: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), floor)


def _relative_spread(values: list[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size <= 1:
        return 0.0
    if float(np.max(np.abs(finite))) < 1e-12:
        return 0.0
    center = float(np.median(finite))
    return float((np.max(finite) - np.min(finite)) / max(abs(center), 1e-10))


def _finite_nonnegative(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) and parsed >= 0.0 else None
