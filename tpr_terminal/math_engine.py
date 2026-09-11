from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.special import ndtr

from .quant_accuracy import QuantEngineMode, sanitize_quote

OptionKind = Literal["call", "put"]
SQRT_2PI = float(np.sqrt(2.0 * np.pi))


@dataclass(frozen=True, slots=True)
class IVResolution:
    volatility: pd.Series
    source: pd.Series
    market_price: pd.Series
    quote_quality: pd.Series
    valid_market: pd.Series
    mark_source: pd.Series
    pricing_residual: pd.Series
    solver_iterations: pd.Series
    solver_status: pd.Series


@dataclass(frozen=True, slots=True)
class IVSolveResult:
    volatility: float | None
    residual: float | None
    iterations: int
    converged: bool
    status: str


def _normal_pdf(values: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * values**2) / SQRT_2PI


def black_scholes_price(
    *,
    option_kind: OptionKind,
    spot: float,
    strikes: np.ndarray | float,
    time_years: np.ndarray | float,
    volatility: np.ndarray | float,
    risk_free_rate: np.ndarray | float,
    dividend_yield: np.ndarray | float,
) -> np.ndarray:
    strikes_a = np.asarray(strikes, dtype=float)
    time_a = np.maximum(np.asarray(time_years, dtype=float), 1e-12)
    vol_a = np.maximum(np.asarray(volatility, dtype=float), 1e-12)
    rate_a = np.asarray(risk_free_rate, dtype=float)
    yield_a = np.asarray(dividend_yield, dtype=float)
    spot_f = max(float(spot), 1e-12)

    root_t = np.sqrt(time_a)
    d1 = (
        np.log(spot_f / strikes_a)
        + (rate_a - yield_a + 0.5 * vol_a**2) * time_a
    ) / (vol_a * root_t)
    d2 = d1 - vol_a * root_t
    discounted_spot = spot_f * np.exp(-yield_a * time_a)
    discounted_strike = strikes_a * np.exp(-rate_a * time_a)

    if option_kind == "call":
        return discounted_spot * ndtr(d1) - discounted_strike * ndtr(d2)
    if option_kind == "put":
        return discounted_strike * ndtr(-d2) - discounted_spot * ndtr(-d1)
    raise ValueError(f"Unsupported option kind: {option_kind}")



def black_scholes_delta(
    *,
    option_kind: OptionKind,
    spot: float,
    strikes: np.ndarray,
    time_years: np.ndarray | float,
    volatility: np.ndarray,
    risk_free_rate: np.ndarray | float,
    dividend_yield: np.ndarray | float,
) -> np.ndarray:
    strikes_a = np.asarray(strikes, dtype=float)
    vol_a = np.maximum(np.asarray(volatility, dtype=float), 1e-12)
    time_a = np.maximum(np.asarray(time_years, dtype=float), 1e-12)
    rate_a = np.asarray(risk_free_rate, dtype=float)
    yield_a = np.asarray(dividend_yield, dtype=float)
    spot_f = max(float(spot), 1e-12)
    root_t = np.sqrt(time_a)
    d1 = (
        np.log(spot_f / strikes_a)
        + (rate_a - yield_a + 0.5 * vol_a**2) * time_a
    ) / (vol_a * root_t)
    discount = np.exp(-yield_a * time_a)
    if option_kind == "call":
        delta = discount * ndtr(d1)
    elif option_kind == "put":
        delta = discount * (ndtr(d1) - 1.0)
    else:
        raise ValueError(f"Unsupported option kind: {option_kind}")
    return np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

def black_scholes_gamma(
    *,
    spot: float,
    strikes: np.ndarray,
    time_years: np.ndarray | float,
    volatility: np.ndarray,
    risk_free_rate: np.ndarray | float,
    dividend_yield: np.ndarray | float,
) -> np.ndarray:
    strikes_a = np.asarray(strikes, dtype=float)
    vol_a = np.maximum(np.asarray(volatility, dtype=float), 1e-12)
    time_a = np.maximum(np.asarray(time_years, dtype=float), 1e-12)
    rate_a = np.asarray(risk_free_rate, dtype=float)
    yield_a = np.asarray(dividend_yield, dtype=float)
    spot_f = max(float(spot), 1e-12)
    root_t = np.sqrt(time_a)
    d1 = (
        np.log(spot_f / strikes_a)
        + (rate_a - yield_a + 0.5 * vol_a**2) * time_a
    ) / (vol_a * root_t)
    gamma = np.exp(-yield_a * time_a) * _normal_pdf(d1) / (spot_f * vol_a * root_t)
    return np.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)


def no_arbitrage_bounds(
    *,
    option_kind: OptionKind,
    spot: float,
    strike: float,
    time_years: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> tuple[float, float]:
    discounted_spot = spot * np.exp(-dividend_yield * time_years)
    discounted_strike = strike * np.exp(-risk_free_rate * time_years)
    if option_kind == "call":
        return max(0.0, discounted_spot - discounted_strike), discounted_spot
    return max(0.0, discounted_strike - discounted_spot), discounted_strike


def implied_volatility_brent(
    *,
    option_kind: OptionKind,
    market_price: float,
    spot: float,
    strike: float,
    time_years: float,
    risk_free_rate: float,
    dividend_yield: float,
    lower_vol: float = 1e-4,
    upper_vol: float = 5.0,
) -> float | None:
    return implied_volatility_solve(
        option_kind=option_kind,
        market_price=market_price,
        spot=spot,
        strike=strike,
        time_years=time_years,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        lower_vol=lower_vol,
        upper_vol=upper_vol,
    ).volatility


def implied_volatility_solve(
    *,
    option_kind: OptionKind,
    market_price: float,
    spot: float,
    strike: float,
    time_years: float,
    risk_free_rate: float,
    dividend_yield: float,
    lower_vol: float = 1e-4,
    upper_vol: float = 5.0,
) -> IVSolveResult:
    if not all(np.isfinite(v) for v in (market_price, spot, strike, time_years)):
        return IVSolveResult(None, None, 0, False, "non_finite_input")
    if market_price <= 0 or spot <= 0 or strike <= 0 or time_years <= 0:
        return IVSolveResult(None, None, 0, False, "invalid_input")
    lower_price, upper_price = no_arbitrage_bounds(
        option_kind=option_kind,
        spot=spot,
        strike=strike,
        time_years=time_years,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )
    tolerance = max(1e-7, spot * 1e-8)
    if market_price < lower_price - tolerance or market_price > upper_price + tolerance:
        return IVSolveResult(None, None, 0, False, "outside_no_arbitrage_bounds")

    def objective(volatility: float) -> float:
        model = black_scholes_price(
            option_kind=option_kind,
            spot=spot,
            strikes=strike,
            time_years=time_years,
            volatility=volatility,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
        )
        return float(model) - market_price

    low_value = objective(lower_vol)
    high_value = objective(upper_vol)
    if low_value == 0:
        return IVSolveResult(lower_vol, 0.0, 0, True, "lower_bound_root")
    if high_value == 0:
        return IVSolveResult(upper_vol, 0.0, 0, True, "upper_bound_root")
    if low_value * high_value > 0:
        return IVSolveResult(None, None, 0, False, "root_not_bracketed")
    try:
        root, result = brentq(
            objective,
            lower_vol,
            upper_vol,
            xtol=1e-10,
            rtol=1e-10,
            maxiter=150,
            full_output=True,
            disp=False,
        )
        residual = abs(objective(float(root)))
        converged = bool(result.converged and residual <= max(tolerance, 1e-8))
        return IVSolveResult(
            float(root) if converged else None,
            float(residual),
            int(result.iterations),
            converged,
            "converged" if converged else "residual_too_large",
        )
    except (ValueError, RuntimeError, OverflowError):
        return IVSolveResult(None, None, 0, False, "solver_error")


def clean_option_quotes(
    frame: pd.DataFrame,
    *,
    mode: QuantEngineMode = "legacy",
    option_kind: OptionKind | None = None,
    spot: float | None = None,
    time_years: float | None = None,
    risk_free_rate: float | None = None,
    dividend_yield: float | None = None,
) -> pd.DataFrame:
    result = frame.copy()
    for column in ("bid", "ask", "lastPrice"):
        if column not in result.columns:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")

    if mode != "legacy":
        required = (option_kind, spot, time_years, risk_free_rate, dividend_yield)
        if any(value is None for value in required):
            raise ValueError("Candidate quote sanitation requires the complete pricing context.")
        marks = []
        for _, row in result.iterrows():
            strike = pd.to_numeric(row.get("strike"), errors="coerce")
            if not np.isfinite(strike) or float(strike) <= 0:
                marks.append(None)
                continue
            marks.append(
                sanitize_quote(
                    option_side=option_kind,
                    spot=float(spot),
                    strike=float(strike),
                    time_years=float(time_years),
                    risk_free_rate=float(risk_free_rate),
                    dividend_yield=float(dividend_yield),
                    bid=row.get("bid"),
                    ask=row.get("ask"),
                    last=row.get("lastPrice"),
                )
            )
        result["market_price"] = [None if item is None else item.mark for item in marks]
        result["valid_market"] = [False if item is None else item.valid_market for item in marks]
        result["spread_ratio"] = [np.nan if item is None else item.spread_ratio for item in marks]
        result["quote_quality"] = [0.0 if item is None else item.quote_quality for item in marks]
        result["mark_source"] = ["unusable" if item is None else item.mark_source for item in marks]
        return result

    bid = result["bid"].fillna(0.0).clip(lower=0.0)
    ask = result["ask"].fillna(0.0).clip(lower=0.0)
    last = result["lastPrice"].fillna(0.0).clip(lower=0.0)
    valid_market = (bid > 0) & (ask > 0) & (ask >= bid)
    midpoint = (bid + ask) / 2.0
    spread = (ask - bid).where(valid_market, np.nan)
    spread_ratio = spread / midpoint.clip(lower=0.05)
    selected = midpoint.where(valid_market, last.where(last > 0, np.nan))

    quote_quality = pd.Series(0.0, index=result.index, dtype=float)
    quote_quality.loc[valid_market] = 1.0 / (1.0 + 2.0 * spread_ratio.loc[valid_market].clip(lower=0.0))
    fallback_last = (~valid_market) & (last > 0)
    quote_quality.loc[fallback_last] = 0.25

    result["market_price"] = selected
    result["valid_market"] = valid_market
    result["spread_ratio"] = spread_ratio
    result["quote_quality"] = quote_quality.clip(0.0, 1.0)
    result["mark_source"] = np.where(valid_market, "midpoint", np.where(fallback_last, "last_fallback", "unusable"))
    return result


def resolve_implied_volatility(
    frame: pd.DataFrame,
    *,
    option_kind: OptionKind,
    spot: float,
    time_years: float,
    risk_free_rate: float,
    dividend_yield: float,
    mode: QuantEngineMode = "legacy",
) -> IVResolution:
    quoted = clean_option_quotes(
        frame,
        mode=mode,
        option_kind=option_kind,
        spot=spot,
        time_years=time_years,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )
    strikes = pd.to_numeric(quoted["strike"], errors="coerce")
    vendor_source = (
        quoted["impliedVolatility"]
        if "impliedVolatility" in quoted.columns
        else pd.Series(np.nan, index=quoted.index)
    )
    vendor = pd.to_numeric(vendor_source, errors="coerce")
    resolved = pd.Series(np.nan, index=quoted.index, dtype=float)
    source = pd.Series("missing", index=quoted.index, dtype="object")
    pricing_residual = pd.Series(np.nan, index=quoted.index, dtype=float)
    solver_iterations = pd.Series(0, index=quoted.index, dtype=int)
    solver_status = pd.Series("not_attempted", index=quoted.index, dtype="object")

    for index in quoted.index:
        try:
            price = float(quoted.at[index, "market_price"])
            strike = float(strikes.at[index])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(price) or not np.isfinite(strike):
            continue
        diagnostic = implied_volatility_solve(
            option_kind=option_kind,
            market_price=price,
            spot=spot,
            strike=strike,
            time_years=time_years,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
        )
        pricing_residual.at[index] = diagnostic.residual
        solver_iterations.at[index] = diagnostic.iterations
        solver_status.at[index] = diagnostic.status
        solved = diagnostic.volatility
        if solved is not None and 0.005 <= solved <= 5.0:
            resolved.at[index] = solved
            source.at[index] = "market_solved"

    vendor_valid = vendor.notna() & np.isfinite(vendor) & vendor.between(0.005, 5.0)
    vendor_fill = resolved.isna() & vendor_valid
    resolved.loc[vendor_fill] = vendor.loc[vendor_fill]
    source.loc[vendor_fill] = "vendor"

    valid = resolved.notna() & np.isfinite(resolved) & resolved.between(0.005, 5.0)
    if int(valid.sum()) >= 2:
        x = np.log(strikes.loc[valid].to_numpy(dtype=float) / float(spot))
        y = resolved.loc[valid].to_numpy(dtype=float)
        order = np.argsort(x)
        grouped = (
            pd.DataFrame({"x": x[order], "y": y[order]})
            .groupby("x", as_index=False)["y"]
            .median()
        )
        target = np.log(strikes.to_numpy(dtype=float) / float(spot))
        if len(grouped) >= 3:
            interpolator = PchipInterpolator(
                grouped["x"].to_numpy(dtype=float),
                grouped["y"].to_numpy(dtype=float),
                extrapolate=True,
            )
            fill_values = interpolator(target)
        else:
            fill_values = np.interp(
                target,
                grouped["x"].to_numpy(dtype=float),
                grouped["y"].to_numpy(dtype=float),
            )
        missing = ~valid
        resolved.loc[missing] = np.asarray(fill_values, dtype=float)[missing.to_numpy()]
        source.loc[missing] = "interpolated"
    elif int(valid.sum()) == 1:
        missing = ~valid
        resolved.loc[missing] = float(resolved.loc[valid].iloc[0])
        source.loc[missing] = "flat_fallback"
    else:
        resolved.loc[:] = 0.35
        source.loc[:] = "flat_fallback"

    resolved = resolved.clip(0.01, 4.0).fillna(0.35)
    return IVResolution(
        volatility=resolved,
        source=source,
        market_price=quoted["market_price"],
        quote_quality=quoted["quote_quality"],
        valid_market=quoted["valid_market"],
        mark_source=quoted["mark_source"],
        pricing_residual=pricing_residual,
        solver_iterations=solver_iterations,
        solver_status=solver_status,
    )


def option_mid_price(frame: pd.DataFrame) -> pd.Series:
    return clean_option_quotes(frame)["market_price"].fillna(0.0)


def repair_implied_volatility(
    strikes: pd.Series,
    implied_volatility: pd.Series,
    spot: float,
) -> tuple[pd.Series, int]:
    # Compatibility wrapper retained for older tests and adapters.
    frame = pd.DataFrame(
        {
            "strike": strikes,
            "impliedVolatility": implied_volatility,
            "bid": np.nan,
            "ask": np.nan,
            "lastPrice": np.nan,
        }
    )
    result = resolve_implied_volatility(
        frame,
        option_kind="call",
        spot=spot,
        time_years=30 / 365.25,
        risk_free_rate=0.04,
        dividend_yield=0.0,
    )
    repaired = int((result.source != "vendor").sum())
    return result.volatility, repaired
