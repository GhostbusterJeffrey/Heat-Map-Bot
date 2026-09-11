from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .models import GammaFlip, PinLevel, VolatilitySurfaceResult
from .pricing import black_scholes_delta, black_scholes_gamma
from .volatility_surface import evaluate_surface_iv

_MIN_T = 1.0 / (365.25 * 24 * 60)


def _surface_volatility(
    contracts: pd.DataFrame,
    surface: VolatilitySurfaceResult | None,
    *,
    scenario_spot: float,
    current_spot: float,
    mode: str,
    iv_shift: float,
) -> np.ndarray:
    base = np.maximum(contracts["iv"].to_numpy(dtype=float) + float(iv_shift), 0.01)
    if surface is None or mode == "sticky_strike" or not surface.svi_parameters:
        return base
    output = base.copy()
    for expiration, frame in contracts.groupby("expiration"):
        exp = expiration if isinstance(expiration, date) else pd.Timestamp(expiration).date()
        indexes = frame.index.to_numpy(dtype=int)
        current_forward = surface.forwards.get(exp)
        if current_forward is None or current_forward <= 0:
            continue
        scenario_forward = current_forward * float(scenario_spot) / max(float(current_spot), 1e-12)
        fitted = evaluate_surface_iv(
            surface,
            exp,
            frame["strike"].to_numpy(dtype=float),
            frame["time_years"].to_numpy(dtype=float),
            np.full(len(frame), scenario_forward, dtype=float),
        )
        good = np.isfinite(fitted) & (fitted > 0)
        output[indexes[good]] = np.maximum(fitted[good] + float(iv_shift), 0.01)
    return output


def _root_locations(
    contracts: pd.DataFrame,
    *,
    spot: float,
    multiplier: int,
    surface: VolatilitySurfaceResult | None = None,
    surface_mode: str = "sticky_strike",
    iv_shift: float = 0.0,
    time_shift_years: float = 0.0,
    grid_low_pct: float = 0.85,
    grid_high_pct: float = 1.15,
    grid_points: int = 601,
) -> tuple[pd.DataFrame, tuple[float, ...]]:
    if contracts.empty:
        return pd.DataFrame({"spot": [spot], "gex": [0.0]}), ()
    frame = contracts.reset_index(drop=True)
    strikes = frame["strike"].to_numpy(dtype=float)
    time_years = np.maximum(
        frame["time_years"].to_numpy(dtype=float) - float(time_shift_years),
        _MIN_T,
    )
    rate = frame["rate"].to_numpy(dtype=float)
    dividend = frame["dividend_yield"].to_numpy(dtype=float)
    liquidity = frame["liquidity"].to_numpy(dtype=float)
    polarity = frame["polarity"].to_numpy(dtype=float)

    def exposure_at(scenario: float) -> float:
        iv = _surface_volatility(
            frame,
            surface,
            scenario_spot=scenario,
            current_spot=spot,
            mode=surface_mode,
            iv_shift=iv_shift,
        )
        gamma = black_scholes_gamma(
            spot=scenario,
            strikes=strikes,
            time_years=time_years,
            volatility=iv,
            risk_free_rate=rate,
            dividend_yield=dividend,
        )
        return float(
            np.nansum(liquidity * gamma * multiplier * scenario**2 * 0.01 * polarity)
        )

    scenarios = np.linspace(spot * grid_low_pct, spot * grid_high_pct, int(grid_points))
    totals = np.asarray([exposure_at(float(item)) for item in scenarios], dtype=float)
    roots: list[float] = []
    scale = max(float(np.nanpercentile(np.abs(totals), 75)), 1.0)
    for index in range(len(scenarios) - 1):
        left, right = float(scenarios[index]), float(scenarios[index + 1])
        y_left, y_right = float(totals[index]), float(totals[index + 1])
        if abs(y_left) < scale * 1e-8:
            roots.append(left)
        elif y_left * y_right < 0:
            try:
                roots.append(float(brentq(exposure_at, left, right, xtol=1e-8, maxiter=100)))
            except (ValueError, RuntimeError):
                continue
    unique: list[float] = []
    for root in sorted(roots):
        if not unique or abs(root - unique[-1]) > max(spot * 1e-5, 1e-4):
            unique.append(root)
    return pd.DataFrame({"spot": scenarios, f"gex_{surface_mode}": totals}), tuple(unique)


def build_gamma_flips(
    contracts: pd.DataFrame,
    *,
    spot: float,
    multiplier: int,
    surface: VolatilitySurfaceResult | None = None,
    min_confidence: float = 0.45,
    grid_low_pct: float = 0.85,
    grid_high_pct: float = 1.15,
    grid_points: int = 601,
) -> tuple[pd.DataFrame, tuple[GammaFlip, ...]]:
    profile_strike, roots_strike = _root_locations(
        contracts,
        spot=spot,
        multiplier=multiplier,
        surface=surface,
        surface_mode="sticky_strike",
        grid_low_pct=grid_low_pct,
        grid_high_pct=grid_high_pct,
        grid_points=grid_points,
    )
    profile_delta, roots_delta = _root_locations(
        contracts,
        spot=spot,
        multiplier=multiplier,
        surface=surface,
        surface_mode="sticky_delta",
        grid_low_pct=grid_low_pct,
        grid_high_pct=grid_high_pct,
        grid_points=grid_points,
    )
    profile = profile_strike.merge(profile_delta, on="spot", how="outer").sort_values("spot")
    profile["gex"] = profile["gex_sticky_strike"]
    if not roots_strike:
        return profile, ()

    shock_sets = [
        (-0.01, 0.0),
        (0.01, 0.0),
        (0.0, 1.0 / (365.25 * 24)),
        (0.0, 1.0 / 365.25),
    ]
    shock_roots: list[tuple[float, ...]] = []
    for iv_shift, time_shift in shock_sets:
        _, values = _root_locations(
            contracts,
            spot=spot,
            multiplier=multiplier,
            surface=surface,
            surface_mode="sticky_strike",
            iv_shift=iv_shift,
            time_shift_years=time_shift,
            grid_low_pct=grid_low_pct,
            grid_high_pct=grid_high_pct,
            grid_points=max(301, grid_points // 2),
        )
        shock_roots.append(values)

    x = profile["spot"].to_numpy(dtype=float)
    y = profile["gex_sticky_strike"].to_numpy(dtype=float)
    gex_scale = max(float(np.nanpercentile(np.abs(y), 75)), 1.0)
    h = max(spot * 0.0005, 0.01)
    flips: list[GammaFlip] = []
    mean_quality = (
        float(contracts["contract_quality"].mean())
        if "contract_quality" in contracts
        else 0.5
    )
    for root in roots_strike:
        left = float(np.interp(root - h, x, y))
        right = float(np.interp(root + h, x, y))
        slope = (right - left) / (2.0 * h)
        slope_score = float(np.clip(abs(slope) * spot / gex_scale, 0.0, 1.0))
        matched: list[float] = []
        for values in shock_roots:
            if values:
                nearest = min(values, key=lambda item: abs(item - root))
                if abs(nearest - root) <= spot * 0.03:
                    matched.append(nearest)
        if matched:
            std = float(np.std(matched + [root]))
            shock_stability = float(
                np.clip(1.0 - std / max(spot * 0.01, 1e-6), 0.0, 1.0)
            )
        else:
            shock_stability = 0.0

        sticky_delta = (
            min(roots_delta, key=lambda item: abs(item - root)) if roots_delta else None
        )
        model_spread = abs(sticky_delta - root) if sticky_delta is not None else None
        model_stability = (
            float(np.clip(1.0 - model_spread / max(spot * 0.01, 1e-6), 0.0, 1.0))
            if model_spread is not None
            else 0.25
        )
        combined_stability = 0.70 * shock_stability + 0.30 * model_stability
        confidence = float(
            np.clip(
                0.35 * slope_score
                + 0.35 * combined_stability
                + 0.20 * mean_quality
                + 0.10 * (surface.score / 100.0 if surface is not None else 0.5),
                0.0,
                1.0,
            )
        )
        if confidence < min_confidence:
            continue
        flips.append(
            GammaFlip(
                level=float(root),
                distance=float(root - spot),
                distance_pct=float(root / spot - 1.0),
                slope_at_root=float(slope),
                shock_stability=float(combined_stability),
                confidence=confidence,
                sticky_strike_level=float(root),
                sticky_delta_level=float(sticky_delta) if sticky_delta is not None else None,
                model_spread=float(model_spread) if model_spread is not None else None,
            )
        )
    flips.sort(key=lambda item: abs(item.distance))
    return profile, tuple(flips)


def _delta_for_frame(
    frame: pd.DataFrame,
    scenario_spot: float,
    *,
    current_spot: float,
    surface: VolatilitySurfaceResult | None,
    surface_mode: str = "sticky_delta",
    time_shift: float = 0.0,
) -> np.ndarray:
    normalized = frame.reset_index(drop=True)
    output = np.zeros(len(normalized), dtype=float)
    iv = _surface_volatility(
        normalized,
        surface,
        scenario_spot=scenario_spot,
        current_spot=current_spot,
        mode=surface_mode,
        iv_shift=0.0,
    )
    for side in ("call", "put"):
        mask = normalized["side"].to_numpy() == side
        if not np.any(mask):
            continue
        output[mask] = black_scholes_delta(
            option_kind=side,
            spot=scenario_spot,
            strikes=normalized.loc[mask, "strike"].to_numpy(dtype=float),
            time_years=np.maximum(
                normalized.loc[mask, "time_years"].to_numpy(dtype=float) - time_shift,
                _MIN_T,
            ),
            volatility=iv[mask],
            risk_free_rate=normalized.loc[mask, "rate"].to_numpy(dtype=float),
            dividend_yield=normalized.loc[mask, "dividend_yield"].to_numpy(dtype=float),
        )
    return output


def build_pin_levels(
    contracts: pd.DataFrame,
    strike_totals: pd.DataFrame,
    *,
    spot: float,
    expected_move: float | None,
    multiplier: int,
    surface: VolatilitySurfaceResult | None = None,
    pin_decay_days: float = 7.0,
    pin_shock_pct: float = 0.0025,
    min_score: float = 35.0,
) -> tuple[PinLevel, ...]:
    if contracts.empty or strike_totals.empty:
        return ()
    normalized = contracts.reset_index(drop=True)
    current_delta = _delta_for_frame(
        normalized,
        spot,
        current_spot=spot,
        surface=surface,
        surface_mode="sticky_delta",
    )
    signed_position = (
        normalized["liquidity"].to_numpy(dtype=float)
        * normalized["polarity"].to_numpy(dtype=float)
        * multiplier
    )

    candidates = strike_totals.copy()
    candidates = candidates[candidates["strike"].between(spot * 0.88, spot * 1.12)]
    if candidates.empty:
        return ()
    raw_restoring: list[float] = []
    rows: list[dict[str, float]] = []
    increments = np.diff(np.sort(candidates["strike"].unique()))
    min_increment = float(np.nanmedian(increments)) if increments.size else spot * 0.0025
    if not np.isfinite(min_increment) or min_increment <= 0:
        min_increment = spot * 0.0025
    shock = max(spot * pin_shock_pct, min_increment)

    for strike in candidates["strike"].astype(float):
        below_delta = _delta_for_frame(
            normalized,
            strike - shock,
            current_spot=spot,
            surface=surface,
            surface_mode="sticky_delta",
        )
        above_delta = _delta_for_frame(
            normalized,
            strike + shock,
            current_spot=spot,
            surface=surface,
            surface_mode="sticky_delta",
        )
        flow_below = float(-np.nansum(signed_position * (below_delta - current_delta)))
        flow_above = float(-np.nansum(signed_position * (above_delta - current_delta)))
        restoring = min(max(flow_below, 0.0), max(-flow_above, 0.0))
        raw_restoring.append(restoring)
        related = normalized[np.isclose(normalized["strike"].astype(float), strike)]
        abs_exposure = np.abs(related["gex"].to_numpy(dtype=float))
        if abs_exposure.sum() > 0:
            quality = float(np.average(related["contract_quality"], weights=abs_exposure))
        else:
            quality = float(related["contract_quality"].mean()) if not related.empty else 0.0
        time_weights = (
            np.exp(
                -related["days_to_expiry"].to_numpy(dtype=float)
                / max(pin_decay_days, 1e-6)
            )
            if not related.empty
            else np.array([0.0])
        )
        concentration = (
            float(np.average(time_weights, weights=np.maximum(abs_exposure, 1e-9)))
            if not related.empty
            else 0.0
        )
        breadth = float(
            related["expiration"].nunique() / max(normalized["expiration"].nunique(), 1)
        )
        structural = (
            float(
                candidates.loc[
                    candidates["strike"] == strike, "structural_score"
                ].iloc[0]
            )
            if "structural_score" in candidates.columns
            else 0.5
        )
        rows.append(
            {
                "strike": strike,
                "restoring": restoring,
                "quality": quality,
                "concentration": concentration,
                "breadth": breadth,
                "structural": structural,
            }
        )

    scale = max(
        float(np.nanpercentile(raw_restoring, 90)) if raw_restoring else 0.0,
        1e-9,
    )
    move = expected_move if expected_move and expected_move > 0 else spot * 0.01
    output: list[PinLevel] = []
    for row in rows:
        restoring_score = float(np.clip(row["restoring"] / scale, 0.0, 1.0))
        distance = float(row["strike"] - spot)
        distance_penalty = float(np.exp(-abs(distance) / max(move, 1e-9)))
        surface_stability = surface.score / 100.0 if surface is not None else 0.5
        score = 100.0 * (
            restoring_score * 0.36
            + row["concentration"] * 0.23
            + row["structural"] * 0.14
            + row["breadth"] * 0.09
            + row["quality"] * 0.10
            + surface_stability * 0.08
        ) * distance_penalty
        if score < min_score:
            continue
        label = (
            "very high"
            if score >= 80
            else "high"
            if score >= 65
            else "moderate"
            if score >= 45
            else "low"
        )
        output.append(
            PinLevel(
                strike=float(row["strike"]),
                distance=distance,
                distance_pct=float(distance / spot),
                restoring_flow_score=restoring_score,
                near_term_concentration=float(row["concentration"]),
                expiry_breadth=float(row["breadth"]),
                quality=float(row["quality"]),
                pin_score=float(score),
                label=label,
            )
        )
    output.sort(key=lambda item: item.pin_score, reverse=True)
    return tuple(output)
