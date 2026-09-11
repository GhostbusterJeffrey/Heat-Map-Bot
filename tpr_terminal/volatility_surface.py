from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .models import (
    DataQualityReport,
    ForwardSummary,
    SSVIParameters,
    SVIParameters,
    VolatilitySurfaceResult,
)
from .quant_accuracy import QuantEngineMode

_EPS = 1e-12


def _grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    if score >= 50:
        return "E"
    return "F"


def raw_svi(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
    x = np.asarray(k, dtype=float) - float(m)
    return float(a) + float(b) * (
        float(rho) * x + np.sqrt(x * x + float(sigma) ** 2)
    )


def ssvi_phi(theta: np.ndarray, eta: float, gamma: float) -> np.ndarray:
    th = np.maximum(np.asarray(theta, dtype=float), 1e-8)
    return float(eta) / (th**float(gamma) * (1.0 + th) ** (1.0 - float(gamma)))


def raw_ssvi(
    k: np.ndarray,
    theta: np.ndarray,
    rho: float,
    eta: float,
    gamma: float,
) -> np.ndarray:
    kk = np.asarray(k, dtype=float)
    th = np.maximum(np.asarray(theta, dtype=float), 1e-8)
    phi = ssvi_phi(th, eta, gamma)
    term = phi * kk + float(rho)
    return 0.5 * th * (
        1.0
        + float(rho) * phi * kk
        + np.sqrt(term * term + 1.0 - float(rho) ** 2)
    )


@dataclass(frozen=True, slots=True)
class _SliceData:
    expiration: date
    forward: float
    discount_factor: float
    time_years: float
    k: np.ndarray
    total_variance: np.ndarray
    weights: np.ndarray


def _weights(frame: pd.DataFrame) -> np.ndarray:
    quality = np.clip(frame.get("contract_quality", 0.5).to_numpy(dtype=float), 0.02, 1.0)
    quote = np.clip(frame.get("quote_quality", 0.5).to_numpy(dtype=float), 0.02, 1.0)
    liquidity = np.maximum(frame.get("liquidity", 0.0).to_numpy(dtype=float), 0.0)
    liq_scale = np.log1p(liquidity)
    if np.nanmax(liq_scale) > 0:
        liq_scale = 0.20 + 0.80 * liq_scale / np.nanmax(liq_scale)
    else:
        liq_scale = np.full_like(quality, 0.20)
    return np.clip(quality * quote * liq_scale, 1e-4, 1.0)


def _forward_map(
    contracts: pd.DataFrame,
    forward_summaries: list[ForwardSummary],
    spot: float,
) -> tuple[dict[date, float], dict[date, float]]:
    summaries = {item.expiration: item for item in forward_summaries}
    forwards: dict[date, float] = {}
    dfs: dict[date, float] = {}
    for expiration, frame in contracts.groupby("expiration"):
        exp = expiration if isinstance(expiration, date) else pd.Timestamp(expiration).date()
        rate = float(frame["rate"].median())
        q = float(frame["dividend_yield"].median())
        t = max(float(frame["time_years"].median()), 1e-8)
        fallback_forward = float(spot * np.exp((rate - q) * t))
        summary = summaries.get(exp)
        forwards[exp] = (
            float(summary.forward)
            if summary is not None and summary.reliable and summary.forward > 0
            else fallback_forward
        )
        dfs[exp] = float(np.exp(-rate * t))
    return forwards, dfs


def _slice_data(
    contracts: pd.DataFrame,
    forwards: dict[date, float],
    dfs: dict[date, float],
) -> list[_SliceData]:
    slices: list[_SliceData] = []
    for expiration, frame in contracts.groupby("expiration"):
        exp = expiration if isinstance(expiration, date) else pd.Timestamp(expiration).date()
        fwd = forwards.get(exp)
        if fwd is None or not np.isfinite(fwd) or fwd <= 0:
            continue
        clean = frame[
            np.isfinite(frame["strike"])
            & np.isfinite(frame["iv"])
            & np.isfinite(frame["time_years"])
            & (frame["strike"] > 0)
            & (frame["iv"] > 0.01)
            & (frame["iv"] < 5.0)
            & (frame["time_years"] > 0)
        ].copy()
        if len(clean) < 5:
            continue
        t = float(clean["time_years"].median())
        clean["k"] = np.log(clean["strike"].astype(float) / float(fwd))
        clean["w"] = clean["iv"].astype(float) ** 2 * t
        clean["fit_weight"] = _weights(clean)
        grouped = (
            clean.groupby("k", as_index=False)
            .apply(
                lambda group: pd.Series(
                    {
                        "w": float(
                            np.average(
                                group["w"].to_numpy(dtype=float),
                                weights=group["fit_weight"].to_numpy(dtype=float),
                            )
                        ),
                        "fit_weight": float(group["fit_weight"].sum()),
                    }
                ),
                include_groups=False,
            )
            .reset_index(drop=True)
            .sort_values("k")
        )
        if len(grouped) < 5:
            continue
        slices.append(
            _SliceData(
                expiration=exp,
                forward=float(fwd),
                discount_factor=float(dfs.get(exp, np.exp(-float(clean["rate"].median()) * t))),
                time_years=t,
                k=grouped["k"].to_numpy(dtype=float),
                total_variance=grouped["w"].to_numpy(dtype=float),
                weights=np.clip(grouped["fit_weight"].to_numpy(dtype=float), 1e-4, None),
            )
        )
    return sorted(slices, key=lambda item: item.time_years)


def _fit_svi(slice_data: _SliceData) -> SVIParameters | None:
    k = slice_data.k
    w = slice_data.total_variance
    weights = slice_data.weights / max(float(np.max(slice_data.weights)), _EPS)
    if len(k) < 5 or not np.all(np.isfinite(w)):
        return None

    k_range = max(float(np.ptp(k)), 0.10)
    w_min = max(float(np.nanmin(w)), 1e-8)
    w_max = max(float(np.nanmax(w)), w_min + 1e-6)
    base_b = max((w_max - w_min) / k_range, 0.01)
    m0 = float(k[int(np.nanargmin(w))])
    starts = [
        np.array([w_min * 0.80, base_b, -0.35, m0, 0.15]),
        np.array([w_min * 0.65, base_b * 0.60, -0.10, 0.0, 0.30]),
        np.array([w_min * 0.90, base_b * 1.30, -0.60, m0, 0.08]),
    ]
    lower = np.array([0.0, 1e-7, -0.999, -3.0, 1e-4])
    upper = np.array([max(w_max * 3.0, 3.0), 10.0, 0.999, 3.0, 5.0])

    def residual(params: np.ndarray) -> np.ndarray:
        a, b, rho, m, sigma = params
        model = raw_svi(k, a, b, rho, m, sigma)
        base = np.sqrt(weights) * (model - w)
        left_slope = b * (1.0 - rho)
        right_slope = b * (1.0 + rho)
        penalties = np.asarray(
            [
                max(-float(np.min(model)), 0.0) * 10.0,
                max(left_slope - 2.0, 0.0) * 3.0,
                max(right_slope - 2.0, 0.0) * 3.0,
            ],
            dtype=float,
        )
        return np.concatenate([base, penalties])

    best = None
    best_cost = np.inf
    for start in starts:
        try:
            fit = least_squares(
                residual,
                np.clip(start, lower + 1e-10, upper - 1e-10),
                bounds=(lower, upper),
                loss="soft_l1",
                f_scale=max(float(np.nanmedian(w)) * 0.10, 1e-4),
                max_nfev=4_000,
            )
        except Exception:
            continue
        if fit.cost < best_cost and np.all(np.isfinite(fit.x)):
            best = fit
            best_cost = float(fit.cost)
    if best is None:
        return None
    a, b, rho, m, sigma = (float(v) for v in best.x)
    model = raw_svi(k, a, b, rho, m, sigma)
    rmse = float(np.sqrt(np.average((model - w) ** 2, weights=weights)))
    return SVIParameters(
        a=a,
        b=b,
        rho=rho,
        m=m,
        sigma=sigma,
        weighted_rmse=rmse,
        points_used=len(k),
        success=bool(best.success and np.all(model > 0)),
    )


def _atm_theta(params: SVIParameters) -> float:
    return float(raw_svi(np.asarray([0.0]), params.a, params.b, params.rho, params.m, params.sigma)[0])


def _fit_ssvi(
    slices: list[_SliceData],
    svi: dict[date, SVIParameters],
) -> SSVIParameters | None:
    usable = [item for item in slices if item.expiration in svi]
    if len(usable) < 2:
        return None
    all_k: list[np.ndarray] = []
    all_theta: list[np.ndarray] = []
    all_w: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    for item in usable:
        theta = max(_atm_theta(svi[item.expiration]), 1e-6)
        all_k.append(item.k)
        all_theta.append(np.full_like(item.k, theta))
        all_w.append(item.total_variance)
        all_weights.append(item.weights / max(float(np.max(item.weights)), _EPS))
    k = np.concatenate(all_k)
    theta = np.concatenate(all_theta)
    w = np.concatenate(all_w)
    weights = np.concatenate(all_weights)

    def residual(params: np.ndarray) -> np.ndarray:
        rho, eta, gamma = params
        model = raw_ssvi(k, theta, rho, eta, gamma)
        base = np.sqrt(weights) * (model - w)
        phi = ssvi_phi(theta, eta, gamma)
        cond1 = theta * phi * (1.0 + abs(rho))
        cond2 = theta * phi * phi * (1.0 + abs(rho))
        penalties = np.concatenate(
            [
                np.maximum(cond1 - 4.0, 0.0) * 0.25,
                np.maximum(cond2 - 4.0, 0.0) * 0.25,
                np.maximum(-model, 0.0) * 10.0,
            ]
        )
        return np.concatenate([base, penalties])

    best = None
    for start in (
        np.array([-0.35, 0.70, 0.50]),
        np.array([-0.10, 1.00, 0.25]),
        np.array([-0.60, 0.35, 0.75]),
    ):
        try:
            fit = least_squares(
                residual,
                start,
                bounds=(np.array([-0.999, 1e-5, 0.0]), np.array([0.999, 10.0, 1.0])),
                loss="soft_l1",
                max_nfev=5_000,
            )
        except Exception:
            continue
        if best is None or fit.cost < best.cost:
            best = fit
    if best is None:
        return None
    rho, eta, gamma = (float(value) for value in best.x)
    model = raw_ssvi(k, theta, rho, eta, gamma)
    rmse = float(np.sqrt(np.average((model - w) ** 2, weights=weights)))
    phi = ssvi_phi(theta, eta, gamma)
    valid = bool(
        best.success
        and np.all(model > 0)
        and np.max(theta * phi * (1.0 + abs(rho))) <= 4.05
        and np.max(theta * phi * phi * (1.0 + abs(rho))) <= 4.05
    )
    return SSVIParameters(
        rho=rho,
        eta=eta,
        gamma=gamma,
        weighted_rmse=rmse,
        points_used=len(k),
        success=valid,
    )


def _calendar_violation_rate(
    slices: list[_SliceData],
    svi: dict[date, SVIParameters],
    ssvi: SSVIParameters | None,
) -> float:
    usable = [item for item in slices if item.expiration in svi]
    if len(usable) < 2:
        return 0.0
    grid = np.linspace(-0.35, 0.35, 81)
    curves: list[np.ndarray] = []
    for item in usable:
        theta = max(_atm_theta(svi[item.expiration]), 1e-8)
        if ssvi is not None and ssvi.success:
            curve = raw_ssvi(grid, np.full_like(grid, theta), ssvi.rho, ssvi.eta, ssvi.gamma)
        else:
            p = svi[item.expiration]
            curve = raw_svi(grid, p.a, p.b, p.rho, p.m, p.sigma)
        curves.append(curve)
    checks = 0
    violations = 0
    for earlier, later in zip(curves, curves[1:], strict=False):
        diff = later - earlier
        checks += len(diff)
        violations += int(np.sum(diff < -0.002))
    return violations / checks if checks else 0.0


def _butterfly_violation_rate(
    slices: list[_SliceData],
    svi: dict[date, SVIParameters],
    ssvi: SSVIParameters | None,
) -> float:
    grid = np.linspace(-0.60, 0.60, 241)
    checks = 0
    violations = 0
    for item in slices:
        params = svi.get(item.expiration)
        if params is None:
            continue
        theta = max(_atm_theta(params), 1e-8)
        if ssvi is not None and ssvi.success:
            curve = raw_ssvi(grid, np.full_like(grid, theta), ssvi.rho, ssvi.eta, ssvi.gamma)
        else:
            curve = raw_svi(grid, params.a, params.b, params.rho, params.m, params.sigma)
        second = np.diff(curve, n=2)
        checks += len(second)
        # This is a conservative warning diagnostic, not a full density proof.
        scale = max(float(np.median(np.abs(curve))), 1e-8)
        violations += int(np.sum(second < -0.015 * scale))
    return violations / checks if checks else 0.0


def _surface_score(
    *,
    market_solve_rate: float,
    fitted_ratio: float,
    parity_rmse: float,
    fit_rmse: float,
    calendar_rate: float,
    butterfly_rate: float,
    ssvi_success: bool,
) -> float:
    parity_score = float(np.clip(1.0 - parity_rmse / 0.015, 0.0, 1.0))
    fit_score = float(np.clip(1.0 - fit_rmse / 0.015, 0.0, 1.0))
    score = 100.0 * (
        market_solve_rate * 0.20
        + fitted_ratio * 0.20
        + parity_score * 0.15
        + fit_score * 0.20
        + (1.0 - calendar_rate) * 0.10
        + (1.0 - butterfly_rate) * 0.10
        + (1.0 if ssvi_success else 0.60) * 0.05
    )
    return float(np.clip(score, 0.0, 100.0))


def fit_volatility_surface(
    contracts: pd.DataFrame,
    forward_summaries: list[ForwardSummary],
    quality: DataQualityReport,
    *,
    spot: float,
    quant_engine_mode: QuantEngineMode = "legacy",
) -> tuple[pd.DataFrame, VolatilitySurfaceResult]:
    if contracts.empty:
        empty = VolatilitySurfaceResult(
            mode="raw_iv",
            svi_parameters={},
            ssvi_parameters=None,
            forwards={},
            discount_factors={},
            atm_total_variance={},
            market_solve_rate=0.0,
            fitted_contract_ratio=0.0,
            parity_rmse=0.02,
            butterfly_violation_rate=1.0,
            calendar_violation_rate=1.0,
            weighted_fit_rmse=1.0,
            score=0.0,
            grade="F",
            reasons=("No contracts were available for surface calibration.",),
        )
        return contracts.copy(), empty

    output = contracts.copy()
    forwards, dfs = _forward_map(output, forward_summaries, spot)
    slices = _slice_data(output, forwards, dfs)
    svi: dict[date, SVIParameters] = {}
    for item in slices:
        params = _fit_svi(item)
        if params is not None and params.success:
            svi[item.expiration] = params

    ssvi = _fit_ssvi(slices, svi)
    mode = "ssvi" if ssvi is not None and ssvi.success else "svi" if svi else "raw_iv"
    atm_theta = {expiration: _atm_theta(params) for expiration, params in svi.items()}

    fitted = np.full(len(output), np.nan, dtype=float)
    surface_source = np.full(len(output), "raw_iv", dtype=object)
    for idx, row in output.reset_index(drop=True).iterrows():
        exp = row["expiration"]
        exp = exp if isinstance(exp, date) else pd.Timestamp(exp).date()
        params = svi.get(exp)
        forward = forwards.get(exp)
        t = max(float(row["time_years"]), 1e-8)
        strike = float(row["strike"])
        if params is None or forward is None or strike <= 0 or forward <= 0:
            continue
        k = float(np.log(strike / forward))
        theta = max(atm_theta[exp], 1e-8)
        if ssvi is not None and ssvi.success:
            w = float(raw_ssvi(np.asarray([k]), np.asarray([theta]), ssvi.rho, ssvi.eta, ssvi.gamma)[0])
            source = "ssvi"
        else:
            w = float(raw_svi(np.asarray([k]), params.a, params.b, params.rho, params.m, params.sigma)[0])
            source = "svi"
        if np.isfinite(w) and w > 0:
            fitted[idx] = float(np.clip(np.sqrt(w / t), 0.01, 5.0))
            surface_source[idx] = source

    output = output.reset_index(drop=True)
    valid_fit = np.isfinite(fitted)
    output["raw_iv"] = output["iv"].astype(float)
    output["surface_iv"] = np.where(valid_fit, fitted, output["raw_iv"])
    output["surface_source"] = surface_source
    legacy_input = pd.to_numeric(
        output.get("legacy_input_iv", output["raw_iv"]), errors="coerce"
    ).fillna(output["raw_iv"])
    candidate_input = pd.to_numeric(
        output.get("candidate_input_iv", output["raw_iv"]), errors="coerce"
    ).fillna(output["raw_iv"])
    output["legacy_iv"] = np.where(valid_fit, fitted, legacy_input)

    candidate_source = output.get(
        "candidate_iv_source", output.get("iv_source", "missing")
    )
    candidate_source = pd.Series(candidate_source, index=output.index).astype(str)
    candidate_quality_source = output.get(
        "candidate_quote_quality", output.get("quote_quality", 0.0)
    )
    candidate_quality = pd.to_numeric(
        pd.Series(candidate_quality_source, index=output.index),
        errors="coerce",
    ).fillna(0.0).clip(0.0, 1.0)
    source_weight = candidate_source.map(
        {
            "market_solved": 0.65,
            "vendor": 0.30,
            "interpolated": 0.10,
            "flat_fallback": 0.0,
            "missing": 0.0,
        }
    ).fillna(0.0)
    raw_weight = np.clip(source_weight + 0.25 * candidate_quality, 0.0, 0.90)
    candidate_surface = np.where(valid_fit, fitted, candidate_input)
    output["iv_blend_weight"] = np.where(valid_fit, raw_weight, 1.0)
    output["candidate_iv"] = np.clip(
        output["iv_blend_weight"] * candidate_input
        + (1.0 - output["iv_blend_weight"]) * candidate_surface,
        0.01,
        5.0,
    )
    output["iv"] = (
        output["candidate_iv"]
        if quant_engine_mode == "candidate"
        else output["legacy_iv"]
    )
    output["is_surface_fitted"] = valid_fit
    quality.iv_surface_fitted = int(np.sum(valid_fit))

    used = max(quality.contracts_used, 1)
    market_rate = min(quality.iv_market_solved / used, 1.0)
    fitted_ratio = float(np.mean(valid_fit)) if len(valid_fit) else 0.0
    reliable = [item for item in forward_summaries if item.reliable]
    parity_rmse = (
        float(np.sqrt(np.mean([(item.dispersion_bps / 10_000.0) ** 2 for item in reliable])))
        if reliable
        else 0.02
    )
    fit_values = [params.weighted_rmse for params in svi.values()]
    if ssvi is not None and ssvi.success:
        fit_values.append(ssvi.weighted_rmse)
    fit_rmse = float(np.median(fit_values)) if fit_values else 1.0
    calendar_rate = _calendar_violation_rate(slices, svi, ssvi)
    butterfly_rate = _butterfly_violation_rate(slices, svi, ssvi)
    score = _surface_score(
        market_solve_rate=market_rate,
        fitted_ratio=fitted_ratio,
        parity_rmse=parity_rmse,
        fit_rmse=fit_rmse,
        calendar_rate=calendar_rate,
        butterfly_rate=butterfly_rate,
        ssvi_success=bool(ssvi and ssvi.success),
    )

    reasons: list[str] = []
    if not svi:
        reasons.append("Too few stable volatility slices were available for SVI calibration.")
    elif ssvi is None or not ssvi.success:
        reasons.append("SSVI term-structure calibration was not stable; valid SVI slices were retained.")
    else:
        reasons.append("SSVI term structure calibrated within the current sufficient-condition checks.")
    if fitted_ratio < 0.60:
        reasons.append("Less than 60% of contracts received a fitted surface volatility.")
    if calendar_rate > 0.05:
        reasons.append("Calendar total-variance warnings remain after calibration.")
    if butterfly_rate > 0.05:
        reasons.append("Local convexity warnings remain in parts of the fitted surface.")
    if parity_rmse > 0.01:
        reasons.append("Put-call parity estimates show elevated dispersion.")

    result = VolatilitySurfaceResult(
        mode=mode,
        svi_parameters=svi,
        ssvi_parameters=ssvi,
        forwards=forwards,
        discount_factors=dfs,
        atm_total_variance=atm_theta,
        market_solve_rate=market_rate,
        fitted_contract_ratio=fitted_ratio,
        parity_rmse=parity_rmse,
        butterfly_violation_rate=butterfly_rate,
        calendar_violation_rate=calendar_rate,
        weighted_fit_rmse=fit_rmse,
        score=round(score, 1),
        grade=_grade(score),
        reasons=tuple(reasons),
    )
    return output, result


def evaluate_surface_iv(
    surface: VolatilitySurfaceResult,
    expiration: date,
    strike: np.ndarray,
    time_years: np.ndarray | float,
    forward: np.ndarray | float,
) -> np.ndarray:
    strikes = np.asarray(strike, dtype=float)
    times = np.asarray(time_years, dtype=float)
    forwards = np.asarray(forward, dtype=float)
    params = surface.svi_parameters.get(expiration)
    if params is None:
        return np.full_like(strikes, np.nan, dtype=float)
    k = np.log(np.maximum(strikes, 1e-12) / np.maximum(forwards, 1e-12))
    theta = max(surface.atm_total_variance.get(expiration, _atm_theta(params)), 1e-8)
    if surface.ssvi_parameters is not None and surface.ssvi_parameters.success:
        ssvi = surface.ssvi_parameters
        w = raw_ssvi(k, np.full_like(k, theta), ssvi.rho, ssvi.eta, ssvi.gamma)
    else:
        w = raw_svi(k, params.a, params.b, params.rho, params.m, params.sigma)
    return np.sqrt(np.maximum(w, 1e-12) / np.maximum(times, 1e-12))
