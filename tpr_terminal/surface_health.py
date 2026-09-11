from __future__ import annotations

import numpy as np
import pandas as pd

from .models import (
    DataQualityReport,
    ForwardSummary,
    SurfaceHealthReport,
    VolatilitySurfaceResult,
)


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


def _legacy_report(
    contracts: pd.DataFrame,
    quality: DataQualityReport,
    forward_summaries: list[ForwardSummary] | None,
) -> SurfaceHealthReport:
    used = max(quality.contracts_used, 1)
    market_rate = min(quality.iv_market_solved / used, 1.0)
    vendor_rate = min(quality.iv_vendor_used / used, 1.0)
    repaired_rate = min(
        (quality.iv_values_interpolated + quality.iv_flat_fallback) / used, 1.0
    )
    reliable = [item for item in (forward_summaries or []) if item.reliable]
    parity_rmse = (
        float(np.sqrt(np.mean([(item.dispersion_bps / 10_000.0) ** 2 for item in reliable])))
        if reliable
        else 0.02
    )
    fitted_ratio = market_rate + vendor_rate
    score = 100.0 * (
        market_rate * 0.35
        + min(fitted_ratio, 1.0) * 0.20
        + (1.0 - repaired_rate) * 0.25
        + float(np.clip(1.0 - parity_rmse / 0.015, 0.0, 1.0)) * 0.20
    )
    reasons = (
        "Legacy compatibility surface diagnostic.",
        "A material share of IV inputs required repair." if repaired_rate > 0.20 else "Input IV quality passed the compatibility threshold.",
    )
    return SurfaceHealthReport(
        score=round(float(np.clip(score, 0.0, 100.0)), 1),
        grade=_grade(score),
        mode="raw_iv",
        market_iv_solve_rate=market_rate,
        fitted_contract_ratio=min(fitted_ratio, 1.0),
        vendor_iv_fallback_rate=vendor_rate,
        repaired_iv_rate=repaired_rate,
        parity_rmse=parity_rmse,
        weighted_fit_rmse=0.0,
        smile_roughness=0.0,
        calendar_violation_rate=0.0,
        butterfly_warning_rate=0.0,
        reasons=reasons,
    )


def build_surface_health_report(
    surface: VolatilitySurfaceResult | pd.DataFrame,
    quality: DataQualityReport,
    forward_summaries: list[ForwardSummary] | None = None,
    *,
    spot: float | None = None,
) -> SurfaceHealthReport:
    # Backward-compatible signature for Alpha.2 tests and external integrations.
    if isinstance(surface, pd.DataFrame):
        return _legacy_report(surface, quality, forward_summaries)

    used = max(quality.contracts_used, 1)
    vendor_rate = min(quality.iv_vendor_used / used, 1.0)
    repaired_rate = min(
        (quality.iv_values_interpolated + quality.iv_flat_fallback) / used,
        1.0,
    )
    reasons = list(surface.reasons)
    if repaired_rate > 0.20:
        reasons.append("A material share of pre-surface IV inputs required repair.")
    if surface.fitted_contract_ratio < 0.60:
        reasons.append("Surface coverage is below the preferred 60% threshold.")
    if not reasons:
        reasons.append("Surface diagnostics passed the current Alpha 4.0 thresholds.")

    score = float(np.clip(surface.score - repaired_rate * 8.0, 0.0, 100.0))
    return SurfaceHealthReport(
        score=round(score, 1),
        grade=_grade(score),
        mode=surface.mode,
        market_iv_solve_rate=surface.market_solve_rate,
        fitted_contract_ratio=surface.fitted_contract_ratio,
        vendor_iv_fallback_rate=vendor_rate,
        repaired_iv_rate=repaired_rate,
        parity_rmse=surface.parity_rmse,
        weighted_fit_rmse=surface.weighted_fit_rmse,
        smile_roughness=surface.weighted_fit_rmse,
        calendar_violation_rate=surface.calendar_violation_rate,
        butterfly_warning_rate=surface.butterfly_violation_rate,
        reasons=tuple(dict.fromkeys(reasons)),
    )
