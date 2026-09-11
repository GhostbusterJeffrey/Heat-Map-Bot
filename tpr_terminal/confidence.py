from __future__ import annotations

import numpy as np

from .models import (
    ConfidenceReport,
    DataQualityReport,
    GammaFlip,
    LiquidityReport,
    SurfaceHealthReport,
    WallLevel,
)


def build_liquidity_report(quality: DataQualityReport, mode: str) -> LiquidityReport:
    valid = max(quality.valid_contracts, 1)
    used = max(quality.contracts_used, 1)
    total_abs = max(quality.total_abs_exposure, 1e-9)
    return LiquidityReport(
        mode=mode,
        oi_contracts=quality.reported_oi_contracts,
        volume_fallback_contracts=quality.volume_fallback_contracts,
        fallback_ratio=min(quality.volume_fallback_contracts / used, 1.0),
        usable_ratio=min(quality.contracts_used / max(quality.contracts_seen, 1), 1.0),
        oi_coverage=min(quality.reported_oi_contracts / valid, 1.0),
        volume_coverage=min(quality.contracts_with_volume / valid, 1.0),
        oi_exposure_share=min(quality.oi_abs_exposure / total_abs, 1.0),
        volume_exposure_share=min(quality.volume_abs_exposure / total_abs, 1.0),
    )


def _grade(score: float) -> tuple[str, str]:
    if score >= 90:
        return "A", "High"
    if score >= 80:
        return "B", "Good"
    if score >= 70:
        return "C", "Moderate"
    if score >= 60:
        return "D", "Caution"
    if score >= 50:
        return "E", "Low"
    return "F", "Very low"


def build_confidence_report(
    quality: DataQualityReport,
    surface: SurfaceHealthReport | None = None,
    liquidity: LiquidityReport | None = None,
    positive_wall: WallLevel | None = None,
    negative_wall: WallLevel | None = None,
    primary_flip: GammaFlip | None = None,
) -> ConfidenceReport:
    requested = max(quality.expirations_requested, 1)
    processed = max(quality.expirations_processed, 0)
    seen = max(quality.contracts_seen, 1)
    used = max(quality.contracts_used, 0)

    expiry_coverage = min(processed / requested, 1.0)
    contract_coverage = min(used / seen, 1.0)
    quote_coverage = min(quality.valid_market_quotes / seen, 1.0)
    reported_oi_share = min(quality.reported_oi_contracts / max(used, 1), 1.0)
    non_interpolated = quality.iv_market_solved + quality.iv_vendor_used
    non_interpolated_iv_share = min(non_interpolated / max(used, 1), 1.0)
    parity_coverage = min(quality.parity_expirations / max(processed, 1), 1.0)

    surface_score = (surface.score / 100.0) if surface is not None else non_interpolated_iv_share
    liquidity_coverage = (
        float(np.clip(0.55 * liquidity.usable_ratio + 0.45 * liquidity.oi_exposure_share, 0.0, 1.0))
        if liquidity is not None
        else reported_oi_share
    )
    wall_values = [item for item in (positive_wall, negative_wall) if item is not None]
    wall_stability = (
        float(
            np.mean(
                [
                    0.35 * item.expiry_breadth
                    + 0.30 * item.contract_quality
                    + 0.35 * item.shock_stability
                    for item in wall_values
                ]
            )
        )
        if wall_values
        else 0.45
    )
    flip_stability = primary_flip.shock_stability if primary_flip is not None else 0.45
    contract_quality = 1.0 - min(
        (quality.iv_values_interpolated + quality.iv_flat_fallback) / max(used, 1),
        1.0,
    )

    score = 100.0 * (
        quote_coverage * 0.15
        + liquidity_coverage * 0.15
        + surface_score * 0.20
        + contract_quality * 0.15
        + expiry_coverage * 0.10
        + parity_coverage * 0.10
        + wall_stability * 0.075
        + flip_stability * 0.075
    )

    repaired_rate = min(
        (quality.iv_values_interpolated + quality.iv_flat_fallback) / max(used, 1),
        1.0,
    )
    volume_share = liquidity.volume_exposure_share if liquidity is not None else 0.0
    calendar_rate = surface.calendar_violation_rate if surface is not None else 0.0
    score -= repaired_rate * 12.0
    score -= volume_share * 10.0
    score -= calendar_rate * 10.0
    if contract_coverage < 0.05:
        score -= 30.0
    elif contract_coverage < 0.10:
        score -= 18.0
    if used < 50:
        score -= 12.0
    score = float(np.clip(score, 0.0, 100.0))

    reasons: list[str] = []
    if expiry_coverage < 0.8:
        reasons.append("Not all requested expirations were processed.")
    if contract_coverage < 0.10:
        reasons.append("Usable contract coverage is thin.")
    if quote_coverage < 0.35:
        reasons.append("Many contracts lack a valid two-sided market.")
    if reported_oi_share < 0.50:
        reasons.append("A material share of usable contracts lacks reported open interest.")
    if repaired_rate > 0.20:
        reasons.append("A material share of IV inputs required repair.")
    if volume_share > 0.25:
        reasons.append("Volume fallback contributes materially to estimated exposure.")
    if surface is not None and surface.score < 70:
        reasons.append("Volatility-surface diagnostics are below the preferred threshold.")
    if primary_flip is not None and primary_flip.shock_stability < 0.55:
        reasons.append("The primary gamma flip is sensitive to small IV/time shocks.")
    if not reasons:
        reasons.append("Coverage, surface health, and level stability passed Alpha 4.0 thresholds.")

    grade, label = _grade(score)
    return ConfidenceReport(
        score=round(score, 1),
        grade=grade,
        label=label,
        reasons=tuple(reasons),
        expiry_coverage=expiry_coverage,
        contract_coverage=contract_coverage,
        quote_coverage=quote_coverage,
        reported_oi_share=reported_oi_share,
        non_interpolated_iv_share=non_interpolated_iv_share,
        parity_coverage=parity_coverage,
    )
