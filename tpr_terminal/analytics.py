from __future__ import annotations

import logging
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .canonical_state import build_canonical_contract_ledger
from .confidence import build_confidence_report, build_liquidity_report
from .forward_engine import estimate_forward_from_put_call_parity
from .market_calendar import USMarketCalendar
from .market_data import MarketDataProvider
from .math_engine import resolve_implied_volatility
from .models import (
    AnalysisResult,
    DataQualityReport,
    ExpectedMove,
    ExposureMode,
    ForwardSummary,
    GreekExposureSummary,
    LevelSummary,
    LiquidityMode,
    OptionSide,
    RateCurve,
    WallLevel,
)
from .pricing import black_scholes_gamma, greek_stack
from .quant_accuracy import QuantEngineMode, gamma_exposure_1pct
from .scenario import build_gamma_flips, build_pin_levels
from .surface_health import build_surface_health_report
from .volatility_surface import fit_volatility_surface

logger = logging.getLogger(__name__)


class ExposureAnalyticsEngine:
    """TPR V4 Alpha 4.0 surface-aware public-data options risk engine.

    Alpha 4.0 calibrates SVI slices and attempts an SSVI term structure, reprices
    contract Greeks from the fitted surface, evaluates gamma flips under both
    sticky-strike and sticky-delta assumptions, and retains explicit confidence
    and data-quality disclosures.

    The net view remains a call-positive / put-negative public-data proxy. Open
    interest does not reveal actual dealer ownership or off-exchange inventory.
    """

    def __init__(
        self,
        provider: MarketDataProvider,
        *,
        contract_multiplier: int = 100,
        calendar_overrides: Path | None = None,
        volume_liquidity_weight: float = 0.50,
        max_wall_distance_pct: float = 0.12,
        min_wall_quality: float = 0.40,
        min_wall_percentile: float = 0.70,
        min_expiry_share_for_breadth: float = 0.025,
        min_flip_confidence: float = 0.45,
        flip_grid_low_pct: float = 0.85,
        flip_grid_high_pct: float = 1.15,
        flip_grid_points: int = 601,
        pin_decay_days: float = 7.0,
        pin_shock_pct: float = 0.0025,
        min_pin_score: float = 35.0,
        min_contracts_to_render: int = 50,
        min_usable_ratio: float = 0.05,
        min_confidence_to_render: float = 35.0,
        quant_engine_mode: QuantEngineMode = "legacy",
    ) -> None:
        self._provider = provider
        self._multiplier = int(contract_multiplier)
        self._calendar = USMarketCalendar(calendar_overrides)
        self._volume_weight = float(np.clip(volume_liquidity_weight, 0.0, 1.0))
        self._max_wall_distance_pct = float(max_wall_distance_pct)
        self._min_wall_quality = float(min_wall_quality)
        self._min_wall_percentile = float(min_wall_percentile)
        self._min_expiry_share = float(np.clip(min_expiry_share_for_breadth, 0.0, 1.0))
        self._min_flip_confidence = float(min_flip_confidence)
        self._flip_grid_low_pct = float(flip_grid_low_pct)
        self._flip_grid_high_pct = float(flip_grid_high_pct)
        self._flip_grid_points = int(flip_grid_points)
        self._pin_decay_days = float(pin_decay_days)
        self._pin_shock_pct = float(pin_shock_pct)
        self._min_pin_score = float(min_pin_score)
        self._min_contracts_to_render = int(min_contracts_to_render)
        self._min_usable_ratio = float(min_usable_ratio)
        self._min_confidence_to_render = float(min_confidence_to_render)
        if quant_engine_mode not in {"legacy", "shadow", "candidate"}:
            raise ValueError(f"Unsupported quant engine mode: {quant_engine_mode}")
        self._quant_engine_mode: QuantEngineMode = quant_engine_mode

    def analyze(
        self,
        ticker: str,
        *,
        max_expiries: int,
        liquidity_mode: LiquidityMode,
        exposure_mode: ExposureMode,
        rate_curve: RateCurve,
        now: datetime | None = None,
    ) -> AnalysisResult:
        session = self._provider.open_session(ticker)
        underlying = session.underlying()
        expirations = session.expirations()
        if not expirations:
            raise ValueError(f"No listed option expirations were returned for {ticker}.")

        current = now or datetime.now(UTC)
        current = current.replace(tzinfo=UTC) if current.tzinfo is None else current.astimezone(UTC)

        quality = DataQualityReport(expirations_requested=int(max_expiries))
        contracts: list[pd.DataFrame] = []
        expected_moves: list[ExpectedMove] = []
        forward_summaries: list[ForwardSummary] = []
        selected_modes: list[str] = []

        for expiration in expirations:
            if quality.expirations_processed >= max_expiries:
                break
            try:
                close = self._calendar.session_close(expiration)
                seconds = (close.close_utc - current).total_seconds()
                if seconds <= 0:
                    continue
                days = seconds / 86_400.0
                time_years = max(days / 365.25, 1.0 / (365.25 * 24 * 60))
                rate = rate_curve.rate_for_days(days)
                chain = session.option_chain(expiration)
                expiry_mode = self._select_liquidity_mode(
                    chain.calls,
                    chain.puts,
                    requested=liquidity_mode,
                )
                selected_modes.append(expiry_mode)

                parity = estimate_forward_from_put_call_parity(
                    chain.calls,
                    chain.puts,
                    spot=underlying.spot,
                    time_years=time_years,
                    risk_free_rate=rate,
                )
                carry_yield = underlying.trailing_dividend_yield
                if parity is not None:
                    forward_summaries.append(
                        ForwardSummary(
                            expiration=expiration,
                            forward=parity.forward,
                            dividend_yield=parity.dividend_yield,
                            pairs_used=parity.pairs_used,
                            dispersion_bps=parity.dispersion_bps,
                            reliable=parity.reliable,
                            method=parity.method,
                        )
                    )
                    quality.parity_pairs_used += parity.pairs_used
                    if parity.reliable:
                        carry_yield = parity.dividend_yield
                        quality.parity_expirations += 1

                call_rows = self._prepare_side(
                    chain.calls,
                    side="call",
                    expiration=expiration,
                    spot=underlying.spot,
                    time_years=time_years,
                    days_to_expiry=days,
                    risk_free_rate=rate,
                    dividend_yield=carry_yield,
                    liquidity_mode=expiry_mode,
                    exposure_mode=exposure_mode,
                    quality=quality,
                )
                put_rows = self._prepare_side(
                    chain.puts,
                    side="put",
                    expiration=expiration,
                    spot=underlying.spot,
                    time_years=time_years,
                    days_to_expiry=days,
                    risk_free_rate=rate,
                    dividend_yield=carry_yield,
                    liquidity_mode=expiry_mode,
                    exposure_mode=exposure_mode,
                    quality=quality,
                )
                combined = pd.concat([call_rows, put_rows], ignore_index=True)
                if not combined.empty:
                    contracts.append(combined)

                move = self._expected_move(
                    call_rows,
                    put_rows,
                    expiration=expiration,
                    days_to_expiry=days,
                    spot=underlying.spot,
                    time_years=time_years,
                    horizon_utc=close.close_utc,
                )
                if move is not None:
                    expected_moves.append(move)
                quality.expirations_processed += 1
            except Exception as exc:
                detail = f"{expiration.isoformat()}: {type(exc).__name__}: {exc}"
                quality.expirations_skipped.append(detail)
                logger.exception("Failed to process %s %s", ticker, expiration)

        if not contracts:
            detail = "; ".join(quality.expirations_skipped[:3])
            raise ValueError(
                "No usable option expirations were produced."
                + (f" First failures: {detail}" if detail else "")
            )

        contract_records = pd.concat(contracts, ignore_index=True).reset_index(drop=True)
        if self._quant_engine_mode == "legacy":
            contract_records, surface = fit_volatility_surface(
                contract_records,
                forward_summaries,
                quality,
                spot=underlying.spot,
                quant_engine_mode="legacy",
            )
        else:
            legacy_input = contract_records.copy()
            legacy_input["iv"] = legacy_input["legacy_input_iv"]
            legacy_input["iv_source"] = legacy_input["legacy_iv_source"]
            legacy_input["quote_quality"] = legacy_input["legacy_quote_quality"]
            legacy_input["contract_quality"] = legacy_input["legacy_contract_quality"]
            legacy_records, legacy_surface = fit_volatility_surface(
                legacy_input,
                forward_summaries,
                quality,
                spot=underlying.spot,
                quant_engine_mode="legacy",
            )

            candidate_input = contract_records.copy()
            candidate_input["iv"] = candidate_input["candidate_input_iv"]
            candidate_input["iv_source"] = candidate_input["candidate_iv_source"]
            candidate_input["quote_quality"] = candidate_input["candidate_quote_quality"]
            candidate_input["contract_quality"] = candidate_input["candidate_contract_quality"]
            candidate_records, candidate_surface = fit_volatility_surface(
                candidate_input,
                forward_summaries,
                quality,
                spot=underlying.spot,
                quant_engine_mode="candidate",
            )

            if self._quant_engine_mode == "candidate":
                contract_records = candidate_records
                contract_records["legacy_iv"] = legacy_records["legacy_iv"]
                contract_records["legacy_surface_iv"] = legacy_records["surface_iv"]
                contract_records["legacy_surface_source"] = legacy_records["surface_source"]
                surface = candidate_surface
            else:
                contract_records = legacy_records
                contract_records["candidate_iv"] = candidate_records["candidate_iv"]
                contract_records["candidate_surface_iv"] = candidate_records["surface_iv"]
                contract_records["candidate_surface_source"] = candidate_records["surface_source"]
                contract_records["candidate_iv_blend_weight"] = candidate_records["iv_blend_weight"]
                surface = legacy_surface
            quality.iv_surface_fitted = int(contract_records["is_surface_fitted"].sum())
        contract_records = self._reprice_surface_contracts(
            contract_records,
            spot=underlying.spot,
        )
        quality.total_abs_exposure = float(contract_records["abs_gex"].sum())
        quality.oi_abs_exposure = float(
            contract_records.loc[
                contract_records["liquidity_source"] == "open_interest", "abs_gex"
            ].sum()
        )
        quality.volume_abs_exposure = float(
            contract_records.loc[
                contract_records["liquidity_source"] == "volume_fallback", "abs_gex"
            ].sum()
        )
        matrix_records = self._build_matrix_records(contract_records)
        if matrix_records.empty:
            raise ValueError("Surface repricing produced an empty matrix.")

        strike_totals, raw_positive, raw_negative, positive_wall, negative_wall = (
            self._build_structural_walls(matrix_records, underlying.spot)
        )
        positive_wall = self._apply_wall_shock_stability(
            positive_wall,
            contract_records,
            spot=underlying.spot,
        )
        negative_wall = self._apply_wall_shock_stability(
            negative_wall,
            contract_records,
            spot=underlying.spot,
        )

        profile, flips = build_gamma_flips(
            contract_records,
            spot=underlying.spot,
            multiplier=self._multiplier,
            surface=surface,
            min_confidence=self._min_flip_confidence,
            grid_low_pct=self._flip_grid_low_pct,
            grid_high_pct=self._flip_grid_high_pct,
            grid_points=self._flip_grid_points,
        )
        primary_flip = flips[0] if flips else None
        secondary_flips = flips[1:] if len(flips) > 1 else ()

        expected_sorted = sorted(expected_moves, key=lambda value: value.expiration)
        nearest_move = expected_sorted[0].dollars if expected_sorted else None
        pins = build_pin_levels(
            contract_records,
            strike_totals,
            spot=underlying.spot,
            expected_move=nearest_move,
            multiplier=self._multiplier,
            surface=surface,
            pin_decay_days=self._pin_decay_days,
            pin_shock_pct=self._pin_shock_pct,
            min_score=self._min_pin_score,
        )
        primary_pin = pins[0] if pins else None
        secondary_pins = pins[1:4] if len(pins) > 1 else ()

        effective_mode = self._overall_mode(selected_modes, quality)
        liquidity = build_liquidity_report(quality, effective_mode)
        surface_health = build_surface_health_report(surface, quality)
        confidence = build_confidence_report(
            quality,
            surface_health,
            liquidity,
            positive_wall,
            negative_wall,
            primary_flip,
        )

        usable_ratio = quality.contracts_used / max(quality.contracts_seen, 1)
        if (
            quality.contracts_used < self._min_contracts_to_render
            or usable_ratio < self._min_usable_ratio
        ):
            raise ValueError(
                "Available options data is too incomplete for a reliable matrix. "
                "Try a more liquid ticker or rerun during market hours."
            )
        if confidence.score < self._min_confidence_to_render:
            raise ValueError(
                f"Model/input confidence is only {confidence.score:.0f}/100, below the "
                f"{self._min_confidence_to_render:.0f} render threshold."
            )

        total = float(strike_totals["gex"].sum())
        total_abs = float(strike_totals["abs_gex"].sum())
        neutral_threshold = max(1.0, total_abs * 0.02)
        if abs(total) <= neutral_threshold:
            regime = "neutral gamma proxy"
        else:
            regime = "positive gamma proxy" if total > 0 else "negative gamma proxy"
        if confidence.score < 65:
            regime = "limited-confidence gamma proxy"
            primary_pin = None
            secondary_pins = ()
        if confidence.score < 50:
            positive_wall = None
            negative_wall = None
            primary_flip = None
            secondary_flips = ()

        levels = LevelSummary(
            positive_wall=positive_wall,
            negative_wall=negative_wall,
            raw_positive_peak=raw_positive,
            raw_negative_peak=raw_negative,
            primary_gamma_flip=primary_flip,
            secondary_gamma_flips=secondary_flips,
            primary_pin=primary_pin,
            secondary_pins=secondary_pins,
            total_net_gex=total,
            total_absolute_gex=total_abs,
            regime=regime,
        )
        greek_exposure = self._greek_exposure_summary(contract_records, underlying.spot)
        interpretation = self._interpretation(
            ticker=underlying.ticker,
            spot=underlying.spot,
            levels=levels,
            confidence_score=confidence.score,
            surface_grade=surface_health.grade,
            surface_mode=surface.mode,
        )

        if rate_curve.used_fallback:
            quality.warnings.append("Risk-free curve used the configured static fallback rate.")
        if quality.volume_fallback_contracts:
            quality.warnings.append(
                "Volume substituted for missing/zero OI on some contracts; volume is not position inventory."
            )
        if exposure_mode == "net_proxy":
            quality.warnings.append(
                "Net GEX applies call-positive / put-negative signs; it is not observed dealer inventory."
            )
        quality.warnings.extend(
            [
                f"Assumes a standard {self._multiplier}-share deliverable; adjusted contracts need vendor metadata.",
                "Alpha 4.0 uses fitted SVI/SSVI volatility where calibration passes; American exercise is currently a validation layer, not the primary bulk Greek engine.",
                f"Quant engine mode: {self._quant_engine_mode} (legacy remains the comparison baseline).",
                session.provider_note,
            ]
        )

        result = AnalysisResult(
            ticker=underlying.ticker,
            spot=underlying.spot,
            as_of=underlying.as_of,
            snapshot_id=(
                f"{underlying.ticker}-{current.strftime('%Y%m%dT%H%M%S.%fZ')}"
            ),
            trailing_dividend_yield=underlying.trailing_dividend_yield,
            exposure_mode=exposure_mode,
            requested_liquidity_mode=liquidity_mode,
            effective_liquidity_mode=effective_mode,
            provider_name=session.provider_name,
            provider_note=session.provider_note,
            rate_curve_source=rate_curve.source,
            rate_curve_as_of=rate_curve.as_of,
            matrix_records=matrix_records,
            contract_records=contract_records,
            strike_totals=strike_totals,
            profile=profile,
            expected_moves=expected_sorted,
            forward_summaries=sorted(forward_summaries, key=lambda value: value.expiration),
            levels=levels,
            quality=quality,
            liquidity=liquidity,
            surface=surface,
            surface_health=surface_health,
            confidence=confidence,
            greek_exposure=greek_exposure,
            interpretation=interpretation,
            quant_engine_mode=self._quant_engine_mode,
        )
        result.contract_ledger = build_canonical_contract_ledger(
            result,
            valuation_time=current,
            contract_multiplier=self._multiplier,
        )
        return result

    def slice_result(
        self,
        canonical: AnalysisResult,
        *,
        max_expiries: int,
    ) -> AnalysisResult:
        """Build an expiry-limited view without refitting or repricing contracts.

        A canonical result owns the quote snapshot, carry estimates, calibrated
        volatility surface, and per-contract Greeks. Presets are views over that
        immutable calculation. This guarantees that a shared expiration/strike
        cell has exactly the same GEX in intraday, swing, macro, and custom maps.
        """
        requested = max(int(max_expiries), 1)
        available = sorted(
            {
                value if isinstance(value, date) else pd.Timestamp(value).date()
                for value in canonical.contract_records["expiration"]
            }
        )
        selected = tuple(available[:requested])
        if not selected:
            raise ValueError("The canonical snapshot contains no usable expirations.")

        contracts = canonical.contract_records.loc[
            canonical.contract_records["expiration"].isin(selected)
        ].copy().reset_index(drop=True)
        if contracts.empty:
            raise ValueError("The requested canonical-snapshot view is empty.")

        matrix_records = self._build_matrix_records(contracts)
        strike_totals, raw_positive, raw_negative, positive_wall, negative_wall = (
            self._build_structural_walls(matrix_records, canonical.spot)
        )
        positive_wall = self._apply_wall_shock_stability(
            positive_wall,
            contracts,
            spot=canonical.spot,
        )
        negative_wall = self._apply_wall_shock_stability(
            negative_wall,
            contracts,
            spot=canonical.spot,
        )

        profile, flips = build_gamma_flips(
            contracts,
            spot=canonical.spot,
            multiplier=self._multiplier,
            surface=canonical.surface,
            min_confidence=self._min_flip_confidence,
            grid_low_pct=self._flip_grid_low_pct,
            grid_high_pct=self._flip_grid_high_pct,
            grid_points=self._flip_grid_points,
        )
        primary_flip = flips[0] if flips else None
        secondary_flips = flips[1:] if len(flips) > 1 else ()

        expected_moves = sorted(
            [item for item in canonical.expected_moves if item.expiration in selected],
            key=lambda item: item.expiration,
        )
        forward_summaries = sorted(
            [item for item in canonical.forward_summaries if item.expiration in selected],
            key=lambda item: item.expiration,
        )
        nearest_move = expected_moves[0].dollars if expected_moves else None
        pins = build_pin_levels(
            contracts,
            strike_totals,
            spot=canonical.spot,
            expected_move=nearest_move,
            multiplier=self._multiplier,
            surface=canonical.surface,
            pin_decay_days=self._pin_decay_days,
            pin_shock_pct=self._pin_shock_pct,
            min_score=self._min_pin_score,
        )
        primary_pin = pins[0] if pins else None
        secondary_pins = pins[1:4] if len(pins) > 1 else ()

        quality = self._quality_for_view(
            contracts,
            requested_expiries=requested,
            processed_expiries=len(selected),
            forward_summaries=forward_summaries,
            canonical=canonical,
        )
        selected_modes = self._liquidity_modes_for_view(contracts)
        effective_mode = self._overall_mode(selected_modes, quality)
        liquidity = build_liquidity_report(quality, effective_mode)
        surface_health = build_surface_health_report(canonical.surface, quality)
        confidence = build_confidence_report(
            quality,
            surface_health,
            liquidity,
            positive_wall,
            negative_wall,
            primary_flip,
        )

        total = float(strike_totals["gex"].sum())
        total_abs = float(strike_totals["abs_gex"].sum())
        neutral_threshold = max(1.0, total_abs * 0.02)
        if abs(total) <= neutral_threshold:
            regime = "neutral gamma proxy"
        else:
            regime = "positive gamma proxy" if total > 0 else "negative gamma proxy"
        if confidence.score < 65:
            regime = "limited-confidence gamma proxy"
            primary_pin = None
            secondary_pins = ()
        if confidence.score < 50:
            positive_wall = None
            negative_wall = None
            primary_flip = None
            secondary_flips = ()

        levels = LevelSummary(
            positive_wall=positive_wall,
            negative_wall=negative_wall,
            raw_positive_peak=raw_positive,
            raw_negative_peak=raw_negative,
            primary_gamma_flip=primary_flip,
            secondary_gamma_flips=secondary_flips,
            primary_pin=primary_pin,
            secondary_pins=secondary_pins,
            total_net_gex=total,
            total_absolute_gex=total_abs,
            regime=regime,
        )
        interpretation = self._interpretation(
            ticker=canonical.ticker,
            spot=canonical.spot,
            levels=levels,
            confidence_score=confidence.score,
            surface_grade=surface_health.grade,
            surface_mode=canonical.surface.mode,
        )

        return AnalysisResult(
            ticker=canonical.ticker,
            spot=canonical.spot,
            as_of=canonical.as_of,
            snapshot_id=canonical.snapshot_id,
            trailing_dividend_yield=canonical.trailing_dividend_yield,
            exposure_mode=canonical.exposure_mode,
            requested_liquidity_mode=canonical.requested_liquidity_mode,
            effective_liquidity_mode=effective_mode,
            provider_name=canonical.provider_name,
            provider_note=canonical.provider_note,
            rate_curve_source=canonical.rate_curve_source,
            rate_curve_as_of=canonical.rate_curve_as_of,
            matrix_records=matrix_records,
            contract_records=contracts,
            strike_totals=strike_totals,
            profile=profile,
            expected_moves=expected_moves,
            forward_summaries=forward_summaries,
            levels=levels,
            quality=quality,
            liquidity=liquidity,
            surface=canonical.surface,
            surface_health=surface_health,
            confidence=confidence,
            greek_exposure=self._greek_exposure_summary(contracts, canonical.spot),
            interpretation=interpretation,
            quant_engine_mode=canonical.quant_engine_mode,
            contract_ledger=canonical.contract_ledger,
        )

    @staticmethod
    def _liquidity_modes_for_view(contracts: pd.DataFrame) -> list[str]:
        modes: list[str] = []
        for _, frame in contracts.groupby("expiration", sort=True):
            used = frame.loc[frame["liquidity"] > 0]
            if used.empty:
                modes.append("open_interest")
                continue
            volume_share = float(
                (used["liquidity_source"] == "volume_fallback").mean()
            )
            if volume_share >= 0.999:
                modes.append("volume_dominant")
            elif volume_share > 0:
                modes.append("oi_volume_fallback")
            else:
                modes.append("open_interest")
        return modes

    @staticmethod
    def _quality_for_view(
        contracts: pd.DataFrame,
        *,
        requested_expiries: int,
        processed_expiries: int,
        forward_summaries: list[ForwardSummary],
        canonical: AnalysisResult,
    ) -> DataQualityReport:
        liquidity = pd.to_numeric(contracts["liquidity"], errors="coerce").fillna(0.0)
        open_interest = pd.to_numeric(
            contracts["openInterest"], errors="coerce"
        ).fillna(0.0)
        volume = pd.to_numeric(contracts["volume"], errors="coerce").fillna(0.0)
        used = liquidity > 0
        volume_fallback = contracts["liquidity_source"] == "volume_fallback"
        iv_source = contracts["iv_source"].astype(str)
        abs_gex = pd.to_numeric(contracts["abs_gex"], errors="coerce").fillna(0.0)
        valid_market = contracts["mark_source"].astype(str) != "unusable"
        is_surface_fitted = contracts.get(
            "is_surface_fitted",
            pd.Series(False, index=contracts.index),
        ).astype(bool)

        warnings = list(canonical.quality.warnings)
        warnings.append(
            "Preset view derived from one canonical quote snapshot and volatility surface; "
            "shared contract GEX is invariant across expiry counts."
        )
        return DataQualityReport(
            expirations_requested=int(requested_expiries),
            expirations_processed=int(processed_expiries),
            expirations_skipped=[],
            contracts_seen=len(contracts),
            contracts_used=int(used.sum()),
            valid_market_quotes=int(valid_market.sum()),
            iv_market_solved=int((iv_source == "market_solved").sum()),
            iv_vendor_used=int((iv_source == "vendor").sum()),
            iv_values_interpolated=int((iv_source == "interpolated").sum()),
            iv_flat_fallback=int((iv_source == "flat_fallback").sum()),
            iv_surface_fitted=int(is_surface_fitted.sum()),
            volume_fallback_contracts=int((volume_fallback & used).sum()),
            reported_oi_contracts=int(((~volume_fallback) & (open_interest > 0) & used).sum()),
            zero_oi_contracts=int((open_interest <= 0).sum()),
            contracts_with_volume=int((volume > 0).sum()),
            valid_contracts=len(contracts),
            parity_expirations=sum(item.reliable for item in forward_summaries),
            parity_pairs_used=sum(item.pairs_used for item in forward_summaries),
            oi_abs_exposure=float(abs_gex.loc[~volume_fallback].sum()),
            volume_abs_exposure=float(abs_gex.loc[volume_fallback].sum()),
            total_abs_exposure=float(abs_gex.sum()),
            warnings=list(dict.fromkeys(warnings)),
        )

    @staticmethod
    def _coverage(frame: pd.DataFrame) -> tuple[int, int, int]:
        if frame is None or frame.empty or "strike" not in frame.columns:
            return 0, 0, 0
        strikes = pd.to_numeric(frame["strike"], errors="coerce")
        valid = strikes.notna() & np.isfinite(strikes) & (strikes > 0)
        oi_source = (
            frame["openInterest"]
            if "openInterest" in frame.columns
            else pd.Series(0.0, index=frame.index)
        )
        volume_source = (
            frame["volume"]
            if "volume" in frame.columns
            else pd.Series(0.0, index=frame.index)
        )
        oi = pd.to_numeric(oi_source, errors="coerce").fillna(0.0)
        volume = pd.to_numeric(volume_source, errors="coerce").fillna(0.0)
        return int(valid.sum()), int((valid & (oi > 0)).sum()), int((valid & (volume > 0)).sum())

    def _select_liquidity_mode(
        self,
        calls: pd.DataFrame,
        puts: pd.DataFrame,
        *,
        requested: LiquidityMode,
    ) -> str:
        if requested != "auto":
            return requested
        counts = [self._coverage(calls), self._coverage(puts)]
        valid = sum(item[0] for item in counts)
        oi = sum(item[1] for item in counts)
        volume = sum(item[2] for item in counts)
        if valid <= 0:
            return "open_interest"
        oi_coverage = oi / valid
        volume_coverage = volume / valid
        if oi_coverage >= 0.70:
            return "open_interest"
        if oi_coverage >= 0.20 and volume_coverage >= 0.40:
            return "oi_volume_fallback"
        if volume_coverage >= 0.60:
            return "volume_dominant"
        return "oi_volume_fallback"

    def _prepare_side(
        self,
        source: pd.DataFrame,
        *,
        side: OptionSide,
        expiration: date,
        spot: float,
        time_years: float,
        days_to_expiry: float,
        risk_free_rate: float,
        dividend_yield: float,
        liquidity_mode: str,
        exposure_mode: ExposureMode,
        quality: DataQualityReport,
    ) -> pd.DataFrame:
        if source is None or source.empty:
            return pd.DataFrame()
        frame = source.copy()
        quality.contracts_seen += len(frame)
        for column in (
            "strike",
            "openInterest",
            "volume",
            "impliedVolatility",
            "bid",
            "ask",
            "lastPrice",
        ):
            if column not in frame.columns:
                frame[column] = np.nan
        if "contractSymbol" not in frame.columns:
            frame["contractSymbol"] = None

        frame["strike"] = pd.to_numeric(frame["strike"], errors="coerce")
        frame["openInterest"] = pd.to_numeric(
            frame["openInterest"], errors="coerce"
        ).fillna(0.0)
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
        frame = frame[
            frame["strike"].notna()
            & np.isfinite(frame["strike"])
            & (frame["strike"] > 0)
        ].copy()
        quality.valid_contracts += len(frame)
        quality.contracts_with_volume += int((frame["volume"] > 0).sum())
        if frame.empty:
            return frame

        has_oi = frame["openInterest"] > 0
        quality.zero_oi_contracts += int((~has_oi).sum())
        if liquidity_mode == "open_interest":
            use_volume = pd.Series(False, index=frame.index)
            liquidity = frame["openInterest"].clip(lower=0.0)
        elif liquidity_mode == "oi_volume_fallback":
            use_volume = (~has_oi) & (frame["volume"] > 0)
            liquidity = frame["openInterest"].where(
                ~use_volume,
                frame["volume"] * self._volume_weight,
            ).clip(lower=0.0)
        elif liquidity_mode == "volume_dominant":
            use_volume = frame["volume"] > 0
            liquidity = (frame["volume"] * self._volume_weight).clip(lower=0.0)
        else:
            raise ValueError(f"Unsupported liquidity mode: {liquidity_mode}")

        quality.volume_fallback_contracts += int((use_volume & (liquidity > 0)).sum())
        quality.reported_oi_contracts += int((~use_volume & has_oi & (liquidity > 0)).sum())

        legacy_iv_result = resolve_implied_volatility(
            frame,
            option_kind=side,
            spot=spot,
            time_years=time_years,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            mode="legacy",
        )
        candidate_iv_result = legacy_iv_result
        if self._quant_engine_mode in {"shadow", "candidate"}:
            candidate_iv_result = resolve_implied_volatility(
                frame,
                option_kind=side,
                spot=spot,
                time_years=time_years,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                mode="candidate",
            )
        iv_result = (
            candidate_iv_result
            if self._quant_engine_mode == "candidate"
            else legacy_iv_result
        )
        quality.valid_market_quotes += int(iv_result.valid_market.sum())
        quality.iv_market_solved += int((iv_result.source == "market_solved").sum())
        quality.iv_vendor_used += int((iv_result.source == "vendor").sum())
        quality.iv_values_interpolated += int((iv_result.source == "interpolated").sum())
        quality.iv_flat_fallback += int((iv_result.source == "flat_fallback").sum())

        if exposure_mode == "gross":
            polarity = 1.0
        elif exposure_mode == "net_proxy":
            polarity = 1.0 if side == "call" else -1.0
        else:
            raise ValueError(f"Unsupported exposure mode: {exposure_mode}")

        source_quality_map = {
            "market_solved": 1.0,
            "vendor": 0.80,
            "interpolated": 0.50,
            "flat_fallback": 0.20,
            "missing": 0.0,
        }
        iv_source_quality = iv_result.source.map(
            source_quality_map
        ).fillna(0.0)
        liquidity_quality = pd.Series(0.0, index=frame.index)
        liquidity_quality.loc[(~use_volume) & has_oi & (liquidity > 0)] = 1.0
        liquidity_quality.loc[use_volume & (liquidity > 0)] = 0.55

        frame["liquidity"] = liquidity
        frame["liquidity_source"] = np.where(
            use_volume, "volume_fallback", "open_interest"
        )
        frame["iv"] = iv_result.volatility
        frame["iv_source"] = iv_result.source
        frame["quote_quality"] = iv_result.quote_quality
        frame["mark_source"] = iv_result.mark_source
        frame["iv_pricing_residual"] = iv_result.pricing_residual
        frame["iv_solver_iterations"] = iv_result.solver_iterations
        frame["iv_solver_status"] = iv_result.solver_status
        frame["legacy_input_iv"] = legacy_iv_result.volatility
        frame["legacy_iv_source"] = legacy_iv_result.source
        frame["legacy_quote_quality"] = legacy_iv_result.quote_quality
        frame["candidate_input_iv"] = candidate_iv_result.volatility
        frame["candidate_iv_source"] = candidate_iv_result.source
        frame["candidate_quote_quality"] = candidate_iv_result.quote_quality
        legacy_source_quality = legacy_iv_result.source.map(source_quality_map).fillna(0.0)
        candidate_source_quality = candidate_iv_result.source.map(source_quality_map).fillna(0.0)
        frame["legacy_contract_quality"] = np.sqrt(
            legacy_iv_result.quote_quality.clip(0.0, 1.0)
            * legacy_source_quality.clip(0.0, 1.0)
            * liquidity_quality.clip(0.0, 1.0)
        )
        frame["candidate_contract_quality"] = np.sqrt(
            candidate_iv_result.quote_quality.clip(0.0, 1.0)
            * candidate_source_quality.clip(0.0, 1.0)
            * liquidity_quality.clip(0.0, 1.0)
        )
        frame["contract_quality"] = np.sqrt(
            iv_result.quote_quality.clip(0.0, 1.0)
            * iv_source_quality.clip(0.0, 1.0)
            * liquidity_quality.clip(0.0, 1.0)
        )
        frame["polarity"] = polarity
        frame["mid_price"] = iv_result.market_price.fillna(0.0)
        frame["side"] = side
        frame["expiration"] = expiration
        frame["days_to_expiry"] = days_to_expiry
        frame["time_years"] = time_years
        frame["rate"] = risk_free_rate
        frame["dividend_yield"] = dividend_yield
        frame["is_iv_repaired"] = frame["iv_source"].isin(
            ["interpolated", "flat_fallback"]
        )
        quality.contracts_used += int((frame["liquidity"] > 0).sum())
        return frame[
            [
                "expiration",
                "contractSymbol",
                "strike",
                "side",
                "bid",
                "ask",
                "lastPrice",
                "impliedVolatility",
                "openInterest",
                "volume",
                "liquidity",
                "liquidity_source",
                "iv",
                "iv_source",
                "is_iv_repaired",
                "polarity",
                "mid_price",
                "quote_quality",
                "mark_source",
                "iv_pricing_residual",
                "iv_solver_iterations",
                "iv_solver_status",
                "legacy_input_iv",
                "legacy_iv_source",
                "legacy_quote_quality",
                "legacy_contract_quality",
                "candidate_input_iv",
                "candidate_iv_source",
                "candidate_quote_quality",
                "candidate_contract_quality",
                "contract_quality",
                "days_to_expiry",
                "time_years",
                "rate",
                "dividend_yield",
            ]
        ]

    def _reprice_surface_contracts(self, frame: pd.DataFrame, *, spot: float) -> pd.DataFrame:
        output = frame.copy().reset_index(drop=True)
        for column in (
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
        ):
            output[column] = 0.0
        for side in ("call", "put"):
            mask = output["side"].to_numpy() == side
            if not np.any(mask):
                continue
            stack = greek_stack(
                side,
                spot=spot,
                strikes=output.loc[mask, "strike"].to_numpy(dtype=float),
                time_years=output.loc[mask, "time_years"].to_numpy(dtype=float),
                volatility=output.loc[mask, "iv"].to_numpy(dtype=float),
                risk_free_rate=output.loc[mask, "rate"].to_numpy(dtype=float),
                dividend_yield=output.loc[mask, "dividend_yield"].to_numpy(dtype=float),
            )
            for column in (
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
            ):
                output.loc[mask, column] = getattr(stack, column)

        position_contracts = output["liquidity"].to_numpy(dtype=float)
        polarity = output["polarity"].to_numpy(dtype=float)
        signed_position = position_contracts * polarity * self._multiplier
        output["gex"] = gamma_exposure_1pct(
            output["gamma"].to_numpy(dtype=float),
            position_contracts=position_contracts,
            multiplier=self._multiplier,
            spot=spot,
            polarity=polarity,
        )

        def scenario_gex(iv_column: str) -> np.ndarray:
            values = np.zeros(len(output), dtype=float)
            if iv_column not in output.columns:
                return values
            for scenario_side in ("call", "put"):
                scenario_mask = output["side"].to_numpy() == scenario_side
                if not np.any(scenario_mask):
                    continue
                gamma = black_scholes_gamma(
                    spot=spot,
                    strikes=output.loc[scenario_mask, "strike"].to_numpy(dtype=float),
                    time_years=output.loc[scenario_mask, "time_years"].to_numpy(dtype=float),
                    volatility=output.loc[scenario_mask, iv_column].to_numpy(dtype=float),
                    risk_free_rate=output.loc[scenario_mask, "rate"].to_numpy(dtype=float),
                    dividend_yield=output.loc[scenario_mask, "dividend_yield"].to_numpy(dtype=float),
                )
                values[scenario_mask] = gamma_exposure_1pct(
                    gamma,
                    position_contracts=position_contracts[scenario_mask],
                    multiplier=self._multiplier,
                    spot=spot,
                    polarity=polarity[scenario_mask],
                )
            return values

        output["legacy_gex"] = scenario_gex("legacy_iv")
        output["candidate_gex"] = scenario_gex("candidate_iv")
        output["candidate_minus_legacy_gex"] = (
            output["candidate_gex"] - output["legacy_gex"]
        )
        output["vanna_exposure_1vol"] = signed_position * output["vanna"] * 0.01
        output["charm_exposure_1day"] = signed_position * output["charm"] / 365.25
        output["speed_exposure_1pct"] = (
            signed_position * output["speed"] * (spot * 0.01) * spot**2 * 0.01
        )
        output["color_exposure_1day"] = (
            signed_position * output["color"] / 365.25 * spot**2 * 0.01
        )
        output["zomma_exposure_1vol"] = (
            signed_position * output["zomma"] * 0.01 * spot**2 * 0.01
        )

        # Refresh exposure-source diagnostics after fitted-surface repricing.
        quality_weight = np.abs(output["gex"].to_numpy(dtype=float))
        # These fields are recalculated by the caller through DataQualityReport;
        # retain per-contract information here for aggregation.
        output["abs_gex"] = quality_weight
        return output

    @staticmethod
    def _build_matrix_records(contracts: pd.DataFrame) -> pd.DataFrame:
        data = contracts.copy()
        data["quality_weight"] = data["abs_gex"] * data["contract_quality"]
        data["fallback_weight"] = data["abs_gex"] * (
            data["liquidity_source"] == "volume_fallback"
        ).astype(float)
        data["repair_weight"] = data["abs_gex"] * data["is_iv_repaired"].astype(float)
        grouped = data.groupby(["expiration", "strike"], as_index=False, sort=True).agg(
            gex=("gex", "sum"),
            abs_gex=("abs_gex", "sum"),
            quality_weight=("quality_weight", "sum"),
            fallback_weight=("fallback_weight", "sum"),
            repair_weight=("repair_weight", "sum"),
            days_to_expiry=("days_to_expiry", "median"),
            risk_free_rate=("rate", "median"),
            carry_yield=("dividend_yield", "median"),
        )
        denominator = grouped["abs_gex"].replace(0.0, np.nan)
        grouped["cell_quality"] = (grouped["quality_weight"] / denominator).fillna(0.0)
        grouped["volume_fallback_ratio"] = (
            grouped["fallback_weight"] / denominator
        ).fillna(0.0)
        grouped["iv_repair_ratio"] = (grouped["repair_weight"] / denominator).fillna(0.0)
        grouped["expiry"] = grouped["expiration"].map(lambda value: value.strftime("%m/%d/%Y"))
        return grouped[
            [
                "strike",
                "gex",
                "abs_gex",
                "cell_quality",
                "volume_fallback_ratio",
                "iv_repair_ratio",
                "expiry",
                "expiration",
                "days_to_expiry",
                "risk_free_rate",
                "carry_yield",
            ]
        ]

    def _build_structural_walls(
        self,
        matrix_records: pd.DataFrame,
        spot: float,
    ) -> tuple[
        pd.DataFrame,
        WallLevel | None,
        WallLevel | None,
        WallLevel | None,
        WallLevel | None,
    ]:
        rows: list[dict[str, float]] = []
        total_expiries = max(matrix_records["expiry"].nunique(), 1)
        expiry_floors: dict[str, float] = {}
        for expiry, frame in matrix_records.groupby("expiry"):
            values = frame.loc[frame["gex"] != 0, "gex"].abs().to_numpy(dtype=float)
            expiry_floors[expiry] = (
                max(float(np.percentile(values, 20)), 1.0) if values.size else 1.0
            )

        for strike, frame in matrix_records.groupby("strike"):
            abs_weights = frame["abs_gex"].to_numpy(dtype=float)
            abs_total = float(abs_weights.sum())
            quality = (
                float(np.average(frame["cell_quality"], weights=abs_weights))
                if abs_total > 0
                else float(frame["cell_quality"].mean())
            )
            active = 0
            for _, item in frame.iterrows():
                contribution = abs(float(item["gex"]))
                meaningful_share = contribution >= abs_total * self._min_expiry_share
                meaningful_floor = contribution >= expiry_floors.get(str(item["expiry"]), 1.0)
                if meaningful_share and meaningful_floor:
                    active += 1
            rows.append(
                {
                    "strike": float(strike),
                    "gex": float(frame["gex"].sum()),
                    "abs_gex": abs_total,
                    "contract_quality": float(np.clip(quality, 0.0, 1.0)),
                    "expiry_breadth": active / total_expiries,
                }
            )
        totals = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
        if totals.empty:
            return totals, None, None, None, None
        magnitudes = totals["gex"].abs()
        totals["magnitude_score"] = magnitudes.rank(pct=True, method="average")
        local_scores: list[float] = []
        for index, row in totals.iterrows():
            neighbors = totals.iloc[max(0, index - 2) : min(len(totals), index + 3)]
            neighbors = neighbors[neighbors.index != index]
            baseline = float(neighbors["gex"].abs().median()) if not neighbors.empty else 0.0
            dominance = abs(float(row["gex"])) / max(baseline, 1.0)
            local_scores.append(float(np.clip(dominance / 3.0, 0.0, 1.0)))
        totals["local_dominance"] = local_scores
        totals["structural_score"] = (
            totals["magnitude_score"] * 0.45
            + totals["expiry_breadth"] * 0.20
            + totals["contract_quality"] * 0.20
            + totals["local_dominance"] * 0.15
        )

        def make_wall(row: pd.Series | None) -> WallLevel | None:
            if row is None:
                return None
            breadth = float(row["expiry_breadth"])
            quality = float(row["contract_quality"])
            stability = (
                "broad"
                if breadth >= 0.60 and quality >= 0.70
                else "moderate"
                if breadth >= 0.30
                else "concentrated"
            )
            return WallLevel(
                strike=float(row["strike"]),
                signed_exposure=float(row["gex"]),
                absolute_exposure=float(row["abs_gex"]),
                expiry_breadth=breadth,
                contract_quality=quality,
                local_dominance=float(row["local_dominance"]),
                structural_score=float(row["structural_score"]),
                stability=stability,
            )

        in_window = totals[
            totals["strike"].between(
                spot * (1 - self._max_wall_distance_pct),
                spot * (1 + self._max_wall_distance_pct),
            )
        ]
        positive_all = in_window[in_window["gex"] > 0]
        negative_all = in_window[in_window["gex"] < 0]
        raw_positive = make_wall(
            positive_all.loc[positive_all["gex"].idxmax()]
            if not positive_all.empty
            else None
        )
        raw_negative = make_wall(
            negative_all.loc[negative_all["gex"].idxmin()]
            if not negative_all.empty
            else None
        )
        valid = in_window[
            (in_window["contract_quality"] >= self._min_wall_quality)
            & (in_window["magnitude_score"] >= self._min_wall_percentile)
        ]
        positive = valid[valid["gex"] > 0]
        negative = valid[valid["gex"] < 0]
        positive_wall = make_wall(
            positive.loc[positive["structural_score"].idxmax()]
            if not positive.empty
            else None
        )
        negative_wall = make_wall(
            negative.loc[negative["structural_score"].idxmax()]
            if not negative.empty
            else None
        )
        return totals, raw_positive, raw_negative, positive_wall, negative_wall

    def _apply_wall_shock_stability(
        self,
        wall: WallLevel | None,
        contracts: pd.DataFrame,
        *,
        spot: float,
    ) -> WallLevel | None:
        if wall is None:
            return None
        related = contracts[np.isclose(contracts["strike"].astype(float), wall.strike)].copy()
        if related.empty or abs(wall.signed_exposure) < 1.0:
            return replace(wall, shock_stability=0.0)
        base = wall.signed_exposure
        values: list[float] = []
        for scenario_spot, iv_shift, time_shift in (
            (spot * 0.995, 0.0, 0.0),
            (spot * 1.005, 0.0, 0.0),
            (spot, -0.01, 0.0),
            (spot, 0.01, 0.0),
            (spot, 0.0, 1.0 / (365.25 * 24.0)),
        ):
            gamma = black_scholes_gamma(
                scenario_spot,
                related["strike"].to_numpy(dtype=float),
                np.maximum(
                    related["time_years"].to_numpy(dtype=float) - time_shift,
                    1.0 / (365.25 * 24 * 60),
                ),
                np.maximum(related["iv"].to_numpy(dtype=float) + iv_shift, 0.01),
                related["rate"].to_numpy(dtype=float),
                related["dividend_yield"].to_numpy(dtype=float),
            )
            exposure = float(
                np.nansum(
                    related["liquidity"].to_numpy(dtype=float)
                    * related["polarity"].to_numpy(dtype=float)
                    * self._multiplier
                    * gamma
                    * scenario_spot**2
                    * 0.01
                )
            )
            sign_score = 1.0 if np.sign(exposure) == np.sign(base) else 0.0
            magnitude_score = float(np.exp(-abs(exposure - base) / max(abs(base), 1.0)))
            values.append(0.55 * sign_score + 0.45 * magnitude_score)
        return replace(wall, shock_stability=float(np.mean(values)))

    @staticmethod
    def _expected_move(
        calls: pd.DataFrame,
        puts: pd.DataFrame,
        *,
        expiration: date,
        days_to_expiry: float,
        spot: float,
        time_years: float,
        horizon_utc: datetime,
    ) -> ExpectedMove | None:
        if calls.empty or puts.empty:
            return None
        common = np.intersect1d(
            calls["strike"].to_numpy(dtype=float),
            puts["strike"].to_numpy(dtype=float),
        )
        if common.size == 0:
            return None
        strike = float(common[np.argmin(np.abs(common - spot))])
        call = calls.loc[calls["strike"] == strike].iloc[0]
        put = puts.loc[puts["strike"] == strike].iloc[0]
        straddle = float(call["mid_price"] + put["mid_price"])
        if np.isfinite(straddle) and straddle > 0:
            dollars = straddle
            method = "cleaned ATM straddle market price"
        else:
            atm_iv = float(np.nanmean([call["iv"], put["iv"]]))
            dollars = float(spot * atm_iv * np.sqrt(time_years))
            method = "ATM model IV × sqrt(time) fallback"
        if not np.isfinite(dollars) or dollars <= 0:
            return None
        return ExpectedMove(
            expiration=expiration,
            days_to_expiry=days_to_expiry,
            dollars=dollars,
            percent=dollars / spot,
            lower=max(0.0, spot - dollars),
            upper=spot + dollars,
            method=method,
            horizon_utc=horizon_utc,
        )

    def _greek_exposure_summary(
        self,
        contracts: pd.DataFrame,
        spot: float,
    ) -> GreekExposureSummary:
        return GreekExposureSummary(
            gamma_exposure=float(contracts["gex"].sum()),
            vanna_exposure_1vol=float(contracts["vanna_exposure_1vol"].sum()),
            charm_exposure_1day=float(contracts["charm_exposure_1day"].sum()),
            speed_exposure_1pct=float(contracts["speed_exposure_1pct"].sum()),
            color_exposure_1day=float(contracts["color_exposure_1day"].sum()),
            zomma_exposure_1vol=float(contracts["zomma_exposure_1vol"].sum()),
        )

    @staticmethod
    def _overall_mode(selected_modes: list[str], quality: DataQualityReport) -> str:
        if "volume_dominant" in selected_modes:
            return "volume-dominant estimate"
        if quality.volume_fallback_contracts > 0:
            return "OI + volume fallback"
        return "reported open interest"

    @staticmethod
    def _interpretation(
        *,
        ticker: str,
        spot: float,
        levels: LevelSummary,
        confidence_score: float,
        surface_grade: str,
        surface_mode: str,
    ) -> tuple[str, ...]:
        statements: list[str] = []
        flip = levels.primary_gamma_flip
        if flip is not None:
            relationship = "above" if spot > flip.level else "below"
            statements.append(
                f"{ticker} is trading {relationship} the primary gamma flip at ${flip.level:,.2f}."
            )
            if flip.model_spread is not None:
                statements.append(
                    f"Sticky-strike versus sticky-delta flip dispersion is ${flip.model_spread:,.2f}."
                )
        if levels.positive_wall is not None and levels.negative_wall is not None:
            statements.append(
                f"The strongest structural positive wall is ${levels.positive_wall.strike:,.2f}, "
                f"while the strongest structural negative wall is ${levels.negative_wall.strike:,.2f}."
            )
        elif levels.positive_wall is not None:
            statements.append(
                f"The strongest structural positive wall is ${levels.positive_wall.strike:,.2f}."
            )
        elif levels.negative_wall is not None:
            statements.append(
                f"The strongest structural negative wall is ${levels.negative_wall.strike:,.2f}."
            )
        if levels.primary_pin is not None:
            wording = (
                "estimated pin pressure"
                if levels.primary_pin.pin_score >= 65
                else "a moderate pin candidate"
                if levels.primary_pin.pin_score >= 45
                else "a weak pin candidate"
            )
            statements.append(
                f"The model identifies {wording} near ${levels.primary_pin.strike:,.2f}."
            )
        statements.append(f"The current net sign-convention model is {levels.regime}.")
        label = "high" if confidence_score >= 80 else "moderate" if confidence_score >= 65 else "limited"
        statements.append(
            f"Model/input confidence is {label} at {confidence_score:.0f}/100; "
            f"the {surface_mode.upper()} surface is grade {surface_grade}."
        )
        return tuple(statements[:6])
