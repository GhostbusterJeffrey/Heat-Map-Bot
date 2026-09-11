from __future__ import annotations

from datetime import date
from pathlib import Path

from . import __version__
from .analytics import ExposureAnalyticsEngine
from .contract_analytics import ContractAnalyticsEngine
from .market_data import SnapshottingMarketDataProvider, YahooFinanceProvider
from .models import (
    AnalysisResult,
    ContractGreekReport,
    ExposureMode,
    GreekScope,
    LiquidityMode,
    OptionSide,
    PositionSide,
    RenderedHeatmap,
)
from .rates import TreasuryCurveService
from .renderer import HeatmapRenderer
from .settings import Settings
from .snapshot_store import OIComparison, OptionSnapshotStore, SnapshotStats


class TerminalService:
    def __init__(self, settings: Settings) -> None:
        raw_provider = YahooFinanceProvider()
        provider = raw_provider
        self._snapshot_store: OptionSnapshotStore | None = None
        if settings.snapshot_enabled:
            self._snapshot_store = OptionSnapshotStore(
                settings.snapshot_db_path,
                retention_days=settings.snapshot_retention_days,
            )
            self._snapshot_store.prune()
            provider = SnapshottingMarketDataProvider(
                provider,
                self._snapshot_store,
                model_version=__version__,
            )

        overrides = Path(__file__).resolve().parent.parent / "assets" / "calendar_overrides.json"
        self._provider = provider
        self._analytics = ExposureAnalyticsEngine(
            provider,
            contract_multiplier=settings.contract_multiplier,
            calendar_overrides=overrides,
            volume_liquidity_weight=settings.volume_liquidity_weight,
            max_wall_distance_pct=settings.max_wall_distance_pct,
            min_wall_quality=settings.min_wall_quality,
            min_wall_percentile=settings.min_wall_percentile,
            min_expiry_share_for_breadth=settings.min_expiry_share_for_breadth,
            min_flip_confidence=settings.min_flip_confidence,
            flip_grid_low_pct=settings.flip_grid_low_pct,
            flip_grid_high_pct=settings.flip_grid_high_pct,
            flip_grid_points=settings.flip_grid_points,
            pin_decay_days=settings.pin_decay_days,
            pin_shock_pct=settings.pin_shock_pct,
            min_pin_score=settings.min_pin_score_to_display,
            min_contracts_to_render=settings.min_contracts_to_render,
            min_usable_ratio=settings.min_usable_ratio,
            min_confidence_to_render=settings.min_confidence_to_render,
            quant_engine_mode=settings.quant_engine_mode,
        )
        self._contracts = ContractAnalyticsEngine(
            provider,
            contract_multiplier=settings.contract_multiplier,
            calendar_overrides=overrides,
            quant_engine_mode=settings.quant_engine_mode,
        )
        # Autocomplete lookups deliberately bypass snapshot recording so typing
        # in Discord does not pollute the historical research database.
        self._contract_lookup = ContractAnalyticsEngine(
            raw_provider,
            contract_multiplier=settings.contract_multiplier,
            calendar_overrides=overrides,
            quant_engine_mode=settings.quant_engine_mode,
        )
        self._rates = TreasuryCurveService(
            enabled=settings.use_treasury_curve,
            static_fallback_rate=settings.static_risk_free_rate,
            ttl_seconds=settings.rate_cache_ttl_seconds,
        )
        self._renderer = HeatmapRenderer(assets_dir=settings.assets_dir)

    def analyze(
        self,
        ticker: str,
        *,
        expiries: int,
        liquidity_mode: LiquidityMode,
        exposure_mode: ExposureMode,
    ) -> AnalysisResult:
        return self._analytics.analyze(
            ticker,
            max_expiries=expiries,
            liquidity_mode=liquidity_mode,
            exposure_mode=exposure_mode,
            rate_curve=self._rates.get_curve(),
        )

    def analysis_view(
        self,
        canonical: AnalysisResult,
        *,
        expiries: int,
    ) -> AnalysisResult:
        return self._analytics.slice_result(canonical, max_expiries=expiries)

    def contract_greeks(
        self,
        ticker: str,
        *,
        expiration: date,
        strike: float,
        option_side: OptionSide,
        quantity: int,
        position: PositionSide,
        scope: GreekScope,
        canonical: AnalysisResult | None = None,
    ) -> ContractGreekReport:
        if canonical is not None:
            if canonical.ticker != ticker.upper().strip():
                raise ValueError("Canonical snapshot ticker does not match the requested ticker.")
            if canonical.contract_ledger is None:
                raise ValueError("The canonical snapshot does not contain a contract ledger.")
            state = canonical.contract_ledger.resolve(
                expiration=expiration,
                strike=strike,
                option_side=option_side,
            )
            return self._contracts.analyze_canonical_state(
                state,
                quantity=quantity,
                position=position,
                scope=scope,
            )
        return self._contracts.analyze(
            ticker,
            expiration=expiration,
            strike=strike,
            option_side=option_side,
            quantity=quantity,
            position=position,
            scope=scope,
            rate_curve=self._rates.get_curve(),
        )

    def list_expirations(self, ticker: str, *, limit: int = 25) -> list[date]:
        return self._contract_lookup.list_expirations(ticker, limit=limit)

    def list_strikes(
        self,
        ticker: str,
        *,
        expiration: date,
        option_side: OptionSide,
        limit: int = 200,
    ) -> list[float]:
        return self._contract_lookup.list_strikes(
            ticker,
            expiration=expiration,
            option_side=option_side,
            limit=limit,
        )

    def snapshot_stats(self) -> SnapshotStats | None:
        return None if self._snapshot_store is None else self._snapshot_store.stats()

    def compare_open_interest(
        self,
        ticker: str,
        *,
        expiration: date | None = None,
    ) -> OIComparison:
        if self._snapshot_store is None:
            raise ValueError("Historical snapshots are disabled in configuration.")
        return self._snapshot_store.compare_latest(ticker, expiration=expiration)

    def render(self, result: AnalysisResult, strikes: int) -> RenderedHeatmap:
        png_bytes, displayed = self._renderer.render(result, strikes)
        return RenderedHeatmap(
            png_bytes=png_bytes,
            result=result,
            displayed_strikes=displayed,
        )
