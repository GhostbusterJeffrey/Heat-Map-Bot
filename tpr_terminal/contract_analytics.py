from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .canonical_state import CanonicalContractState
from .forward_engine import estimate_forward_from_put_call_parity
from .market_calendar import USMarketCalendar
from .market_data import MarketDataProvider
from .math_engine import implied_volatility_brent, resolve_implied_volatility
from .models import (
    ContractGreekReport,
    DataQualityReport,
    ForwardSummary,
    GreekScope,
    GreekValueBand,
    OptionSide,
    PositionSide,
    RateCurve,
)
from .pricing import (
    american_binomial_price,
    black_scholes_price,
    classify_instrument,
    greek_stack,
)
from .quant_accuracy import (
    GREEK_DEFINITIONS,
    ContractState,
    QuantEngineMode,
    adaptive_greek_validation,
    gamma_exposure_1pct,
    sanitize_quote,
)
from .volatility_surface import fit_volatility_surface

_GREEK_NAMES = (
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
)

_UNIT_LABELS = GREEK_DEFINITIONS

_SCOPE_LABELS = {
    "per_share": "per option share",
    "per_contract": "per standard contract",
    "custom_position": "for the requested position",
    "market_open_interest": "for listed OI under the public sign proxy",
}


class ContractAnalyticsEngine:
    """Resolve a listed contract and calculate a quote-aware Greek stack."""

    def __init__(
        self,
        provider: MarketDataProvider,
        *,
        contract_multiplier: int,
        calendar_overrides: Path | None = None,
        quant_engine_mode: QuantEngineMode = "legacy",
    ) -> None:
        self._provider = provider
        self._multiplier = int(contract_multiplier)
        self._calendar = USMarketCalendar(calendar_overrides)
        if quant_engine_mode not in {"legacy", "shadow", "candidate"}:
            raise ValueError(f"Unsupported quant engine mode: {quant_engine_mode}")
        self._quant_engine_mode: QuantEngineMode = quant_engine_mode

    def list_expirations(self, ticker: str, *, limit: int = 25) -> list[date]:
        return self._provider.open_session(ticker).expirations()[: max(int(limit), 1)]

    def list_strikes(
        self,
        ticker: str,
        *,
        expiration: date,
        option_side: OptionSide,
        limit: int = 200,
    ) -> list[float]:
        session = self._provider.open_session(ticker)
        if expiration not in session.expirations():
            return []
        chain = session.option_chain(expiration)
        frame = chain.calls if option_side == "call" else chain.puts
        strikes = pd.to_numeric(frame.get("strike"), errors="coerce").dropna()
        return sorted({float(value) for value in strikes if np.isfinite(value) and value > 0})[:limit]

    def analyze_canonical_state(
        self,
        state: CanonicalContractState,
        *,
        quantity: int,
        position: PositionSide,
        scope: GreekScope,
    ) -> ContractGreekReport:
        """Build a command report from the exact state used by the heatmap."""

        if state.contract_multiplier != self._multiplier:
            raise ValueError(
                "Canonical contract multiplier does not match the configured analytics engine."
            )

        quantity_i = max(int(quantity), 1)
        scale_factor = self._scope_scale(
            scope,
            option_side=state.option_side,
            quantity=quantity_i,
            position=position,
            open_interest=state.open_interest,
        )
        iv_bid = self._solve_quote_iv(
            state.bid,
            option_side=state.option_side,
            spot=state.spot,
            strike=state.strike,
            time_years=state.time_years,
            rate=state.risk_free_rate,
            dividend_yield=state.dividend_yield,
        )
        iv_ask = self._solve_quote_iv(
            state.ask,
            option_side=state.option_side,
            spot=state.spot,
            strike=state.strike,
            time_years=state.time_years,
            rate=state.risk_free_rate,
            dividend_yield=state.dividend_yield,
        )

        normalized_mid = self._normalize_raw_greeks(state.greeks)
        normalized: dict[str, dict[str, float]] = {"mid": normalized_mid}
        if iv_bid is not None:
            normalized["bid"] = self._normalized_greeks(
                option_side=state.option_side,
                spot=state.spot,
                strike=state.strike,
                time_years=state.time_years,
                volatility=iv_bid,
                rate=state.risk_free_rate,
                dividend_yield=state.dividend_yield,
            )
        if iv_ask is not None:
            normalized["ask"] = self._normalized_greeks(
                option_side=state.option_side,
                spot=state.spot,
                strike=state.strike,
                time_years=state.time_years,
                volatility=iv_ask,
                rate=state.risk_free_rate,
                dividend_yield=state.dividend_yield,
            )

        validation = adaptive_greek_validation(
            option_side=state.option_side,
            spot=state.spot,
            strike=state.strike,
            time_years=state.time_years,
            volatility=state.selected_implied_volatility,
            risk_free_rate=state.risk_free_rate,
            dividend_yield=state.dividend_yield,
            analytic={name: state.greek(name) for name in _GREEK_NAMES},
        )

        unit_suffix = _SCOPE_LABELS[scope]
        values: dict[str, GreekValueBand] = {}
        for name in _GREEK_NAMES:
            mid_value = normalized_mid[name] * scale_factor
            candidates = [case[name] * scale_factor for case in normalized.values()]
            low, high = _interval(candidates)
            values[name] = GreekValueBand(
                low=low,
                mid=mid_value,
                high=high,
                unit=f"{_UNIT_LABELS[name]} — {unit_suffix}",
                stable=name not in validation.unstable_greeks,
                relative_spread=float(validation.higher_order_spreads.get(name, 0.0)),
            )

        gamma_candidates = [
            case["gamma"] * scale_factor * state.spot**2 * 0.01
            for case in normalized.values()
        ]
        gamma_low, gamma_high = _interval(gamma_candidates)
        dollar_gamma = GreekValueBand(
            low=gamma_low,
            mid=float(
                gamma_exposure_1pct(
                    normalized_mid["gamma"],
                    position_contracts=scale_factor,
                    multiplier=1.0,
                    spot=state.spot,
                )
            ),
            high=gamma_high,
            unit=f"dollar gamma for a 1% spot move — {unit_suffix}",
        )

        legacy_validation_error = self._validation_error(
            option_side=state.option_side,
            spot=state.spot,
            strike=state.strike,
            time_years=state.time_years,
            volatility=state.selected_implied_volatility,
            rate=state.risk_free_rate,
            dividend_yield=state.dividend_yield,
            analytic_delta=state.greek("delta"),
            analytic_gamma=state.greek("gamma"),
        )
        validation_error = (
            validation.maximum_core_error
            if self._quant_engine_mode == "candidate"
            else legacy_validation_error
        )

        instrument_type = classify_instrument(state.ticker)
        model_label = (
            "Canonical-snapshot Black-Scholes-Merton analytic "
            "+ adaptive finite-difference validation"
        )
        warnings: list[str] = [
            f"Resolved from canonical snapshot {state.snapshot_id} "
            f"and contract state {state.contract_state_id}.",
            f"Assumes a {self._multiplier}-share standard deliverable unless vendor metadata says otherwise.",
        ]
        if scope == "market_open_interest":
            warnings.append(
                "Market-OI scope applies the public call-positive/put-negative sign convention; it is not observed dealer inventory."
            )
            if state.liquidity_source != "open_interest":
                warnings.append(
                    "The heatmap used a provisional volume fallback for this node; the report's market-OI scope still uses reported OI."
                )
        if instrument_type in {"american_equity", "american_etf"}:
            model_label += " + CRR American price check"
            european_price = float(
                black_scholes_price(
                    state.option_side,
                    state.spot,
                    np.asarray([state.strike]),
                    np.asarray([state.time_years]),
                    np.asarray([state.selected_implied_volatility]),
                    np.asarray([state.risk_free_rate]),
                    np.asarray([state.dividend_yield]),
                )[0]
            )
            american_price = american_binomial_price(
                state.option_side,
                spot=state.spot,
                strike=state.strike,
                time_years=state.time_years,
                volatility=state.selected_implied_volatility,
                risk_free_rate=state.risk_free_rate,
                dividend_yield=state.dividend_yield,
                steps=250,
            )
            exercise_gap = american_price - european_price
            if exercise_gap > max(0.02, abs(european_price) * 0.02):
                warnings.append(
                    f"American exercise premium is material (${exercise_gap:.3f}/share); "
                    "the reported Greeks remain BSM point estimates in this development build."
                )
        if iv_bid is None or iv_ask is None:
            warnings.append(
                "A complete bid/ask-consistent IV interval was unavailable; missing interval endpoints are omitted."
            )
        if self._quant_engine_mode == "candidate" and validation.unstable_greeks:
            warnings.append(
                "Numerically unstable higher Greeks were flagged: "
                + ", ".join(validation.unstable_greeks)
                + "."
            )

        return ContractGreekReport(
            ticker=state.ticker,
            expiration=state.expiration,
            strike=state.strike,
            option_side=state.option_side,
            spot=state.spot,
            as_of=state.underlying_as_of,
            instrument_type=instrument_type,
            model_label=model_label,
            provider_name=state.provider_name,
            contract_symbol=state.contract_symbol,
            bid=state.bid,
            ask=state.ask,
            last_price=state.last_price,
            open_interest=state.open_interest,
            volume=state.volume,
            quote_quality=state.quote_quality,
            implied_volatility_bid=iv_bid,
            implied_volatility_mid=state.selected_implied_volatility,
            implied_volatility_ask=iv_ask,
            implied_volatility_source=state.implied_volatility_source,
            time_years=state.time_years,
            days_to_expiry=state.days_to_expiry,
            risk_free_rate=state.risk_free_rate,
            dividend_yield=state.dividend_yield,
            contract_multiplier=self._multiplier,
            quantity=quantity_i,
            position=position,
            scope=scope,
            scale_factor=scale_factor,
            values=values,
            dollar_gamma_1pct=dollar_gamma,
            validation_error=validation_error,
            validation_label=_validation_label(validation_error),
            warnings=tuple(warnings),
            quant_engine_mode=self._quant_engine_mode,
            core_validation_errors=dict(validation.core_errors),
            unstable_greeks=validation.unstable_greeks,
            forward_price=state.forward_price,
            selected_mark=state.selected_mark,
            mark_source=state.mark_source,
            settlement_time=state.settlement_time,
            legacy_implied_volatility=state.legacy_implied_volatility,
            candidate_implied_volatility=state.candidate_implied_volatility,
            snapshot_id=state.snapshot_id,
            contract_state_id=state.contract_state_id,
            contract_state_source="canonical_ledger",
            heatmap_gamma_per_share=state.greek("gamma"),
            heatmap_gex_1pct=state.gamma_exposure_1pct,
        )

    def analyze(
        self,
        ticker: str,
        *,
        expiration: date,
        strike: float,
        option_side: OptionSide,
        quantity: int,
        position: PositionSide,
        scope: GreekScope,
        rate_curve: RateCurve,
        now: datetime | None = None,
    ) -> ContractGreekReport:
        symbol = ticker.upper().strip()
        session = self._provider.open_session(symbol)
        underlying = session.underlying()
        expirations = session.expirations()
        if expiration not in expirations:
            nearest = _nearest_dates(expiration, expirations)
            hint = ", ".join(value.isoformat() for value in nearest) or "none returned"
            raise ValueError(
                f"{expiration.isoformat()} is not a listed {symbol} expiration. Nearest: {hint}."
            )

        current = now or datetime.now(UTC)
        current = current.replace(tzinfo=UTC) if current.tzinfo is None else current.astimezone(UTC)
        close = self._calendar.session_close(expiration)
        seconds = (close.close_utc - current).total_seconds()
        if seconds <= 0:
            raise ValueError("That contract has already expired at the configured settlement time.")
        days = seconds / 86_400.0
        time_years = max(days / 365.25, 1.0 / (365.25 * 24 * 60))
        rate = rate_curve.rate_for_days(days)

        chain = session.option_chain(expiration)
        source = chain.calls if option_side == "call" else chain.puts
        row = self._resolve_strike(source, strike)

        carry_yield = underlying.trailing_dividend_yield
        parity = estimate_forward_from_put_call_parity(
            chain.calls,
            chain.puts,
            spot=underlying.spot,
            time_years=time_years,
            risk_free_rate=rate,
        )
        if parity is not None and parity.reliable:
            carry_yield = parity.dividend_yield

        one_row = row.to_frame().T
        legacy_resolution = resolve_implied_volatility(
            one_row,
            option_kind=option_side,
            spot=underlying.spot,
            time_years=time_years,
            risk_free_rate=rate,
            dividend_yield=carry_yield,
            mode="legacy",
        )
        candidate_resolution = legacy_resolution
        candidate_location = legacy_resolution.volatility.index[0]
        if self._quant_engine_mode in {"shadow", "candidate"}:
            candidate_resolution = resolve_implied_volatility(
                source,
                option_kind=option_side,
                spot=underlying.spot,
                time_years=time_years,
                risk_free_rate=rate,
                dividend_yield=carry_yield,
                mode="candidate",
            )
            candidate_location = row.name
        legacy_iv_mid = float(legacy_resolution.volatility.iloc[0])
        candidate_iv_mid = float(candidate_resolution.volatility.loc[candidate_location])
        candidate_iv_source = str(candidate_resolution.source.loc[candidate_location])
        if self._quant_engine_mode in {"shadow", "candidate"}:
            surface_value = self._candidate_surface_iv(
                chain,
                expiration=expiration,
                target_side=option_side,
                target_strike=float(strike),
                spot=underlying.spot,
                time_years=time_years,
                days_to_expiry=days,
                rate=rate,
                dividend_yield=carry_yield,
                parity=parity,
            )
            if surface_value is not None:
                candidate_iv_mid, candidate_iv_source = surface_value

        if self._quant_engine_mode == "candidate":
            iv_mid = candidate_iv_mid
            iv_source = candidate_iv_source
            quote_quality = float(candidate_resolution.quote_quality.loc[candidate_location])
        else:
            iv_mid = legacy_iv_mid
            iv_source = str(legacy_resolution.source.iloc[0])
            quote_quality = float(legacy_resolution.quote_quality.iloc[0])

        bid = _finite_positive(row.get("bid"))
        ask = _finite_positive(row.get("ask"))
        last = _finite_positive(row.get("lastPrice"))
        sanitized = sanitize_quote(
            option_side=option_side,
            spot=underlying.spot,
            strike=float(strike),
            time_years=time_years,
            risk_free_rate=rate,
            dividend_yield=carry_yield,
            bid=row.get("bid"),
            ask=row.get("ask"),
            last=row.get("lastPrice"),
        )
        iv_bid = self._solve_quote_iv(
            bid,
            option_side=option_side,
            spot=underlying.spot,
            strike=float(strike),
            time_years=time_years,
            rate=rate,
            dividend_yield=carry_yield,
        )
        iv_ask = self._solve_quote_iv(
            ask,
            option_side=option_side,
            spot=underlying.spot,
            strike=float(strike),
            time_years=time_years,
            rate=rate,
            dividend_yield=carry_yield,
        )

        quantity_i = max(int(quantity), 1)
        open_interest = max(int(_finite_nonnegative(row.get("openInterest")) or 0), 0)
        volume = max(int(_finite_nonnegative(row.get("volume")) or 0), 0)
        scale_factor = self._scope_scale(
            scope,
            option_side=option_side,
            quantity=quantity_i,
            position=position,
            open_interest=open_interest,
        )

        iv_cases: dict[str, float] = {"mid": iv_mid}
        if iv_bid is not None:
            iv_cases["bid"] = iv_bid
        if iv_ask is not None:
            iv_cases["ask"] = iv_ask
        normalized = {
            key: self._normalized_greeks(
                option_side=option_side,
                spot=underlying.spot,
                strike=float(strike),
                time_years=time_years,
                volatility=vol,
                rate=rate,
                dividend_yield=carry_yield,
            )
            for key, vol in iv_cases.items()
        }

        values: dict[str, GreekValueBand] = {}
        raw_stack = greek_stack(
            option_side,
            spot=underlying.spot,
            strikes=np.asarray([float(strike)]),
            time_years=np.asarray([time_years]),
            volatility=np.asarray([iv_mid]),
            risk_free_rate=np.asarray([rate]),
            dividend_yield=np.asarray([carry_yield]),
        )
        raw_mid = {name: float(getattr(raw_stack, name)[0]) for name in _GREEK_NAMES}
        validation = adaptive_greek_validation(
            option_side=option_side,
            spot=underlying.spot,
            strike=float(strike),
            time_years=time_years,
            volatility=iv_mid,
            risk_free_rate=rate,
            dividend_yield=carry_yield,
            analytic=raw_mid,
        )
        unit_suffix = _SCOPE_LABELS[scope]
        for name in _GREEK_NAMES:
            mid_value = normalized["mid"][name] * scale_factor
            candidates = [case[name] * scale_factor for case in normalized.values()]
            low, high = _interval(candidates)
            values[name] = GreekValueBand(
                low=low,
                mid=mid_value,
                high=high,
                unit=f"{_UNIT_LABELS[name]} — {unit_suffix}",
                stable=name not in validation.unstable_greeks,
                relative_spread=float(validation.higher_order_spreads.get(name, 0.0)),
            )

        gamma_candidates = [
            case["gamma"] * scale_factor * underlying.spot**2 * 0.01
            for case in normalized.values()
        ]
        gamma_low, gamma_high = _interval(gamma_candidates)
        dollar_gamma = GreekValueBand(
            low=gamma_low,
            mid=float(
                gamma_exposure_1pct(
                    normalized["mid"]["gamma"],
                    position_contracts=scale_factor,
                    multiplier=1.0,
                    spot=underlying.spot,
                )
            ),
            high=gamma_high,
            unit=f"dollar gamma for a 1% spot move — {unit_suffix}",
        )

        legacy_validation_error = self._validation_error(
            option_side=option_side,
            spot=underlying.spot,
            strike=float(strike),
            time_years=time_years,
            volatility=iv_mid,
            rate=rate,
            dividend_yield=carry_yield,
            analytic_delta=normalized["mid"]["delta"],
            analytic_gamma=normalized["mid"]["gamma"],
        )
        validation_error = (
            validation.maximum_core_error
            if self._quant_engine_mode == "candidate"
            else legacy_validation_error
        )
        validation_label = _validation_label(validation_error)

        instrument_type = classify_instrument(symbol)
        model_label = "Black-Scholes-Merton analytic + adaptive finite-difference validation"
        warnings: list[str] = [
            f"Assumes a {self._multiplier}-share standard deliverable unless vendor metadata says otherwise.",
        ]
        if scope == "market_open_interest":
            warnings.append(
                "Market-OI scope applies the public call-positive/put-negative sign convention; it is not observed dealer inventory."
            )
        if instrument_type in {"american_equity", "american_etf"}:
            model_label += " + CRR American price check"
            european_price = float(
                black_scholes_price(
                    option_side,
                    underlying.spot,
                    np.asarray([float(strike)]),
                    np.asarray([time_years]),
                    np.asarray([iv_mid]),
                    np.asarray([rate]),
                    np.asarray([carry_yield]),
                )[0]
            )
            american_price = american_binomial_price(
                option_side,
                spot=underlying.spot,
                strike=float(strike),
                time_years=time_years,
                volatility=iv_mid,
                risk_free_rate=rate,
                dividend_yield=carry_yield,
                steps=250,
            )
            exercise_gap = american_price - european_price
            if exercise_gap > max(0.02, abs(european_price) * 0.02):
                warnings.append(
                    f"American exercise premium is material (${exercise_gap:.3f}/share); the reported Greeks remain BSM point estimates in Alpha 4.0."
                )
        if iv_bid is None or iv_ask is None:
            warnings.append(
                "A complete bid/ask-consistent IV interval was unavailable; missing interval endpoints are omitted."
            )
        if rate_curve.used_fallback:
            warnings.append("The configured static risk-free-rate fallback was used.")
        if self._quant_engine_mode == "candidate" and validation.unstable_greeks:
            warnings.append(
                "Numerically unstable higher Greeks were flagged: "
                + ", ".join(validation.unstable_greeks)
                + "."
            )

        forward_price = (
            float(parity.forward)
            if parity is not None and parity.reliable
            else float(underlying.spot * np.exp((rate - carry_yield) * time_years))
        )
        state = ContractState(
            ticker=symbol,
            expiration=expiration,
            valuation_time=current,
            settlement_time=close.close_utc,
            option_side=option_side,
            spot=underlying.spot,
            strike=float(strike),
            time_years=time_years,
            risk_free_rate=rate,
            dividend_yield=carry_yield,
            forward=forward_price,
            quote=sanitized,
            vendor_iv=_finite_positive(row.get("impliedVolatility")),
            selected_iv=iv_mid,
            iv_source=iv_source,
            open_interest=open_interest,
            volume=volume,
            multiplier=self._multiplier,
        )

        contract_symbol = row.get("contractSymbol")
        return ContractGreekReport(
            ticker=symbol,
            expiration=expiration,
            strike=float(strike),
            option_side=option_side,
            spot=underlying.spot,
            as_of=underlying.as_of,
            instrument_type=instrument_type,
            model_label=model_label,
            provider_name=session.provider_name,
            contract_symbol=str(contract_symbol) if contract_symbol is not None else None,
            bid=bid,
            ask=ask,
            last_price=last,
            open_interest=open_interest,
            volume=volume,
            quote_quality=quote_quality,
            implied_volatility_bid=iv_bid,
            implied_volatility_mid=iv_mid,
            implied_volatility_ask=iv_ask,
            implied_volatility_source=iv_source,
            time_years=time_years,
            days_to_expiry=days,
            risk_free_rate=rate,
            dividend_yield=carry_yield,
            contract_multiplier=self._multiplier,
            quantity=quantity_i,
            position=position,
            scope=scope,
            scale_factor=scale_factor,
            values=values,
            dollar_gamma_1pct=dollar_gamma,
            validation_error=validation_error,
            validation_label=validation_label,
            warnings=tuple(warnings),
            quant_engine_mode=self._quant_engine_mode,
            core_validation_errors=dict(validation.core_errors),
            unstable_greeks=validation.unstable_greeks,
            forward_price=state.forward,
            selected_mark=state.quote.mark,
            mark_source=state.quote.mark_source,
            settlement_time=state.settlement_time,
            legacy_implied_volatility=legacy_iv_mid,
            candidate_implied_volatility=candidate_iv_mid,
        )

    @staticmethod
    def _resolve_strike(frame: pd.DataFrame, requested: float) -> pd.Series:
        if frame is None or frame.empty or "strike" not in frame.columns:
            raise ValueError("The selected option side returned no strikes.")
        strikes = pd.to_numeric(frame["strike"], errors="coerce")
        tolerance = max(abs(float(requested)) * 1e-8, 1e-6)
        matches = frame.loc[(strikes - float(requested)).abs() <= tolerance]
        if matches.empty:
            valid = strikes.dropna().to_numpy(dtype=float)
            nearest = sorted(valid, key=lambda value: abs(value - requested))[:3]
            hint = ", ".join(f"{value:g}" for value in nearest) or "none"
            raise ValueError(f"Strike {requested:g} is not listed. Nearest strikes: {hint}.")
        return matches.iloc[0]

    @staticmethod
    def _candidate_surface_iv(
        chain,
        *,
        expiration: date,
        target_side: OptionSide,
        target_strike: float,
        spot: float,
        time_years: float,
        days_to_expiry: float,
        rate: float,
        dividend_yield: float,
        parity,
    ) -> tuple[float, str] | None:
        rows: list[pd.DataFrame] = []
        total_market_solved = 0
        for side, source in (("call", chain.calls), ("put", chain.puts)):
            if source is None or source.empty:
                continue
            frame = source.copy()
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
            frame["strike"] = pd.to_numeric(frame["strike"], errors="coerce")
            frame = frame.loc[
                frame["strike"].notna()
                & np.isfinite(frame["strike"])
                & (frame["strike"] > 0)
            ].copy()
            if frame.empty:
                continue
            resolution = resolve_implied_volatility(
                frame,
                option_kind=side,
                spot=spot,
                time_years=time_years,
                risk_free_rate=rate,
                dividend_yield=dividend_yield,
                mode="candidate",
            )
            total_market_solved += int((resolution.source == "market_solved").sum())
            oi = pd.to_numeric(frame["openInterest"], errors="coerce").fillna(0.0).clip(lower=0.0)
            volume = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0).clip(lower=0.0)
            use_volume = (oi <= 0.0) & (volume > 0.0)
            liquidity = oi.where(~use_volume, volume * 0.50)
            source_quality = resolution.source.map(
                {
                    "market_solved": 1.0,
                    "vendor": 0.80,
                    "interpolated": 0.50,
                    "flat_fallback": 0.20,
                    "missing": 0.0,
                }
            ).fillna(0.0)
            liquidity_quality = np.where(liquidity > 0.0, np.where(use_volume, 0.55, 1.0), 0.0)
            prepared = pd.DataFrame(
                {
                    "expiration": expiration,
                    "strike": frame["strike"].to_numpy(dtype=float),
                    "side": side,
                    "openInterest": oi.to_numpy(dtype=float),
                    "volume": volume.to_numpy(dtype=float),
                    "liquidity": liquidity.to_numpy(dtype=float),
                    "liquidity_source": np.where(use_volume, "volume_fallback", "open_interest"),
                    "iv": resolution.volatility.to_numpy(dtype=float),
                    "iv_source": resolution.source.to_numpy(),
                    "candidate_input_iv": resolution.volatility.to_numpy(dtype=float),
                    "candidate_iv_source": resolution.source.to_numpy(),
                    "candidate_quote_quality": resolution.quote_quality.to_numpy(dtype=float),
                    "is_iv_repaired": resolution.source.isin(["interpolated", "flat_fallback"]).to_numpy(),
                    "polarity": 1.0 if side == "call" else -1.0,
                    "mid_price": resolution.market_price.fillna(0.0).to_numpy(dtype=float),
                    "quote_quality": resolution.quote_quality.to_numpy(dtype=float),
                    "contract_quality": np.sqrt(
                        resolution.quote_quality.to_numpy(dtype=float)
                        * source_quality.to_numpy(dtype=float)
                        * np.asarray(liquidity_quality, dtype=float)
                    ),
                    "days_to_expiry": days_to_expiry,
                    "time_years": time_years,
                    "rate": rate,
                    "dividend_yield": dividend_yield,
                }
            )
            rows.append(prepared)

        if not rows:
            return None
        contracts = pd.concat(rows, ignore_index=True)
        summaries: list[ForwardSummary] = []
        if parity is not None:
            summaries.append(
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
        quality = DataQualityReport(
            expirations_requested=1,
            expirations_processed=1,
            contracts_seen=len(contracts),
            valid_contracts=len(contracts),
            contracts_used=int((contracts["liquidity"] > 0.0).sum()),
            valid_market_quotes=int((contracts["quote_quality"] > 0.0).sum()),
            iv_market_solved=total_market_solved,
            parity_expirations=int(bool(parity is not None and parity.reliable)),
        )
        fitted, _ = fit_volatility_surface(
            contracts,
            summaries,
            quality,
            spot=spot,
            quant_engine_mode="candidate",
        )
        mask = (fitted["side"] == target_side) & np.isclose(
            fitted["strike"].to_numpy(dtype=float), target_strike, rtol=0.0, atol=max(target_strike * 1e-8, 1e-6)
        )
        matches = fitted.loc[mask]
        if matches.empty:
            return None
        selected = matches.iloc[0]
        source = (
            f"surface_blend:{selected['surface_source']}"
            if bool(selected["is_surface_fitted"])
            else str(selected.get("candidate_iv_source", "candidate"))
        )
        return float(selected["candidate_iv"]), source

    @staticmethod
    def _solve_quote_iv(
        price: float | None,
        *,
        option_side: OptionSide,
        spot: float,
        strike: float,
        time_years: float,
        rate: float,
        dividend_yield: float,
    ) -> float | None:
        if price is None:
            return None
        solved = implied_volatility_brent(
            option_kind=option_side,
            market_price=price,
            spot=spot,
            strike=strike,
            time_years=time_years,
            risk_free_rate=rate,
            dividend_yield=dividend_yield,
        )
        if solved is None or not 0.005 <= solved <= 5.0:
            return None
        return float(solved)

    def _scope_scale(
        self,
        scope: GreekScope,
        *,
        option_side: OptionSide,
        quantity: int,
        position: PositionSide,
        open_interest: int,
    ) -> float:
        if scope == "per_share":
            return 1.0
        if scope == "per_contract":
            return float(self._multiplier)
        if scope == "custom_position":
            sign = 1.0 if position == "long" else -1.0
            return sign * self._multiplier * quantity
        if scope == "market_open_interest":
            polarity = 1.0 if option_side == "call" else -1.0
            return polarity * self._multiplier * open_interest
        raise ValueError(f"Unsupported Greek scope: {scope}")

    @staticmethod
    def _normalized_greeks(
        *,
        option_side: OptionSide,
        spot: float,
        strike: float,
        time_years: float,
        volatility: float,
        rate: float,
        dividend_yield: float,
    ) -> dict[str, float]:
        stack = greek_stack(
            option_side,
            spot=spot,
            strikes=np.asarray([strike]),
            time_years=np.asarray([time_years]),
            volatility=np.asarray([volatility]),
            risk_free_rate=np.asarray([rate]),
            dividend_yield=np.asarray([dividend_yield]),
        )
        raw = {name: float(getattr(stack, name)[0]) for name in _GREEK_NAMES}
        return {
            "delta": raw["delta"],
            "gamma": raw["gamma"],
            "theta": raw["theta"] / 365.25,
            "vega": raw["vega"] * 0.01,
            "rho": raw["rho"] * 0.01,
            "vanna": raw["vanna"] * 0.01,
            "charm": raw["charm"] / 365.25,
            "vomma": raw["vomma"] * 0.0001,
            "speed": raw["speed"],
            "color": raw["color"] / 365.25,
            "zomma": raw["zomma"] * 0.01,
            "ultima": raw["ultima"] * 0.000001,
        }

    @staticmethod
    def _normalize_raw_greeks(raw: Mapping[str, float]) -> dict[str, float]:
        return {
            "delta": float(raw["delta"]),
            "gamma": float(raw["gamma"]),
            "theta": float(raw["theta"]) / 365.25,
            "vega": float(raw["vega"]) * 0.01,
            "rho": float(raw["rho"]) * 0.01,
            "vanna": float(raw["vanna"]) * 0.01,
            "charm": float(raw["charm"]) / 365.25,
            "vomma": float(raw["vomma"]) * 0.0001,
            "speed": float(raw["speed"]),
            "color": float(raw["color"]) / 365.25,
            "zomma": float(raw["zomma"]) * 0.01,
            "ultima": float(raw["ultima"]) * 0.000001,
        }

    @staticmethod
    def _validation_error(
        *,
        option_side: OptionSide,
        spot: float,
        strike: float,
        time_years: float,
        volatility: float,
        rate: float,
        dividend_yield: float,
        analytic_delta: float,
        analytic_gamma: float,
    ) -> float:
        estimates: list[tuple[float, float]] = []
        base = float(
            black_scholes_price(
                option_side,
                spot,
                np.asarray([strike]),
                np.asarray([time_years]),
                np.asarray([volatility]),
                np.asarray([rate]),
                np.asarray([dividend_yield]),
            )[0]
        )
        for fraction in (5e-5, 1e-4, 2e-4, 5e-4):
            bump = max(spot * fraction, 1e-4)
            plus = float(
                black_scholes_price(
                    option_side,
                    spot + bump,
                    np.asarray([strike]),
                    np.asarray([time_years]),
                    np.asarray([volatility]),
                    np.asarray([rate]),
                    np.asarray([dividend_yield]),
                )[0]
            )
            minus = float(
                black_scholes_price(
                    option_side,
                    max(spot - bump, 1e-8),
                    np.asarray([strike]),
                    np.asarray([time_years]),
                    np.asarray([volatility]),
                    np.asarray([rate]),
                    np.asarray([dividend_yield]),
                )[0]
            )
            estimates.append(((plus - minus) / (2.0 * bump), (plus - 2.0 * base + minus) / bump**2))
        fd_delta = float(np.median([value[0] for value in estimates]))
        fd_gamma = float(np.median([value[1] for value in estimates]))
        delta_error = abs(fd_delta - analytic_delta) / max(abs(analytic_delta), abs(fd_delta), 1e-8)
        gamma_error = abs(fd_gamma - analytic_gamma) / max(abs(analytic_gamma), abs(fd_gamma), 1e-8)
        return float(max(delta_error, gamma_error))


def _interval(values: Iterable[float]) -> tuple[float | None, float | None]:
    finite = [float(value) for value in values if np.isfinite(value)]
    if len(finite) <= 1:
        return None, None
    return min(finite), max(finite)


def _finite_positive(value) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) and parsed > 0 else None


def _finite_nonnegative(value) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) and parsed >= 0 else None


def _nearest_dates(target: date, values: list[date], count: int = 3) -> list[date]:
    return sorted(values, key=lambda value: abs((value - target).days))[:count]


def _validation_label(error: float) -> str:
    if error <= 0.0025:
        return "excellent"
    if error <= 0.01:
        return "good"
    if error <= 0.03:
        return "fair"
    return "weak"
