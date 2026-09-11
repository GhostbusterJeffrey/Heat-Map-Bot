from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from hashlib import sha256
from types import MappingProxyType

import numpy as np
import pandas as pd

from .models import AnalysisResult, OptionSide

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


@dataclass(frozen=True, slots=True)
class CanonicalContractState:
    """One immutable, snapshot-bound contract input and calculation state."""

    contract_state_id: str
    snapshot_id: str
    ticker: str
    expiration: date
    strike: float
    option_side: OptionSide
    valuation_time: datetime
    settlement_time: datetime
    underlying_as_of: datetime
    spot: float
    time_years: float
    days_to_expiry: float
    risk_free_rate: float
    dividend_yield: float
    forward_price: float
    provider_name: str
    contract_symbol: str | None
    bid: float | None
    ask: float | None
    last_price: float | None
    selected_mark: float | None
    mark_source: str
    quote_quality: float
    vendor_implied_volatility: float | None
    selected_implied_volatility: float
    implied_volatility_source: str
    legacy_implied_volatility: float | None
    candidate_implied_volatility: float | None
    surface_source: str
    is_surface_fitted: bool
    open_interest: int
    volume: int
    liquidity: float
    liquidity_source: str
    polarity: float
    contract_multiplier: int
    greeks: Mapping[str, float]
    gamma_exposure_1pct: float

    def greek(self, name: str) -> float:
        try:
            return float(self.greeks[name])
        except KeyError as exc:
            raise ValueError(f"Unknown canonical Greek: {name}") from exc


class CanonicalContractLedger:
    """Read-only index of every contract calculated for one canonical snapshot."""

    def __init__(
        self,
        *,
        ticker: str,
        snapshot_id: str,
        valuation_time: datetime,
        states: Iterable[CanonicalContractState],
    ) -> None:
        materialized = tuple(states)
        if not materialized:
            raise ValueError("A canonical contract ledger cannot be empty.")
        if any(state.ticker != ticker for state in materialized):
            raise ValueError("Canonical contract ledger contains more than one ticker.")
        if any(state.snapshot_id != snapshot_id for state in materialized):
            raise ValueError("Canonical contract ledger contains more than one snapshot.")
        ids = tuple(state.contract_state_id for state in materialized)
        if len(ids) != len(set(ids)):
            raise ValueError("Canonical contract state IDs must be unique.")

        self.ticker = ticker
        self.snapshot_id = snapshot_id
        self.valuation_time = valuation_time
        self._states = materialized
        grouped: dict[tuple[date, OptionSide], list[CanonicalContractState]] = {}
        for state in materialized:
            grouped.setdefault((state.expiration, state.option_side), []).append(state)
        self._grouped = {
            key: tuple(
                sorted(
                    values,
                    key=lambda item: (
                        item.strike,
                        -item.quote_quality,
                        -item.liquidity,
                        item.contract_symbol or "",
                    ),
                )
            )
            for key, values in grouped.items()
        }

    @property
    def states(self) -> tuple[CanonicalContractState, ...]:
        return self._states

    @property
    def expirations(self) -> tuple[date, ...]:
        return tuple(sorted({state.expiration for state in self._states}))

    def __len__(self) -> int:
        return len(self._states)

    def resolve(
        self,
        *,
        expiration: date,
        strike: float,
        option_side: OptionSide,
    ) -> CanonicalContractState:
        candidates = self._grouped.get((expiration, option_side), ())
        if not candidates:
            available = ", ".join(value.isoformat() for value in self.expirations)
            raise ValueError(
                f"{expiration.isoformat()} is outside canonical snapshot {self.snapshot_id}. "
                f"Available expirations: {available or 'none'}."
            )

        requested = float(strike)
        tolerance = max(abs(requested) * 1e-8, 1e-6)
        matches = [
            state
            for state in candidates
            if abs(float(state.strike) - requested) <= tolerance
        ]
        if not matches:
            nearest = sorted(
                {float(state.strike) for state in candidates},
                key=lambda value: abs(value - requested),
            )[:3]
            hint = ", ".join(f"{value:g}" for value in nearest) or "none"
            raise ValueError(
                f"Strike {requested:g} is not in canonical snapshot "
                f"{self.snapshot_id}. Nearest strikes: {hint}."
            )

        # Adjusted or duplicate listed contracts can share a strike. Until the
        # command accepts a contract symbol, resolve deterministically to the
        # strongest quote/liquidity observation and retain its symbol in the state.
        return max(
            matches,
            key=lambda state: (
                state.quote_quality,
                state.liquidity,
                state.open_interest,
                state.contract_symbol or "",
            ),
        )


def build_canonical_contract_ledger(
    result: AnalysisResult,
    *,
    valuation_time: datetime,
    contract_multiplier: int,
) -> CanonicalContractLedger:
    """Materialize the exact contract inputs and Greeks used by a heatmap."""

    forwards = {
        summary.expiration: float(summary.forward)
        for summary in result.forward_summaries
        if np.isfinite(summary.forward) and summary.forward > 0.0
    }
    occurrences: dict[tuple[date, OptionSide, float, str], int] = {}
    states: list[CanonicalContractState] = []

    for _, row in result.contract_records.iterrows():
        expiration = _as_date(row["expiration"])
        side = str(row["side"]).lower()
        if side not in {"call", "put"}:
            raise ValueError(f"Unsupported canonical option side: {side}")
        option_side: OptionSide = side  # type: ignore[assignment]
        strike = float(row["strike"])
        symbol = _optional_text(row.get("contractSymbol"))
        occurrence_key = (expiration, option_side, strike, symbol or "")
        occurrence = occurrences.get(occurrence_key, 0)
        occurrences[occurrence_key] = occurrence + 1

        state_payload = "|".join(
            (
                result.snapshot_id,
                expiration.isoformat(),
                option_side,
                f"{strike:.12g}",
                symbol or "",
                str(occurrence),
            )
        )
        state_id = f"cs-{sha256(state_payload.encode('utf-8')).hexdigest()[:20]}"

        time_years = float(row["time_years"])
        days = float(row["days_to_expiry"])
        rate = float(row["rate"])
        dividend_yield = float(row["dividend_yield"])
        forward = forwards.get(
            expiration,
            float(result.spot * np.exp((rate - dividend_yield) * time_years)),
        )
        settlement_time = valuation_time + timedelta(days=time_years * 365.25)
        selected_mark = _optional_positive(row.get("mid_price"))
        mark_source = str(row.get("mark_source", "unknown"))
        if mark_source == "unusable":
            selected_mark = None

        greek_values = MappingProxyType(
            {name: float(row[name]) for name in _GREEK_NAMES}
        )
        selected_iv = float(row["iv"])
        surface_source = str(row.get("surface_source", row.get("iv_source", "unknown")))
        is_surface_fitted = bool(row.get("is_surface_fitted", False))
        iv_source = (
            f"surface:{surface_source}"
            if is_surface_fitted
            else str(row.get("iv_source", surface_source))
        )

        states.append(
            CanonicalContractState(
                contract_state_id=state_id,
                snapshot_id=result.snapshot_id,
                ticker=result.ticker,
                expiration=expiration,
                strike=strike,
                option_side=option_side,
                valuation_time=valuation_time,
                settlement_time=settlement_time,
                underlying_as_of=result.as_of,
                spot=float(result.spot),
                time_years=time_years,
                days_to_expiry=days,
                risk_free_rate=rate,
                dividend_yield=dividend_yield,
                forward_price=forward,
                provider_name=result.provider_name,
                contract_symbol=symbol,
                bid=_optional_positive(row.get("bid")),
                ask=_optional_positive(row.get("ask")),
                last_price=_optional_positive(row.get("lastPrice")),
                selected_mark=selected_mark,
                mark_source=mark_source,
                quote_quality=float(row.get("quote_quality", 0.0)),
                vendor_implied_volatility=_optional_positive(
                    row.get("impliedVolatility")
                ),
                selected_implied_volatility=selected_iv,
                implied_volatility_source=iv_source,
                legacy_implied_volatility=_optional_positive(
                    row.get("legacy_iv", row.get("legacy_input_iv"))
                ),
                candidate_implied_volatility=_optional_positive(
                    row.get("candidate_iv", row.get("candidate_input_iv"))
                ),
                surface_source=surface_source,
                is_surface_fitted=is_surface_fitted,
                open_interest=max(int(float(row.get("openInterest", 0.0))), 0),
                volume=max(int(float(row.get("volume", 0.0))), 0),
                liquidity=max(float(row.get("liquidity", 0.0)), 0.0),
                liquidity_source=str(row.get("liquidity_source", "unknown")),
                polarity=float(row.get("polarity", 1.0)),
                contract_multiplier=int(contract_multiplier),
                greeks=greek_values,
                gamma_exposure_1pct=float(row["gex"]),
            )
        )

    return CanonicalContractLedger(
        ticker=result.ticker,
        snapshot_id=result.snapshot_id,
        valuation_time=valuation_time,
        states=states,
    )


def _as_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def _optional_positive(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) and parsed > 0.0 else None


def _optional_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None
