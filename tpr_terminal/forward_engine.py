from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .math_engine import clean_option_quotes


@dataclass(frozen=True, slots=True)
class ForwardEstimate:
    forward: float
    dividend_yield: float
    pairs_used: int
    dispersion_bps: float
    method: str
    reliable: bool


def estimate_forward_from_put_call_parity(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    *,
    spot: float,
    time_years: float,
    risk_free_rate: float,
) -> ForwardEstimate | None:
    if calls is None or puts is None or calls.empty or puts.empty or time_years <= 0:
        return None
    call_quotes = clean_option_quotes(calls)
    put_quotes = clean_option_quotes(puts)
    call_quotes["strike"] = pd.to_numeric(call_quotes["strike"], errors="coerce")
    put_quotes["strike"] = pd.to_numeric(put_quotes["strike"], errors="coerce")
    merged = call_quotes[["strike", "market_price", "quote_quality"]].merge(
        put_quotes[["strike", "market_price", "quote_quality"]],
        on="strike",
        suffixes=("_call", "_put"),
    )
    merged = merged.dropna(subset=["strike", "market_price_call", "market_price_put"])
    if merged.empty:
        return None
    merged = merged[
        merged["strike"].between(spot * 0.85, spot * 1.15)
        & (merged["market_price_call"] > 0)
        & (merged["market_price_put"] > 0)
    ].copy()
    if merged.empty:
        return None

    discount = float(np.exp(-risk_free_rate * time_years))
    merged["forward"] = merged["strike"] + (
        merged["market_price_call"] - merged["market_price_put"]
    ) / discount
    merged["weight"] = (
        merged["quote_quality_call"].clip(0.05, 1.0)
        * merged["quote_quality_put"].clip(0.05, 1.0)
        / (1.0 + 8.0 * np.abs(np.log(merged["strike"] / spot)))
    )
    merged = merged[
        np.isfinite(merged["forward"])
        & merged["forward"].between(spot * 0.70, spot * 1.30)
        & (merged["weight"] > 0)
    ]
    if merged.empty:
        return None

    values = merged["forward"].to_numpy(dtype=float)
    weights = merged["weight"].to_numpy(dtype=float)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) / np.sum(weights)
    forward = float(values[np.searchsorted(cumulative, 0.5)])
    absolute_deviation = np.abs(values - forward)
    dispersion = float(np.median(absolute_deviation) / max(forward, 1e-9) * 10_000.0)
    dividend_yield = float(risk_free_rate - np.log(forward / spot) / time_years)
    dividend_yield = float(np.clip(dividend_yield, -0.15, 0.50))
    pairs = int(len(values))
    reliable = pairs >= 3 and dispersion <= 100.0
    return ForwardEstimate(
        forward=forward,
        dividend_yield=dividend_yield,
        pairs_used=pairs,
        dispersion_bps=dispersion,
        method="put-call parity weighted median",
        reliable=reliable,
    )
