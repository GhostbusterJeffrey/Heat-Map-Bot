from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.special import ndtr

from .models import InstrumentType

OptionKind = Literal["call", "put"]
_SQRT_2PI = float(np.sqrt(2.0 * np.pi))


def normal_pdf(value: np.ndarray) -> np.ndarray:
    values = np.asarray(value, dtype=float)
    return np.exp(-0.5 * values * values) / _SQRT_2PI


def classify_instrument(ticker: str) -> InstrumentType:
    symbol = ticker.upper().strip()
    if symbol in {"SPX", "^SPX", "NDX", "^NDX", "RUT", "^RUT", "VIX", "^VIX"}:
        return "european_index"
    if symbol in {"SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "SLV", "XLF", "XLK"}:
        return "american_etf"
    if symbol and symbol[0].isalpha():
        return "american_equity"
    return "unknown"


def _d1_d2(
    spot: float | np.ndarray,
    strikes: np.ndarray,
    time_years: float | np.ndarray,
    volatility: np.ndarray,
    risk_free_rate: float | np.ndarray,
    dividend_yield: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    s = np.maximum(np.asarray(spot, dtype=float), 1e-12)
    k = np.maximum(np.asarray(strikes, dtype=float), 1e-12)
    t = np.maximum(np.asarray(time_years, dtype=float), 1e-10)
    vol = np.maximum(np.asarray(volatility, dtype=float), 1e-8)
    r = np.asarray(risk_free_rate, dtype=float)
    q = np.asarray(dividend_yield, dtype=float)
    sqrt_t = np.sqrt(t)
    d1 = (np.log(s / k) + (r - q + 0.5 * vol * vol) * t) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    return d1, d2


def black_scholes_price(
    option_kind: OptionKind,
    spot: float | np.ndarray,
    strikes: np.ndarray,
    time_years: float | np.ndarray,
    volatility: np.ndarray,
    risk_free_rate: float | np.ndarray,
    dividend_yield: float | np.ndarray,
) -> np.ndarray:
    s = np.asarray(spot, dtype=float)
    k = np.asarray(strikes, dtype=float)
    t = np.maximum(np.asarray(time_years, dtype=float), 1e-10)
    r = np.asarray(risk_free_rate, dtype=float)
    q = np.asarray(dividend_yield, dtype=float)
    d1, d2 = _d1_d2(s, k, t, volatility, r, q)
    df_r = np.exp(-r * t)
    df_q = np.exp(-q * t)
    if option_kind == "call":
        return s * df_q * ndtr(d1) - k * df_r * ndtr(d2)
    return k * df_r * ndtr(-d2) - s * df_q * ndtr(-d1)


def black_scholes_delta(
    option_kind: OptionKind,
    spot: float | np.ndarray,
    strikes: np.ndarray,
    time_years: float | np.ndarray,
    volatility: np.ndarray,
    risk_free_rate: float | np.ndarray,
    dividend_yield: float | np.ndarray,
) -> np.ndarray:
    t = np.maximum(np.asarray(time_years, dtype=float), 1e-10)
    q = np.asarray(dividend_yield, dtype=float)
    d1, _ = _d1_d2(spot, strikes, t, volatility, risk_free_rate, q)
    factor = np.exp(-q * t)
    return factor * ndtr(d1) if option_kind == "call" else factor * (ndtr(d1) - 1.0)


def black_scholes_gamma(
    spot: float | np.ndarray,
    strikes: np.ndarray,
    time_years: float | np.ndarray,
    volatility: np.ndarray,
    risk_free_rate: float | np.ndarray,
    dividend_yield: float | np.ndarray,
) -> np.ndarray:
    s = np.maximum(np.asarray(spot, dtype=float), 1e-12)
    t = np.maximum(np.asarray(time_years, dtype=float), 1e-10)
    vol = np.maximum(np.asarray(volatility, dtype=float), 1e-8)
    q = np.asarray(dividend_yield, dtype=float)
    d1, _ = _d1_d2(s, strikes, t, vol, risk_free_rate, q)
    return np.exp(-q * t) * normal_pdf(d1) / (s * vol * np.sqrt(t))


def black_scholes_vega(
    spot: float | np.ndarray,
    strikes: np.ndarray,
    time_years: float | np.ndarray,
    volatility: np.ndarray,
    risk_free_rate: float | np.ndarray,
    dividend_yield: float | np.ndarray,
) -> np.ndarray:
    s = np.asarray(spot, dtype=float)
    t = np.maximum(np.asarray(time_years, dtype=float), 1e-10)
    q = np.asarray(dividend_yield, dtype=float)
    d1, _ = _d1_d2(s, strikes, t, volatility, risk_free_rate, q)
    return s * np.exp(-q * t) * normal_pdf(d1) * np.sqrt(t)


def black_scholes_theta(
    option_kind: OptionKind,
    spot: float | np.ndarray,
    strikes: np.ndarray,
    time_years: float | np.ndarray,
    volatility: np.ndarray,
    risk_free_rate: float | np.ndarray,
    dividend_yield: float | np.ndarray,
) -> np.ndarray:
    """Calendar-time theta in option-value units per year.

    The returned sign follows the market convention: change in option value as
    calendar time passes, holding the other Black--Scholes inputs constant.
    Divide by 365.25 for a per-calendar-day value.
    """
    s = np.maximum(np.asarray(spot, dtype=float), 1e-12)
    k = np.maximum(np.asarray(strikes, dtype=float), 1e-12)
    t = np.maximum(np.asarray(time_years, dtype=float), 1e-10)
    vol = np.maximum(np.asarray(volatility, dtype=float), 1e-8)
    r = np.asarray(risk_free_rate, dtype=float)
    q = np.asarray(dividend_yield, dtype=float)
    d1, d2 = _d1_d2(s, k, t, vol, r, q)
    diffusion = -(s * np.exp(-q * t) * normal_pdf(d1) * vol) / (2.0 * np.sqrt(t))
    if option_kind == "call":
        return diffusion - r * k * np.exp(-r * t) * ndtr(d2) + q * s * np.exp(-q * t) * ndtr(d1)
    return diffusion + r * k * np.exp(-r * t) * ndtr(-d2) - q * s * np.exp(-q * t) * ndtr(-d1)


def black_scholes_rho(
    option_kind: OptionKind,
    spot: float | np.ndarray,
    strikes: np.ndarray,
    time_years: float | np.ndarray,
    volatility: np.ndarray,
    risk_free_rate: float | np.ndarray,
    dividend_yield: float | np.ndarray,
) -> np.ndarray:
    """Rate sensitivity per unit change in continuously compounded rate."""
    k = np.maximum(np.asarray(strikes, dtype=float), 1e-12)
    t = np.maximum(np.asarray(time_years, dtype=float), 1e-10)
    r = np.asarray(risk_free_rate, dtype=float)
    _, d2 = _d1_d2(spot, k, t, volatility, r, dividend_yield)
    if option_kind == "call":
        return k * t * np.exp(-r * t) * ndtr(d2)
    return -k * t * np.exp(-r * t) * ndtr(-d2)


def _central_difference(
    plus: np.ndarray,
    minus: np.ndarray,
    step: np.ndarray | float,
) -> np.ndarray:
    return (np.asarray(plus, dtype=float) - np.asarray(minus, dtype=float)) / (
        2.0 * np.maximum(np.asarray(step, dtype=float), 1e-12)
    )


@dataclass(frozen=True, slots=True)
class GreekStack:
    delta: np.ndarray
    gamma: np.ndarray
    theta: np.ndarray
    vega: np.ndarray
    rho: np.ndarray
    vanna: np.ndarray
    charm: np.ndarray
    vomma: np.ndarray
    speed: np.ndarray
    color: np.ndarray
    zomma: np.ndarray
    ultima: np.ndarray


def greek_stack(
    option_kind: OptionKind,
    *,
    spot: float,
    strikes: np.ndarray,
    time_years: np.ndarray,
    volatility: np.ndarray,
    risk_free_rate: np.ndarray,
    dividend_yield: np.ndarray,
    bump_scale: float = 1.0,
) -> GreekStack:
    strikes = np.asarray(strikes, dtype=float)
    t = np.maximum(np.asarray(time_years, dtype=float), 1e-8)
    vol = np.maximum(np.asarray(volatility, dtype=float), 0.01)
    r = np.asarray(risk_free_rate, dtype=float)
    q = np.asarray(dividend_yield, dtype=float)
    delta = black_scholes_delta(option_kind, spot, strikes, t, vol, r, q)
    gamma = black_scholes_gamma(spot, strikes, t, vol, r, q)
    theta = black_scholes_theta(option_kind, spot, strikes, t, vol, r, q)
    vega = black_scholes_vega(spot, strikes, t, vol, r, q)
    rho = black_scholes_rho(option_kind, spot, strikes, t, vol, r, q)

    scale = float(np.clip(bump_scale, 0.10, 10.0))
    vol_step = np.maximum(vol * 1e-3 * scale, 1e-4 * scale)
    spot_step = max(float(spot) * 1e-4 * scale, 1e-4 * scale)
    time_step = np.minimum(
        np.maximum(t * 1e-3 * scale, scale / (365.25 * 24.0)),
        t * 0.45,
    )

    delta_vol_plus = black_scholes_delta(option_kind, spot, strikes, t, vol + vol_step, r, q)
    delta_vol_minus = black_scholes_delta(
        option_kind, spot, strikes, t, np.maximum(vol - vol_step, 0.005), r, q
    )
    vanna = _central_difference(delta_vol_plus, delta_vol_minus, vol_step)

    delta_t_plus = black_scholes_delta(option_kind, spot, strikes, t + time_step, vol, r, q)
    delta_t_minus = black_scholes_delta(
        option_kind, spot, strikes, np.maximum(t - time_step, 1e-8), vol, r, q
    )
    # Charm is the change in delta as calendar time passes, i.e. -dDelta/dT.
    charm = -_central_difference(delta_t_plus, delta_t_minus, time_step)

    vega_plus = black_scholes_vega(spot, strikes, t, vol + vol_step, r, q)
    vega_minus = black_scholes_vega(
        spot, strikes, t, np.maximum(vol - vol_step, 0.005), r, q
    )
    vomma = _central_difference(vega_plus, vega_minus, vol_step)

    gamma_spot_plus = black_scholes_gamma(spot + spot_step, strikes, t, vol, r, q)
    gamma_spot_minus = black_scholes_gamma(max(spot - spot_step, 1e-6), strikes, t, vol, r, q)
    speed = _central_difference(gamma_spot_plus, gamma_spot_minus, spot_step)

    gamma_t_plus = black_scholes_gamma(spot, strikes, t + time_step, vol, r, q)
    gamma_t_minus = black_scholes_gamma(
        spot, strikes, np.maximum(t - time_step, 1e-8), vol, r, q
    )
    color = -_central_difference(gamma_t_plus, gamma_t_minus, time_step)

    gamma_vol_plus = black_scholes_gamma(spot, strikes, t, vol + vol_step, r, q)
    gamma_vol_minus = black_scholes_gamma(
        spot, strikes, t, np.maximum(vol - vol_step, 0.005), r, q
    )
    zomma = _central_difference(gamma_vol_plus, gamma_vol_minus, vol_step)

    wider_step = np.maximum(vol_step * 2.0, 2e-4)
    vega_outer_plus = black_scholes_vega(spot, strikes, t, vol + wider_step, r, q)
    vega_outer_minus = black_scholes_vega(
        spot, strikes, t, np.maximum(vol - wider_step, 0.005), r, q
    )
    vomma_plus = _central_difference(
        black_scholes_vega(spot, strikes, t, vol + wider_step + vol_step, r, q),
        black_scholes_vega(spot, strikes, t, vol + wider_step - vol_step, r, q),
        vol_step,
    )
    lower_center = np.maximum(vol - wider_step, 0.005 + vol_step)
    vomma_minus = _central_difference(
        black_scholes_vega(spot, strikes, t, lower_center + vol_step, r, q),
        black_scholes_vega(spot, strikes, t, np.maximum(lower_center - vol_step, 0.005), r, q),
        vol_step,
    )
    # Third derivative of price with respect to volatility.  The unused outer
    # vega values deliberately force the same stable volatility domain checks
    # as the nested calculation above and are retained for numerical clarity.
    _ = (vega_outer_plus, vega_outer_minus)
    ultima = _central_difference(vomma_plus, vomma_minus, wider_step)

    return GreekStack(
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        rho=rho,
        vanna=vanna,
        charm=charm,
        vomma=vomma,
        speed=speed,
        color=color,
        zomma=zomma,
        ultima=ultima,
    )


def american_binomial_price(
    option_kind: OptionKind,
    *,
    spot: float,
    strike: float,
    time_years: float,
    volatility: float,
    risk_free_rate: float,
    dividend_yield: float,
    steps: int = 200,
) -> float:
    """CRR American option price used for validation and exercise-risk diagnostics."""
    n = max(int(steps), 10)
    t = max(float(time_years), 1e-8)
    dt = t / n
    vol = max(float(volatility), 1e-6)
    up = float(np.exp(vol * np.sqrt(dt)))
    down = 1.0 / up
    growth = float(np.exp((risk_free_rate - dividend_yield) * dt))
    probability = float(np.clip((growth - down) / max(up - down, 1e-12), 0.0, 1.0))
    discount = float(np.exp(-risk_free_rate * dt))
    nodes = np.arange(n + 1)
    terminal = float(spot) * up ** (n - nodes) * down**nodes
    values = (
        np.maximum(terminal - strike, 0.0)
        if option_kind == "call"
        else np.maximum(strike - terminal, 0.0)
    )
    for step in range(n - 1, -1, -1):
        values = discount * (probability * values[:-1] + (1.0 - probability) * values[1:])
        nodes = np.arange(step + 1)
        prices = float(spot) * up ** (step - nodes) * down**nodes
        exercise = (
            np.maximum(prices - strike, 0.0)
            if option_kind == "call"
            else np.maximum(strike - prices, 0.0)
        )
        values = np.maximum(values, exercise)
    return float(values[0])
