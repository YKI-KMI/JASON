"""Pure quant signal functions (numpy/pandas only — no I/O, no MCP here).

Kept separate from the MCP server wrapper so the math is trivially unit-testable
against hand-computed expected values, and reusable outside MCP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SignalResult:
    """A single computed signal with sample size, for calibrated language."""

    name: str
    value: float
    n_observations: int
    metadata: dict | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "n_observations": self.n_observations,
            "metadata": self.metadata or {},
        }


def compute_momentum(
    prices: pd.Series,
    lookback_days: int,
    skip_days: int = 0,
) -> SignalResult:
    """Standard momentum: total return over `lookback_days` ending `skip_days` ago.

    Momentum research convention (Jegadeesh & Titman 1993) skips the most
    recent `skip_days` to avoid short-term reversal effects. With skip_days=0
    this is simply P_t / P_{t-lookback} - 1.

    Raises ValueError on insufficient data (caller surfaces 'no data' honestly).
    """
    prices = _clean_series(prices)
    need = lookback_days + skip_days + 1
    if len(prices) < need:
        raise ValueError(
            f"insufficient data: need at least {need} price points for "
            f"lookback_days={lookback_days}, skip_days={skip_days}; got {len(prices)}"
        )
    p_now = prices.iloc[-1 - skip_days]
    p_then = prices.iloc[-1 - skip_days - lookback_days]
    momentum = p_now / p_then - 1.0
    return SignalResult(
        name="momentum",
        value=float(momentum),
        n_observations=lookback_days,
        metadata={
            "definition": f"P(t)/P(t-{lookback_days}) - 1",
            "skip_days": skip_days,
            "start_price": float(p_then),
            "end_price": float(p_now),
        },
    )


def compute_volatility(
    prices: pd.Series,
    window: int = 20,
    periods_per_year: int = 252,
) -> SignalResult:
    """Annualized realized volatility = std(daily log returns) * sqrt(252).

    Computed over the last `window` daily returns. Raises ValueError if the
    series is too short.
    """
    prices = _clean_series(prices)
    if len(prices) < window + 1:
        raise ValueError(
            f"insufficient data: need at least {window + 1} price points for "
            f"window={window}; got {len(prices)}"
        )
    log_returns = np.log(prices / prices.shift(1)).dropna()
    recent = log_returns.iloc[-window:]
    std = float(recent.std(ddof=1))
    annualized = std * math.sqrt(periods_per_year)
    return SignalResult(
        name="realized_volatility",
        value=float(annualized),
        n_observations=window,
        metadata={
            "definition": "std(daily log returns, ddof=1) * sqrt(252)",
            "daily_std": std,
            "periods_per_year": periods_per_year,
        },
    )


def compute_zscore(current_value: float, historical_series: pd.Series) -> SignalResult:
    """Statistical z-score of current value vs. a historical sample.

    z = (x - mean(hist)) / std(hist, ddof=1). If historical std is 0 the result
    is 0.0 when x equals the mean, otherwise undefined -> raises ValueError
    (honest failure rather than an invented number).
    """
    hist = _clean_series(historical_series).astype(float)
    if len(hist) < 2:
        raise ValueError("z-score needs at least 2 historical observations")
    mean = float(hist.mean())
    std = float(hist.std(ddof=1))
    if std == 0.0:
        if current_value == mean:
            return SignalResult("zscore", 0.0, len(hist), {"note": "zero variance, x == mean"})
        raise ValueError("z-score undefined: zero variance and x != mean")
    z = (current_value - mean) / std
    return SignalResult(
        name="zscore",
        value=float(z),
        n_observations=len(hist),
        metadata={"mean": mean, "std": std, "current_value": float(current_value)},
    )


def compute_correlation_matrix(
    tickers: list[str],
    price_data: dict[str, pd.Series],
    min_overlap: int = 10,
) -> dict:
    """Pairwise Pearson correlations of daily log returns over overlapping dates.

    Tickers with missing data are excluded and reported; pairs with fewer than
    `min_overlap` shared dates get NaN (reported as None in JSON) — never a
    fabricated number.
    """
    if not tickers:
        return {"tickers": [], "matrix": [], "excluded": [], "min_overlap": min_overlap}

    returns: dict[str, pd.Series] = {}
    for t in tickers:
        series = price_data.get(t)
        if series is None or len(series) < 2:
            continue
        returns[t] = np.log(series / series.shift(1)).dropna()

    excluded = [t for t in tickers if t not in returns]
    kept = [t for t in tickers if t in returns]

    matrix: list[list[float | None]] = []
    for a in kept:
        row: list[float | None] = []
        for b in kept:
            if a == b:
                row.append(1.0)
                continue
            ra, rb = returns[a], returns[b]
            joined = pd.concat([ra, rb], axis=1, join="inner").dropna()
            joined.columns = ["a", "b"]
            if len(joined) < min_overlap:
                row.append(None)
            else:
                row.append(float(joined["a"].corr(joined["b"])))
        matrix.append(row)

    return {
        "tickers": kept,
        "matrix": matrix,
        "excluded": excluded,
        "min_overlap": min_overlap,
    }


def _clean_series(series: pd.Series) -> pd.Series:
    """Drop NaNs, coerce numeric, sort by index if dates. Raises on empty."""
    s = pd.Series(series).dropna()
    s = pd.to_numeric(s, errors="coerce").dropna()
    if isinstance(s.index, pd.DatetimeIndex):
        s = s.sort_index()
    if len(s) == 0:
        raise ValueError("no valid numeric observations in series")
    return s
