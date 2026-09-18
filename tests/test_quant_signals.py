"""Math-correctness tests for quant signals.

Every expected value here is hand-computed from a small synthetic dataset —
these tests verify the MATH, not just that the functions run.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mcp_servers.signals import (
    compute_correlation_matrix,
    compute_momentum,
    compute_volatility,
    compute_zscore,
)


# --------------------------------------------------------------------------
# Helpers to build synthetic series
# --------------------------------------------------------------------------
def series(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype=float)


# --------------------------------------------------------------------------
# Momentum
# --------------------------------------------------------------------------
class TestMomentum:
    def test_simple_positive_momentum(self) -> None:
        # Prices grow 1.0 -> 1.2 over 20 steps: (1.2/1.0) - 1 = 0.2 exactly.
        prices = series(np.linspace(1.0, 1.2, 21).tolist())
        res = compute_momentum(prices, lookback_days=20)
        assert res.value == pytest.approx(0.2, abs=1e-12)
        assert res.n_observations == 20
        assert res.metadata["start_price"] == pytest.approx(1.0)
        assert res.metadata["end_price"] == pytest.approx(1.2)

    def test_negative_momentum(self) -> None:
        # 100 -> 90 over 30 days: 90/100 - 1 = -0.10
        prices = series(np.linspace(100.0, 90.0, 31).tolist())
        res = compute_momentum(prices, lookback_days=30)
        assert res.value == pytest.approx(-0.10, abs=1e-12)

    def test_skip_days_excludes_recent_bar(self) -> None:
        # 20-day momentum measured 5 days ago: P(t-5)/P(t-25) - 1.
        # Build: days 0..25 linear 1.0 -> 1.25 (so P(t-25)=1.0, P(t-5)=1.20).
        prices = series(np.linspace(1.0, 1.25, 26).tolist())
        res = compute_momentum(prices, lookback_days=20, skip_days=5)
        assert res.value == pytest.approx(1.20 / 1.0 - 1.0, abs=1e-12)

    def test_insufficient_data_raises(self) -> None:
        with pytest.raises(ValueError, match="insufficient data"):
            compute_momentum(series([1.0, 2.0, 3.0]), lookback_days=10)

    def test_nan_values_are_dropped_before_check(self) -> None:
        clean = [1.0, 1.1, 1.2, 1.3]
        dirty = clean[:2] + [float("nan")] * 3 + clean[2:]
        res = compute_momentum(series(dirty), lookback_days=3)
        assert res.value == pytest.approx(1.3 / 1.0 - 1.0)

    def test_constant_price_series_has_zero_momentum(self) -> None:
        res = compute_momentum(series([50.0] * 21), lookback_days=20)
        assert res.value == pytest.approx(0.0, abs=1e-15)


# --------------------------------------------------------------------------
# Volatility
# --------------------------------------------------------------------------
class TestVolatility:
    def test_hand_computed_realized_vol(self) -> None:
        # Daily log returns alternate +1% / -1% for 5 days (window=5).
        # mean = 0.002; deviations: {0.008, -0.012, 0.008, -0.012, 0.008}
        # sample variance (ddof=1) = (3*0.008^2 + 2*0.012^2)/4 = 0.00012
        # daily std = sqrt(0.00012) = 0.0109544511501...
        # annualized = daily_std * sqrt(252)
        daily_std = math.sqrt((3 * 0.008**2 + 2 * 0.012**2) / 4)
        prices = [100.0]
        for r in [0.01, -0.01, 0.01, -0.01, 0.01]:
            prices.append(prices[-1] * math.exp(r))
        res = compute_volatility(series(prices), window=5)
        assert res.value == pytest.approx(daily_std * math.sqrt(252), rel=1e-9)
        assert res.metadata["daily_std"] == pytest.approx(daily_std, rel=1e-9)
        assert res.n_observations == 5

    def test_flat_series_has_zero_vol(self) -> None:
        res = compute_volatility(series([100.0] * 31), window=30)
        assert res.value == pytest.approx(0.0, abs=1e-15)

    def test_vol_scales_with_sqrt_of_period(self) -> None:
        # Same return series, monthly periods (12/yr) vs daily (252/yr):
        # ratio must be sqrt(252/12).
        prices = [100.0]
        for r in [0.01, -0.01, 0.01, -0.01, 0.01]:
            prices.append(prices[-1] * math.exp(r))
        daily = compute_volatility(series(prices), window=5, periods_per_year=252)
        monthly = compute_volatility(series(prices), window=5, periods_per_year=12)
        assert daily.value / monthly.value == pytest.approx(math.sqrt(252 / 12), rel=1e-9)

    def test_insufficient_data_raises(self) -> None:
        with pytest.raises(ValueError, match="insufficient data"):
            compute_volatility(series([100.0, 101.0]), window=20)


# --------------------------------------------------------------------------
# Z-score
# --------------------------------------------------------------------------
class TestZScore:
    def test_hand_computed_zscore(self) -> None:
        # Hist = [2, 4, 6, 8]: mean=5, sample std (ddof=1) = sqrt(20/3)
        # x=8 -> z = (8-5)/sqrt(20/3) = 3/2.5819... = 1.1619...
        hist = series([2.0, 4.0, 6.0, 8.0])
        res = compute_zscore(8.0, hist)
        expected = 3.0 / math.sqrt(20.0 / 3.0)
        assert res.value == pytest.approx(expected, rel=1e-9)
        assert res.n_observations == 4

    def test_zscore_negative(self) -> None:
        hist = series([2.0, 4.0, 6.0, 8.0])
        res = compute_zscore(2.0, hist)
        assert res.value == pytest.approx(-3.0 / math.sqrt(20.0 / 3.0), rel=1e-9)

    def test_at_mean_is_zero(self) -> None:
        hist = series([1.0, 2.0, 3.0])
        assert compute_zscore(2.0, hist).value == pytest.approx(0.0, abs=1e-12)

    def test_zero_variance_at_mean(self) -> None:
        res = compute_zscore(5.0, series([5.0, 5.0, 5.0]))
        assert res.value == 0.0

    def test_zero_variance_off_mean_raises(self) -> None:
        with pytest.raises(ValueError, match="zero variance"):
            compute_zscore(6.0, series([5.0, 5.0, 5.0]))

    def test_too_few_observations_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            compute_zscore(1.0, series([1.0]))


# --------------------------------------------------------------------------
# Correlation matrix
# --------------------------------------------------------------------------
class TestCorrelationMatrix:
    def test_perfect_positive_correlation(self) -> None:
        a = series([100.0, 101.0, 102.0, 103.0, 104.0])
        b = a * 2.0  # identical returns -> perfectly correlated
        out = compute_correlation_matrix(["A", "B"], {"A": a, "B": b}, min_overlap=3)
        assert out["tickers"] == ["A", "B"]
        assert out["matrix"][0][1] == pytest.approx(1.0, abs=1e-12)
        assert out["matrix"][1][0] == pytest.approx(1.0, abs=1e-12)
        assert out["matrix"][0][0] == 1.0
        assert out["excluded"] == []

    def test_perfect_negative_correlation(self) -> None:
        # B's daily log returns are the exact negation of A's -> corr = -1.
        # (Prices moving opposite over TIME is not enough: the RETURN series
        # must mirror each other; note ln(1.01) != -ln(0.99).)
        a_vals = np.array([100.0, 101.0, 103.02, 106.111056])
        b_vals = 200.0 * np.cumprod(np.concatenate([[1.0], a_vals[:-1] / a_vals[1:]]))
        out = compute_correlation_matrix(
            ["A", "B"], {"A": series(a_vals.tolist()), "B": series(b_vals.tolist())}, min_overlap=2
        )
        assert out["matrix"][0][1] == pytest.approx(-1.0, abs=1e-9)

    def test_missing_ticker_is_excluded_not_fabricated(self) -> None:
        a = series([100.0, 101.0, 102.0, 103.0])
        out = compute_correlation_matrix(["A", "MISSING"], {"A": a})
        assert out["excluded"] == ["MISSING"]
        assert out["tickers"] == ["A"]

    def test_insufficient_overlap_is_null(self) -> None:
        a = series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0])
        b = series([50.0, 51.0, 52.0])  # only 1 overlapping return -> < min_overlap
        out = compute_correlation_matrix(["A", "B"], {"A": a, "B": b}, min_overlap=5)
        assert out["matrix"][0][1] is None

    def test_hand_computed_correlation(self) -> None:
        # Returns of A: [+0.01, +0.01, +0.01]; B: [+0.02, 0.0, -0.02].
        # corr is negative. Compute exactly:
        # A centered: [.00667, .00667, .00667] (all equal -> wait: mean .01, devs 0)
        # A devs are all 0 -> std=0 -> corr undefined -> NaN from pandas.
        # Use a non-degenerate A instead: [+0.01, +0.02, +0.03] vs B [+0.03, +0.01, -0.01]
        a = series([100.0, 101.0, 103.0, 106.09])  # returns .01,.019803..,.029126..
        b = series([200.0, 206.0, 208.0, 208.0])  # returns .03,.009709.., 0.0
        out = compute_correlation_matrix(["A", "B"], {"A": a, "B": b}, min_overlap=2)
        ra = np.diff(np.log(np.array([100.0, 101.0, 103.0, 106.09])))
        rb = np.diff(np.log(np.array([200.0, 206.0, 208.0, 208.0])))
        expected = float(np.corrcoef(ra, rb)[0, 1])
        assert out["matrix"][0][1] == pytest.approx(expected, rel=1e-9)

    def test_empty_input(self) -> None:
        out = compute_correlation_matrix([], {})
        assert out["tickers"] == [] and out["matrix"] == []
