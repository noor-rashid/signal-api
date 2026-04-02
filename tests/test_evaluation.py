"""Tests for signal evaluation framework."""

import numpy as np
import pandas as pd

from signal_api.evaluation.forward_returns import compute_forward_returns
from signal_api.evaluation.signal_tester import (
    hit_rate,
    ic_p_value,
    information_coefficient,
    quantile_returns,
)


def test_forward_returns_basic():
    df = pd.DataFrame({"close": [100.0, 102.0, 101.0, 105.0, 103.0]})
    result = compute_forward_returns(df, horizons=[1, 2])
    assert "fwd_ret_1h" in result.columns
    assert "fwd_ret_2h" in result.columns
    # fwd_ret_1h at index 0 = (102 - 100) / 100 = 0.02
    assert abs(result["fwd_ret_1h"].iloc[0] - 0.02) < 1e-10
    # Last row should be NaN
    assert pd.isna(result["fwd_ret_1h"].iloc[-1])


def test_forward_returns_no_lookahead():
    """Verify forward returns don't use future data."""
    df = pd.DataFrame({"close": list(range(1, 101))}, dtype=float)
    result = compute_forward_returns(df, horizons=[5])
    # Last 5 rows should be NaN
    assert result["fwd_ret_5h"].iloc[-5:].isna().all()
    # First row should not be NaN
    assert pd.notna(result["fwd_ret_5h"].iloc[0])


def test_ic_perfect_signal():
    """A perfect signal should have IC close to 1."""
    rng = np.random.default_rng(42)
    n = 1000
    signal = pd.Series(rng.normal(0, 1, n))
    # Forward return perfectly correlated with signal
    fwd_ret = signal * 0.5 + rng.normal(0, 0.01, n)
    ic = information_coefficient(signal, fwd_ret)
    assert ic > 0.95


def test_ic_random_signal():
    """Random noise should have IC close to 0."""
    rng = np.random.default_rng(42)
    n = 5000
    signal = pd.Series(rng.normal(0, 1, n))
    fwd_ret = pd.Series(rng.normal(0, 1, n))
    ic = information_coefficient(signal, fwd_ret)
    assert abs(ic) < 0.05


def test_ic_p_value_significant():
    """Strong signal should have low p-value."""
    rng = np.random.default_rng(42)
    n = 1000
    signal = pd.Series(rng.normal(0, 1, n))
    fwd_ret = signal * 0.3 + rng.normal(0, 0.5, n)
    ic = information_coefficient(signal, fwd_ret)
    p = ic_p_value(ic, n)
    assert p < 0.001


def test_hit_rate_perfect():
    signal = pd.Series([1.0, -1.0, 1.0, -1.0, 1.0] * 20)
    fwd_ret = pd.Series([0.5, -0.3, 0.2, -0.1, 0.4] * 20)
    hr = hit_rate(signal, fwd_ret)
    assert hr == 1.0


def test_hit_rate_random():
    rng = np.random.default_rng(42)
    n = 10000
    signal = pd.Series(rng.normal(0, 1, n))
    fwd_ret = pd.Series(rng.normal(0, 1, n))
    hr = hit_rate(signal, fwd_ret)
    assert 0.45 < hr < 0.55  # Should be close to 50%


def test_quantile_returns_shape():
    rng = np.random.default_rng(42)
    n = 1000
    signal = pd.Series(rng.normal(0, 1, n))
    fwd_ret = pd.Series(rng.normal(0, 1, n))
    qr = quantile_returns(signal, fwd_ret, n_quantiles=5)
    assert len(qr) == 5
    assert "mean" in qr.columns
    assert "count" in qr.columns


def test_quantile_returns_monotonic_with_signal():
    """Strong signal should produce monotonic quantile returns."""
    rng = np.random.default_rng(42)
    n = 5000
    signal = pd.Series(rng.normal(0, 1, n))
    fwd_ret = signal * 0.3 + rng.normal(0, 0.5, n)
    qr = quantile_returns(signal, fwd_ret, n_quantiles=5)
    # Q5 mean should be > Q1 mean
    assert qr["mean"].iloc[-1] > qr["mean"].iloc[0]
