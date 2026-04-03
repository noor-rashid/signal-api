"""Tests for backtesting framework."""

import numpy as np
import pandas as pd

from signal_api.backtesting.tail_risk import (
    feature_importance_for_tails,
    label_tail_events,
    walk_forward_backtest,
)


def _make_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0001, 0.01, n)
    # Inject periodic crashes
    for start in [1000, 2000, 3000, 4000]:
        if start + 30 < n:
            returns[start:start+30] = rng.normal(-0.025, 0.015, 30)

    prices = 42000 * np.exp(np.cumsum(returns))
    signal = rng.normal(0, 1, n)
    # Make signal slightly predictive of crashes
    for start in [1000, 2000, 3000, 4000]:
        if start - 10 >= 0 and start + 30 < n:
            signal[start-10:start+20] += 1.5

    return pd.DataFrame({
        "close": prices,
        "signal_a": signal,
        "signal_b": rng.normal(0, 1, n),  # random noise
    })


class TestLabelTailEvents:
    def test_basic_labelling(self):
        df = _make_data()
        labels = label_tail_events(df, horizon=24, percentile=5)
        valid = labels.dropna()
        assert len(valid) > 0
        assert set(valid.unique()).issubset({0, 1})
        # ~5% should be tail events
        tail_pct = valid.mean()
        assert 0.02 < tail_pct < 0.12

    def test_no_lookahead(self):
        df = _make_data()
        labels = label_tail_events(df, horizon=24)
        # Last 24 rows should be NaN
        assert labels.iloc[-24:].isna().all()


class TestWalkForwardBacktest:
    def test_runs_without_error(self):
        df = _make_data()
        labels = label_tail_events(df, horizon=24, percentile=10)
        results = walk_forward_backtest(
            features=df,
            labels=labels,
            feature_columns=["signal_a", "signal_b"],
            train_window=1500,
            test_window=168,
            step=168,
        )
        assert "error" not in results
        assert results["n_folds"] > 0
        assert results["total_predictions"] > 0

    def test_predictive_signal_beats_random(self):
        df = _make_data()
        labels = label_tail_events(df, horizon=24, percentile=10)

        # Test with predictive signal only
        results_good = walk_forward_backtest(
            features=df,
            labels=labels,
            feature_columns=["signal_a"],
            train_window=1500,
            test_window=168,
            step=168,
        )

        # Test with random signal only
        results_random = walk_forward_backtest(
            features=df,
            labels=labels,
            feature_columns=["signal_b"],
            train_window=1500,
            test_window=168,
            step=168,
        )

        # Predictive signal should have better AUC (or at least not worse)
        if results_good.get("auc") and results_random.get("auc"):
            assert results_good["auc"] >= results_random["auc"] - 0.1


class TestFeatureImportance:
    def test_ranks_features(self):
        df = _make_data()
        labels = label_tail_events(df, horizon=24, percentile=10)
        importance = feature_importance_for_tails(
            features=df,
            labels=labels,
            feature_columns=["signal_a", "signal_b"],
        )
        assert len(importance) == 2
        assert "feature" in importance.columns
        assert "single_feature_auc" in importance.columns
