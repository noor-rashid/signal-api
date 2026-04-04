"""Tests for model training and serving."""

import numpy as np
import pandas as pd
import pytest

from signal_api.backtesting.tail_risk import label_tail_events
from signal_api.models.train import VALIDATED_FEATURES, _evaluate_model


def _make_synthetic_data(n: int = 3000, seed: int = 42):
    """Create synthetic data with known signal."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0001, 0.01, n)
    for start in [500, 1000, 1500, 2000, 2500]:
        if start + 20 < n:
            returns[start:start+20] = rng.normal(-0.025, 0.015, 20)

    prices = 42000 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({
        "close": prices,
        "volume": rng.uniform(50, 200, n),
        "taker_buy_volume": rng.uniform(25, 150, n),
    })

    # Add mock features
    for feat in VALIDATED_FEATURES:
        df[feat] = rng.normal(0, 1, n)
        # Make some features slightly predictive of crashes
        for start in [500, 1000, 1500, 2000, 2500]:
            if start - 5 >= 0 and start + 20 < n:
                df.loc[start-5:start+15, feat] += 1.0

    return df


class TestEvaluateModel:
    def test_metrics_keys(self):
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (200, 5))
        y = (rng.random(200) > 0.9).astype(int)
        model = LogisticRegression(class_weight="balanced", max_iter=1000)
        model.fit(X, y)
        metrics = _evaluate_model(model, X, y)
        assert "auc" in metrics
        assert "precision_tail" in metrics
        assert "recall_tail" in metrics
        assert "f1_tail" in metrics
        assert "precision_at_50" in metrics


class TestLabelConsistency:
    def test_labels_are_binary(self):
        df = _make_synthetic_data()
        labels = label_tail_events(df, horizon=4, percentile=5)
        valid = labels.dropna()
        assert set(valid.unique()).issubset({0, 1})

    def test_tail_event_ratio(self):
        df = _make_synthetic_data()
        labels = label_tail_events(df, horizon=4, percentile=10)
        valid = labels.dropna()
        ratio = valid.mean()
        assert 0.03 < ratio < 0.20
