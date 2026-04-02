"""Tests for feature computation."""

import numpy as np
import pandas as pd

from signal_api.features.spot import TakerBuyRatio, TradeIntensity


def _make_spot_df(n: int = 500) -> pd.DataFrame:
    """Create synthetic spot data."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "open_time": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "open": 42000 + rng.normal(0, 100, n).cumsum(),
        "high": 42500 + rng.normal(0, 100, n).cumsum(),
        "low": 41800 + rng.normal(0, 100, n).cumsum(),
        "close": 42200 + rng.normal(0, 100, n).cumsum(),
        "volume": rng.uniform(50, 200, n),
        "taker_buy_volume": rng.uniform(25, 150, n),
        "trades": rng.integers(100, 1000, n),
        "symbol": "BTCUSDT",
    })


class TestTakerBuyRatio:
    def test_output_shape(self):
        df = _make_spot_df()
        feature = TakerBuyRatio()
        result = feature.compute(df)
        assert len(result) == len(df)
        assert result.name == "taker_buy_ratio_zscore"

    def test_is_zscore_like(self):
        df = _make_spot_df(1000)
        feature = TakerBuyRatio()
        result = feature.compute(df).dropna()
        # Z-scores should have mean ~0 and std ~1
        assert abs(result.mean()) < 0.5
        assert 0.5 < result.std() < 2.0

    def test_missing_column_raises(self):
        df = pd.DataFrame({"open": [1, 2, 3]})
        feature = TakerBuyRatio()
        try:
            feature.compute(df)
            assert False, "Should have raised"
        except ValueError as e:
            assert "taker_buy_volume" in str(e)


class TestTradeIntensity:
    def test_output_shape(self):
        df = _make_spot_df()
        feature = TradeIntensity()
        result = feature.compute(df)
        assert len(result) == len(df)
        assert result.name == "trade_intensity_zscore"

    def test_is_zscore_like(self):
        df = _make_spot_df(1000)
        feature = TradeIntensity()
        result = feature.compute(df).dropna()
        assert abs(result.mean()) < 0.5
        assert 0.5 < result.std() < 2.0
