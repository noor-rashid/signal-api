"""Tests for volatility and tail risk features."""

import numpy as np
import pandas as pd

from signal_api.features.volatility import (
    DownsideVolRatio,
    RealizedVolatility,
    ReturnSkewness,
    TailConcentration,
    VolatilityZScore,
    VolOfVol,
)


def _make_price_df(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0001, 0.01, n)
    # Inject a crash period (rows 1000-1050)
    returns[1000:1050] = rng.normal(-0.03, 0.02, 50)
    prices = 42000 * np.exp(np.cumsum(returns))
    return pd.DataFrame({
        "open_time": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "close": prices,
        "volume": rng.uniform(50, 200, n),
        "taker_buy_volume": rng.uniform(25, 150, n),
        "trades": rng.integers(100, 1000, n),
    })


class TestRealizedVolatility:
    def test_output_shape(self):
        df = _make_price_df()
        result = RealizedVolatility().compute(df)
        assert len(result) == len(df)
        assert result.name == "realized_vol"

    def test_vol_spikes_during_crash(self):
        df = _make_price_df()
        vol = RealizedVolatility().compute(df)
        # Vol around the crash (index ~1020) should be higher than before
        pre_crash = vol.iloc[900:950].mean()
        crash_period = vol.iloc[1020:1060].mean()
        assert crash_period > pre_crash * 1.5


class TestVolatilityZScore:
    def test_zscore_spikes_during_crash(self):
        df = _make_price_df()
        vz = VolatilityZScore().compute(df)
        # Z-score should be elevated during crash
        crash_max = vz.iloc[1020:1060].max()
        assert crash_max > 1.0


class TestVolOfVol:
    def test_output(self):
        df = _make_price_df()
        result = VolOfVol().compute(df)
        assert len(result) == len(df)
        valid = result.dropna()
        assert len(valid) > 0


class TestDownsideVolRatio:
    def test_crash_elevates_ratio(self):
        df = _make_price_df()
        ratio = DownsideVolRatio().compute(df)
        # During/after crash, downside vol ratio should be elevated
        crash_ratio = ratio.iloc[1050:1200].mean()
        assert crash_ratio > 0.5


class TestReturnSkewness:
    def test_negative_skew_during_crash(self):
        df = _make_price_df()
        skew = ReturnSkewness().compute(df)
        # Skewness should go negative around the crash period
        post_crash = skew.iloc[1050:1200].mean()
        assert post_crash < 0


class TestTailConcentration:
    def test_concentration_rises_during_crash(self):
        df = _make_price_df()
        tc = TailConcentration().compute(df)
        valid = tc.dropna()
        if len(valid) > 100:
            # After crash, tail concentration should be elevated
            post_crash = tc.iloc[1100:1200].mean()
            pre_crash = tc.iloc[800:900].mean()
            assert post_crash > pre_crash
