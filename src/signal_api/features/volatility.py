"""Volatility and left-tail risk features."""

import numpy as np
import pandas as pd

from signal_api.features.base import Feature


class RealizedVolatility(Feature):
    """Rolling realized volatility (annualised std of log returns).

    Regime indicator — high vol begets high vol.
    """

    name = "realized_vol"
    required_columns = ["close"]
    lookback_periods = 168

    def compute(self, df: pd.DataFrame, window: int = 24) -> pd.Series:
        self.validate_input(df)
        log_ret = np.log(df["close"] / df["close"].shift(1))
        # Annualised: sqrt(8760) for hourly data
        vol = log_ret.rolling(window, min_periods=window // 2).std() * np.sqrt(8760)
        return vol.rename(self.name)


class VolatilityZScore(Feature):
    """Z-score of current realized vol vs its own history.

    Elevated vol z-score = market stress, higher probability of tail events.
    """

    name = "vol_zscore"
    required_columns = ["close"]
    lookback_periods = 336  # 14 days

    def compute(self, df: pd.DataFrame, fast: int = 24, slow: int = 336) -> pd.Series:
        self.validate_input(df)
        log_ret = np.log(df["close"] / df["close"].shift(1))
        fast_vol = log_ret.rolling(fast, min_periods=fast // 2).std()
        slow_mean = fast_vol.rolling(slow, min_periods=slow // 2).mean()
        slow_std = fast_vol.rolling(slow, min_periods=slow // 2).std()
        zscore = (fast_vol - slow_mean) / slow_std.replace(0, float("nan"))
        return zscore.rename(self.name)


class VolOfVol(Feature):
    """Volatility of volatility — instability of the vol regime itself.

    High vol-of-vol precedes regime shifts and tail events.
    """

    name = "vol_of_vol"
    required_columns = ["close"]
    lookback_periods = 336

    def compute(self, df: pd.DataFrame, vol_window: int = 24, vov_window: int = 168) -> pd.Series:
        self.validate_input(df)
        log_ret = np.log(df["close"] / df["close"].shift(1))
        vol = log_ret.rolling(vol_window, min_periods=vol_window // 2).std()
        vov = vol.rolling(vov_window, min_periods=vov_window // 2).std()
        # Z-score the vol-of-vol
        mean = vov.rolling(336, min_periods=168).mean()
        std = vov.rolling(336, min_periods=168).std()
        zscore = (vov - mean) / std.replace(0, float("nan"))
        return zscore.rename(self.name)


class DownsideVolRatio(Feature):
    """Ratio of downside volatility to total volatility.

    >0.5 means negative returns are more volatile than positive.
    Elevated ratio = asymmetric risk skewed to the left tail.
    """

    name = "downside_vol_ratio"
    required_columns = ["close"]
    lookback_periods = 168

    def compute(self, df: pd.DataFrame, window: int = 168) -> pd.Series:
        self.validate_input(df)
        log_ret = np.log(df["close"] / df["close"].shift(1))
        downside = log_ret.where(log_ret < 0, 0)
        down_vol = downside.rolling(window, min_periods=window // 2).std()
        total_vol = log_ret.rolling(window, min_periods=window // 2).std()
        ratio = down_vol / total_vol.replace(0, float("nan"))
        return ratio.rename(self.name)


class ReturnSkewness(Feature):
    """Rolling skewness of returns.

    Negative skew = fat left tail. Increasingly negative = crash risk building.
    """

    name = "return_skewness"
    required_columns = ["close"]
    lookback_periods = 168

    def compute(self, df: pd.DataFrame, window: int = 168) -> pd.Series:
        self.validate_input(df)
        log_ret = np.log(df["close"] / df["close"].shift(1))
        skew = log_ret.rolling(window, min_periods=window // 2).skew()
        return skew.rename(self.name)


class TailConcentration(Feature):
    """Fraction of returns in the bottom 10th percentile over rolling window.

    Rising concentration = clustering of extreme losses (tail dependence).
    """

    name = "tail_concentration"
    required_columns = ["close"]
    lookback_periods = 720  # 30 days

    def compute(self, df: pd.DataFrame, window: int = 168, percentile: float = 10) -> pd.Series:
        self.validate_input(df)
        log_ret = np.log(df["close"] / df["close"].shift(1))
        # Use expanding percentile threshold (avoids lookahead)
        threshold = log_ret.expanding(min_periods=720).quantile(percentile / 100)
        is_tail = (log_ret <= threshold).astype(float)
        concentration = is_tail.rolling(window, min_periods=window // 2).mean()
        return concentration.rename(self.name)


class FundingVolSpread(Feature):
    """Interaction: funding rate z-score * volatility z-score.

    High leverage (extreme funding) during high vol = fragile market.
    This combination is the classic crash setup in crypto.
    """

    name = "funding_vol_spread"
    required_columns = ["fr_fundingRate", "close"]
    lookback_periods = 336

    def compute(self, df: pd.DataFrame) -> pd.Series:
        self.validate_input(df)
        # Funding z-score
        rate = df["fr_fundingRate"]
        fr_mean = rate.rolling(168, min_periods=84).mean()
        fr_std = rate.rolling(168, min_periods=84).std()
        fr_z = (rate - fr_mean) / fr_std.replace(0, float("nan"))

        # Vol z-score
        log_ret = np.log(df["close"] / df["close"].shift(1))
        fast_vol = log_ret.rolling(24, min_periods=12).std()
        slow_mean = fast_vol.rolling(336, min_periods=168).mean()
        slow_std = fast_vol.rolling(336, min_periods=168).std()
        vol_z = (fast_vol - slow_mean) / slow_std.replace(0, float("nan"))

        # Interaction — both elevated = danger
        spread = fr_z.abs() * vol_z
        return spread.rename(self.name)
