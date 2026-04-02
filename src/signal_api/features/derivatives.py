"""Features derived from Binance Futures derivatives data."""

import numpy as np
import pandas as pd

from signal_api.features.base import Feature


class OIZScoreChange(Feature):
    """Z-score of Open Interest percentage changes.

    Sudden OI spikes relative to recent history indicate new positions
    entering the market, often preceding volatility or reversals.
    """

    name = "oi_zscore_change"
    required_columns = ["oi_sumOpenInterest"]
    lookback_periods = 168

    def compute(self, df: pd.DataFrame, window: int = 168) -> pd.Series:
        self.validate_input(df)
        oi = df["oi_sumOpenInterest"]
        oi_pct = oi.pct_change()
        mean = oi_pct.rolling(window, min_periods=window // 2).mean()
        std = oi_pct.rolling(window, min_periods=window // 2).std()
        zscore = (oi_pct - mean) / std.replace(0, float("nan"))
        return zscore.rename(self.name)


class FundingRateZScore(Feature):
    """Z-score of funding rate over a rolling window.

    Extreme funding = overcrowded positioning = mean reversion.
    Positive funding means longs pay shorts (bullish crowding).
    """

    name = "funding_rate_zscore"
    required_columns = ["fr_fundingRate"]
    lookback_periods = 168

    def compute(self, df: pd.DataFrame, window: int = 168) -> pd.Series:
        self.validate_input(df)
        rate = df["fr_fundingRate"]
        mean = rate.rolling(window, min_periods=window // 2).mean()
        std = rate.rolling(window, min_periods=window // 2).std()
        zscore = (rate - mean) / std.replace(0, float("nan"))
        return zscore.rename(self.name)


class OIPriceDivergence(Feature):
    """Divergence between OI direction and price direction.

    OI rising + price falling = shorts entering aggressively (bearish pressure).
    OI rising + price rising = longs entering (bullish, but fragile if overcrowded).

    Quantified as: OI_change_zscore * -sign(price_change).
    Positive = divergence (OI up, price down or vice versa).
    """

    name = "oi_price_divergence"
    required_columns = ["oi_sumOpenInterest", "close"]
    lookback_periods = 24

    def compute(self, df: pd.DataFrame, period: int = 4) -> pd.Series:
        self.validate_input(df)
        oi_change = df["oi_sumOpenInterest"].pct_change(period)
        price_change = df["close"].pct_change(period)

        # Normalize OI change to z-score
        oi_mean = oi_change.rolling(168, min_periods=84).mean()
        oi_std = oi_change.rolling(168, min_periods=84).std()
        oi_z = (oi_change - oi_mean) / oi_std.replace(0, float("nan"))

        divergence = oi_z * -np.sign(price_change)
        return divergence.rename(self.name)


class FundingRateMomentum(Feature):
    """Rate of change of funding rate.

    Rapidly rising funding = leverage building quickly = fragile market.
    Computed across multiple funding periods (8h each).
    """

    name = "funding_rate_momentum"
    required_columns = ["fr_fundingRate"]
    lookback_periods = 168

    def compute(self, df: pd.DataFrame, period: int = 24) -> pd.Series:
        self.validate_input(df)
        rate = df["fr_fundingRate"]
        momentum = rate.diff(period)
        mean = momentum.rolling(168, min_periods=84).mean()
        std = momentum.rolling(168, min_periods=84).std()
        zscore = (momentum - mean) / std.replace(0, float("nan"))
        return zscore.rename(self.name)


class LongShortRatioExtremes(Feature):
    """Z-score of long/short account ratio — contrarian signal.

    When the ratio is extreme (everyone long), price tends to reverse.
    """

    name = "ls_ratio_zscore"
    required_columns = ["ls_longShortRatio"]
    lookback_periods = 168

    def compute(self, df: pd.DataFrame, window: int = 168) -> pd.Series:
        self.validate_input(df)
        ratio = df["ls_longShortRatio"]
        mean = ratio.rolling(window, min_periods=window // 2).mean()
        std = ratio.rolling(window, min_periods=window // 2).std()
        zscore = (ratio - mean) / std.replace(0, float("nan"))
        return zscore.rename(self.name)


class VolumeWeightedOIChange(Feature):
    """OI change per unit volume — disentangles positioning from churn.

    High values = genuine new positioning. Low values = high-volume churn
    without meaningful position changes.
    """

    name = "volume_weighted_oi_change"
    required_columns = ["oi_sumOpenInterest", "volume"]
    lookback_periods = 168

    def compute(self, df: pd.DataFrame, window: int = 168) -> pd.Series:
        self.validate_input(df)
        oi_change = df["oi_sumOpenInterest"].diff().abs()
        ratio = oi_change / df["volume"].replace(0, float("nan"))
        mean = ratio.rolling(window, min_periods=window // 2).mean()
        std = ratio.rolling(window, min_periods=window // 2).std()
        zscore = (ratio - mean) / std.replace(0, float("nan"))
        return zscore.rename(self.name)
