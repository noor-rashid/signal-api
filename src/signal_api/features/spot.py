"""Features derived from spot OHLCV data (no additional API calls needed)."""

import pandas as pd

from signal_api.features.base import Feature


class TakerBuyRatio(Feature):
    """Ratio of taker buy volume to total volume, z-scored.

    High values = aggressive buying dominance.
    Extreme values often precede mean reversion.
    """

    name = "taker_buy_ratio_zscore"
    required_columns = ["taker_buy_volume", "volume"]
    lookback_periods = 168  # 7 days of hourly data

    def compute(self, df: pd.DataFrame, window: int = 168) -> pd.Series:
        self.validate_input(df)
        ratio = df["taker_buy_volume"] / df["volume"].replace(0, float("nan"))
        mean = ratio.rolling(window, min_periods=window // 2).mean()
        std = ratio.rolling(window, min_periods=window // 2).std()
        zscore = (ratio - mean) / std.replace(0, float("nan"))
        return zscore.rename(self.name)


class TradeIntensity(Feature):
    """Number of trades per unit volume — proxy for order fragmentation.

    High = many small orders (retail). Low = few large orders (whale).
    """

    name = "trade_intensity_zscore"
    required_columns = ["trades", "volume"]
    lookback_periods = 168

    def compute(self, df: pd.DataFrame, window: int = 168) -> pd.Series:
        self.validate_input(df)
        intensity = df["trades"] / df["volume"].replace(0, float("nan"))
        mean = intensity.rolling(window, min_periods=window // 2).mean()
        std = intensity.rolling(window, min_periods=window // 2).std()
        zscore = (intensity - mean) / std.replace(0, float("nan"))
        return zscore.rename(self.name)
