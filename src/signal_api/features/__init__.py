"""Feature engineering for crypto signal detection."""

import logging

import pandas as pd

from signal_api.features.base import Feature
from signal_api.features.derivatives import (
    FundingRateMomentum,
    FundingRateZScore,
    LongShortRatioExtremes,
    OIPriceDivergence,
    OIZScoreChange,
    VolumeWeightedOIChange,
)
from signal_api.features.spot import TakerBuyRatio, TradeIntensity

logger = logging.getLogger(__name__)

ALL_SPOT_FEATURES: list[Feature] = [
    TakerBuyRatio(),
    TradeIntensity(),
]

ALL_DERIVATIVES_FEATURES: list[Feature] = [
    OIZScoreChange(),
    FundingRateZScore(),
    OIPriceDivergence(),
    FundingRateMomentum(),
    LongShortRatioExtremes(),
    VolumeWeightedOIChange(),
]


def compute_spot_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all spot-based features and return augmented DataFrame."""
    result = df.copy()
    for feature in ALL_SPOT_FEATURES:
        result[feature.name] = feature.compute(df)
    return result


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features (spot + derivatives) on a merged DataFrame."""
    result = df.copy()

    for feature in ALL_SPOT_FEATURES:
        result[feature.name] = feature.compute(result)

    for feature in ALL_DERIVATIVES_FEATURES:
        try:
            feature.validate_input(result)
            result[feature.name] = feature.compute(result)
        except ValueError:
            logger.debug(f"Skipping {feature.name}: missing required columns")

    return result


def build_feature_matrix(
    spot_df: pd.DataFrame,
    oi_df: pd.DataFrame | None = None,
    funding_df: pd.DataFrame | None = None,
    ls_df: pd.DataFrame | None = None,
    taker_futures_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge spot + futures data and compute all features.

    All data is aligned to the spot 1h candle grid using merge_asof.
    """
    result = spot_df.copy().sort_values("open_time").reset_index(drop=True)

    # Merge futures data if provided
    for futures_df, prefix in [
        (oi_df, "oi_"),
        (funding_df, "fr_"),
        (ls_df, "ls_"),
        (taker_futures_df, "tk_"),
    ]:
        if futures_df is not None and not futures_df.empty:
            fdf = futures_df.copy().sort_values("timestamp")
            # Rename columns with prefix to avoid collisions
            rename_cols = {
                c: f"{prefix}{c}"
                for c in fdf.columns
                if c not in ("timestamp", "symbol")
            }
            fdf = fdf.rename(columns=rename_cols)
            result = pd.merge_asof(
                result,
                fdf.drop(columns=["symbol"], errors="ignore"),
                left_on="open_time",
                right_on="timestamp",
                direction="backward",
            )
            # Drop the timestamp column to avoid collisions on next merge
            result = result.drop(columns=["timestamp"], errors="ignore")

    # Compute all features
    result = compute_all_features(result)

    return result
