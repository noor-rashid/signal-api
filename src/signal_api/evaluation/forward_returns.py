"""Compute forward returns as prediction targets."""

import pandas as pd


def compute_forward_returns(
    df: pd.DataFrame,
    horizons: list[int] | None = None,
    price_col: str = "close",
) -> pd.DataFrame:
    """Add forward return columns for each horizon.

    Forward return at time t = (close[t+h] - close[t]) / close[t].
    Last h rows will be NaN (no lookahead).

    Args:
        df: DataFrame with a price column.
        horizons: List of forward periods in rows (e.g. [1, 4, 24] for 1h, 4h, 24h).
        price_col: Column to compute returns from.

    Returns:
        DataFrame with added fwd_ret_{h}h columns.
    """
    if horizons is None:
        horizons = [1, 4, 24]

    result = df.copy()
    for h in horizons:
        result[f"fwd_ret_{h}h"] = (
            result[price_col].shift(-h) / result[price_col] - 1
        )
    return result
