"""Statistical tests for signal predictive power."""

import math

import numpy as np
import pandas as pd
from scipy import stats


def information_coefficient(signal: pd.Series, forward_return: pd.Series) -> float:
    """Spearman rank correlation between signal and forward return.

    IC > 0.02 is interesting in crypto; IC > 0.05 is strong.
    """
    mask = signal.notna() & forward_return.notna()
    if mask.sum() < 30:
        return float("nan")
    corr, _ = stats.spearmanr(signal[mask], forward_return[mask])
    return float(corr)


def ic_t_statistic(ic: float, n: int) -> float:
    """T-statistic for IC significance."""
    if abs(ic) >= 1.0 or n < 3:
        return float("nan")
    return ic * math.sqrt(n - 2) / math.sqrt(1 - ic**2)


def ic_p_value(ic: float, n: int) -> float:
    """Two-sided p-value for IC."""
    t = ic_t_statistic(ic, n)
    if math.isnan(t):
        return float("nan")
    return float(2 * stats.t.sf(abs(t), df=n - 2))


def hit_rate(signal: pd.Series, forward_return: pd.Series) -> float:
    """Percentage of times sign(signal) matches sign(forward_return).

    50% = coin flip; >52% is potentially exploitable.
    """
    mask = signal.notna() & forward_return.notna() & (signal != 0) & (forward_return != 0)
    if mask.sum() < 30:
        return float("nan")
    correct = (np.sign(signal[mask]) == np.sign(forward_return[mask])).mean()
    return float(correct)


def rolling_ic(
    signal: pd.Series,
    forward_return: pd.Series,
    window: int = 168,
) -> pd.Series:
    """Rolling IC over a window to check consistency.

    Args:
        signal: Feature values.
        forward_return: Forward returns.
        window: Rolling window size (default 168 = 1 week hourly).

    Returns:
        Series of rolling IC values.
    """
    combined = pd.DataFrame({"signal": signal, "ret": forward_return}).dropna()
    if len(combined) < window:
        return pd.Series(dtype=float)

    ics = []
    for i in range(window, len(combined) + 1):
        chunk = combined.iloc[i - window : i]
        corr, _ = stats.spearmanr(chunk["signal"], chunk["ret"])
        ics.append(corr)

    return pd.Series(
        ics,
        index=combined.index[window - 1 :],
        name="rolling_ic",
    )


def quantile_returns(
    signal: pd.Series,
    forward_return: pd.Series,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Mean forward return per signal quantile.

    Should see monotonic increase from Q1 to Q5 for a good signal.
    """
    mask = signal.notna() & forward_return.notna()
    if mask.sum() < n_quantiles * 10:
        return pd.DataFrame()

    combined = pd.DataFrame({"signal": signal[mask], "ret": forward_return[mask]})
    combined["quantile"] = pd.qcut(
        combined["signal"], q=n_quantiles, labels=False, duplicates="drop"
    )
    result = combined.groupby("quantile")["ret"].agg(["mean", "std", "count"])
    result.index = [f"Q{i+1}" for i in range(len(result))]
    return result


def long_short_pnl(
    signal: pd.Series,
    forward_return: pd.Series,
    n_quantiles: int = 5,
) -> pd.Series:
    """Cumulative return of long top quantile, short bottom quantile."""
    mask = signal.notna() & forward_return.notna()
    if mask.sum() < n_quantiles * 10:
        return pd.Series(dtype=float)

    combined = pd.DataFrame({
        "signal": signal[mask],
        "ret": forward_return[mask],
    })
    combined["quantile"] = pd.qcut(
        combined["signal"], q=n_quantiles, labels=False, duplicates="drop"
    )

    top = combined[combined["quantile"] == combined["quantile"].max()]["ret"]
    bottom = combined[combined["quantile"] == combined["quantile"].min()]["ret"]

    # Align indices
    common = top.index.intersection(bottom.index)
    ls_return = top.loc[common] - bottom.loc[common]
    return ls_return.cumsum().rename("long_short_cumulative")
