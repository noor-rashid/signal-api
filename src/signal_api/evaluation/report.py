"""Evaluate all features and generate summary report."""

import pandas as pd

from signal_api.evaluation.signal_tester import (
    hit_rate,
    ic_p_value,
    information_coefficient,
    quantile_returns,
    rolling_ic,
)


def evaluate_feature(
    signal: pd.Series,
    forward_returns: dict[str, pd.Series],
) -> dict:
    """Evaluate a single feature across multiple horizons."""
    results = {"feature": signal.name}

    for horizon_name, fwd_ret in forward_returns.items():
        mask = signal.notna() & fwd_ret.notna()
        n = mask.sum()

        ic = information_coefficient(signal, fwd_ret)
        hr = hit_rate(signal, fwd_ret)
        p_val = ic_p_value(ic, n)

        # IC stability — std of rolling weekly IC
        ric = rolling_ic(signal, fwd_ret, window=168)
        ic_std = float(ric.std()) if len(ric) > 0 else float("nan")

        # Quantile monotonicity — correlation of quantile rank with mean return
        qr = quantile_returns(signal, fwd_ret)
        if not qr.empty:
            q_mono = float(qr["mean"].corr(pd.Series(range(len(qr)))))
        else:
            q_mono = float("nan")

        results[f"{horizon_name}_ic"] = round(ic, 4)
        results[f"{horizon_name}_p"] = round(p_val, 4)
        results[f"{horizon_name}_hit"] = round(hr, 4)
        results[f"{horizon_name}_ic_std"] = round(ic_std, 4)
        results[f"{horizon_name}_n"] = n
        results[f"{horizon_name}_mono"] = round(q_mono, 4)

    return results


def evaluate_all_features(
    feature_matrix: pd.DataFrame,
    feature_columns: list[str],
    forward_return_columns: list[str],
) -> pd.DataFrame:
    """Evaluate all features and return a summary DataFrame.

    Args:
        feature_matrix: DataFrame with features and forward returns.
        feature_columns: List of feature column names.
        forward_return_columns: List of forward return column names.

    Returns:
        Summary DataFrame sorted by best absolute IC.
    """
    fwd_rets = {col: feature_matrix[col] for col in forward_return_columns}

    rows = []
    for col in feature_columns:
        result = evaluate_feature(feature_matrix[col], fwd_rets)
        rows.append(result)

    summary = pd.DataFrame(rows)

    # Sort by best absolute IC across horizons
    ic_cols = [c for c in summary.columns if c.endswith("_ic")]
    summary["best_abs_ic"] = summary[ic_cols].abs().max(axis=1)
    summary = summary.sort_values("best_abs_ic", ascending=False).drop(
        columns=["best_abs_ic"]
    )

    return summary


def print_report(summary: pd.DataFrame, significance_level: float = 0.05) -> None:
    """Print a formatted evaluation report with keep/drop recommendations."""
    print("\n" + "=" * 80)
    print("SIGNAL EVALUATION REPORT")
    print("=" * 80)

    ic_cols = [c for c in summary.columns if c.endswith("_ic") and not c.endswith("_ic_std")]
    p_cols = [c for c in summary.columns if c.endswith("_p")]
    hit_cols = [c for c in summary.columns if c.endswith("_hit")]
    n_cols = [c for c in summary.columns if c.endswith("_n")]

    for _, row in summary.iterrows():
        print(f"\n{'─' * 60}")
        print(f"  {row['feature']}")
        print(f"{'─' * 60}")

        any_significant = False
        for ic_col, p_col, hit_col, n_col in zip(ic_cols, p_cols, hit_cols, n_cols):
            horizon = ic_col.replace("_ic", "")
            ic = row[ic_col]
            p = row[p_col]
            hr = row[hit_col]
            n = row[n_col]
            ic_std = row.get(f"{horizon}_ic_std", float("nan"))

            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < significance_level else ""
            if p < significance_level:
                any_significant = True

            print(f"  {horizon:>8s}:  IC={ic:+.4f}{sig:3s}  p={p:.4f}  "
                  f"hit={hr:.1%}  IC_std={ic_std:.4f}  N={n}")

        verdict = "KEEP" if any_significant else "DROP"
        marker = "✓" if any_significant else "✗"
        print(f"\n  Verdict: {marker} {verdict}")

    print(f"\n{'=' * 80}")
    kept = []
    dropped = []
    for _, row in summary.iterrows():
        sig = any(row[c] < significance_level for c in p_cols if pd.notna(row[c]))
        if sig:
            kept.append(row["feature"])
        else:
            dropped.append(row["feature"])

    print(f"  KEEP ({len(kept)}): {', '.join(kept) if kept else 'none'}")
    print(f"  DROP ({len(dropped)}): {', '.join(dropped) if dropped else 'none'}")
    print("=" * 80 + "\n")
