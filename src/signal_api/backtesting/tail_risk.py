"""Left tail risk prediction and backtesting."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)


def label_tail_events(
    df: pd.DataFrame,
    horizon: int = 24,
    percentile: float = 5.0,
    price_col: str = "close",
) -> pd.Series:
    """Label future left-tail events (worst N% of returns).

    Args:
        df: DataFrame with price column.
        horizon: Forward window in periods.
        percentile: Bottom percentile threshold (e.g. 5 = worst 5%).
        price_col: Column for returns.

    Returns:
        Binary series: 1 = tail event, 0 = normal.
    """
    fwd_ret = df[price_col].shift(-horizon) / df[price_col] - 1
    # Use expanding threshold to avoid lookahead
    threshold = fwd_ret.expanding(min_periods=500).quantile(percentile / 100)
    labels = (fwd_ret <= threshold).astype(int)
    # NaN out the last `horizon` rows and the warmup period
    labels.iloc[-horizon:] = np.nan
    labels.iloc[:500] = np.nan
    return labels.rename(f"tail_{percentile}pct_{horizon}h")


def walk_forward_backtest(
    features: pd.DataFrame,
    labels: pd.Series,
    feature_columns: list[str],
    train_window: int = 2160,  # 90 days hourly
    test_window: int = 168,    # 7 days hourly
    step: int = 168,           # Step forward 7 days
) -> dict:
    """Walk-forward backtest for tail event prediction.

    Uses a simple logistic-style scoring (rank-based) to avoid overfitting
    with small feature sets. Scores each test window using training-period
    feature quantiles.

    Returns dict with predictions, actuals, and metrics per fold.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    mask = labels.notna()
    for col in feature_columns:
        mask &= features[col].notna()

    valid_idx = mask[mask].index
    if len(valid_idx) < train_window + test_window:
        return {"error": "Insufficient data for walk-forward backtest"}

    all_preds = []
    all_actuals = []
    all_probas = []
    fold_metrics = []

    start = 0
    fold = 0

    while start + train_window + test_window <= len(valid_idx):
        train_idx = valid_idx[start : start + train_window]
        test_idx = valid_idx[start + train_window : start + train_window + test_window]

        X_train = features.loc[train_idx, feature_columns].values
        y_train = labels.loc[train_idx].values
        X_test = features.loc[test_idx, feature_columns].values
        y_test = labels.loc[test_idx].values

        # Skip folds with no tail events in training
        if y_train.sum() < 5:
            start += step
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=0.1,
            random_state=42,
        )
        model.fit(X_train_s, y_train)

        probas = model.predict_proba(X_test_s)[:, 1]
        preds = (probas > 0.5).astype(int)

        all_preds.extend(preds)
        all_actuals.extend(y_test)
        all_probas.extend(probas)

        # Per-fold metrics
        if y_test.sum() > 0 and len(np.unique(y_test)) > 1:
            try:
                auc = roc_auc_score(y_test, probas)
            except ValueError:
                auc = float("nan")
        else:
            auc = float("nan")

        fold_metrics.append({
            "fold": fold,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "tail_events_train": int(y_train.sum()),
            "tail_events_test": int(y_test.sum()),
            "auc": round(auc, 4) if not np.isnan(auc) else None,
        })

        start += step
        fold += 1

    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)
    all_probas = np.array(all_probas)

    # Aggregate metrics
    results = {
        "n_folds": fold,
        "total_predictions": len(all_preds),
        "total_tail_events": int(all_actuals.sum()),
        "predicted_tail_events": int(all_preds.sum()),
        "fold_metrics": fold_metrics,
    }

    if len(all_actuals) > 0 and all_actuals.sum() > 0:
        try:
            results["auc"] = round(float(roc_auc_score(all_actuals, all_probas)), 4)
        except ValueError:
            results["auc"] = None

        # Precision at various recall levels
        precision, recall, thresholds = precision_recall_curve(all_actuals, all_probas)
        for target_recall in [0.3, 0.5, 0.7]:
            idx = np.argmin(np.abs(recall - target_recall))
            results[f"precision_at_{int(target_recall*100)}pct_recall"] = round(
                float(precision[idx]), 4
            )

        results["classification_report"] = classification_report(
            all_actuals, all_preds, target_names=["normal", "tail"], zero_division=0,
        )

    results["predictions"] = pd.DataFrame({
        "actual": all_actuals,
        "predicted": all_preds,
        "probability": all_probas,
    })

    return results


def feature_importance_for_tails(
    features: pd.DataFrame,
    labels: pd.Series,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Rank features by their importance for predicting tail events.

    Uses both IC with tail labels and conditional means.
    """
    from scipy import stats

    rows = []
    mask = labels.notna()
    for col in feature_columns:
        col_mask = mask & features[col].notna()
        signal = features.loc[col_mask, col]
        target = labels[col_mask]

        if len(signal) < 100:
            continue

        # Point-biserial correlation (equivalent to Pearson for binary target)
        corr, p_val = stats.pointbiserialr(target, signal)

        # Mean feature value during tail events vs normal
        tail_mean = signal[target == 1].mean()
        normal_mean = signal[target == 0].mean()

        # AUC for this single feature
        try:
            auc = roc_auc_score(target, signal)
        except ValueError:
            auc = float("nan")

        rows.append({
            "feature": col,
            "correlation": round(float(corr), 4),
            "p_value": round(float(p_val), 6),
            "tail_mean": round(float(tail_mean), 4),
            "normal_mean": round(float(normal_mean), 4),
            "mean_diff": round(float(tail_mean - normal_mean), 4),
            "single_feature_auc": round(float(auc), 4),
            "n_obs": int(col_mask.sum()),
        })

    result = pd.DataFrame(rows).sort_values("single_feature_auc", ascending=False, key=abs)
    # Re-sort: AUC further from 0.5 is better (either direction)
    result["auc_edge"] = (result["single_feature_auc"] - 0.5).abs()
    result = result.sort_values("auc_edge", ascending=False).drop(columns=["auc_edge"])
    return result.reset_index(drop=True)
