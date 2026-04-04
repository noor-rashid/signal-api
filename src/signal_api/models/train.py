"""Model training pipeline with MLflow tracking."""

import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from signal_api.backtesting.tail_risk import label_tail_events
from signal_api.data.cache import ParquetCache
from signal_api.features import (
    ALL_DERIVATIVES_FEATURES,
    ALL_SPOT_FEATURES,
    ALL_VOLATILITY_FEATURES,
    build_feature_matrix,
)

logger = logging.getLogger(__name__)

# Features validated as having predictive power
VALIDATED_FEATURES = [
    # Spot (deep history, strong IC)
    "taker_buy_ratio_zscore",
    # Derivatives (significant IC)
    "funding_rate_zscore",
    # Volatility (strong tail AUC)
    "realized_vol",
    "vol_zscore",
    "vol_of_vol",
    "downside_vol_ratio",
    "return_skewness",
    "tail_concentration",
    "funding_vol_spread",
]

MODELS = {
    "logistic_regression": lambda: LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        C=0.1,
        random_state=42,
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "gradient_boosting": lambda: GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=50,
        subsample=0.8,
        random_state=42,
    ),
}


def prepare_dataset(
    symbol: str = "BTCUSDT",
    horizon: int = 4,
    tail_percentile: float = 5.0,
    data_dir: Path = Path("data/raw"),
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load data, build features, create labels.

    Returns (feature_matrix, labels, feature_column_names).
    """
    cache = ParquetCache(data_dir)
    spot = cache.load(symbol, "1h")
    funding = cache.load(symbol, "all", prefix="FUNDING_")
    oi = cache.load(symbol, "1h", prefix="OI_")
    ls = cache.load(symbol, "1h", prefix="LSRATIO_")

    matrix = build_feature_matrix(spot, oi, funding, ls)
    labels = label_tail_events(matrix, horizon=horizon, percentile=tail_percentile)

    # Filter to validated features that exist
    feature_cols = [f for f in VALIDATED_FEATURES if f in matrix.columns]
    logger.info(f"Using {len(feature_cols)} features: {feature_cols}")

    return matrix, labels, feature_cols


def time_series_train_test_split(
    matrix: pd.DataFrame,
    labels: pd.Series,
    feature_cols: list[str],
    test_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Index, pd.Index]:
    """Split data chronologically — no shuffle, no lookahead."""
    mask = labels.notna()
    for col in feature_cols:
        mask &= matrix[col].notna()

    valid_idx = mask[mask].index
    split_point = int(len(valid_idx) * (1 - test_ratio))

    train_idx = valid_idx[:split_point]
    test_idx = valid_idx[split_point:]

    X_train = matrix.loc[train_idx, feature_cols].values
    X_test = matrix.loc[test_idx, feature_cols].values
    y_train = labels.loc[train_idx].values
    y_test = labels.loc[test_idx].values

    return X_train, X_test, y_train, y_test, train_idx, test_idx


def train_and_evaluate(
    symbol: str = "BTCUSDT",
    horizon: int = 4,
    tail_percentile: float = 5.0,
    experiment_name: str = "tail_risk_detection",
    track_mlflow: bool = True,
) -> dict:
    """Full training pipeline with MLflow tracking.

    Trains multiple models, evaluates with time-series CV,
    and logs everything to MLflow.
    """
    matrix, labels, feature_cols = prepare_dataset(
        symbol=symbol, horizon=horizon, tail_percentile=tail_percentile,
    )

    X_train, X_test, y_train, y_test, train_idx, test_idx = (
        time_series_train_test_split(matrix, labels, feature_cols)
    )

    logger.info(
        f"Train: {len(X_train)} ({y_train.sum():.0f} tail events) | "
        f"Test: {len(X_test)} ({y_test.sum():.0f} tail events)"
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if track_mlflow:
        mlflow.set_experiment(experiment_name)

    results = {}

    for model_name, model_factory in MODELS.items():
        logger.info(f"Training {model_name}...")

        model = model_factory()

        if track_mlflow:
            with mlflow.start_run(run_name=f"{symbol}_{model_name}_{horizon}h_{tail_percentile}pct"):
                mlflow.log_params({
                    "symbol": symbol,
                    "horizon": horizon,
                    "tail_percentile": tail_percentile,
                    "model": model_name,
                    "n_features": len(feature_cols),
                    "features": ",".join(feature_cols),
                    "train_size": len(X_train),
                    "test_size": len(X_test),
                    "tail_events_train": int(y_train.sum()),
                    "tail_events_test": int(y_test.sum()),
                })

                model.fit(X_train_s, y_train)
                metrics = _evaluate_model(model, X_test_s, y_test)
                mlflow.log_metrics(metrics)

                # Log model
                mlflow.sklearn.log_model(model, artifact_path="model")

                # Log feature importance if available
                if hasattr(model, "feature_importances_"):
                    importance = dict(zip(feature_cols, model.feature_importances_))
                    for fname, imp in importance.items():
                        mlflow.log_metric(f"importance_{fname}", imp)

                results[model_name] = {**metrics, "run_id": mlflow.active_run().info.run_id}
        else:
            model.fit(X_train_s, y_train)
            metrics = _evaluate_model(model, X_test_s, y_test)
            results[model_name] = metrics

    # Time-series cross-validation for best model
    best_model_name = max(results, key=lambda k: results[k].get("auc", 0))
    logger.info(f"Best model: {best_model_name} (AUC: {results[best_model_name].get('auc', 'N/A')})")

    tscv_results = _time_series_cv(
        MODELS[best_model_name](), X_train_s, y_train, feature_cols,
    )
    results["cv_results"] = tscv_results

    return {
        "best_model": best_model_name,
        "feature_cols": feature_cols,
        "results": results,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }


def _evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Compute evaluation metrics for a trained model."""
    probas = model.predict_proba(X_test)[:, 1]
    preds = (probas > 0.5).astype(int)

    metrics: dict[str, float] = {}

    try:
        metrics["auc"] = round(float(roc_auc_score(y_test, probas)), 4)
    except ValueError:
        metrics["auc"] = 0.5

    metrics["precision_tail"] = round(float(precision_score(y_test, preds, zero_division=0)), 4)
    metrics["recall_tail"] = round(float(recall_score(y_test, preds, zero_division=0)), 4)
    metrics["f1_tail"] = round(float(f1_score(y_test, preds, zero_division=0)), 4)

    # Precision at higher thresholds (more conservative predictions)
    for threshold in [0.3, 0.5, 0.7]:
        preds_t = (probas > threshold).astype(int)
        metrics[f"precision_at_{int(threshold*100)}"] = round(
            float(precision_score(y_test, preds_t, zero_division=0)), 4,
        )
        metrics[f"recall_at_{int(threshold*100)}"] = round(
            float(recall_score(y_test, preds_t, zero_division=0)), 4,
        )

    return metrics


def _time_series_cv(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: list[str],
    n_splits: int = 5,
) -> dict:
    """Time-series cross-validation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        if y_tr.sum() < 5 or y_val.sum() < 2:
            continue

        model.fit(X_tr, y_tr)
        probas = model.predict_proba(X_val)[:, 1]

        try:
            auc = roc_auc_score(y_val, probas)
            fold_aucs.append(auc)
        except ValueError:
            continue

    return {
        "cv_folds": len(fold_aucs),
        "cv_auc_mean": round(float(np.mean(fold_aucs)), 4) if fold_aucs else 0,
        "cv_auc_std": round(float(np.std(fold_aucs)), 4) if fold_aucs else 0,
        "cv_aucs": [round(a, 4) for a in fold_aucs],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        print(f"\n{'='*60}")
        print(f"  Training for {symbol}")
        print(f"{'='*60}")

        results = train_and_evaluate(
            symbol=symbol,
            horizon=4,
            tail_percentile=5.0,
            track_mlflow=False,
        )

        print(f"\n  Best model: {results['best_model']}")
        print(f"  Features: {results['feature_cols']}")
        for model_name, metrics in results["results"].items():
            if model_name == "cv_results":
                print(f"\n  CV Results: AUC {metrics['cv_auc_mean']:.4f} +/- {metrics['cv_auc_std']:.4f}")
                continue
            print(f"\n  {model_name}:")
            for k, v in metrics.items():
                if k != "run_id":
                    print(f"    {k}: {v}")
