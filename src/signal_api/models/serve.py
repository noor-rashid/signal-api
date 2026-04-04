"""Model serving — load trained model and make predictions."""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from signal_api.data.cache import ParquetCache
from signal_api.features import build_feature_matrix, compute_all_features
from signal_api.models.train import MODELS, VALIDATED_FEATURES, prepare_dataset, time_series_train_test_split

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")


class SignalPredictor:
    """Load a trained model and predict tail risk for new data."""

    def __init__(self, model=None, scaler: StandardScaler | None = None, feature_cols: list[str] | None = None):
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols or VALIDATED_FEATURES

    def train_and_save(
        self,
        symbol: str = "BTCUSDT",
        model_name: str = "gradient_boosting",
        horizon: int = 4,
        tail_percentile: float = 5.0,
    ) -> dict:
        """Train on all available data and save model to disk."""
        matrix, labels, feature_cols = prepare_dataset(
            symbol=symbol, horizon=horizon, tail_percentile=tail_percentile,
        )
        self.feature_cols = feature_cols

        X_train, X_test, y_train, y_test, _, _ = time_series_train_test_split(
            matrix, labels, feature_cols,
        )

        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        self.model = MODELS[model_name]()
        self.model.fit(X_train_s, y_train)

        # Evaluate
        probas = self.model.predict_proba(X_test_s)[:, 1]
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(y_test, probas)
        except ValueError:
            auc = 0.5

        # Save
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / f"{symbol}_{model_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "feature_cols": self.feature_cols,
                "symbol": symbol,
                "model_name": model_name,
                "horizon": horizon,
                "tail_percentile": tail_percentile,
                "test_auc": auc,
            }, f)

        logger.info(f"Model saved to {model_path} (AUC: {auc:.4f})")
        return {"path": str(model_path), "auc": round(auc, 4)}

    def load(self, path: str | Path) -> None:
        """Load a saved model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_cols = data["feature_cols"]
        logger.info(f"Loaded model from {path}")

    def predict(self, features: pd.DataFrame) -> dict:
        """Predict tail risk probability for a feature row.

        Args:
            features: DataFrame with at least one row containing required feature columns.

        Returns:
            Dict with probability, risk_level, and feature values.
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load() or train_and_save() first.")

        available = [f for f in self.feature_cols if f in features.columns]
        if len(available) < len(self.feature_cols):
            missing = set(self.feature_cols) - set(available)
            logger.warning(f"Missing features: {missing}")

        row = features[self.feature_cols].iloc[[-1]]
        if row.isna().any().any():
            nan_cols = row.columns[row.isna().any()].tolist()
            logger.warning(f"NaN in features: {nan_cols}")
            row = row.fillna(0)

        X = self.scaler.transform(row.values)
        proba = float(self.model.predict_proba(X)[0, 1])

        # Risk levels
        if proba > 0.7:
            risk_level = "CRITICAL"
        elif proba > 0.5:
            risk_level = "HIGH"
        elif proba > 0.3:
            risk_level = "ELEVATED"
        else:
            risk_level = "NORMAL"

        return {
            "tail_probability": round(proba, 4),
            "risk_level": risk_level,
            "features_used": {
                col: round(float(features[col].iloc[-1]), 4)
                for col in self.feature_cols
                if col in features.columns and pd.notna(features[col].iloc[-1])
            },
        }
