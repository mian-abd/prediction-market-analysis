"""XGBoost classifier for prediction market outcome prediction."""

import logging
import numpy as np
from pathlib import Path
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)
MODEL_PATH = Path("ml/saved_models/xgboost_ensemble.joblib")


class XGBoostModel:
    def __init__(self):
        self.model: XGBClassifier | None = None
        self._is_trained = False
        self.brier_score: float | None = None
        self.feature_names: list[str] = []

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None):
        """Train XGBoost binary classifier.

        Moderate hyperparameters for N~2000-5000:
        - max_depth=4: enough depth for meaningful feature interactions
        - n_estimators=150: sufficient trees with low learning rate
        - learning_rate=0.05: slower learning for better generalization
        - Moderate L1/L2 regularization
        """
        pos_count = y.sum()
        neg_count = len(y) - pos_count
        scale_pos_weight = neg_count / max(pos_count, 1)

        self.model = XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.3,
            reg_lambda=1.5,
            min_child_weight=5,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )

        self.feature_names = feature_names or []

        # Out-of-fold Brier score via temporal CV (expanding window).
        # Data MUST be sorted by resolved_at before reaching here.
        # TimeSeriesSplit ensures we never train on future data.
        n_splits = min(5, len(y) // 20)
        if n_splits < 2:
            self.model.fit(X, y)
            self.brier_score = brier_score_loss(y, self.model.predict_proba(X)[:, 1])
            self._is_trained = True
            logger.info(f"XGBoost train Brier (no CV): {self.brier_score:.4f}")
            return

        oof_preds = np.full(len(y), np.nan)
        tscv = TimeSeriesSplit(n_splits=n_splits)

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            if y_train.sum() == 0 or y_train.sum() == len(y_train):
                continue
            fold_model = XGBClassifier(**self.model.get_params())
            fold_model.fit(X_train, y_train)
            oof_preds[val_idx] = fold_model.predict_proba(X_val)[:, 1]

        mask = ~np.isnan(oof_preds)
        if mask.sum() > 0:
            self.brier_score = brier_score_loss(y[mask], oof_preds[mask])
        else:
            self.brier_score = 1.0
        logger.info(f"XGBoost OOF Brier (temporal): {self.brier_score:.4f} ({n_splits}-fold, n={mask.sum()})")

        # Final model on all data
        self.model.fit(X, y)
        self._is_trained = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(YES) for each sample."""
        if not self._is_trained:
            raise RuntimeError("Model not trained")
        return self.model.predict_proba(np.atleast_2d(X))[:, 1]

    def predict_single(self, features: np.ndarray) -> float:
        """Predict P(YES) for a single sample."""
        return float(self.predict_proba(features.reshape(1, -1))[0])

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance scores."""
        if not self._is_trained:
            return {}
        importances = self.model.feature_importances_
        names = self.feature_names or [f"f{i}" for i in range(len(importances))]
        return dict(sorted(zip(names, importances.tolist()), key=lambda x: -x[1]))

    def save(self, path: Path | None = None):
        path = path or MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "brier_score": self.brier_score,
            "feature_names": self.feature_names,
        }, path)
        logger.info(f"XGBoost model saved to {path}")

    def load(self, path: Path | None = None):
        path = path or MODEL_PATH
        if path.exists():
            data = joblib.load(path)
            self.model = data["model"]
            self.brier_score = data.get("brier_score")
            self.feature_names = data.get("feature_names", [])
            self._is_trained = True
            logger.info(f"XGBoost model loaded from {path}")
            return True
        logger.warning(f"No saved XGBoost model at {path}")
        return False
