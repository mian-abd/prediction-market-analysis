"""LightGBM classifier for prediction market outcome prediction."""

import logging
import numpy as np
from pathlib import Path
import joblib
from lightgbm import LGBMClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)
MODEL_PATH = Path("ml/saved_models/lightgbm_ensemble.joblib")


class LightGBMModel:
    def __init__(self):
        self.model: LGBMClassifier | None = None
        self._is_trained = False
        self.brier_score: float | None = None
        self.feature_names: list[str] = []

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None):
        """Train LightGBM binary classifier.

        LightGBM uses leaf-wise growth (vs XGBoost's depth-wise), so we
        control complexity via num_leaves and min_child_samples.
        Conservative for small data.
        """
        pos_count = y.sum()
        neg_count = len(y) - pos_count
        scale_pos_weight = neg_count / max(pos_count, 1)

        self.model = LGBMClassifier(
            n_estimators=80,
            num_leaves=8,
            max_depth=3,
            learning_rate=0.1,
            min_child_samples=10,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=3,
            lambda_l1=0.5,
            lambda_l2=2.0,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=-1,
        )

        self.feature_names = feature_names or []

        # Out-of-fold Brier score
        n_splits = min(5, int(min(pos_count, neg_count)))
        if n_splits < 2:
            self.model.fit(X, y)
            self.brier_score = brier_score_loss(y, self.model.predict_proba(X)[:, 1])
            self._is_trained = True
            logger.info(f"LightGBM train Brier (no CV): {self.brier_score:.4f}")
            return

        oof_preds = np.zeros(len(y))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            fold_model = LGBMClassifier(**self.model.get_params())
            fold_model.fit(X_train, y_train)
            oof_preds[val_idx] = fold_model.predict_proba(X_val)[:, 1]

        self.brier_score = brier_score_loss(y, oof_preds)
        logger.info(f"LightGBM OOF Brier: {self.brier_score:.4f} ({n_splits}-fold)")

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
        logger.info(f"LightGBM model saved to {path}")

    def load(self, path: Path | None = None):
        path = path or MODEL_PATH
        if path.exists():
            data = joblib.load(path)
            self.model = data["model"]
            self.brier_score = data.get("brier_score")
            self.feature_names = data.get("feature_names", [])
            self._is_trained = True
            logger.info(f"LightGBM model loaded from {path}")
            return True
        logger.warning(f"No saved LightGBM model at {path}")
        return False
