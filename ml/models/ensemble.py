"""Ensemble model blending Calibration + XGBoost + LightGBM predictions.

Weights are inversely proportional to each model's Brier score,
computed during training via out-of-fold evaluation.
"""

import logging
import numpy as np
from pathlib import Path
import joblib

from ml.models.calibration_model import CalibrationModel
from ml.models.xgboost_model import XGBoostModel
from ml.models.lightgbm_model import LightGBMModel
from ml.features.training_features import (
    extract_features_from_market,
    features_to_array,
    ENSEMBLE_FEATURE_NAMES,
)

logger = logging.getLogger(__name__)
WEIGHTS_PATH = Path("ml/saved_models/ensemble_weights.joblib")


class EnsembleModel:
    """Weighted blend of CalibrationModel + XGBoost + LightGBM."""

    def __init__(self):
        self.calibration = CalibrationModel()
        self.xgboost = XGBoostModel()
        self.lightgbm = LightGBMModel()
        self.weights: dict[str, float] = {
            "calibration": 1 / 3,
            "xgboost": 1 / 3,
            "lightgbm": 1 / 3,
        }
        self.metrics: dict = {}
        self._is_loaded = False

    def load_all(self) -> bool:
        """Load all sub-models and ensemble weights.

        Returns True if ensemble is fully loaded, False if falling back
        to calibration-only.
        """
        # Calibration always loads (trains from historical data if no saved model)
        self.calibration.load()

        xgb_ok = self.xgboost.load()
        lgb_ok = self.lightgbm.load()

        if WEIGHTS_PATH.exists():
            data = joblib.load(WEIGHTS_PATH)
            self.weights = data.get("weights", self.weights)
            self.metrics = data.get("metrics", {})
            logger.info(f"Ensemble weights: {self.weights}")

        self._is_loaded = xgb_ok and lgb_ok
        if not self._is_loaded:
            logger.warning("Ensemble not fully loaded, will use calibration-only fallback")
        return self._is_loaded

    def predict_market(self, market) -> dict:
        """Generate ensemble prediction for a Market ORM object.

        Returns dict with individual model predictions, weights,
        and blended ensemble probability.
        """
        price = market.price_yes if market.price_yes else 0.5

        # Calibration prediction (always available)
        cal_pred = self.calibration.predict_single(price)

        if not self._is_loaded:
            # Fallback: calibration only
            delta = cal_pred - price
            return {
                "ensemble_probability": cal_pred,
                "market_price": price,
                "delta": delta,
                "delta_pct": delta * 100,
                "direction": _direction(delta),
                "edge_estimate": abs(delta),
                "model_predictions": {
                    "calibration": {"probability": cal_pred, "weight": 1.0},
                },
                "features_used": 1,
                "ensemble_active": False,
            }

        # Feature-based predictions
        features = extract_features_from_market(market)
        feat_array = features_to_array(features)

        xgb_pred = self.xgboost.predict_single(feat_array)
        lgb_pred = self.lightgbm.predict_single(feat_array)

        # Weighted blend
        ensemble_pred = (
            self.weights["calibration"] * cal_pred
            + self.weights["xgboost"] * xgb_pred
            + self.weights["lightgbm"] * lgb_pred
        )

        # Clip to valid probability range
        ensemble_pred = max(0.01, min(0.99, ensemble_pred))

        delta = ensemble_pred - price

        return {
            "ensemble_probability": ensemble_pred,
            "market_price": price,
            "delta": delta,
            "delta_pct": delta * 100,
            "direction": _direction(delta),
            "edge_estimate": abs(delta),
            "model_predictions": {
                "calibration": {
                    "probability": round(cal_pred, 4),
                    "weight": round(self.weights["calibration"], 3),
                },
                "xgboost": {
                    "probability": round(xgb_pred, 4),
                    "weight": round(self.weights["xgboost"], 3),
                },
                "lightgbm": {
                    "probability": round(lgb_pred, 4),
                    "weight": round(self.weights["lightgbm"], 3),
                },
            },
            "features_used": len(ENSEMBLE_FEATURE_NAMES),
            "ensemble_active": True,
        }

    @staticmethod
    def compute_weights(brier_scores: dict[str, float]) -> dict[str, float]:
        """Compute weights inversely proportional to Brier scores.

        Lower Brier = better calibration = higher weight.
        """
        inv_brier = {
            name: 1.0 / max(score, 0.001) for name, score in brier_scores.items()
        }
        total = sum(inv_brier.values())
        return {name: round(val / total, 4) for name, val in inv_brier.items()}

    def save_weights(self, weights: dict[str, float], metrics: dict):
        """Save ensemble weights and training metrics."""
        WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"weights": weights, "metrics": metrics}, WEIGHTS_PATH)
        self.weights = weights
        self.metrics = metrics
        logger.info(f"Ensemble weights saved: {weights}")


def _direction(delta: float) -> str:
    if delta > 0.02:
        return "underpriced"
    elif delta < -0.02:
        return "overpriced"
    return "fair"
