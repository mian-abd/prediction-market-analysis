"""Ensemble model blending Calibration + XGBoost + LightGBM predictions.

Weights are inversely proportional to each model's Brier score,
computed during training via walk-forward out-of-fold evaluation.

Only models that significantly beat calibration are included.
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
        self.post_calibrator = None  # Post-ensemble calibrator
        self.weights: dict[str, float] = {
            "calibration": 1 / 3,
            "xgboost": 1 / 3,
            "lightgbm": 1 / 3,
        }
        self.metrics: dict = {}
        self._is_loaded = False
        self._active_features: list[str] = []  # Set by load_all from saved metrics

    def load_all(self) -> bool:
        """Load all sub-models and ensemble weights.

        Returns True if ensemble is fully loaded, False if falling back
        to calibration-only.
        """
        # Calibration always loads (trains from historical data if no saved model)
        self.calibration.load()

        xgb_ok = self.xgboost.load()
        lgb_ok = self.lightgbm.load()

        # Load Post-Calibrator model
        post_cal_path = Path("ml/saved_models/post_calibrator.joblib")
        if post_cal_path.exists():
            try:
                self.post_calibrator = CalibrationModel()
                self.post_calibrator.load(post_cal_path)
                logger.info("Loaded post_calibrator.joblib")
            except Exception as e:
                logger.warning(f"Failed to load Post-Calibrator: {e}")
                self.post_calibrator = None
        else:
            logger.info("post_calibrator.joblib not found, using raw ensemble predictions")
            self.post_calibrator = None

        if WEIGHTS_PATH.exists():
            data = joblib.load(WEIGHTS_PATH)
            self.weights = data.get("weights", self.weights)
            self.metrics = data.get("metrics", {})
            logger.info(f"Ensemble weights: {self.weights}")

            # Use the feature list that was active during training
            self._active_features = self.metrics.get("feature_names", [])

            # Feature-count mismatch guard: if saved model was trained with
            # different feature count than what we'd extract, fall back
            if self._active_features:
                # Verify all saved features still exist in current ENSEMBLE_FEATURE_NAMES
                missing = [f for f in self._active_features if f not in ENSEMBLE_FEATURE_NAMES]
                if missing:
                    logger.warning(
                        f"Feature mismatch: saved model uses features not in current code: {missing}. "
                        f"Falling back to calibration-only until retrained."
                    )
                    self._is_loaded = False
                    return False
        else:
            self._active_features = list(ENSEMBLE_FEATURE_NAMES)

        # Only need XGB + LGB if they have weight > 0 in saved weights
        needs_xgb = self.weights.get("xgboost", 0) > 0
        needs_lgb = self.weights.get("lightgbm", 0) > 0

        self._is_loaded = (
            (not needs_xgb or xgb_ok) and
            (not needs_lgb or lgb_ok)
        )
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

        # Feature-based predictions using the active feature subset from training
        features = extract_features_from_market(market)
        active = self._active_features or list(ENSEMBLE_FEATURE_NAMES)
        feat_array = np.array([features[name] for name in active])

        model_preds = {"calibration": cal_pred}
        model_weights = {"calibration": self.weights.get("calibration", 1.0)}

        if self.weights.get("xgboost", 0) > 0:
            xgb_pred = self.xgboost.predict_single(feat_array)
            model_preds["xgboost"] = xgb_pred
            model_weights["xgboost"] = self.weights["xgboost"]

        if self.weights.get("lightgbm", 0) > 0:
            lgb_pred = self.lightgbm.predict_single(feat_array)
            model_preds["lightgbm"] = lgb_pred
            model_weights["lightgbm"] = self.weights["lightgbm"]

        # Weighted blend
        raw_ensemble_pred = sum(
            model_weights[name] * model_preds[name]
            for name in model_preds
        )

        # Apply post-ensemble calibration if available
        if self.post_calibrator:
            ensemble_pred = self.post_calibrator.predict_single(raw_ensemble_pred)
        else:
            ensemble_pred = raw_ensemble_pred

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
                name: {
                    "probability": round(pred, 4),
                    "weight": round(model_weights[name], 3),
                }
                for name, pred in model_preds.items()
            },
            "features_used": len(active),
            "ensemble_active": True,
        }

    @staticmethod
    def compute_weights(brier_scores: dict[str, float]) -> dict[str, float]:
        """Compute weights inversely proportional to Brier scores.

        Lower Brier = better calibration = higher weight.
        Calibration model is capped at 5% max weight to limit its drag
        on ensemble performance (its Brier is worse than baseline).
        """
        inv_brier = {
            name: 1.0 / max(score, 0.001) for name, score in brier_scores.items()
        }
        total = sum(inv_brier.values())
        weights = {name: round(val / total, 4) for name, val in inv_brier.items()}

        # Cap calibration weight at 2.5% — Phase 2.3: Further reduce negative-edge model
        # Calibration Brier (0.0892) > baseline (0.0843), so minimize its drag
        # Previous testing: 2% too low, 5% too high, 3% middle ground → now 2.5% for optimization
        CAL_MAX_WEIGHT = 0.025
        if "calibration" in weights and weights["calibration"] > CAL_MAX_WEIGHT:
            excess = weights["calibration"] - CAL_MAX_WEIGHT
            weights["calibration"] = CAL_MAX_WEIGHT
            remaining = {k: v for k, v in weights.items() if k != "calibration"}
            total_remaining = sum(remaining.values())
            if total_remaining > 0:
                for k in remaining:
                    weights[k] = round(weights[k] + excess * (remaining[k] / total_remaining), 4)
            logger.info(f"Calibration weight capped at {CAL_MAX_WEIGHT}: {weights}")

        return weights

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
