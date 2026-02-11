"""Calibration model using isotonic regression.
Maps market probability to true (calibrated) probability.
Exploits systematic overconfidence at price extremes.

No AI/LLM needed. Trains locally for free."""

import logging
import numpy as np
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
import joblib

from ml.features.calibration_features import HISTORICAL_CALIBRATION

logger = logging.getLogger(__name__)

MODEL_PATH = Path("ml/saved_models/calibration_iso.joblib")


class CalibrationModel:
    def __init__(self):
        self.model: IsotonicRegression | None = None
        self._is_trained = False

    def train(
        self,
        market_prices: np.ndarray,
        resolutions: np.ndarray,
    ):
        """Train isotonic regression on (market_price, did_resolve_yes) pairs.

        Args:
            market_prices: Array of market probabilities [0, 1]
            resolutions: Array of 0/1 (did the market resolve YES?)
        """
        self.model = IsotonicRegression(
            y_min=0.01,
            y_max=0.99,
            out_of_bounds="clip",
        )
        self.model.fit(market_prices, resolutions)
        self._is_trained = True

        # Evaluate
        calibrated = self.model.predict(market_prices)
        brier_before = brier_score_loss(resolutions, market_prices)
        brier_after = brier_score_loss(resolutions, calibrated)
        logger.info(
            f"Calibration model trained. Brier score: {brier_before:.4f} -> {brier_after:.4f} "
            f"(improvement: {(brier_before - brier_after) / brier_before * 100:.1f}%)"
        )

    def train_from_historical(self):
        """Train from published calibration research data.
        Uses the well-documented overconfidence bias in prediction markets.
        """
        # Generate synthetic training data from research
        prices = []
        outcomes = []

        for market_price, actual_rate in HISTORICAL_CALIBRATION.items():
            # Generate N samples per bucket proportional to density
            n_samples = 100
            n_yes = int(n_samples * actual_rate)
            n_no = n_samples - n_yes

            # Add noise to prices (markets aren't exactly at bucket centers)
            for _ in range(n_yes):
                p = market_price + np.random.normal(0, 0.015)
                p = max(0.01, min(0.99, p))
                prices.append(p)
                outcomes.append(1)

            for _ in range(n_no):
                p = market_price + np.random.normal(0, 0.015)
                p = max(0.01, min(0.99, p))
                prices.append(p)
                outcomes.append(0)

        self.train(np.array(prices), np.array(outcomes))

    def predict(self, market_prices: np.ndarray) -> np.ndarray:
        """Predict calibrated probability from market price."""
        if not self._is_trained:
            self.train_from_historical()
        return self.model.predict(np.atleast_1d(market_prices))

    def predict_single(self, market_price: float) -> float:
        """Predict calibrated probability for a single market price."""
        return float(self.predict(np.array([market_price]))[0])

    def get_mispricing(self, market_price: float) -> dict:
        """Get mispricing analysis for a single market."""
        calibrated = self.predict_single(market_price)
        delta = calibrated - market_price

        return {
            "market_price": market_price,
            "calibrated_price": calibrated,
            "delta": delta,
            "delta_pct": delta * 100,
            "direction": "UNDERPRICED" if delta > 0.02 else ("OVERPRICED" if delta < -0.02 else "FAIR"),
            "edge_estimate": abs(delta),
        }

    def save(self, path: Path | None = None):
        path = path or MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Calibration model saved to {path}")

    def load(self, path: Path | None = None):
        path = path or MODEL_PATH
        if path.exists():
            self.model = joblib.load(path)
            self._is_trained = True
            logger.info(f"Calibration model loaded from {path}")
        else:
            logger.info("No saved model found, training from historical data")
            self.train_from_historical()

    def get_calibration_curve(self, n_points: int = 20) -> list[dict]:
        """Get the calibration curve for visualization."""
        if not self._is_trained:
            self.train_from_historical()

        prices = np.linspace(0.05, 0.95, n_points)
        calibrated = self.predict(prices)

        return [
            {
                "market_price": float(p),
                "calibrated_price": float(c),
                "bias": float(c - p),
            }
            for p, c in zip(prices, calibrated)
        ]
