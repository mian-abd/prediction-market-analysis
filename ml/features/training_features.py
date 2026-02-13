"""Unified feature extraction for ML training and inference.

Produces a consistent feature vector from Market ORM objects.
Handles both resolved (training) and active (inference) markets.

LEAKAGE RULES:
- Markets with volume_total = 0 are excluded from training
  (their price_yes is post-settlement zeros, not informative)
- price_yes is ONLY used when it represents a real traded price
"""

import math
import numpy as np
from datetime import datetime

from ml.features.calibration_features import (
    compute_calibration_features,
    get_calibration_estimate,
)
from ml.features.market_features import compute_market_features

# Canonical ordered list of features for ensemble models
ENSEMBLE_FEATURE_NAMES: list[str] = [
    # Price features (only meaningful when volume > 0)
    "price_yes",
    "price_bucket",
    "price_distance_from_50",
    # Volume/liquidity features
    "log_volume_total",
    "log_open_interest",
    # Calibration features (derived from price_yes)
    "calibration_bias",
    "overconfidence_delta",
    "historical_resolution",
    "spread_percentile",
    # Time features
    "time_to_resolution_hrs",
    "volume_per_day",
]

N_FEATURES = len(ENSEMBLE_FEATURE_NAMES)


def extract_features_from_market(market, for_training: bool = False) -> dict[str, float]:
    """Extract features from a Market ORM object.

    Args:
        market: Market ORM object (or any object with the right attributes)
        for_training: If True, compute time features relative to market's
                      own timeline (not current time).

    Returns:
        Dict of {feature_name: float_value}
    """
    price = market.price_yes if market.price_yes else 0.5

    # Volume features
    vol_total = float(market.volume_total or 0)
    liquidity = float(market.liquidity or 0)
    log_volume_total = math.log1p(vol_total)
    log_open_interest = math.log1p(liquidity)

    # Price-derived features
    price_bucket = min(19, int(price / 0.05)) if price > 0 else 0
    price_distance_from_50 = abs(price - 0.5)

    # Calibration features
    cf = compute_calibration_features(market_price=price)

    # Time features
    if for_training and market.end_date:
        # For training: use market duration (end_date - created_at)
        created = getattr(market, "created_at", None)
        if created and market.end_date:
            try:
                end_naive = market.end_date.replace(tzinfo=None) if market.end_date.tzinfo else market.end_date
                created_naive = created.replace(tzinfo=None) if created.tzinfo else created
                delta = end_naive - created_naive
                time_hrs = max(0, delta.total_seconds() / 3600)
            except (TypeError, AttributeError):
                time_hrs = 168.0  # Default 1 week
        else:
            time_hrs = 168.0
    elif market.end_date:
        # For inference: time remaining until resolution
        now = datetime.utcnow()
        try:
            end_naive = market.end_date.replace(tzinfo=None) if market.end_date.tzinfo else market.end_date
            delta = end_naive - now
            time_hrs = max(0, delta.total_seconds() / 3600)
        except (TypeError, AttributeError):
            time_hrs = 168.0
    else:
        time_hrs = 8760.0  # Default 1 year

    # Volume per day (engagement density)
    duration_days = max(time_hrs / 24, 1.0)
    volume_per_day = vol_total / duration_days

    return {
        "price_yes": price,
        "price_bucket": float(price_bucket),
        "price_distance_from_50": price_distance_from_50,
        "log_volume_total": log_volume_total,
        "log_open_interest": log_open_interest,
        "calibration_bias": cf["calibration_bias"],
        "overconfidence_delta": cf["overconfidence_delta"],
        "historical_resolution": cf["historical_resolution"],
        "spread_percentile": cf["spread_percentile"],
        "time_to_resolution_hrs": time_hrs,
        "volume_per_day": volume_per_day,
    }


def features_to_array(features: dict[str, float]) -> np.ndarray:
    """Convert feature dict to numpy array in canonical order."""
    return np.array([features[name] for name in ENSEMBLE_FEATURE_NAMES])


def build_training_matrix(markets: list) -> tuple[np.ndarray, np.ndarray]:
    """Build X (features) and y (labels) from resolved markets.

    Filters out markets with:
    - No resolution_value
    - No price (price_yes is None)
    - Zero volume (price_yes is post-settlement, not informative)

    Returns:
        (X, y) where X is (n_samples, n_features), y is (n_samples,) binary
    """
    X_rows = []
    y_rows = []
    skipped = {"no_resolution": 0, "no_price": 0, "zero_volume": 0}

    for m in markets:
        if m.resolution_value is None:
            skipped["no_resolution"] += 1
            continue
        if m.price_yes is None:
            skipped["no_price"] += 1
            continue
        if (m.volume_total or 0) <= 0:
            skipped["zero_volume"] += 1
            continue

        feat = extract_features_from_market(m, for_training=True)
        X_rows.append(features_to_array(feat))
        y_rows.append(m.resolution_value)

    if skipped["zero_volume"] > 0 or skipped["no_resolution"] > 0:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(
            f"Training matrix: {len(X_rows)} usable, "
            f"skipped {skipped['zero_volume']} zero-volume, "
            f"{skipped['no_resolution']} no-resolution, "
            f"{skipped['no_price']} no-price"
        )

    return np.array(X_rows), np.array(y_rows)
