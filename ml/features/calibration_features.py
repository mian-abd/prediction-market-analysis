"""Calibration features - detect systematic market bias.
Markets at 80% historically resolve at ~74% (6pp overconfidence)."""

import numpy as np


# Historical calibration data from published research
# (price_bucket_center, actual_resolution_rate)
# Source: Multiple studies on Polymarket/Kalshi calibration
HISTORICAL_CALIBRATION = {
    0.05: 0.08,   # 5% priced events resolve YES 8% of the time
    0.10: 0.13,
    0.15: 0.17,
    0.20: 0.26,   # 20% events → 26% actual (underpriced)
    0.25: 0.29,
    0.30: 0.33,
    0.35: 0.37,
    0.40: 0.42,
    0.45: 0.46,
    0.50: 0.50,   # 50% is well-calibrated
    0.55: 0.54,
    0.60: 0.58,
    0.65: 0.62,
    0.70: 0.66,
    0.75: 0.71,
    0.80: 0.74,   # 80% events → 74% actual (overpriced)
    0.85: 0.80,
    0.90: 0.87,
    0.95: 0.92,
}


def get_calibration_estimate(market_price: float) -> float:
    """Get historical resolution rate for a given market price.
    Uses linear interpolation between known calibration points.
    """
    if market_price <= 0.05:
        return HISTORICAL_CALIBRATION[0.05]
    if market_price >= 0.95:
        return HISTORICAL_CALIBRATION[0.95]

    # Find surrounding buckets
    buckets = sorted(HISTORICAL_CALIBRATION.keys())
    for i in range(len(buckets) - 1):
        if buckets[i] <= market_price <= buckets[i + 1]:
            # Linear interpolation
            t = (market_price - buckets[i]) / (buckets[i + 1] - buckets[i])
            low_val = HISTORICAL_CALIBRATION[buckets[i]]
            high_val = HISTORICAL_CALIBRATION[buckets[i + 1]]
            return low_val + t * (high_val - low_val)

    return market_price  # Fallback


def compute_calibration_features(
    market_price: float,
    price_history_24h: list[float] | None = None,
    market_age_days: float = 0,
) -> dict:
    """Extract 6 calibration features."""
    # Feature 1: Calibration bias (historical_rate - market_price)
    calibrated = get_calibration_estimate(market_price)
    calibration_bias = calibrated - market_price

    # Feature 2: Overconfidence delta (positive = market is overconfident)
    overconfidence_delta = market_price - calibrated

    # Feature 3: Historical resolution rate for this price level
    historical_resolution = calibrated

    # Feature 4: Market age in days
    # Feature 5: Price stability (max - min in 24h)
    if price_history_24h and len(price_history_24h) >= 2:
        price_stability = max(price_history_24h) - min(price_history_24h)
    else:
        price_stability = 0.0

    # Feature 6: Spread percentile (simplified - based on distance from 0.5)
    # Markets near 50% tend to have tighter spreads
    spread_percentile = abs(market_price - 0.5) * 2  # 0 at 50%, 1 at extremes

    return {
        "calibration_bias": calibration_bias,
        "overconfidence_delta": overconfidence_delta,
        "historical_resolution": historical_resolution,
        "market_age_days": market_age_days,
        "price_stability_24h": price_stability,
        "spread_percentile": spread_percentile,
    }
