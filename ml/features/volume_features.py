"""Volume pattern features for ML models.

Volume dynamics can signal market confidence and resolution proximity:
- Increasing volume suggests more information arrival
- Decreasing volume suggests market converging to equilibrium
- Volume spikes often precede resolution
"""

import numpy as np
from typing import Optional


def compute_volume_features(
    volume_history: list[dict],  # [{"timestamp": dt, "volume": float}, ...]
    current_volume: float = 0.0,
    open_interest: float = 0.0,
) -> dict:
    """Extract 4 volume pattern features from historical volume data.

    Args:
        volume_history: List of dicts with "timestamp" and "volume" keys,
                        sorted by timestamp (oldest to newest)
        current_volume: Total volume traded (for volume_to_liquidity_ratio)
        open_interest: Current open interest/liquidity (for ratio)

    Returns:
        Dict with 4 volume features
    """
    if not volume_history or len(volume_history) < 2:
        return _empty_features()

    volumes = np.array([v["volume"] for v in volume_history])

    # Feature 1: Volume trend (7-day linear regression slope)
    # Positive slope = increasing volume (more activity)
    # Negative slope = decreasing volume (market quieting)
    volume_trend_7d = _compute_linear_slope(volumes)

    # Feature 2: Volume volatility (std dev of volumes)
    # High volatility = erratic trading activity
    # Low volatility = stable engagement
    volume_volatility = float(np.std(volumes)) if len(volumes) > 1 else 0.0

    # Feature 3: Volume acceleration (2nd derivative)
    # Positive = volume trend increasing (accelerating interest)
    # Negative = volume trend decreasing (slowing down)
    volume_acceleration = _compute_acceleration(volumes)

    # Feature 4: Volume to liquidity ratio
    # Trading intensity relative to available liquidity
    # High ratio = lots of turnover (price discovery ongoing)
    # Low ratio = thin trading (price may be stale)
    if open_interest > 0:
        volume_to_liquidity_ratio = current_volume / open_interest
    else:
        volume_to_liquidity_ratio = 0.0

    return {
        "volume_trend_7d": volume_trend_7d,
        "volume_volatility": volume_volatility,
        "volume_acceleration": volume_acceleration,
        "volume_to_liquidity_ratio": volume_to_liquidity_ratio,
    }


def _compute_linear_slope(values: np.ndarray) -> float:
    """Compute linear regression slope (trend direction)."""
    if len(values) < 2:
        return 0.0

    x = np.arange(len(values))
    # Normalize to [-1, 1] range for consistent feature scaling
    if np.std(values) > 0:
        # Fit linear regression y = mx + b
        coef = np.polyfit(x, values, deg=1)[0]
        # Normalize by mean volume to get percentage change per period
        mean_vol = np.mean(values)
        if mean_vol > 0:
            normalized_slope = coef / mean_vol
            # Clip to reasonable range
            return float(np.clip(normalized_slope, -1.0, 1.0))
    return 0.0


def _compute_acceleration(values: np.ndarray) -> float:
    """Compute 2nd derivative (change in trend).

    Measures if volume is accelerating (getting more intense)
    or decelerating (tapering off).
    """
    if len(values) < 3:
        return 0.0

    # Split into two halves and compare slopes
    mid = len(values) // 2
    first_half = values[:mid]
    second_half = values[mid:]

    slope1 = _compute_linear_slope(first_half)
    slope2 = _compute_linear_slope(second_half)

    # Acceleration = change in slope
    acceleration = slope2 - slope1

    # Clip to [-1, 1]
    return float(np.clip(acceleration, -1.0, 1.0))


def _empty_features() -> dict:
    """Default values when no volume history is available."""
    return {
        "volume_trend_7d": 0.0,
        "volume_volatility": 0.0,
        "volume_acceleration": 0.0,
        "volume_to_liquidity_ratio": 0.0,
    }
