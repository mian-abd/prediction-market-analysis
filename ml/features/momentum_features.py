"""Momentum and mean-reversion features from price history."""

import numpy as np
from typing import Optional


def compute_momentum_features(
    prices: list[float],
    timestamps: list[float] | None = None,
) -> dict:
    """Extract 10 momentum features from price time series.
    prices: list of prices ordered oldest to newest.
    """
    if len(prices) < 2:
        return _empty_features()

    prices_arr = np.array(prices, dtype=float)
    current = prices_arr[-1]

    # Returns over various lookbacks
    def safe_return(lookback: int) -> float:
        if len(prices_arr) > lookback and prices_arr[-(lookback+1)] > 0:
            return (current - prices_arr[-(lookback+1)]) / prices_arr[-(lookback+1)]
        return 0.0

    # Feature 1-6: Returns at different horizons
    return_1 = safe_return(1)     # ~1 interval ago
    return_5 = safe_return(5)     # ~5 intervals
    return_15 = safe_return(15)   # ~15 intervals
    return_60 = safe_return(60)   # ~60 intervals
    return_240 = safe_return(240) # ~240 intervals
    return_1440 = safe_return(min(1440, len(prices_arr) - 1))  # ~1 day if 1-min data

    # Feature 7: Rolling volatility (20-period)
    if len(prices_arr) >= 20:
        returns = np.diff(prices_arr[-21:]) / prices_arr[-21:-1]
        returns = returns[np.isfinite(returns)]
        volatility_20 = float(np.std(returns)) if len(returns) > 1 else 0.0
    else:
        volatility_20 = 0.0

    # Feature 8: Z-score vs 24h SMA (mean-reversion signal)
    lookback_24h = min(1440, len(prices_arr))
    if lookback_24h >= 10:
        sma = np.mean(prices_arr[-lookback_24h:])
        std = np.std(prices_arr[-lookback_24h:])
        zscore_24h = float((current - sma) / std) if std > 0 else 0.0
    else:
        zscore_24h = 0.0

    # Feature 9: Momentum RSI-like (14 periods)
    if len(prices_arr) >= 15:
        changes = np.diff(prices_arr[-15:])
        gains = np.mean(changes[changes > 0]) if np.any(changes > 0) else 0.0
        losses = abs(np.mean(changes[changes < 0])) if np.any(changes < 0) else 0.0
        if losses > 0:
            rs = gains / losses
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 100.0 if gains > 0 else 50.0
        momentum_rsi = rsi / 100.0  # Normalize to 0-1
    else:
        momentum_rsi = 0.5

    # Feature 10: Volume spike (would need volume data, placeholder)
    volume_spike = 0.0

    return {
        "return_1": return_1,
        "return_5": return_5,
        "return_15": return_15,
        "return_60": return_60,
        "return_240": return_240,
        "return_1440": return_1440,
        "volatility_20": volatility_20,
        "zscore_24h": zscore_24h,
        "momentum_rsi": momentum_rsi,
        "volume_spike": volume_spike,
    }


def _empty_features() -> dict:
    return {
        "return_1": 0.0,
        "return_5": 0.0,
        "return_15": 0.0,
        "return_60": 0.0,
        "return_240": 0.0,
        "return_1440": 0.0,
        "volatility_20": 0.0,
        "zscore_24h": 0.0,
        "momentum_rsi": 0.5,
        "volume_spike": 0.0,
    }
