"""Platform- and category-aware calibration lookup with fallback hierarchy.

Provides data-driven calibration estimates trained on our own resolved market data
instead of generic static tables.  Falls back gracefully when sample sizes are
insufficient or the curves file doesn't exist yet.

Fallback order (most specific → least specific):
    1. (platform, category) — if ≥200 samples in that bucket
    2. (platform, "default") — platform-level aggregate
    3. "global"              — all-platform aggregate
    4. HISTORICAL_CALIBRATION — static research table (always available)

The curves file is produced by:
    python scripts/build_calibration_curves.py

File format (ml/saved_models/calibration_curves.json):
    {
        "polymarket": {
            "politics": {"buckets": [...], "rates": [...], "n_samples": 412},
            "sports":   {"buckets": [...], "rates": [...], "n_samples": 231},
            "default":  {"buckets": [...], "rates": [...], "n_samples": 1803}
        },
        "kalshi": {
            "default": {"buckets": [...], "rates": [...], "n_samples": 87}
        },
        "global": {"buckets": [...], "rates": [...], "n_samples": 3600}
    }
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_CURVES_PATH = Path(__file__).parent.parent / "saved_models" / "calibration_curves.json"
_MIN_SAMPLES = 200  # Minimum samples to trust a bucket-level curve

# Module-level cache — loaded once, reloaded if file changes
_cache: dict | None = None
_cache_lock = threading.Lock()
_cache_mtime: float = 0.0


def _load_curves() -> dict:
    """Load calibration curves from disk, using in-memory cache."""
    global _cache, _cache_mtime

    if not _CURVES_PATH.exists():
        return {}

    try:
        mtime = _CURVES_PATH.stat().st_mtime
    except OSError:
        return {}

    with _cache_lock:
        if _cache is not None and mtime <= _cache_mtime:
            return _cache
        try:
            with open(_CURVES_PATH) as f:
                _cache = json.load(f)
            _cache_mtime = mtime
            logger.debug(f"Loaded calibration curves from {_CURVES_PATH}")
        except Exception as e:
            logger.warning(f"Failed to load calibration_curves.json: {e}")
            _cache = {}
        return _cache or {}


def _interpolate(price: float, buckets: list[float], rates: list[float]) -> float | None:
    """Linear interpolation over (buckets, rates) arrays.

    Returns None if arrays are empty or don't bracket the price.
    """
    if not buckets or not rates or len(buckets) != len(rates):
        return None

    if price <= buckets[0]:
        return rates[0]
    if price >= buckets[-1]:
        return rates[-1]

    for i in range(len(buckets) - 1):
        if buckets[i] <= price <= buckets[i + 1]:
            span = buckets[i + 1] - buckets[i]
            if span == 0:
                return rates[i]
            t = (price - buckets[i]) / span
            return rates[i] + t * (rates[i + 1] - rates[i])

    return None


def _lookup_curve(curves: dict, key: str, price: float) -> float | None:
    """Try to find a curve under `curves[key]` and interpolate."""
    curve_data = curves.get(key)
    if not curve_data:
        return None
    if isinstance(curve_data, dict) and "buckets" in curve_data:
        n = curve_data.get("n_samples", 0)
        if n < _MIN_SAMPLES:
            return None
        return _interpolate(price, curve_data["buckets"], curve_data["rates"])
    return None


def get_calibration_estimate(
    price: float,
    platform: str | None = None,
    category: str | None = None,
) -> float:
    """Return calibrated resolution probability for a given market price.

    Uses the fallback hierarchy described in the module docstring.
    Always returns a float in [0, 1].

    Args:
        price:    Market-implied YES probability (0–1).
        platform: Platform name string, e.g. "polymarket", "kalshi".
                  Case-insensitive. Pass None to skip platform-specific lookup.
        category: Normalised category string, e.g. "politics", "sports".
                  Case-insensitive. Pass None to use platform default.

    Returns:
        Calibrated resolution probability estimate.
    """
    # Clamp input
    price = max(0.0, min(1.0, float(price)))

    curves = _load_curves()
    pname = (platform or "").lower().strip()
    cat = (category or "").lower().strip()

    if curves:
        # 1. Platform + category
        if pname and cat:
            est = _lookup_curve(curves.get(pname, {}), cat, price)
            if est is not None:
                return max(0.0, min(1.0, est))

        # 2. Platform default
        if pname:
            est = _lookup_curve(curves.get(pname, {}), "default", price)
            if est is not None:
                return max(0.0, min(1.0, est))

        # 3. Global
        est = _lookup_curve(curves, "global", price)
        if est is not None:
            return max(0.0, min(1.0, est))

    # 4. Static HISTORICAL_CALIBRATION fallback (always available)
    from ml.features.calibration_features import (
        HISTORICAL_CALIBRATION,
        get_calibration_estimate as _static_estimate,
    )
    return _static_estimate(price)


def is_curves_file_available() -> bool:
    """Return True if the calibration curves file exists and is non-empty."""
    return _CURVES_PATH.exists() and _CURVES_PATH.stat().st_size > 10
