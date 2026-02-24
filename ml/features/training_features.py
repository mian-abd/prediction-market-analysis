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
from ml.features.momentum_features import compute_momentum_features
from ml.features.orderbook_features import compute_orderbook_features
from ml.features.volume_features import compute_volume_features

# Category encoding map (matches market_features.py)
CATEGORY_MAP = {
    "politics": 0, "crypto": 1, "sports": 2, "science": 3,
    "entertainment": 4, "economics": 5, "technology": 6,
    "weather": 7, "culture": 8, "other": 9,
}

# Canonical ordered list of features for ensemble models
# NOTE: 3 redundant features removed (overconfidence_delta = -calibration_bias,
# historical_resolution = deterministic(price_yes), spread_percentile = 2*price_distance_from_50)
ENSEMBLE_FEATURE_NAMES: list[str] = [
    # Price features (only meaningful when volume > 0)
    "price_yes",
    "price_bucket",
    "price_distance_from_50",
    # Volume/liquidity features
    # EXCLUDED 2026-02-14: log_volume_total (0.664 correlation with resolution - contaminated)
    # "log_volume_total",
    "log_open_interest",  # Keep open_interest (less contaminated)
    # Calibration features (derived from price_yes)
    "calibration_bias",
    # Time features
    "time_to_resolution_hrs",
    # EXCLUDED 2026-02-14: volume_per_day (derived from contaminated volume_24h)
    # "volume_per_day",
    # Market features (from market_features.py)
    "category_encoded",
    "is_weekend",
    # Momentum features (from momentum_features.py, need price_snapshots)
    "return_1h",
    "volatility_20",
    "zscore_24h",
    # Orderbook features (from orderbook_features.py, need orderbook_snapshot)
    "obi_level1",
    "obi_weighted_5",
    "bid_ask_spread_abs",
    "bid_ask_spread_rel",
    "depth_ratio",
    "bid_depth_usd",
    "ask_depth_usd",
    "vwap_deviation",
    # Volume pattern features (from volume_features.py)
    # EXCLUDED 2026-02-14: All volume pattern features show high correlation with resolution
    # - volume_trend_7d: 0.809 correlation (post-resolution spike contamination)
    # - volume_volatility: 0.853 correlation (SEVERE contamination, was 55.6% XGBoost importance)
    # - volume_acceleration: likely contaminated (not tested but derived from volume_24h)
    # - volume_to_liquidity_ratio: derived from contaminated volume
    # "volume_trend_7d",
    # "volume_volatility",
    # "volume_acceleration",
    # "volume_to_liquidity_ratio",
    # Cross-platform features (Phase 2.4)
    "cross_platform_spread",  # poly_price - kalshi_price (informed trading signal)
]

N_FEATURES = len(ENSEMBLE_FEATURE_NAMES)  # 20 (added cross_platform_spread 2026-02-22)

# Feature quality metadata — "real" = derived from live market data,
# "proxy" = synthetic approximation from snapshot-level data
FEATURE_QUALITY: dict[str, str] = {
    "price_yes": "real",
    "price_bucket": "real",
    "price_distance_from_50": "real",
    "log_open_interest": "real",
    "calibration_bias": "real",
    "time_to_resolution_hrs": "real",
    "category_encoded": "real",
    "is_weekend": "real",
    "return_1h": "proxy",       # defaults to 0 when <2 snapshots
    "volatility_20": "proxy",   # defaults to 0 when <2 snapshots
    "zscore_24h": "proxy",      # defaults to 0 when <2 snapshots
    "obi_level1": "proxy",      # defaults to 0 when no orderbook
    "obi_weighted_5": "proxy",
    "bid_ask_spread_abs": "proxy",
    "bid_ask_spread_rel": "proxy",
    "depth_ratio": "proxy",     # defaults to 1.0 when no orderbook
    "bid_depth_usd": "proxy",
    "ask_depth_usd": "proxy",
    "vwap_deviation": "proxy",
}


def get_feature_quality_summary(features: dict[str, float] | None = None) -> dict:
    """Summarize feature quality for API responses."""
    total = len(FEATURE_QUALITY)
    real = sum(1 for v in FEATURE_QUALITY.values() if v == "real")
    proxy = total - real

    result = {
        "total_features": total,
        "real_features": real,
        "proxy_features": proxy,
        "real_pct": round(real / total * 100, 1),
    }

    # If features provided, count how many proxy features are at defaults
    if features:
        proxy_at_default = 0
        for name, quality in FEATURE_QUALITY.items():
            if quality == "proxy" and name in features:
                val = features[name]
                if val == 0.0 or (name == "depth_ratio" and val == 1.0):
                    proxy_at_default += 1
        result["proxy_at_default"] = proxy_at_default
        result["features_with_real_data"] = real + (proxy - proxy_at_default)

    return result


def extract_features_from_market(
    market,
    for_training: bool = False,
    price_snapshots: list[float] | None = None,
    orderbook_snapshot = None,
    price_yes_override: float | None = None,
    matched_market_price: float | None = None,
) -> dict[str, float]:
    """Extract features from a Market ORM object.

    Args:
        market: Market ORM object (or any object with the right attributes)
        for_training: If True, compute time features relative to market's
                      own timeline (not current time).
        price_snapshots: Optional list of historical prices (oldest to newest)
                         for momentum feature computation.
        orderbook_snapshot: Optional OrderbookSnapshot ORM object with
                            bids_json/asks_json for orderbook features.
        price_yes_override: Optional price override for training (as_of enforcement).
                            If provided, use this instead of market.price_yes.
                            This prevents leakage from using resolved outcome price.
        matched_market_price: Optional cross-platform matched market price (Kalshi if
                             current is Polymarket, or vice versa). Used to compute
                             cross_platform_spread feature. Defaults to 0.0 if not provided.

    Returns:
        Dict of {feature_name: float_value}
    """
    # Use override if provided (for as_of enforcement), else market.price_yes
    price = price_yes_override if price_yes_override is not None else (market.price_yes if market.price_yes else 0.5)

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

    # Category feature (uses normalized_category if available, falls back to category)
    category = getattr(market, "normalized_category", None) or getattr(market, "category", "other") or "other"
    category_encoded = float(CATEGORY_MAP.get(category.lower(), 9))

    # Weekend feature
    if for_training:
        # For training: check if market was created on weekend
        created = getattr(market, "created_at", None)
        is_weekend = 1.0 if (created and created.weekday() >= 5) else 0.0
    else:
        is_weekend = 1.0 if datetime.utcnow().weekday() >= 5 else 0.0

    # Momentum features (from price snapshots)
    if price_snapshots and len(price_snapshots) >= 2:
        mom = compute_momentum_features(price_snapshots)
        return_1h = mom["return_1"]
        volatility_20 = mom["volatility_20"]
        zscore_24h = mom["zscore_24h"]
    else:
        return_1h = 0.0
        volatility_20 = 0.0
        zscore_24h = 0.0

    # Orderbook features (from orderbook snapshot)
    if orderbook_snapshot and hasattr(orderbook_snapshot, 'bids_json') and hasattr(orderbook_snapshot, 'asks_json'):
        bids = orderbook_snapshot.bids_json or []
        asks = orderbook_snapshot.asks_json or []
        if bids and asks:
            ob_features = compute_orderbook_features(bids, asks)
        else:
            ob_features = {
                "obi_level1": 0.0,
                "obi_weighted_5": 0.0,
                "bid_ask_spread_abs": 0.0,
                "bid_ask_spread_rel": 0.0,
                "depth_ratio": 1.0,
                "bid_depth_usd": 0.0,
                "ask_depth_usd": 0.0,
                "vwap_deviation": 0.0,
            }
    else:
        # Default to neutral values when no orderbook data
        ob_features = {
            "obi_level1": 0.0,
            "obi_weighted_5": 0.0,
            "bid_ask_spread_abs": 0.0,
            "bid_ask_spread_rel": 0.0,
            "depth_ratio": 1.0,
            "bid_depth_usd": 0.0,
            "ask_depth_usd": 0.0,
            "vwap_deviation": 0.0,
        }

    # Volume pattern features (simplified - no time series required)
    volume_24h = float(market.volume_24h or 0)
    # volume_trend_7d: Use 24h volume relative to total as proxy for trend
    # If 24h volume is high relative to total, market is trending up
    if vol_total > 0:
        volume_trend = (volume_24h / vol_total) - (1.0 / 7.0)  # Compare to 1/7th baseline
        volume_trend_7d = float(np.clip(volume_trend, -1.0, 1.0))
    else:
        volume_trend_7d = 0.0

    # volume_volatility: Can't compute without time series, use volume_24h variability as proxy
    # Normalize by duration to get intensity
    volume_volatility = math.log1p(volume_24h / max(duration_days, 1.0))

    # volume_acceleration: Use second-order difference (volume_24h - expected steady-state)
    # If volume_24h >> volume_total/duration, it's accelerating
    expected_daily = vol_total / max(duration_days, 1.0)
    if expected_daily > 0:
        acceleration_ratio = (volume_24h - expected_daily) / expected_daily
        volume_acceleration = float(np.clip(acceleration_ratio, -1.0, 1.0))
    else:
        volume_acceleration = 0.0

    # volume_to_liquidity_ratio: Trading intensity
    if liquidity > 0:
        volume_to_liquidity_ratio = vol_total / liquidity
    else:
        volume_to_liquidity_ratio = 0.0

    # Cross-platform spread feature (Phase 2.4)
    # Positive spread = Polymarket price > Kalshi price (potential informed buying on Poly)
    # Negative spread = Polymarket price < Kalshi price (potential informed buying on Kalshi)
    # Zero = no cross-platform match or prices equal
    if matched_market_price is not None:
        cross_platform_spread = price - matched_market_price
    else:
        cross_platform_spread = 0.0

    return {
        "price_yes": price,
        "price_bucket": float(price_bucket),
        "price_distance_from_50": price_distance_from_50,
        "log_volume_total": log_volume_total,
        "log_open_interest": log_open_interest,
        "calibration_bias": cf["calibration_bias"],
        "time_to_resolution_hrs": time_hrs,
        "volume_per_day": volume_per_day,
        "category_encoded": category_encoded,
        "is_weekend": is_weekend,
        "return_1h": return_1h,
        "volatility_20": volatility_20,
        "zscore_24h": zscore_24h,
        "obi_level1": ob_features["obi_level1"],
        "obi_weighted_5": ob_features["obi_weighted_5"],
        "bid_ask_spread_abs": ob_features["bid_ask_spread_abs"],
        "bid_ask_spread_rel": ob_features["bid_ask_spread_rel"],
        "depth_ratio": ob_features["depth_ratio"],
        "bid_depth_usd": ob_features["bid_depth_usd"],
        "ask_depth_usd": ob_features["ask_depth_usd"],
        "vwap_deviation": ob_features["vwap_deviation"],
        "volume_trend_7d": volume_trend_7d,
        "volume_volatility": volume_volatility,
        "volume_acceleration": volume_acceleration,
        "volume_to_liquidity_ratio": volume_to_liquidity_ratio,
        "cross_platform_spread": cross_platform_spread,
    }


def prune_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Remove features with zero variance or near-constant values.

    Hard gates:
        - Zero variance (constant columns) → dropped
        - <5% unique values → dropped (near-constant, e.g. all defaults)

    Diagnostic only (logged but NOT auto-dropped):
        - Univariate AUC — helps understand signal but interaction-only
          features would be wrongly killed by an AUC gate.

    Returns:
        (X_pruned, active_feature_names, dropped_descriptions)
    """
    import logging
    from sklearn.metrics import roc_auc_score

    logger = logging.getLogger(__name__)
    active_indices: list[int] = []
    dropped: list[str] = []

    for i, name in enumerate(feature_names):
        col = X[:, i]

        # Hard gate 1: zero variance
        if np.var(col) < 1e-10:
            dropped.append(f"{name} (zero variance)")
            continue

        # Hard gate 2: near-constant (<5% unique values)
        n_unique = len(np.unique(col))
        if n_unique / len(col) < 0.05:
            dropped.append(f"{name} (near-constant, {n_unique} unique/{len(col)})")
            continue

        # Diagnostic: univariate AUC (info only, don't auto-drop)
        try:
            auc = roc_auc_score(y, col)
            logger.info(f"  {name}: AUC={auc:.3f}")
        except ValueError:
            logger.info(f"  {name}: AUC=N/A")

        active_indices.append(i)

    X_pruned = X[:, active_indices]
    active_names = [feature_names[i] for i in active_indices]

    if dropped:
        logger.info(f"Pruned {len(dropped)} features: {dropped}")
    logger.info(f"Active features: {len(active_names)}/{len(feature_names)}")

    return X_pruned, active_names, dropped


def features_to_array(features: dict[str, float]) -> np.ndarray:
    """Convert feature dict to numpy array in canonical order."""
    return np.array([features[name] for name in ENSEMBLE_FEATURE_NAMES])


def build_training_matrix(
    markets: list,
    price_snapshots_map: dict[int, list[float]] | None = None,
    price_at_as_of_map: dict[int, float] | None = None,
    orderbook_snapshots_map: dict[int, any] | None = None,
    snapshot_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build X (features) and y (labels) from resolved markets.

    Filters out markets with:
    - No resolution_value
    - No price (price_yes is None or no price at as_of)
    - Zero volume (price_yes is post-settlement, not informative)

    Args:
        markets: List of Market ORM objects
        price_snapshots_map: Optional {market_id: [prices]} for momentum features (filtered to as_of).
                             If None, momentum features default to 0.0.
        price_at_as_of_map: Optional {market_id: price_at_as_of} for as_of enforcement.
                            If provided, use this as price_yes_override (prevents leakage).
                            Markets without an as_of price are skipped (strict mode).
        orderbook_snapshots_map: Optional {market_id: OrderbookSnapshot} for orderbook features.
                                 If None, orderbook features default to 0.0/neutral.
        snapshot_only: If True, ONLY include markets with a real as_of snapshot price
                       (skips the market.price_yes fallback entirely). This eliminates
                       leakage from resolved-market prices at the cost of fewer samples.
                       Recommended once as_of coverage exceeds ~20%.

    Returns:
        (X, y) where X is (n_samples, n_features), y is (n_samples,) binary
    """
    X_rows = []
    y_rows = []
    skipped = {"no_resolution": 0, "no_price": 0, "zero_volume": 0, "fallback_price": 0, "no_snapshot": 0}
    as_of_used = 0  # Count markets that used clean as_of price (not fallback)
    snapshots_map = price_snapshots_map or {}
    as_of_map = price_at_as_of_map or {}
    ob_snapshots_map = orderbook_snapshots_map or {}

    for m in markets:
        if m.resolution_value is None:
            skipped["no_resolution"] += 1
            continue

        if as_of_map and m.id not in as_of_map:
            if snapshot_only:
                # Strict mode: skip markets without real as_of price to avoid leakage.
                # market.price_yes for resolved markets reflects the outcome (≈0 or ≈1),
                # which contaminates training with spurious signal.
                skipped["no_snapshot"] += 1
                continue
            # Graceful fallback: use market.price_yes for non-backfilled markets.
            # Useful when as_of coverage is low (<20%), but introduces leakage.
            if m.price_yes is None:
                skipped["no_price"] += 1
                continue
            skipped["fallback_price"] += 1
            # Continue without skipping — price_override will be None,
            # so extract_features_from_market uses market.price_yes

        if not as_of_map and m.price_yes is None:
            skipped["no_price"] += 1
            continue

        if (m.volume_total or 0) <= 0:
            skipped["zero_volume"] += 1
            continue

        # Get price snapshots for this market (already filtered to as_of)
        snapshots = snapshots_map.get(m.id)
        # Get orderbook snapshot for this market (if available)
        ob_snapshot = ob_snapshots_map.get(m.id)
        # Get price at as_of (overrides market.price_yes to prevent leakage)
        price_override = as_of_map.get(m.id) if as_of_map else None

        feat = extract_features_from_market(
            m,
            for_training=True,
            price_snapshots=snapshots,
            orderbook_snapshot=ob_snapshot,
            price_yes_override=price_override,
        )
        X_rows.append(features_to_array(feat))
        y_rows.append(m.resolution_value)
        if as_of_map and m.id in as_of_map:
            as_of_used += 1

    if skipped["zero_volume"] > 0 or skipped["no_resolution"] > 0 or skipped["fallback_price"] > 0:
        import logging
        logger = logging.getLogger(__name__)
        skip_msg = f"Training matrix: {len(X_rows)} usable, skipped "
        skip_msg += f"{skipped['zero_volume']} zero-volume, "
        skip_msg += f"{skipped['no_resolution']} no-resolution, "
        skip_msg += f"{skipped['no_price']} no-price"
        if skipped.get("no_snapshot"):
            skip_msg += f", {skipped['no_snapshot']} no-snapshot (snapshot_only=True)"
        if as_of_map:
            n_fallback = len(X_rows) - as_of_used
            if n_fallback > 0:
                skip_msg += f" | {as_of_used} with as_of price, {n_fallback} using market.price_yes fallback"
            else:
                skip_msg += f" | {as_of_used} with as_of price, 0 fallbacks (clean)"
        logger.info(skip_msg)
        if snapshots_map:
            with_momentum = sum(1 for m in markets if m.id in snapshots_map and len(snapshots_map[m.id]) >= 2)
            logger.info(f"Markets with momentum data: {with_momentum}/{len(X_rows)}")
        if ob_snapshots_map:
            with_orderbook = sum(1 for m in markets if m.id in ob_snapshots_map)
            logger.info(f"Markets with orderbook data: {with_orderbook}/{len(X_rows)}")

    return np.array(X_rows), np.array(y_rows)
