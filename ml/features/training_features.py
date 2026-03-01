"""Unified feature extraction for ML training and inference.

Produces a consistent feature vector from Market ORM objects.
Both training and serving call the SAME function with the SAME interface.
The only difference is the `as_of` timestamp:
  - Training: as_of = resolved_at - timedelta(days=N)
  - Serving:  as_of = datetime.utcnow()

LEAKAGE RULES:
- Markets with volume_total = 0 are excluded from training
  (their price_yes is post-settlement zeros, not informative)
- price_yes is ONLY used when it represents a real traded price
- ALL features are computed relative to as_of — no future data
"""

import math
import numpy as np
from datetime import datetime, timedelta

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
    # Volume/liquidity features — ALL EXCLUDED
    # EXCLUDED 2026-02-14: log_volume_total (0.664 correlation with resolution)
    # "log_volume_total",
    # EXCLUDED 2026-02-27: log_open_interest — market.liquidity is the FINAL value;
    # for resolved markets it collapses toward 0, creating leakage. No historical
    # liquidity snapshots exist to provide a clean as_of value.
    # "log_open_interest",
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
    # Order flow features (from trade history, computed in orderflow_features.py)
    # These capture informed trading patterns: buy/sell imbalance, trade intensity,
    # large-trade signals, VWAP deviation. Default to 0.0 when no trade data available.
    "oflow_buy_sell_ratio_1h",
    "oflow_buy_sell_ratio_24h",
    "oflow_trade_intensity_1h",
    "oflow_trade_intensity_24h",
    "oflow_avg_trade_size_24h",
    "oflow_large_trade_count_24h",
    "oflow_large_trade_count_7d",
    "oflow_vwap_deviation_24h",
    "oflow_volume_acceleration_6h",
    "oflow_total_volume_24h",
]

N_FEATURES = len(ENSEMBLE_FEATURE_NAMES)  # 28

FEATURE_QUALITY: dict[str, str] = {
    "price_yes": "real",
    "price_bucket": "real",
    "price_distance_from_50": "real",
    "calibration_bias": "real",
    "time_to_resolution_hrs": "real",
    "category_encoded": "real",
    "is_weekend": "real",
    "return_1h": "proxy",
    "volatility_20": "proxy",
    "zscore_24h": "proxy",
    "obi_level1": "proxy",
    "obi_weighted_5": "proxy",
    "bid_ask_spread_abs": "proxy",
    "bid_ask_spread_rel": "proxy",
    "depth_ratio": "proxy",
    "bid_depth_usd": "proxy",
    "ask_depth_usd": "proxy",
    "vwap_deviation": "proxy",
    "oflow_buy_sell_ratio_1h": "real",
    "oflow_buy_sell_ratio_24h": "real",
    "oflow_trade_intensity_1h": "real",
    "oflow_trade_intensity_24h": "real",
    "oflow_avg_trade_size_24h": "real",
    "oflow_large_trade_count_24h": "real",
    "oflow_large_trade_count_7d": "real",
    "oflow_vwap_deviation_24h": "real",
    "oflow_volume_acceleration_6h": "real",
    "oflow_total_volume_24h": "real",
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
    as_of: datetime | None = None,
    price_snapshots: list[float] | None = None,
    orderbook_snapshot=None,
    price_yes_override: float | None = None,
    matched_market_price: float | None = None,
    for_training: bool = False,
    trades: list[dict] | None = None,
) -> dict[str, float]:
    """Extract features from a Market ORM object.

    CRITICAL: Both training and serving MUST call this function with the same
    interface. The `as_of` parameter is the single source of truth for "what
    time is it?" — all time-dependent features are computed relative to it.

    Args:
        market: Market ORM object (or any object with the right attributes)
        as_of: Point-in-time for feature computation. REQUIRED for honest features.
               Training: as_of = resolved_at - timedelta(days=1) (or snapshot time)
               Serving:  as_of = datetime.utcnow()
               If None, defaults to datetime.utcnow() for backward compatibility.
        price_snapshots: Optional list of historical prices (oldest to newest)
                         for momentum feature computation. Must be filtered to
                         only include snapshots BEFORE as_of.
        orderbook_snapshot: Optional OrderbookSnapshot ORM object with
                            bids_json/asks_json for orderbook features.
                            Must be the latest snapshot BEFORE as_of.
        price_yes_override: Optional price override (as_of enforcement).
                            If provided, use this instead of market.price_yes.
                            This prevents leakage from using resolved outcome price.
        matched_market_price: Optional cross-platform matched market price.
        for_training: DEPRECATED. Kept for backward compatibility only.
                      Use as_of instead. Will be removed in future version.

    Returns:
        Dict of {feature_name: float_value}
    """
    reference_time = as_of or datetime.utcnow()

    # Use override if provided (for as_of enforcement), else market.price_yes
    price = price_yes_override if price_yes_override is not None else (float(market.price_yes) if market.price_yes is not None else 0.5)

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

    # Time features: ALWAYS computed as (end_date - reference_time)
    # This is identical for training and serving — the only difference is
    # what reference_time is (as_of for training, now() for serving).
    if market.end_date:
        try:
            end_naive = market.end_date.replace(tzinfo=None) if market.end_date.tzinfo else market.end_date
            ref_naive = reference_time.replace(tzinfo=None) if reference_time.tzinfo else reference_time
            delta = end_naive - ref_naive
            time_hrs = max(0, delta.total_seconds() / 3600)
        except (TypeError, AttributeError):
            time_hrs = 168.0
    else:
        time_hrs = 8760.0

    # Volume per day (engagement density)
    duration_days = max(time_hrs / 24, 1.0)
    volume_per_day = vol_total / duration_days

    # Category feature (uses normalized_category if available, falls back to category)
    category = getattr(market, "normalized_category", None) or getattr(market, "category", "other") or "other"
    category_encoded = float(CATEGORY_MAP.get(category.lower(), 9))

    # Weekend feature: ALWAYS computed from reference_time
    is_weekend = 1.0 if reference_time.weekday() >= 5 else 0.0

    # Momentum features (from price snapshots)
    # Accept either plain floats or PriceSnapshot ORM objects
    if price_snapshots and len(price_snapshots) >= 2:
        if hasattr(price_snapshots[0], "price_yes"):
            raw_prices = [float(s.price_yes) for s in price_snapshots if s.price_yes is not None]
        else:
            raw_prices = [float(p) for p in price_snapshots if p is not None]
        mom = compute_momentum_features(raw_prices) if len(raw_prices) >= 2 else None
        if mom is None:
            return_1h = volatility_20 = zscore_24h = 0.0
        else:
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
    if matched_market_price is not None:
        cross_platform_spread = price - matched_market_price
    else:
        cross_platform_spread = 0.0

    # Order flow features (from trade history — optional, defaults to 0.0)
    from ml.features.orderflow_features import compute_orderflow_features, ORDERFLOW_FEATURE_NAMES
    if trades:
        oflow = compute_orderflow_features(trades, as_of=reference_time)
    else:
        oflow = {name: 0.0 for name in ORDERFLOW_FEATURE_NAMES}

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
        **oflow,
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

        # Hard gate 2: dominated by single value (>97% one class)
        # Old gate used unique_ratio < 5%, which incorrectly killed binary features
        # like is_weekend (2 unique values / 200 samples = 1%).
        unique_vals, counts = np.unique(col, return_counts=True)
        if len(unique_vals) <= 1:
            dropped.append(f"{name} (single value)")
            continue
        dominant_frac = counts.max() / len(col)
        if dominant_frac > 0.97:
            dropped.append(f"{name} (near-constant, {dominant_frac:.1%} dominant)")
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
    tradeable_range: tuple[float, float] | None = None,
    as_of_days: int = 1,
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
        tradeable_range: Optional (min_price, max_price) tuple. When set, only include markets
                         whose as_of price is within this range. Use (0.05, 0.95) to exclude
                         near-decided markets (price near 0 or 1 at as_of time). These are
                         trivially predictable and inflate AUC/Brier; removing them forces the
                         model to learn genuine signal for uncertain, actively-traded markets.
                         Only applies when a real as_of price is available (snapshot markets).

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

        # Tradeable-range filter: skip near-decided markets if range specified.
        # Apply to ALL markets (both snapshot and fallback) to prevent:
        # 1. Snapshot markets: prices near 0/1 are trivially predictable
        # 2. Fallback markets: market.price_yes IS the settlement price (0 or 1),
        #    so near-extreme prices are direct target leakage.
        if tradeable_range is not None:
            check_price = price_override if price_override is not None else (float(m.price_yes) if m.price_yes is not None else None)
            if check_price is not None:
                lo, hi = tradeable_range
                if check_price < lo or check_price > hi:
                    skipped["near_decided"] = skipped.get("near_decided", 0) + 1
                    continue

        # as_of must match the snapshot lookup window used in train_ensemble.py.
        # If snapshots were loaded at resolved_at - N days, features must also be
        # computed at that same point in time. Otherwise price features reflect one
        # time horizon while time features reflect another.
        market_as_of = None
        if m.resolved_at:
            market_as_of = m.resolved_at - timedelta(days=as_of_days)
        elif m.end_date:
            market_as_of = m.end_date - timedelta(days=as_of_days)
        elif m.created_at:
            market_as_of = m.created_at

        feat = extract_features_from_market(
            m,
            as_of=market_as_of,
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
        if skipped.get("near_decided"):
            skip_msg += f", {skipped['near_decided']} near-decided (tradeable_range filter)"
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


async def load_serving_context(session, market_id: int, max_snapshots: int = 48):
    """Load price snapshots and orderbook data for serving-time feature parity.

    This ensures the serving path has the same feature inputs as training.
    Without this, momentum and orderbook features default to 0.0 at inference
    while having real values during training — a critical train/serve skew.

    Args:
        session: AsyncSession for DB queries
        market_id: Market ID to load context for
        max_snapshots: Maximum number of recent price snapshots to load

    Returns:
        (price_snapshots, orderbook_snapshot) tuple
    """
    from db.models import PriceSnapshot, OrderbookSnapshot
    from sqlalchemy import select

    snapshots_result = await session.execute(
        select(PriceSnapshot)
        .where(PriceSnapshot.market_id == market_id)
        .order_by(PriceSnapshot.timestamp.desc())
        .limit(max_snapshots)
    )
    price_snapshots = list(reversed(snapshots_result.scalars().all()))

    ob_result = await session.execute(
        select(OrderbookSnapshot)
        .where(OrderbookSnapshot.market_id == market_id)
        .order_by(OrderbookSnapshot.timestamp.desc())
        .limit(1)
    )
    orderbook_snapshot = ob_result.scalar_one_or_none()

    return price_snapshots, orderbook_snapshot


# ── Variable as_of Horizons ──────────────────────────────────────────

VARIABLE_AS_OF_DAYS = [1, 2, 3, 5, 7, 14, 30]


def build_variable_as_of_matrix(
    markets: list,
    raw_price_snapshots: dict[int, list[tuple]],
    raw_orderbook_snapshots: dict[int, list] | None = None,
    raw_trades: dict[int, list[dict]] | None = None,
    tradeable_range: tuple[float, float] | None = None,
    horizons: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build training matrix with multiple as_of time horizons per market.

    For each resolved market, creates samples at multiple time points before
    resolution, producing natural variance in time_to_resolution_hrs and
    price-at-different-horizons. This multiplies effective training set size
    by up to len(horizons)x while maintaining temporal integrity.

    Args:
        markets: List of Market ORM objects, ordered by resolved_at
        raw_price_snapshots: {market_id: [(timestamp, price), ...]} — ALL snapshots
                             (unfiltered). Will be filtered per-horizon internally.
        raw_orderbook_snapshots: {market_id: [(timestamp, ob_dict), ...]} — ALL OB snapshots.
        tradeable_range: Optional (lo, hi) price filter (same as build_training_matrix).
        horizons: List of as_of days [1, 2, 3, 5, 7, 14, 30]. Defaults to VARIABLE_AS_OF_DAYS.

    Returns:
        (X, y, market_indices) where:
          X: (n_samples, n_features) feature matrix
          y: (n_samples,) binary labels
          market_indices: (n_samples,) index into `markets` list for each sample.
                          Use this for temporal split: group by market, not sample.
    """
    import logging
    logger = logging.getLogger(__name__)

    if horizons is None:
        horizons = VARIABLE_AS_OF_DAYS

    ob_map = raw_orderbook_snapshots or {}
    trade_map = raw_trades or {}

    X_rows = []
    y_rows = []
    idx_rows = []  # maps each sample back to market index

    stats = {
        "total_samples": 0,
        "markets_used": 0,
        "samples_per_horizon": {h: 0 for h in horizons},
        "skipped_no_resolution": 0,
        "skipped_zero_volume": 0,
        "skipped_no_snapshots": 0,
        "skipped_near_decided": 0,
        "skipped_too_early": 0,
    }

    for market_idx, m in enumerate(markets):
        if m.resolution_value is None:
            stats["skipped_no_resolution"] += 1
            continue

        if (m.volume_total or 0) <= 0:
            stats["skipped_zero_volume"] += 1
            continue

        raw_snaps = raw_price_snapshots.get(m.id)
        if not raw_snaps or len(raw_snaps) < 1:
            stats["skipped_no_snapshots"] += 1
            continue

        ref_date = m.resolved_at or m.end_date
        if not ref_date:
            continue

        market_created = m.created_at or (ref_date - timedelta(days=365))
        added_for_this_market = False

        for h in horizons:
            as_of = ref_date - timedelta(days=h)

            if as_of < market_created:
                stats["skipped_too_early"] += 1
                continue

            filtered = [(ts, p) for (ts, p) in raw_snaps if ts <= as_of]
            if not filtered:
                continue

            filtered.sort(key=lambda x: x[0])
            price_at_as_of = filtered[-1][1]

            if tradeable_range is not None:
                lo, hi = tradeable_range
                if price_at_as_of < lo or price_at_as_of > hi:
                    stats["skipped_near_decided"] += 1
                    continue

            price_list = [p for (_, p) in filtered]

            ob_snapshot = None
            raw_obs = ob_map.get(m.id)
            if raw_obs:
                valid_obs = [(ts, ob) for (ts, ob) in raw_obs if ts <= as_of]
                if valid_obs:
                    valid_obs.sort(key=lambda x: x[0])
                    ob_snapshot = valid_obs[-1][1]

            market_trades = trade_map.get(m.id)
            trades_for_feature = None
            if market_trades:
                trades_for_feature = [
                    t for t in market_trades
                    if t.get("timestamp") and t["timestamp"] <= as_of
                ]

            feat = extract_features_from_market(
                m,
                as_of=as_of,
                price_snapshots=price_list,
                orderbook_snapshot=ob_snapshot,
                price_yes_override=price_at_as_of,
                trades=trades_for_feature,
            )
            X_rows.append(features_to_array(feat))
            y_rows.append(m.resolution_value)
            idx_rows.append(market_idx)
            stats["samples_per_horizon"][h] += 1
            stats["total_samples"] += 1
            added_for_this_market = True

        if added_for_this_market:
            stats["markets_used"] += 1

    logger.info(
        f"Variable as_of matrix: {stats['total_samples']} samples from "
        f"{stats['markets_used']} markets ({len(markets)} total)"
    )
    for h in horizons:
        cnt = stats["samples_per_horizon"][h]
        if cnt > 0:
            logger.info(f"  as_of={h}d: {cnt} samples")
    if stats["skipped_no_snapshots"] > 0:
        logger.info(f"  Skipped: {stats['skipped_no_snapshots']} no snapshots, "
                     f"{stats['skipped_near_decided']} near-decided, "
                     f"{stats['skipped_too_early']} as_of before creation")

    if not X_rows:
        return np.array([]).reshape(0, N_FEATURES), np.array([]), np.array([], dtype=int)

    return np.array(X_rows), np.array(y_rows), np.array(idx_rows, dtype=int)
