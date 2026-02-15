"""Analytics endpoints for correlations and performance metrics."""

import logging
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np
from collections import defaultdict

from db.database import get_session
from db.models import Market, PriceSnapshot, Platform

router = APIRouter(tags=["analytics"])
logger = logging.getLogger(__name__)


@router.get("/analytics/correlations")
async def get_market_correlations(
    category: str | None = None,
    min_correlation: float = Query(0.3, ge=-1.0, le=1.0),
    lookback_days: int = Query(30, ge=1, le=365),
    session: AsyncSession = Depends(get_session),
):
    """Compute price RETURN correlations between markets.

    IMPORTANT: Correlates price CHANGES (returns), not raw price levels.
    Raw level correlation produces spurious results because markets trending
    toward 0 or 1 appear correlated even when their movements are independent.

    Uses hourly resampling for short lookbacks (<=14d), daily for longer periods.
    """
    # Get active markets with sufficient volume (optionally filtered by category)
    query = select(Market).where(
        Market.is_active == True,  # noqa
        Market.price_yes != None,  # noqa
    )
    if category:
        query = query.where(
            or_(
                Market.normalized_category == category,
                Market.category == category,
            )
        )

    result = await session.execute(query.limit(25))
    markets = result.scalars().all()

    if len(markets) < 2:
        return {
            "markets": [], "correlations": [], "total_pairs": 0,
            "lookback_days": lookback_days, "min_correlation": min_correlation,
            "message": "Not enough active markets for correlation analysis.",
        }

    cutoff_time = datetime.utcnow() - timedelta(days=lookback_days)

    # Choose time bucket granularity based on lookback period
    use_hourly = lookback_days <= 14

    # Get platform map for display
    platforms_result = await session.execute(select(Platform))
    platform_map = {p.id: p.name for p in platforms_result.scalars().all()}

    # Batch-fetch all price snapshots at once (much faster than per-market queries)
    market_ids_list = [m.id for m in markets]
    snap_result = await session.execute(
        select(PriceSnapshot)
        .where(
            PriceSnapshot.market_id.in_(market_ids_list),
            PriceSnapshot.timestamp >= cutoff_time,
            PriceSnapshot.price_yes != None,  # noqa
        )
        .order_by(PriceSnapshot.timestamp)
    )
    all_snapshots = snap_result.scalars().all()

    # Group snapshots by market_id
    snaps_by_market: dict[int, list] = defaultdict(list)
    for snap in all_snapshots:
        snaps_by_market[snap.market_id].append(snap)

    # Build time-bucketed price RETURN series for each market
    market_data: dict[int, dict] = {}
    market_obj_map = {m.id: m for m in markets}

    for market_id, snapshots in snaps_by_market.items():
        if len(snapshots) < 4:
            continue

        # Resample to time buckets: average price per bucket
        bucket_prices: dict[str, list[float]] = defaultdict(list)
        for snap in snapshots:
            if use_hourly:
                bucket_key = snap.timestamp.strftime("%Y-%m-%d %H:00")
            else:
                bucket_key = snap.timestamp.strftime("%Y-%m-%d")
            bucket_prices[bucket_key].append(snap.price_yes)

        # Compute bucket averages and sort chronologically
        sorted_buckets = sorted(bucket_prices.keys())
        bucket_avg = {b: float(np.mean(bucket_prices[b])) for b in sorted_buckets}

        # Compute RETURNS (price changes) instead of raw levels
        # This avoids spurious correlation from common trends
        if len(sorted_buckets) >= 4:
            returns = {}
            for k in range(1, len(sorted_buckets)):
                prev_price = bucket_avg[sorted_buckets[k - 1]]
                curr_price = bucket_avg[sorted_buckets[k]]
                if prev_price > 0.001:
                    returns[sorted_buckets[k]] = curr_price - prev_price
            if len(returns) >= 3:
                market_data[market_id] = {
                    "returns": returns,
                    "buckets": set(returns.keys()),
                }

    bucket_type = "hours" if use_hourly else "days"

    if len(market_data) < 2:
        return {
            "markets": [], "correlations": [],
            "lookback_days": lookback_days, "min_correlation": min_correlation,
            "total_pairs": 0,
            "message": f"Need at least 2 markets with 4+ {bucket_type} of price data "
                      f"in the last {lookback_days} days. Found {len(market_data)} qualifying.",
        }

    # Compute pairwise correlations on RETURNS
    data_ids = list(market_data.keys())
    n = len(data_ids)
    correlations = []

    for i in range(n):
        for j in range(i + 1, n):
            id_a, id_b = data_ids[i], data_ids[j]
            da, db = market_data[id_a], market_data[id_b]

            # Only correlate returns at COMMON time buckets
            common = sorted(da["buckets"] & db["buckets"])
            if len(common) < 3:
                continue

            returns_a = np.array([da["returns"][b] for b in common])
            returns_b = np.array([db["returns"][b] for b in common])

            # Need non-zero variance in both return series
            if returns_a.std() < 1e-8 or returns_b.std() < 1e-8:
                continue

            corr = float(np.corrcoef(returns_a, returns_b)[0, 1])
            if np.isnan(corr):
                continue

            if abs(corr) >= min_correlation:
                ma = market_obj_map.get(id_a)
                mb = market_obj_map.get(id_b)
                if ma and mb:
                    correlations.append({
                        "market_a_id": id_a,
                        "market_a_question": ma.question[:80],
                        "market_a_platform": platform_map.get(ma.platform_id, "unknown"),
                        "market_b_id": id_b,
                        "market_b_question": mb.question[:80],
                        "market_b_platform": platform_map.get(mb.platform_id, "unknown"),
                        "correlation": round(corr, 4),
                        "common_points": len(common),
                    })

    correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    # Build market list for graph
    market_list = []
    for mid in data_ids:
        m = market_obj_map.get(mid)
        if m:
            market_list.append({
                "id": m.id,
                "question": m.question[:50],
                "platform": platform_map.get(m.platform_id, "unknown"),
                "category": m.category,
            })

    return {
        "markets": market_list,
        "correlations": correlations,
        "lookback_days": lookback_days,
        "min_correlation": min_correlation,
        "total_pairs": len(correlations),
    }
