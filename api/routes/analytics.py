"""Analytics endpoints for correlations and performance metrics."""

from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np
from collections import defaultdict

from db.database import get_session
from db.models import Market, PriceSnapshot, Platform

router = APIRouter(tags=["analytics"])


@router.get("/analytics/correlations")
async def get_market_correlations(
    category: str | None = None,
    min_correlation: float = Query(0.3, ge=-1.0, le=1.0),
    lookback_days: int = Query(7, ge=1, le=365),
    session: AsyncSession = Depends(get_session),
):
    """Compute price correlations between markets using timestamp-aligned daily prices.

    Returns pairwise correlations showing how markets move together.
    Uses daily resampling to ensure prices are compared at the same point in time.
    """
    # Get active markets (optionally filtered by category)
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

    result = await session.execute(query.limit(30))
    markets = result.scalars().all()

    if len(markets) < 2:
        return {"markets": [], "correlations": [], "message": "Not enough markets for correlation"}

    cutoff_time = datetime.utcnow() - timedelta(days=lookback_days)

    # Build DAILY price series for each market (timestamp-aligned)
    # Key: market_id -> {date_str: avg_price}
    market_daily_prices: dict[int, dict] = {}

    for market in markets:
        price_result = await session.execute(
            select(PriceSnapshot)
            .where(
                PriceSnapshot.market_id == market.id,
                PriceSnapshot.timestamp >= cutoff_time,
            )
            .order_by(PriceSnapshot.timestamp)
        )
        snapshots = price_result.scalars().all()

        if not snapshots:
            continue

        # Resample to daily frequency: average price per date
        daily_prices: dict[str, list[float]] = defaultdict(list)
        for snap in snapshots:
            if snap.price_yes is not None:
                date_key = snap.timestamp.strftime("%Y-%m-%d")
                daily_prices[date_key].append(snap.price_yes)

        # Compute daily averages
        daily_avg = {date: np.mean(prices) for date, prices in daily_prices.items()}

        # Need at least 3 unique days for meaningful correlation
        if len(daily_avg) >= 3:
            market_daily_prices[market.id] = {
                "market": market,
                "daily_prices": daily_avg,
                "dates": set(daily_avg.keys()),
                "num_days": len(daily_avg),
            }

    if len(market_daily_prices) < 2:
        return {
            "markets": [],
            "correlations": [],
            "lookback_days": lookback_days,
            "min_correlation": min_correlation,
            "total_pairs": 0,
            "message": f"Need at least 2 markets with 3+ days of price data in the last {lookback_days} days. "
                      f"Found {len(market_daily_prices)} qualifying markets.",
        }

    # Compute pairwise correlations using TIMESTAMP-ALIGNED prices
    market_ids = list(market_daily_prices.keys())
    n = len(market_ids)
    correlations = []

    # Get platform map for display
    platforms_result = await session.execute(select(Platform))
    platform_map = {p.id: p.name for p in platforms_result.scalars().all()}

    for i in range(n):
        for j in range(i + 1, n):
            market_a_id = market_ids[i]
            market_b_id = market_ids[j]

            data_a = market_daily_prices[market_a_id]
            data_b = market_daily_prices[market_b_id]

            # Find COMMON dates (only compare prices at the same time)
            common_dates = sorted(data_a["dates"] & data_b["dates"])

            # Need at least 5 common dates for statistically meaningful correlation
            if len(common_dates) < 5:
                continue

            # Extract aligned price arrays (same dates, same order)
            prices_a = np.array([data_a["daily_prices"][d] for d in common_dates])
            prices_b = np.array([data_b["daily_prices"][d] for d in common_dates])

            # Compute Pearson correlation on properly aligned data
            if prices_a.std() > 1e-6 and prices_b.std() > 1e-6:
                correlation = float(np.corrcoef(prices_a, prices_b)[0, 1])

                if abs(correlation) >= min_correlation:
                    market_a = data_a["market"]
                    market_b = data_b["market"]

                    correlations.append({
                        "market_a_id": market_a_id,
                        "market_a_question": market_a.question[:80],
                        "market_a_platform": platform_map.get(market_a.platform_id, "unknown"),
                        "market_b_id": market_b_id,
                        "market_b_question": market_b.question[:80],
                        "market_b_platform": platform_map.get(market_b.platform_id, "unknown"),
                        "correlation": correlation,
                        "common_days": len(common_dates),
                    })

    # Sort by absolute correlation (highest first)
    correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    # Build market list for graph rendering
    market_list = []
    for market_id in market_ids:
        market = market_daily_prices[market_id]["market"]
        market_list.append({
            "id": market.id,
            "question": market.question[:50],
            "platform": platform_map.get(market.platform_id, "unknown"),
            "category": market.category,
        })

    return {
        "markets": market_list,
        "correlations": correlations,
        "lookback_days": lookback_days,
        "min_correlation": min_correlation,
        "total_pairs": len(correlations),
    }
