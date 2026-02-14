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
    lookback_days: int = Query(7, ge=1, le=90),
    session: AsyncSession = Depends(get_session),
):
    """Compute price correlations between markets.

    Returns NxN correlation matrix showing how markets move together.
    Useful for finding related markets for hedging or identifying anomalies.
    """
    # Get active markets (optionally filtered by category)
    query = select(Market).where(
        Market.is_active == True,  # noqa
        Market.price_yes != None,  # noqa
    )
    if category:
        # Use normalized_category for consistent filtering
        query = query.where(
            or_(
                Market.normalized_category == category,
                Market.category == category,
            )
        )

    result = await session.execute(query.limit(50))  # Limit to 50 markets for performance
    markets = result.scalars().all()

    if len(markets) < 2:
        return {"markets": [], "correlations": [], "message": "Not enough markets for correlation"}

    # Get price history for each market (last N days)
    cutoff_time = datetime.utcnow() - timedelta(days=lookback_days)

    # Build price series for each market
    market_prices = {}
    for market in markets:
        price_result = await session.execute(
            select(PriceSnapshot)
            .where(
                PriceSnapshot.market_id == market.id,
                PriceSnapshot.timestamp >= cutoff_time,
            )
            .order_by(PriceSnapshot.timestamp)
        )
        prices = price_result.scalars().all()

        if len(prices) < 5:  # Need at least 5 data points
            continue

        # Extract YES prices
        price_series = [p.price_yes for p in prices if p.price_yes is not None]
        if len(price_series) >= 5:
            market_prices[market.id] = {
                "market": market,
                "prices": np.array(price_series),
            }

    if len(market_prices) < 2:
        return {"markets": [], "correlations": [], "message": "Not enough price data for correlation"}

    # Compute pairwise correlations using Pearson correlation
    market_ids = list(market_prices.keys())
    n = len(market_ids)
    correlations = []

    # Get platform map for display
    platforms_result = await session.execute(select(Platform))
    platform_map = {p.id: p.name for p in platforms_result.scalars().all()}

    for i in range(n):
        for j in range(i + 1, n):  # Only upper triangle (avoid duplicates)
            market_a_id = market_ids[i]
            market_b_id = market_ids[j]

            prices_a = market_prices[market_a_id]["prices"]
            prices_b = market_prices[market_b_id]["prices"]

            # Align series to same length (use minimum)
            min_len = min(len(prices_a), len(prices_b))
            prices_a_aligned = prices_a[-min_len:]
            prices_b_aligned = prices_b[-min_len:]

            # Compute Pearson correlation
            if len(prices_a_aligned) > 1 and prices_a_aligned.std() > 0 and prices_b_aligned.std() > 0:
                correlation = np.corrcoef(prices_a_aligned, prices_b_aligned)[0, 1]

                # Only include if above minimum threshold
                if abs(correlation) >= min_correlation:
                    market_a = market_prices[market_a_id]["market"]
                    market_b = market_prices[market_b_id]["market"]

                    correlations.append({
                        "market_a_id": market_a_id,
                        "market_a_question": market_a.question[:80],
                        "market_a_platform": platform_map.get(market_a.platform_id, "unknown"),
                        "market_b_id": market_b_id,
                        "market_b_question": market_b.question[:80],
                        "market_b_platform": platform_map.get(market_b.platform_id, "unknown"),
                        "correlation": float(correlation),
                    })

    # Sort by absolute correlation (highest first)
    correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    # Build market list for matrix rendering
    market_list = []
    for market_id in market_ids:
        market = market_prices[market_id]["market"]
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
