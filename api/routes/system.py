"""System health and status endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_session
from db.models import (
    Market, Platform, PriceSnapshot, OrderbookSnapshot,
    ArbitrageOpportunity, CrossPlatformMatch,
)

router = APIRouter(tags=["system"])


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/system/stats")
async def system_stats(session: AsyncSession = Depends(get_session)):
    """Pipeline and data statistics."""
    # Market counts by platform
    platform_counts = {}
    platforms_result = await session.execute(select(Platform))
    for p in platforms_result.scalars().all():
        count_result = await session.execute(
            select(func.count(Market.id)).where(
                Market.platform_id == p.id,
                Market.is_active == True,  # noqa
            )
        )
        platform_counts[p.name] = count_result.scalar() or 0

    total_markets = sum(platform_counts.values())

    # Price snapshots count
    price_count = (await session.execute(
        select(func.count(PriceSnapshot.id))
    )).scalar() or 0

    # Orderbook snapshots count
    ob_count = (await session.execute(
        select(func.count(OrderbookSnapshot.id))
    )).scalar() or 0

    # Cross-platform matches
    match_count = (await session.execute(
        select(func.count(CrossPlatformMatch.id))
    )).scalar() or 0

    # Active arbitrage opportunities
    arb_count = (await session.execute(
        select(func.count(ArbitrageOpportunity.id)).where(
            ArbitrageOpportunity.expired_at == None  # noqa
        )
    )).scalar() or 0

    # Most recent market update
    latest_result = await session.execute(
        select(Market.last_fetched_at)
        .order_by(Market.last_fetched_at.desc())
        .limit(1)
    )
    latest_fetch = latest_result.scalar()

    return {
        "total_active_markets": total_markets,
        "markets_by_platform": platform_counts,
        "price_snapshots": price_count,
        "orderbook_snapshots": ob_count,
        "cross_platform_matches": match_count,
        "active_arbitrage_opportunities": arb_count,
        "last_data_fetch": latest_fetch.isoformat() if latest_fetch else None,
    }
