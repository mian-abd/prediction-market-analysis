"""System health and status endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_session
from db.models import (
    Market, Platform, PriceSnapshot, OrderbookSnapshot,
    ArbitrageOpportunity, CrossPlatformMatch,
)
from ml.evaluation.confidence_adjuster import get_adjuster_stats

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

@router.get("/system/websocket-status")
async def websocket_status():
    """Real-time WebSocket streaming status (Phase 2.1)."""
    try:
        # Import here to avoid circular dependency
        from data_pipeline.scheduler import _price_cache, _polymarket_stream
        
        if not _price_cache or not _polymarket_stream:
            return {
                "status": "not_initialized",
                "connected": False,
                "subscribed_markets": 0,
                "messages_processed": 0,
                "last_message": None,
                "pending_arbitrage_signals": 0,
            }
        
        # Get stream stats
        stream_stats = await _polymarket_stream.get_stats()
        cache_stats = await _price_cache.get_stats()
        
        return {
            "status": "connected" if stream_stats["connected"] else "disconnected",
            "connected": stream_stats["connected"],
            "subscribed_markets": stream_stats["subscribed_markets"],
            "messages_processed": stream_stats["messages_processed"],
            "last_message": stream_stats["last_message"],
            "pending_arbitrage_signals": cache_stats.get("pending_signals", 0),
            "cache": {
                "polymarket_cached": cache_stats.get("polymarket_cached", 0),
                "kalshi_cached": cache_stats.get("kalshi_cached", 0),
                "total_cached": cache_stats.get("total_cached", 0),
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "connected": False,
        }


@router.get("/system/confidence-adjuster")
async def confidence_adjuster_stats():
    """Adaptive confidence adjuster statistics (Phase 2.5)."""
    try:
        stats = get_adjuster_stats()
        return stats
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }
