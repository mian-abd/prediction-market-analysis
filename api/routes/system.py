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


@router.get("/system/truth-dashboard")
async def truth_dashboard(session: AsyncSession = Depends(get_session)):
    """Truth dashboard — comprehensive view of data quality and model health.

    Returns data coverage, model deployment status, feature quality,
    and collection pipeline health in a single response.
    """
    from datetime import datetime, timedelta
    from sqlalchemy import text
    from db.models import Trade

    now = datetime.utcnow()
    cutoff_24h = (now - timedelta(hours=24)).isoformat()

    def q(sql):
        return session.execute(text(sql))

    # Data coverage
    total_markets = (await q("SELECT COUNT(*) FROM markets")).scalar()
    resolved = (await q("SELECT COUNT(*) FROM markets WHERE resolution_value IS NOT NULL")).scalar()
    active = (await q("SELECT COUNT(*) FROM markets WHERE is_active = 1")).scalar()

    total_snaps = (await q("SELECT COUNT(*) FROM price_snapshots")).scalar()
    markets_with_snaps = (await q("SELECT COUNT(DISTINCT market_id) FROM price_snapshots")).scalar()
    snaps_24h = (await q(f"SELECT COUNT(*) FROM price_snapshots WHERE timestamp >= '{cutoff_24h}'")).scalar()

    total_obs = (await q("SELECT COUNT(*) FROM orderbook_snapshots")).scalar()
    obs_24h = (await q(f"SELECT COUNT(*) FROM orderbook_snapshots WHERE timestamp >= '{cutoff_24h}'")).scalar()

    total_trades = (await q("SELECT COUNT(*) FROM trades")).scalar()

    newest_snap = (await q("SELECT MAX(timestamp) FROM price_snapshots")).scalar()
    newest_ob = (await q("SELECT MAX(timestamp) FROM orderbook_snapshots")).scalar()

    # Model health
    try:
        from ml.evaluation.deployment_gate import get_model_health_summary
        model_health = get_model_health_summary()
    except Exception as e:
        model_health = {"error": str(e)}

    # Pipeline freshness
    snap_age_min = None
    if newest_snap:
        try:
            ts = datetime.fromisoformat(str(newest_snap))
            snap_age_min = round((now - ts).total_seconds() / 60, 1)
        except Exception:
            pass

    ob_age_min = None
    if newest_ob:
        try:
            ts = datetime.fromisoformat(str(newest_ob))
            ob_age_min = round((now - ts).total_seconds() / 60, 1)
        except Exception:
            pass

    return {
        "generated_at": now.isoformat(),
        "data_coverage": {
            "markets": {"total": total_markets, "resolved": resolved, "active": active},
            "price_snapshots": {
                "total": total_snaps,
                "markets_covered": markets_with_snaps,
                "coverage_pct": round(markets_with_snaps / max(total_markets, 1) * 100, 1),
                "last_24h": snaps_24h,
                "newest": str(newest_snap) if newest_snap else None,
                "age_minutes": snap_age_min,
            },
            "orderbook_snapshots": {
                "total": total_obs,
                "last_24h": obs_24h,
                "newest": str(newest_ob) if newest_ob else None,
                "age_minutes": ob_age_min,
            },
            "trades": {"total": total_trades},
        },
        "pipeline_health": {
            "price_collection_active": snaps_24h > 0,
            "orderbook_collection_active": obs_24h > 0,
            "stale_warning": (snap_age_min or 999) > 60,
        },
        "model_health": model_health,
    }
