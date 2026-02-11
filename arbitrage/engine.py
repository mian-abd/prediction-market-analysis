"""Main arbitrage engine - orchestrates all scanning strategies."""

import logging
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import ArbitrageOpportunity
from arbitrage.strategies.single_market import scan_single_market_arb
from arbitrage.strategies.cross_platform import scan_cross_platform_arb

logger = logging.getLogger(__name__)


async def run_full_scan(session: AsyncSession) -> list[dict]:
    """Run all arbitrage strategies and return combined opportunities."""
    all_opportunities = []

    # Strategy 1: Single-market rebalancing
    try:
        single_opps = await scan_single_market_arb(session)
        all_opportunities.extend(single_opps)
        logger.info(f"Single-market: {len(single_opps)} opportunities")
    except Exception as e:
        logger.error(f"Single-market scan failed: {e}")

    # Strategy 2: Cross-platform arbitrage
    try:
        cross_opps = await scan_cross_platform_arb(session)
        all_opportunities.extend(cross_opps)
        logger.info(f"Cross-platform: {len(cross_opps)} opportunities")
    except Exception as e:
        logger.error(f"Cross-platform scan failed: {e}")

    # Sort all by net profit
    all_opportunities.sort(key=lambda x: x["net_profit_pct"], reverse=True)

    # Persist to DB
    for opp in all_opportunities:
        db_opp = ArbitrageOpportunity(
            strategy_type=opp["strategy_type"],
            detected_at=opp["detected_at"],
            market_ids=opp["market_ids"],
            prices_snapshot=opp.get("prices_snapshot"),
            gross_spread=opp["gross_spread"],
            total_fees=opp["total_fees"],
            net_profit_pct=opp["net_profit_pct"],
            estimated_profit_usd=opp.get("estimated_profit_usd", 0),
        )
        session.add(db_opp)

    if all_opportunities:
        await session.commit()

    logger.info(f"Total arbitrage opportunities: {len(all_opportunities)}")
    return all_opportunities


async def get_summary(session: AsyncSession) -> dict:
    """Get arbitrage summary stats."""
    from sqlalchemy import select, func
    from db.models import ArbitrageOpportunity

    # Active opportunities
    active_count = (await session.execute(
        select(func.count(ArbitrageOpportunity.id)).where(
            ArbitrageOpportunity.expired_at == None  # noqa
        )
    )).scalar() or 0

    # Total historical
    total_count = (await session.execute(
        select(func.count(ArbitrageOpportunity.id))
    )).scalar() or 0

    # Best current opportunity
    best = (await session.execute(
        select(ArbitrageOpportunity)
        .where(ArbitrageOpportunity.expired_at == None)  # noqa
        .order_by(ArbitrageOpportunity.net_profit_pct.desc())
        .limit(1)
    )).scalar_one_or_none()

    # Sum of estimated profits
    total_profit = (await session.execute(
        select(func.sum(ArbitrageOpportunity.estimated_profit_usd)).where(
            ArbitrageOpportunity.expired_at == None  # noqa
        )
    )).scalar() or 0

    return {
        "active_opportunities": active_count,
        "total_historical": total_count,
        "best_net_profit_pct": best.net_profit_pct if best else 0,
        "total_estimated_profit_usd": total_profit,
    }
