"""Arbitrage opportunity endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_session
from db.models import ArbitrageOpportunity, Market

router = APIRouter(tags=["arbitrage"])


@router.get("/arbitrage/opportunities")
async def list_opportunities(
    strategy_type: str | None = None,
    min_profit_pct: float = 0.0,
    limit: int = Query(default=50, le=200),
    session: AsyncSession = Depends(get_session),
):
    """List current arbitrage opportunities."""
    query = (
        select(ArbitrageOpportunity)
        .where(ArbitrageOpportunity.expired_at == None)  # noqa
        .where(ArbitrageOpportunity.net_profit_pct >= min_profit_pct)
    )

    if strategy_type:
        query = query.where(ArbitrageOpportunity.strategy_type == strategy_type)

    query = query.order_by(ArbitrageOpportunity.net_profit_pct.desc()).limit(limit)

    result = await session.execute(query)
    opportunities = result.scalars().all()

    # Enrich with market details
    enriched = []
    for opp in opportunities:
        market_details = []
        for mid in (opp.market_ids or []):
            m = await session.get(Market, mid)
            if m:
                market_details.append({
                    "id": m.id,
                    "question": m.question,
                    "price_yes": m.price_yes,
                    "price_no": m.price_no,
                })

        enriched.append({
            "id": opp.id,
            "strategy_type": opp.strategy_type,
            "detected_at": opp.detected_at.isoformat() if opp.detected_at else None,
            "markets": market_details,
            "prices_snapshot": opp.prices_snapshot,
            "gross_spread": opp.gross_spread,
            "total_fees": opp.total_fees,
            "net_profit_pct": opp.net_profit_pct,
            "estimated_profit_usd": opp.estimated_profit_usd,
            "was_executed": opp.was_executed,
        })

    return {"opportunities": enriched, "count": len(enriched)}


@router.get("/arbitrage/history")
async def arbitrage_history(
    strategy_type: str | None = None,
    limit: int = Query(default=100, le=500),
    session: AsyncSession = Depends(get_session),
):
    """Historical arbitrage opportunities."""
    query = select(ArbitrageOpportunity).order_by(
        ArbitrageOpportunity.detected_at.desc()
    ).limit(limit)

    if strategy_type:
        query = query.where(ArbitrageOpportunity.strategy_type == strategy_type)

    result = await session.execute(query)
    opportunities = result.scalars().all()

    return {
        "history": [
            {
                "id": opp.id,
                "strategy_type": opp.strategy_type,
                "detected_at": opp.detected_at.isoformat() if opp.detected_at else None,
                "expired_at": opp.expired_at.isoformat() if opp.expired_at else None,
                "net_profit_pct": opp.net_profit_pct,
                "estimated_profit_usd": opp.estimated_profit_usd,
                "was_executed": opp.was_executed,
                "actual_profit_usd": opp.actual_profit_usd,
            }
            for opp in opportunities
        ],
        "count": len(opportunities),
    }
