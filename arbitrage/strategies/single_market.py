"""Single-market rebalancing arbitrage.
If YES + NO < $1.00, buy both = guaranteed profit at resolution.
Pure math, zero AI cost."""

import logging
from datetime import datetime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Market, Platform
from arbitrage.fee_calculator import FeeCalculator
from config.settings import settings

logger = logging.getLogger(__name__)


async def scan_single_market_arb(
    session: AsyncSession,
    min_profit_pct: float | None = None,
) -> list[dict]:
    """Scan all active binary markets for YES + NO < $1 opportunities."""
    if min_profit_pct is None:
        min_profit_pct = settings.min_single_market_profit_pct

    # Get platform map
    platforms_result = await session.execute(select(Platform))
    platform_map = {p.id: p.name for p in platforms_result.scalars().all()}

    # Get all active markets with valid prices
    result = await session.execute(
        select(Market).where(
            Market.is_active == True,  # noqa
            Market.price_yes != None,  # noqa
            Market.price_no != None,   # noqa
            Market.price_yes > 0,
            Market.price_no > 0,
        )
    )
    markets = result.scalars().all()

    opportunities = []
    for market in markets:
        yes_price = market.price_yes
        no_price = market.price_no
        total = yes_price + no_price

        # Only interested if total < 1.0 (guaranteed profit)
        if total >= 1.0:
            continue

        platform_name = platform_map.get(market.platform_id, "unknown")

        # Calculate with fees
        fee_result = FeeCalculator.single_market_arb_fees(
            yes_price=yes_price,
            no_price=no_price,
            platform=platform_name,
        )

        if not fee_result["profitable"]:
            continue

        if fee_result["net_pct"] < min_profit_pct:
            continue

        opportunity = {
            "strategy_type": "single_market",
            "market_ids": [market.id],
            "detected_at": datetime.utcnow(),
            "platform": platform_name,
            "market_question": market.question,
            "market_id": market.id,
            "yes_price": yes_price,
            "no_price": no_price,
            "total_cost": total,
            "prices_snapshot": {
                f"market_{market.id}_yes": yes_price,
                f"market_{market.id}_no": no_price,
            },
            "gross_spread": fee_result["gross_pct"],
            "total_fees": fee_result["fees"],
            "net_profit_pct": fee_result["net_pct"],
            "estimated_profit_usd": fee_result["net_profit"] * 100,  # per $100 position
            "volume_24h": market.volume_24h or 0,
            "liquidity": market.liquidity or 0,
        }
        opportunities.append(opportunity)

    # Sort by net profit descending
    opportunities.sort(key=lambda x: x["net_profit_pct"], reverse=True)

    logger.info(
        f"Single-market scan: checked {len(markets)} markets, "
        f"found {len(opportunities)} opportunities"
    )
    return opportunities
