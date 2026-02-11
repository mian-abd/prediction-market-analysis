"""Cross-platform arbitrage.
Same event priced differently on Polymarket vs Kalshi.
Pure math after initial matching. Need >2.5% spread to overcome fees."""

import logging
from datetime import datetime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Market, Platform, CrossPlatformMatch
from arbitrage.fee_calculator import FeeCalculator
from config.settings import settings

logger = logging.getLogger(__name__)


async def scan_cross_platform_arb(
    session: AsyncSession,
    min_spread_pct: float | None = None,
) -> list[dict]:
    """Scan cross-platform matches for price discrepancies."""
    if min_spread_pct is None:
        min_spread_pct = settings.min_cross_platform_spread_pct

    # Get all confirmed or high-similarity matches
    result = await session.execute(
        select(CrossPlatformMatch).where(
            CrossPlatformMatch.similarity_score >= 0.45,
        )
    )
    matches = result.scalars().all()

    if not matches:
        logger.info("No cross-platform matches found. Run market matcher first.")
        return []

    # Get platform map
    platforms_result = await session.execute(select(Platform))
    platform_map = {p.id: p.name for p in platforms_result.scalars().all()}

    opportunities = []
    for match in matches:
        market_a = await session.get(Market, match.market_id_a)
        market_b = await session.get(Market, match.market_id_b)

        if not market_a or not market_b:
            continue
        if not market_a.is_active or not market_b.is_active:
            continue
        if market_a.price_yes is None or market_b.price_yes is None:
            continue

        platform_a = platform_map.get(market_a.platform_id, "unknown")
        platform_b = platform_map.get(market_b.platform_id, "unknown")

        price_a_yes = market_a.price_yes
        price_b_yes = market_b.price_yes

        # Handle inverted matches (YES on A = NO on B)
        if match.is_inverted:
            price_b_yes = 1.0 - price_b_yes

        # Check both directions
        # Direction 1: Buy YES on A (cheaper), buy NO on B (= sell YES on B)
        spread_1 = price_b_yes - price_a_yes
        # Direction 2: Buy YES on B (cheaper), buy NO on A (= sell YES on A)
        spread_2 = price_a_yes - price_b_yes

        for spread, buy_market, sell_market, buy_platform, sell_platform, buy_yes_price in [
            (spread_1, market_a, market_b, platform_a, platform_b, price_a_yes),
            (spread_2, market_b, market_a, platform_b, platform_a, price_b_yes),
        ]:
            if spread <= 0:
                continue

            # Sell complement price = 1 - sell_market_yes_price
            sell_complement = 1.0 - (market_b.price_yes if buy_market == market_a else market_a.price_yes)
            if match.is_inverted:
                sell_complement = market_b.price_yes if buy_market == market_a else (1.0 - market_a.price_yes)

            fee_result = FeeCalculator.cross_platform_arb_fees(
                price_buy=buy_yes_price,
                price_sell_complement=1.0 - (buy_yes_price + spread),
                platform_buy=buy_platform,
                platform_sell=sell_platform,
            )

            if not fee_result["profitable"]:
                continue
            if fee_result["net_pct"] < min_spread_pct:
                continue

            opportunity = {
                "strategy_type": "cross_platform",
                "market_ids": [buy_market.id, sell_market.id],
                "detected_at": datetime.utcnow(),
                "buy_platform": buy_platform,
                "sell_platform": sell_platform,
                "buy_market_question": buy_market.question,
                "sell_market_question": sell_market.question,
                "buy_yes_price": buy_yes_price,
                "sell_yes_price": buy_yes_price + spread,
                "raw_spread": spread,
                "raw_spread_pct": spread * 100,
                "similarity_score": match.similarity_score,
                "prices_snapshot": {
                    f"market_{buy_market.id}_yes": buy_market.price_yes,
                    f"market_{sell_market.id}_yes": sell_market.price_yes,
                },
                "gross_spread": fee_result["gross_pct"],
                "total_fees": fee_result["total_fees"],
                "net_profit_pct": fee_result["net_pct"],
                "estimated_profit_usd": fee_result["net_profit"] * 100,
            }
            opportunities.append(opportunity)

    opportunities.sort(key=lambda x: x["net_profit_pct"], reverse=True)
    logger.info(
        f"Cross-platform scan: checked {len(matches)} matches, "
        f"found {len(opportunities)} opportunities"
    )
    return opportunities
