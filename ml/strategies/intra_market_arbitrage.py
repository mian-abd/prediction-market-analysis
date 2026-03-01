"""Intra-market arbitrage detector.

Finds sum-to-one violations in multi-outcome Polymarket markets.
This is the most defensible edge: when YES + NO prices don't sum to 1.0
(or multi-way outcome prices don't sum to 1.0), there's a guaranteed
arbitrage opportunity.

Sources of profit:
1. YES + NO > 1.0 (sell both sides, pocket the excess)
2. YES + NO < 1.0 (buy both sides, guaranteed profit at resolution)
3. Multi-outcome: sum of all outcome prices != 1.0

The key advantage: this requires no forecasting model. The edge is
mathematical, not statistical. It survives all friction if the spread
exceeds fees + slippage.
"""

import logging
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config.constants import compute_polymarket_fee
from db.models import Market

logger = logging.getLogger(__name__)


@dataclass
class ArbitrageOpportunity:
    """Detected intra-market arbitrage."""
    market_id: int
    question: str
    arb_type: str  # "sum_under_one", "sum_over_one", "multi_outcome"
    price_yes: float
    price_no: float
    implied_total: float
    raw_profit_pct: float
    net_profit_pct: float  # After fees + slippage
    entry_cost: float  # Total cost to enter both legs
    expected_profit: float  # Dollar profit per $100 notional
    detected_at: datetime


# Minimum net profit threshold (after fees + slippage) to trigger a signal
MIN_NET_PROFIT_PCT = 0.5  # 0.5%
SLIPPAGE_BPS = 100  # 1% slippage per leg


async def scan_binary_arbitrage(
    session: AsyncSession,
    min_liquidity: float = 5000.0,
    min_volume: float = 1000.0,
) -> list[ArbitrageOpportunity]:
    """Scan for YES + NO sum violations in binary markets.

    IMPORTANT: price_yes and price_no from the DB are typically mid-prices
    or last-traded prices, NOT best ask. To buy both sides, you must cross
    the spread on both legs. We use orderbook data when available, otherwise
    conservatively add spread + slippage estimates.

    Real arbitrage only exists when:
      ask_yes + ask_no + fees + slippage < 1.0  (buy both sides)
    or:
      bid_yes + bid_no - fees - slippage > 1.0  (sell both sides)
    """
    from db.models import OrderbookSnapshot

    result = await session.execute(
        select(Market)
        .where(
            Market.is_active == True,  # noqa: E712
            Market.price_yes.isnot(None),
            Market.price_no.isnot(None),
            Market.liquidity >= min_liquidity,
        )
    )
    markets = result.scalars().all()

    opportunities = []

    for market in markets:
        price_yes = float(market.price_yes or 0)
        price_no = float(market.price_no or 0)

        if price_yes <= 0.01 or price_no <= 0.01:
            continue
        if price_yes >= 0.99 or price_no >= 0.99:
            continue

        # Try to get orderbook data for realistic execution prices
        ob_result = await session.execute(
            select(OrderbookSnapshot)
            .where(OrderbookSnapshot.market_id == market.id)
            .order_by(OrderbookSnapshot.timestamp.desc())
            .limit(1)
        )
        ob = ob_result.scalar_one_or_none()

        slippage_rate = SLIPPAGE_BPS / 10000

        if ob and ob.asks_json and ob.bids_json:
            # Use orderbook: best ask for buying, best bid for selling
            asks = ob.asks_json or []
            bids = ob.bids_json or []
            if asks and bids:
                best_ask_yes = float(asks[0].get("price", price_yes))
                best_bid_yes = float(bids[0].get("price", price_yes))
                # For NO: best ask = 1 - best_bid_yes, best bid = 1 - best_ask_yes
                # (on Polymarket, NO tokens have their own orderbook, but this is a reasonable approx)
                ask_no = 1.0 - best_bid_yes
                bid_no = 1.0 - best_ask_yes
            else:
                # Fallback: estimate ask as mid + half spread
                est_spread = 0.02
                best_ask_yes = price_yes + est_spread / 2
                ask_no = price_no + est_spread / 2
                best_bid_yes = price_yes - est_spread / 2
                bid_no = price_no - est_spread / 2
        else:
            # No orderbook: conservatively estimate spread at 2%
            est_spread = 0.02
            best_ask_yes = price_yes + est_spread / 2
            ask_no = price_no + est_spread / 2
            best_bid_yes = price_yes - est_spread / 2
            bid_no = price_no - est_spread / 2

        # BUY BOTH: cost = ask_yes + ask_no + slippage on both legs
        buy_cost = best_ask_yes * (1 + slippage_rate) + ask_no * (1 + slippage_rate)
        fee_bps = getattr(market, 'taker_fee_bps', 0) or 0
        fee_per_leg = compute_polymarket_fee(price_yes, 1.0, fee_bps, is_maker=False)
        total_buy_cost = buy_cost + fee_per_leg * 2

        if total_buy_cost < 1.0:
            net_profit = 1.0 - total_buy_cost
            net_profit_pct = net_profit / total_buy_cost * 100

            if net_profit_pct >= MIN_NET_PROFIT_PCT:
                opportunities.append(ArbitrageOpportunity(
                    market_id=market.id,
                    question=market.question or "",
                    arb_type="sum_under_one",
                    price_yes=price_yes,
                    price_no=price_no,
                    implied_total=best_ask_yes + ask_no,
                    raw_profit_pct=(1.0 - (price_yes + price_no)) / (price_yes + price_no) * 100,
                    net_profit_pct=net_profit_pct,
                    entry_cost=total_buy_cost * 100,
                    expected_profit=net_profit * 100,
                    detected_at=datetime.utcnow(),
                ))

        # SELL BOTH: revenue = bid_yes + bid_no - slippage - fees
        sell_revenue = best_bid_yes * (1 - slippage_rate) + bid_no * (1 - slippage_rate)
        total_sell_revenue = sell_revenue - fee_per_leg * 2

        if total_sell_revenue > 1.0:
            net_profit = total_sell_revenue - 1.0
            net_profit_pct = net_profit / 1.0 * 100

            if net_profit_pct >= MIN_NET_PROFIT_PCT:
                opportunities.append(ArbitrageOpportunity(
                    market_id=market.id,
                    question=market.question or "",
                    arb_type="sum_over_one",
                    price_yes=price_yes,
                    price_no=price_no,
                    implied_total=best_bid_yes + bid_no,
                    raw_profit_pct=((price_yes + price_no) - 1.0) / 1.0 * 100,
                    net_profit_pct=net_profit_pct,
                    entry_cost=100.0,
                    expected_profit=net_profit * 100,
                    detected_at=datetime.utcnow(),
                ))

    if opportunities:
        logger.info(
            f"Intra-market arb scan: {len(opportunities)} opportunities found "
            f"(from {len(markets)} markets)"
        )
        for opp in opportunities[:5]:
            logger.info(
                f"  [{opp.arb_type}] {opp.question[:60]}... "
                f"total={opp.implied_total:.4f} net={opp.net_profit_pct:.2f}%"
            )

    return opportunities


async def scan_multi_outcome_arbitrage(
    session: AsyncSession,
) -> list[ArbitrageOpportunity]:
    """Scan for sum violations in multi-outcome markets.

    Multi-outcome markets (e.g., "Who will win the election?" with 5+ candidates)
    should have all outcome prices sum to 1.0. When they don't, there's
    arbitrage.

    This requires markets to be grouped by condition_id / event.
    """
    # Multi-outcome markets share the same condition_id
    # but have different token IDs. We need to group by condition_id.
    result = await session.execute(
        select(Market)
        .where(
            Market.is_active == True,  # noqa: E712
            Market.condition_id.isnot(None),
            Market.price_yes.isnot(None),
        )
    )
    markets = result.scalars().all()

    # Group by slug prefix (multi-outcome markets share a slug base)
    from collections import defaultdict
    slug_groups = defaultdict(list)
    for m in markets:
        if m.slug:
            base_slug = m.slug.rsplit("-", 1)[0] if "-" in (m.slug or "") else m.slug
            slug_groups[base_slug].append(m)

    opportunities = []

    for slug, group in slug_groups.items():
        if len(group) < 3:
            continue

        prices = [float(m.price_yes or 0) for m in group]
        total = sum(prices)

        if total < 0.95 or total > 1.05:
            raw_profit_pct = abs(total - 1.0) / total * 100 if total > 0 else 0
            if raw_profit_pct >= 1.0:
                arb_type = "multi_under" if total < 1.0 else "multi_over"
                opportunities.append(ArbitrageOpportunity(
                    market_id=group[0].id,
                    question=f"Multi-outcome group: {slug} ({len(group)} outcomes)",
                    arb_type=arb_type,
                    price_yes=total,
                    price_no=0.0,
                    implied_total=total,
                    raw_profit_pct=raw_profit_pct,
                    net_profit_pct=raw_profit_pct - 2.0,  # rough fee estimate
                    entry_cost=total * 100,
                    expected_profit=(abs(total - 1.0) - 0.02) * 100,
                    detected_at=datetime.utcnow(),
                ))

    if opportunities:
        logger.info(f"Multi-outcome arb scan: {len(opportunities)} groups with violations")

    return opportunities
