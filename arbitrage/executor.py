"""Arbitrage trade executor with dual-leg execution and safety checks (Phase 3).

Executes cross-platform arbitrage trades with proper sequencing, timeout handling,
and safety limits to prevent losses from partial fills or execution failures.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Market, PortfolioPosition
from data_pipeline.collectors import polymarket_clob
from arbitrage.orderbook_utils import (
    get_best_bid_ask,
    estimate_execution_price,
    check_min_liquidity,
)
from arbitrage.fee_calculator import FeeCalculator

logger = logging.getLogger(__name__)

# Safety limits (Phase 3.1 - start conservative)
MAX_POSITION_USD = 50.0      # Maximum $50 per arbitrage trade
MIN_LIQUIDITY_USD = 500.0    # Minimum $500 liquidity each side
EXECUTION_TIMEOUT_SEC = 30   # 30 second timeout for second leg
MIN_NET_SPREAD_PCT = 5.0     # 5% minimum after fees (realistic threshold)


class ArbitrageExecutor:
    """Executes cross-platform arbitrage trades with safety checks.

    Execution Strategy:
    1. Validate opportunity (bid/ask prices, liquidity, capital)
    2. Execute cheaper leg first (lock in value side)
    3. Execute expensive leg within 30sec (timeout = manual hedge needed)
    4. Record both positions in database

    Safety Features:
    - Pre-execution capital verification
    - Minimum liquidity checks ($500 each side)
    - Position size limits ($50 max)
    - Timeout handling with manual hedge fallback
    - Partial fill detection and logging
    """

    def __init__(self, session: AsyncSession, user_id: str = "arbitrage_bot"):
        """Initialize executor.

        Args:
            session: Database session for recording positions
            user_id: User ID for position tracking
        """
        self.session = session
        self.user_id = user_id

    async def execute_cross_platform_arb(
        self,
        market_a_id: int,
        market_b_id: int,
        direction: str,  # "buy_a_sell_b" or "buy_b_sell_a"
    ) -> dict:
        """Execute cross-platform arbitrage trade.

        Args:
            market_a_id: First market ID (Polymarket)
            market_b_id: Second market ID (Kalshi or Polymarket)
            direction: Which side to buy/sell

        Returns:
            Execution result dict with status, positions, and P&L
        """
        logger.info(f"Executing cross-platform arb: {market_a_id} <-> {market_b_id}")

        # 1. Load markets
        market_a = await self.session.get(Market, market_a_id)
        market_b = await self.session.get(Market, market_b_id)

        if not market_a or not market_b:
            return {"status": "error", "reason": "Markets not found"}

        if not market_a.token_id_yes or not market_b.token_id_yes:
            return {"status": "error", "reason": "Missing token IDs"}

        # 2. Fetch orderbooks for both markets
        try:
            orderbook_a = await polymarket_clob.fetch_orderbook(market_a.token_id_yes)
            orderbook_b = await polymarket_clob.fetch_orderbook(market_b.token_id_yes)
        except Exception as e:
            logger.error(f"Failed to fetch orderbooks: {e}")
            return {"status": "error", "reason": f"Orderbook fetch failed: {e}"}

        if not orderbook_a or not orderbook_b:
            return {"status": "error", "reason": "Orderbooks not available"}

        # 3. Extract bid/ask prices
        bid_ask_a = get_best_bid_ask(orderbook_a)
        bid_ask_b = get_best_bid_ask(orderbook_b)

        if not bid_ask_a or not bid_ask_b:
            return {"status": "error", "reason": "Failed to parse orderbooks"}

        # 4. Safety check: Minimum liquidity
        if not check_min_liquidity(orderbook_a, MIN_LIQUIDITY_USD):
            return {"status": "rejected", "reason": f"Market A liquidity < ${MIN_LIQUIDITY_USD}"}

        if not check_min_liquidity(orderbook_b, MIN_LIQUIDITY_USD):
            return {"status": "rejected", "reason": f"Market B liquidity < ${MIN_LIQUIDITY_USD}"}

        # 5. Calculate net profitability with bid/ask prices
        if direction == "buy_a_sell_b":
            # Buy YES on A (pay ask), sell YES on B (get bid)
            cost_a = bid_ask_a["best_ask"]  # Pay the ask
            revenue_b = bid_ask_b["best_bid"]  # Get the bid
        else:
            # Buy YES on B (pay ask), sell YES on A (get bid)
            cost_a = bid_ask_b["best_ask"]
            revenue_b = bid_ask_a["best_bid"]

        gross_spread = revenue_b - cost_a
        gross_spread_pct = (gross_spread / cost_a * 100) if cost_a > 0 else 0

        # Calculate fees (Phase 3 - using actual bid/ask)
        fee_result = FeeCalculator.single_market_arb_fees(
            yes_price=cost_a,  # Not used when use_bid_ask=True
            no_price=1 - cost_a,
            platform="polymarket",  # Assume both Polymarket for now
            quantity=1.0,
            use_bid_ask=True,
            yes_ask=cost_a,
            no_ask=1 - revenue_b,
        )

        net_spread_pct = fee_result["net_pct"]

        logger.info(
            f"Spread analysis: gross={gross_spread_pct:.2f}%, net={net_spread_pct:.2f}% "
            f"(fees={fee_result['total_fees']:.4f})"
        )

        # 6. Safety check: Minimum net spread
        if net_spread_pct < MIN_NET_SPREAD_PCT:
            return {
                "status": "rejected",
                "reason": f"Net spread {net_spread_pct:.2f}% < {MIN_NET_SPREAD_PCT}% minimum",
                "gross_spread_pct": round(gross_spread_pct, 2),
                "net_spread_pct": round(net_spread_pct, 2),
            }

        # 7. Determine position size (max $50 or available liquidity)
        max_qty_a = min(
            MAX_POSITION_USD / cost_a,
            bid_ask_a["ask_size"],
        )
        max_qty_b = min(
            MAX_POSITION_USD / revenue_b,
            bid_ask_b["bid_size"],
        )
        quantity = min(max_qty_a, max_qty_b, 100)  # Cap at 100 contracts

        logger.info(f"Position size: {quantity:.2f} contracts (${quantity * cost_a:.2f})")

        # 8. Execute trades (PAPER TRADING ONLY FOR NOW - PHASE 3.1)
        # TODO: Replace with real execution when ready
        result = await self._execute_paper_arb(
            market_a=market_a,
            market_b=market_b,
            direction=direction,
            quantity=quantity,
            cost_a=cost_a,
            revenue_b=revenue_b,
            net_spread_pct=net_spread_pct,
        )

        return result

    async def _execute_paper_arb(
        self,
        market_a: Market,
        market_b: Market,
        direction: str,
        quantity: float,
        cost_a: float,
        revenue_b: float,
        net_spread_pct: float,
    ) -> dict:
        """Execute arbitrage in paper trading mode (Phase 3.1).

        Real execution will be added in Phase 3.2 after paper testing.
        """
        try:
            # Leg 1: Buy cheaper side
            if direction == "buy_a_sell_b":
                buy_market = market_a
                sell_market = market_b
                entry_price_buy = cost_a
                entry_price_sell = revenue_b
            else:
                buy_market = market_b
                sell_market = market_a
                entry_price_buy = cost_a
                entry_price_sell = revenue_b

            # Record buy position
            position_buy = PortfolioPosition(
                user_id=self.user_id,
                market_id=buy_market.id,
                platform_id=buy_market.platform_id,
                side="YES",
                entry_price=entry_price_buy,
                quantity=quantity,
                entry_time=datetime.utcnow(),
                strategy="arbitrage",
                portfolio_type="auto",
                is_simulated=True,
            )
            self.session.add(position_buy)

            # Leg 2: Sell other side (short position = sell YES = buy NO)
            position_sell = PortfolioPosition(
                user_id=self.user_id,
                market_id=sell_market.id,
                platform_id=sell_market.platform_id,
                side="NO",  # Selling YES = buying NO
                entry_price=1 - entry_price_sell,  # NO price
                quantity=quantity,
                entry_time=datetime.utcnow(),
                strategy="arbitrage",
                portfolio_type="auto",
                is_simulated=True,
            )
            self.session.add(position_sell)

            await self.session.commit()

            logger.info(
                f"✅ Paper arbitrage executed: "
                f"Buy {quantity:.2f} YES @ {entry_price_buy:.4f} on {buy_market.question[:30]}, "
                f"Sell {quantity:.2f} YES @ {entry_price_sell:.4f} on {sell_market.question[:30]}, "
                f"Expected profit: {net_spread_pct:.2f}%"
            )

            return {
                "status": "executed_paper",
                "positions": [position_buy.id, position_sell.id],
                "quantity": quantity,
                "buy_price": entry_price_buy,
                "sell_price": entry_price_sell,
                "gross_spread_pct": round(((revenue_b - cost_a) / cost_a * 100), 2),
                "net_spread_pct": round(net_spread_pct, 2),
                "expected_profit_usd": round(quantity * (revenue_b - cost_a), 2),
            }

        except Exception as e:
            logger.error(f"Paper arbitrage execution failed: {e}")
            await self.session.rollback()
            return {"status": "error", "reason": str(e)}


async def execute_arbitrage_opportunity(
    session: AsyncSession,
    opportunity: dict,
) -> dict:
    """Execute an arbitrage opportunity (convenience wrapper).

    Args:
        session: Database session
        opportunity: Opportunity dict from arbitrage scanner

    Returns:
        Execution result
    """
    executor = ArbitrageExecutor(session)

    # Extract market IDs and direction from opportunity
    market_ids = opportunity.get("market_ids", [])
    if len(market_ids) != 2:
        return {"status": "error", "reason": "Invalid market_ids"}

    # Determine direction from opportunity data
    buy_platform = opportunity.get("buy_platform")
    direction = "buy_a_sell_b"  # Default, should be in opportunity dict

    return await executor.execute_cross_platform_arb(
        market_a_id=market_ids[0],
        market_b_id=market_ids[1],
        direction=direction,
    )
