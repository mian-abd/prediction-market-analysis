"""Orderbook utilities for accurate arbitrage execution pricing (Phase 3).

Extracts best bid/ask from orderbook data to calculate real execution costs
instead of mid-prices, eliminating false positives.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_best_bid_ask(orderbook: dict) -> Optional[dict]:
    """Extract best bid and ask prices from orderbook.

    Args:
        orderbook: Dict with "bids" and "asks" arrays from CLOB API
            Format: {"bids": [{"price": "0.65", "size": "100"}, ...],
                     "asks": [{"price": "0.67", "size": "80"}, ...]}

    Returns:
        Dict with best prices and sizes, or None if orderbook empty:
        {
            "best_bid": 0.65,
            "best_ask": 0.67,
            "bid_size": 100.0,
            "ask_size": 80.0,
            "spread_abs": 0.02,
            "spread_pct": 3.08,
        }
    """
    if not orderbook:
        return None

    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])

    if not bids or not asks:
        logger.debug("Empty orderbook (no bids or asks)")
        return None

    try:
        # Best bid = highest buy price (first in sorted bids)
        best_bid = float(bids[0]["price"])
        bid_size = float(bids[0]["size"])

        # Best ask = lowest sell price (first in sorted asks)
        best_ask = float(asks[0]["price"])
        ask_size = float(asks[0]["size"])

        # Calculate spread
        spread_abs = best_ask - best_bid
        spread_pct = (spread_abs / best_bid) * 100 if best_bid > 0 else 0

        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "bid_size": bid_size,
            "ask_size": ask_size,
            "spread_abs": round(spread_abs, 4),
            "spread_pct": round(spread_pct, 2),
        }

    except (KeyError, IndexError, ValueError, TypeError) as e:
        logger.warning(f"Failed to parse orderbook: {e}")
        return None


def estimate_execution_price(
    orderbook: dict,
    side: str,
    quantity: float,
) -> Optional[dict]:
    """Estimate execution price for a quantity, accounting for orderbook depth.

    Args:
        orderbook: CLOB orderbook dict
        side: "buy" (use asks) or "sell" (use bids)
        quantity: Number of contracts to execute

    Returns:
        Dict with execution details:
        {
            "avg_price": 0.671,  # Volume-weighted average price
            "total_cost": 67.1,
            "filled": 100.0,     # Actual contracts filled
            "unfilled": 0.0,     # Contracts not filled (insufficient liquidity)
            "slippage_pct": 0.15,  # % worse than best price
        }
    """
    if not orderbook or quantity <= 0:
        return None

    # Use asks for buying, bids for selling
    levels = orderbook.get("asks" if side == "buy" else "bids", [])
    if not levels:
        return None

    try:
        best_price = float(levels[0]["price"])
        filled = 0.0
        total_cost = 0.0

        # Walk through price levels until quantity filled
        for level in levels:
            price = float(level["price"])
            size = float(level["size"])

            if filled >= quantity:
                break

            # How much can we fill at this level?
            fill_at_level = min(size, quantity - filled)
            filled += fill_at_level
            total_cost += fill_at_level * price

        # Calculate metrics
        avg_price = total_cost / filled if filled > 0 else 0
        unfilled = max(0, quantity - filled)
        slippage_pct = ((avg_price - best_price) / best_price * 100) if best_price > 0 else 0

        return {
            "avg_price": round(avg_price, 6),
            "total_cost": round(total_cost, 2),
            "filled": round(filled, 2),
            "unfilled": round(unfilled, 2),
            "slippage_pct": round(slippage_pct, 3),
        }

    except (KeyError, IndexError, ValueError, TypeError) as e:
        logger.warning(f"Failed to estimate execution: {e}")
        return None


def check_min_liquidity(
    orderbook: dict,
    min_usd: float = 500.0,
) -> bool:
    """Check if orderbook has minimum liquidity for safe execution.

    Args:
        orderbook: CLOB orderbook dict
        min_usd: Minimum USD liquidity required (default $500)

    Returns:
        True if both bid and ask sides have sufficient liquidity
    """
    bid_ask = get_best_bid_ask(orderbook)
    if not bid_ask:
        return False

    # Estimate USD liquidity at best prices
    bid_liquidity_usd = bid_ask["bid_size"] * bid_ask["best_bid"]
    ask_liquidity_usd = bid_ask["ask_size"] * bid_ask["best_ask"]

    return bid_liquidity_usd >= min_usd and ask_liquidity_usd >= min_usd
