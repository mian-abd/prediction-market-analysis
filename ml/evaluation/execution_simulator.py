"""Execution simulator for realistic fill modeling.

Simulates both taker (market order) and maker (limit order) execution,
modeling fill probability, queue position, and adverse selection.

This is the critical prerequisite for the market making engine —
maker profitability depends on fill rates and queue priority that
can only be validated through honest simulation.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class SimulatedOrder:
    """A single simulated order."""
    order_id: str
    market_id: int
    side: OrderSide
    order_type: OrderType
    price: float
    quantity: float
    timestamp: datetime
    filled: bool = False
    fill_price: float = 0.0
    fill_quantity: float = 0.0
    fill_time: datetime | None = None
    slippage: float = 0.0
    queue_position: int = 0


@dataclass
class SimulatedFill:
    """Result of a simulated fill."""
    order_id: str
    market_id: int
    side: str
    order_type: str
    limit_price: float
    fill_price: float
    quantity: float
    slippage_bps: float
    fill_latency_ms: float
    was_adverse: bool = False


@dataclass
class OrderbookState:
    """Snapshot of an orderbook for simulation."""
    bids: list[tuple[float, float]]  # [(price, size), ...]
    asks: list[tuple[float, float]]
    timestamp: datetime
    midpoint: float = 0.0
    spread: float = 0.0

    def __post_init__(self):
        if self.bids and self.asks:
            best_bid = self.bids[0][0]
            best_ask = self.asks[0][0]
            self.midpoint = (best_bid + best_ask) / 2
            self.spread = best_ask - best_bid


class ExecutionSimulator:
    """Simulates order execution with realistic fill modeling.

    Models:
    - Market orders: cross spread + size-dependent slippage
    - Limit orders: fill probability based on queue position and time
    - Adverse selection: information-driven fills (maker risk)
    """

    def __init__(
        self,
        default_fill_rate: float = 0.3,
        adverse_selection_rate: float = 0.15,
        latency_ms: float = 100,
        seed: int = 42,
    ):
        self.default_fill_rate = default_fill_rate
        self.adverse_selection_rate = adverse_selection_rate
        self.latency_ms = latency_ms
        self.fills: list[SimulatedFill] = []
        self.rng = np.random.RandomState(seed)

    def simulate_market_order(
        self,
        order: SimulatedOrder,
        orderbook: OrderbookState,
    ) -> SimulatedFill:
        """Simulate a market order crossing the spread.

        Market orders always fill but face slippage proportional to
        order size relative to available liquidity.
        """
        if order.side == OrderSide.BUY:
            levels = orderbook.asks
        else:
            levels = orderbook.bids

        if not levels:
            # No orderbook data — use midpoint + default slippage
            fill_price = orderbook.midpoint * (1.02 if order.side == OrderSide.BUY else 0.98)
            slippage_bps = 200
        else:
            # Walk the book to compute volume-weighted average fill price
            remaining = order.quantity
            total_cost = 0.0

            for price, size in levels:
                fill_at_level = min(remaining, size)
                total_cost += fill_at_level * price
                remaining -= fill_at_level
                if remaining <= 0:
                    break

            if remaining > 0:
                # Order larger than available liquidity — fill remainder at worst price + penalty
                worst_price = levels[-1][0] if levels else orderbook.midpoint
                penalty = 0.03 * (remaining / order.quantity)
                if order.side == OrderSide.BUY:
                    total_cost += remaining * (worst_price + penalty)
                else:
                    total_cost += remaining * (worst_price - penalty)

            fill_price = total_cost / order.quantity
            if order.side == OrderSide.BUY:
                slippage_bps = max(0, (fill_price - orderbook.midpoint) / orderbook.midpoint * 10000)
            else:
                slippage_bps = max(0, (orderbook.midpoint - fill_price) / orderbook.midpoint * 10000)

        fill = SimulatedFill(
            order_id=order.order_id,
            market_id=order.market_id,
            side=order.side.value,
            order_type=order.order_type.value,
            limit_price=order.price,
            fill_price=fill_price,
            quantity=order.quantity,
            slippage_bps=slippage_bps,
            fill_latency_ms=self.latency_ms,
            was_adverse=False,
        )
        self.fills.append(fill)
        return fill

    def simulate_limit_order(
        self,
        order: SimulatedOrder,
        orderbook: OrderbookState,
        time_in_book_minutes: float = 30,
        price_moved_towards: bool = False,
    ) -> SimulatedFill | None:
        """Simulate a limit order with fill probability modeling.

        Limit orders control price but may not fill. Fill probability depends on:
        - Distance from midpoint (closer = more likely to fill)
        - Time in book (longer = more likely)
        - Whether price moved towards the order (adverse selection signal)

        Returns None if order doesn't fill.
        """
        if order.side == OrderSide.BUY:
            distance = orderbook.midpoint - order.price
        else:
            distance = order.price - orderbook.midpoint

        distance_pct = distance / orderbook.midpoint if orderbook.midpoint > 0 else 0.5

        # Fill probability model
        # Base rate * distance decay * time factor
        distance_factor = math.exp(-5 * abs(distance_pct))
        time_factor = 1 - math.exp(-0.05 * time_in_book_minutes)
        fill_prob = self.default_fill_rate * distance_factor * time_factor

        # Adverse selection: if price moved towards our order, the fill is
        # more likely but also more likely to be "toxic" (informed trader)
        is_adverse = False
        if price_moved_towards:
            fill_prob *= 1.5
            is_adverse = self.rng.random() < self.adverse_selection_rate

        if self.rng.random() > fill_prob:
            return None

        fill_ratio = self.rng.uniform(0.5, 1.0)
        fill_quantity = order.quantity * fill_ratio

        fill = SimulatedFill(
            order_id=order.order_id,
            market_id=order.market_id,
            side=order.side.value,
            order_type=order.order_type.value,
            limit_price=order.price,
            fill_price=order.price,
            quantity=fill_quantity,
            slippage_bps=0,
            fill_latency_ms=self.latency_ms + self.rng.exponential(500),
            was_adverse=is_adverse,
        )
        self.fills.append(fill)
        return fill

    def get_fill_statistics(self) -> dict:
        """Compute aggregate statistics across all simulated fills."""
        if not self.fills:
            return {"total_fills": 0}

        market_fills = [f for f in self.fills if f.order_type == "market"]
        limit_fills = [f for f in self.fills if f.order_type == "limit"]
        adverse_fills = [f for f in self.fills if f.was_adverse]

        avg_slippage = np.mean([f.slippage_bps for f in market_fills]) if market_fills else 0

        return {
            "total_fills": len(self.fills),
            "market_fills": len(market_fills),
            "limit_fills": len(limit_fills),
            "avg_market_slippage_bps": float(avg_slippage),
            "adverse_fill_rate": len(adverse_fills) / max(len(limit_fills), 1),
            "avg_fill_latency_ms": float(np.mean([f.fill_latency_ms for f in self.fills])),
        }

    def reset(self):
        """Clear all fills for a new simulation run."""
        self.fills = []
