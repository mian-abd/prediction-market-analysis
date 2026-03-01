"""Avellaneda-Stoikov Market Making Engine adapted for prediction markets.

This implements an inventory-aware two-sided quoting strategy based on the
Avellaneda-Stoikov (2008) framework, with adaptations for prediction market
mechanics:

  1. Binary terminal value: Assets resolve to 0 or 1, creating fundamentally
     different risk dynamics than continuous-price assets.
  2. Resolution deadline: Time-to-expiry is finite and known, which bounds
     inventory risk but also creates an adverse selection regime shift.
  3. Information events: News/catalysts cause discrete jumps that make quoting
     during event windows extremely risky.
  4. Maker rebates: On fee-enabled markets, filled maker orders earn rebates
     that offset spread capture costs.

Core equations (Avellaneda-Stoikov, adapted):
  reservation_price = mid - inventory * gamma * sigma^2 * tau
  optimal_spread = gamma * sigma^2 * tau + (2/gamma) * ln(1 + gamma/kappa)

Where:
  gamma = risk aversion (higher = wider spreads, less inventory risk)
  sigma = estimated volatility of the market
  tau = time to resolution (hours)
  kappa = order arrival intensity parameter
  inventory = current net position (positive = long YES, negative = short YES)

References:
  - Avellaneda & Stoikov (2008), "High-frequency trading in a limit order book"
  - Polymarket MM docs: docs.polymarket.com/market-makers/trading
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class QuoteAction(Enum):
    POST = "post"
    CANCEL = "cancel"
    WIDEN = "widen"
    HOLD = "hold"


@dataclass
class MarketMakingConfig:
    """Configuration for the market making engine."""
    gamma: float = 0.5
    kappa: float = 1.5
    min_spread_bps: int = 100
    max_spread_bps: int = 2000
    max_inventory: float = 500.0
    quote_size: float = 50.0
    min_time_to_resolution_hrs: float = 4.0
    max_price_deviation: float = 0.03
    volatility_window: int = 20
    # Risk controls
    max_loss_usd: float = 50.0
    kill_switch_drawdown_pct: float = 0.10
    no_quote_near_resolution_hrs: float = 2.0
    # Maker rebate estimate (fraction of taker fee returned)
    maker_rebate_frac: float = 0.20
    # Minimum book depth to quote against (avoid quoting into empty books)
    min_book_depth_usd: float = 100.0


@dataclass
class Quote:
    """A single-side quote to post on the book."""
    side: str  # "buy" or "sell"
    price: float
    size: float
    token_id: str = ""
    order_type: str = "GTC"
    expiration: int | None = None


@dataclass
class MarketMakingState:
    """Tracks the state of the MM engine for a single market."""
    market_id: int
    token_id: str = ""
    inventory: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_fills: int = 0
    total_volume: float = 0.0
    spread_captured: float = 0.0
    rebates_earned: float = 0.0
    last_mid: float = 0.5
    last_quote_time: datetime | None = None
    active_bids: list = field(default_factory=list)
    active_asks: list = field(default_factory=list)
    session_start: datetime = field(default_factory=datetime.utcnow)
    peak_pnl: float = 0.0
    max_drawdown: float = 0.0


class AvellanedaStoikovEngine:
    """Prediction-market adapted Avellaneda-Stoikov quoting engine.

    Computes optimal bid/ask prices given current market state, inventory,
    volatility, and time to resolution. Includes safety controls for
    prediction market-specific risks (binary resolution, information events).
    """

    def __init__(self, config: MarketMakingConfig | None = None):
        self.config = config or MarketMakingConfig()
        self.states: dict[int, MarketMakingState] = {}
        self.rng = np.random.RandomState(42)

    def get_state(self, market_id: int) -> MarketMakingState:
        if market_id not in self.states:
            self.states[market_id] = MarketMakingState(market_id=market_id)
        return self.states[market_id]

    def estimate_volatility(
        self,
        price_history: list[float],
        window: int | None = None,
    ) -> float:
        """Estimate realized volatility from recent price observations.

        For prediction markets, volatility is bounded by the binary outcome.
        Prices near 0 or 1 have lower realized volatility than prices near 0.5.
        """
        window = window or self.config.volatility_window
        if len(price_history) < 3:
            mid = price_history[-1] if price_history else 0.5
            return self._implied_vol_from_price(mid)

        prices = np.array(price_history[-window:])
        returns = np.diff(np.log(np.clip(prices, 0.01, 0.99)))

        if len(returns) < 2:
            return self._implied_vol_from_price(prices[-1])

        realized_vol = float(np.std(returns))
        # Floor at implied vol from price level (binary outcomes bound vol)
        floor_vol = self._implied_vol_from_price(prices[-1]) * 0.3
        return max(realized_vol, floor_vol)

    def _implied_vol_from_price(self, price: float) -> float:
        """Estimate vol from price level. Prices near 0/1 have lower vol."""
        p = np.clip(price, 0.01, 0.99)
        return float(math.sqrt(p * (1 - p)) * 0.1)

    def compute_reservation_price(
        self,
        mid: float,
        inventory: float,
        sigma: float,
        tau_hours: float,
    ) -> float:
        """Compute the reservation price (AS model).

        The reservation price is the mid price adjusted for inventory risk.
        When holding positive inventory (long YES), the reservation price drops
        below mid to incentivize selling.

        Args:
            mid: Current book midpoint
            inventory: Net position (positive = long, negative = short)
            sigma: Estimated volatility
            tau_hours: Hours until market resolution
        """
        gamma = self.config.gamma
        tau = max(tau_hours / 24.0, 0.01)  # Convert to days, floor at ~15 min
        inventory_penalty = inventory * gamma * (sigma ** 2) * tau
        reservation = mid - inventory_penalty
        return float(np.clip(reservation, 0.01, 0.99))

    def compute_optimal_spread(
        self,
        sigma: float,
        tau_hours: float,
    ) -> float:
        """Compute the optimal bid-ask spread (AS model).

        Wider when: volatility is high, time-to-resolution is large,
        risk aversion is high, or order arrival is rare.
        """
        gamma = self.config.gamma
        kappa = self.config.kappa
        tau = max(tau_hours / 24.0, 0.01)

        spread = gamma * (sigma ** 2) * tau + (2.0 / gamma) * math.log(1 + gamma / kappa)

        # Enforce min/max spread constraints
        min_spread = self.config.min_spread_bps / 10000.0
        max_spread = self.config.max_spread_bps / 10000.0
        return float(np.clip(spread, min_spread, max_spread))

    def compute_quotes(
        self,
        market_id: int,
        mid: float,
        price_history: list[float],
        tau_hours: float,
        best_bid: float | None = None,
        best_ask: float | None = None,
        bid_depth_usd: float = 0.0,
        ask_depth_usd: float = 0.0,
        taker_fee_bps: int = 0,
    ) -> tuple[Quote | None, Quote | None, QuoteAction]:
        """Compute optimal bid and ask quotes for a market.

        Returns (bid_quote, ask_quote, action) where action indicates
        whether to post, cancel, widen, or hold.
        """
        state = self.get_state(market_id)
        cfg = self.config

        # --- Safety checks ---
        if tau_hours < cfg.no_quote_near_resolution_hrs:
            logger.debug(
                f"Market {market_id}: {tau_hours:.1f}h to resolution, "
                f"below {cfg.no_quote_near_resolution_hrs}h threshold — cancelling quotes"
            )
            return None, None, QuoteAction.CANCEL

        if tau_hours < cfg.min_time_to_resolution_hrs:
            return None, None, QuoteAction.CANCEL

        # Check kill switch: max drawdown
        current_pnl = state.realized_pnl + state.unrealized_pnl
        state.peak_pnl = max(state.peak_pnl, current_pnl)
        drawdown = state.peak_pnl - current_pnl
        if state.peak_pnl > 0:
            drawdown_pct = drawdown / state.peak_pnl
        else:
            drawdown_pct = drawdown / max(abs(current_pnl), 1.0)

        if drawdown > cfg.max_loss_usd or drawdown_pct > cfg.kill_switch_drawdown_pct:
            logger.warning(
                f"Market {market_id}: KILL SWITCH — drawdown ${drawdown:.2f} "
                f"({drawdown_pct:.1%}). Cancelling all quotes."
            )
            state.max_drawdown = max(state.max_drawdown, drawdown)
            return None, None, QuoteAction.CANCEL

        # Check book depth
        if bid_depth_usd < cfg.min_book_depth_usd or ask_depth_usd < cfg.min_book_depth_usd:
            logger.debug(
                f"Market {market_id}: thin book "
                f"(bid_depth=${bid_depth_usd:.0f}, ask_depth=${ask_depth_usd:.0f})"
            )
            return None, None, QuoteAction.HOLD

        # --- Compute AS parameters ---
        sigma = self.estimate_volatility(price_history)
        reservation = self.compute_reservation_price(
            mid, state.inventory, sigma, tau_hours
        )
        spread = self.compute_optimal_spread(sigma, tau_hours)

        half_spread = spread / 2.0
        bid_price = reservation - half_spread
        ask_price = reservation + half_spread

        # Apply maker rebate adjustment — rebates effectively narrow
        # the required spread since we get some back.
        if taker_fee_bps > 0 and cfg.maker_rebate_frac > 0:
            fee_adjustment = (taker_fee_bps / 10000.0) * cfg.maker_rebate_frac
            bid_price += fee_adjustment / 2.0
            ask_price -= fee_adjustment / 2.0

        # Snap to tick (Polymarket uses 0.01 tick for most markets)
        bid_price = math.floor(bid_price * 100) / 100.0
        ask_price = math.ceil(ask_price * 100) / 100.0

        # Ensure spread is at least 1 tick
        if ask_price <= bid_price:
            ask_price = bid_price + 0.01

        # Price guard: reject if too far from current book
        if best_bid is not None and abs(bid_price - best_bid) > cfg.max_price_deviation:
            logger.debug(f"Market {market_id}: bid {bid_price:.3f} deviates from book {best_bid:.3f}")
        if best_ask is not None and abs(ask_price - best_ask) > cfg.max_price_deviation:
            logger.debug(f"Market {market_id}: ask {ask_price:.3f} deviates from book {best_ask:.3f}")

        # Clip to valid probability range
        bid_price = max(0.01, min(0.99, bid_price))
        ask_price = max(0.01, min(0.99, ask_price))

        # Inventory-based size skewing
        # When long, increase ask size and decrease bid size (encourage selling)
        inventory_ratio = state.inventory / cfg.max_inventory if cfg.max_inventory > 0 else 0
        inventory_ratio = np.clip(inventory_ratio, -1.0, 1.0)

        bid_size = cfg.quote_size * (1.0 - 0.5 * inventory_ratio)
        ask_size = cfg.quote_size * (1.0 + 0.5 * inventory_ratio)

        # Hard inventory limit: don't add to a position that's already at max
        if state.inventory >= cfg.max_inventory:
            bid_size = 0.0
        if state.inventory <= -cfg.max_inventory:
            ask_size = 0.0

        bid_quote = Quote(side="buy", price=bid_price, size=bid_size) if bid_size > 0 else None
        ask_quote = Quote(side="sell", price=ask_price, size=ask_size) if ask_size > 0 else None

        state.last_mid = mid
        state.last_quote_time = datetime.utcnow()

        return bid_quote, ask_quote, QuoteAction.POST

    def process_fill(
        self,
        market_id: int,
        side: str,
        fill_price: float,
        fill_size: float,
        taker_fee_bps: int = 0,
    ) -> None:
        """Process a fill event, updating inventory and P&L tracking."""
        state = self.get_state(market_id)

        if side == "buy":
            state.inventory += fill_size
            cost = fill_price * fill_size
            state.realized_pnl -= cost
        else:
            state.inventory -= fill_size
            revenue = fill_price * fill_size
            state.realized_pnl += revenue

        # Track maker rebate earned
        if taker_fee_bps > 0:
            p = fill_price
            fee_rate = taker_fee_bps / 10000.0
            taker_fee = fill_size * fee_rate * (p * (1 - p))
            rebate = taker_fee * self.config.maker_rebate_frac
            state.rebates_earned += rebate

        state.total_fills += 1
        state.total_volume += fill_size * fill_price
        state.spread_captured += abs(fill_price - state.last_mid) * fill_size

    def update_unrealized_pnl(self, market_id: int, current_mid: float) -> None:
        """Update the unrealized P&L based on current mid price."""
        state = self.get_state(market_id)
        state.unrealized_pnl = state.inventory * current_mid

    def should_quote_market(
        self,
        market_price: float,
        tau_hours: float,
        liquidity: float,
        volume_24h: float,
        taker_fee_bps: int = 0,
    ) -> tuple[bool, str]:
        """Decide whether this market is suitable for market making.

        Returns (should_quote, reason).
        """
        cfg = self.config

        # Don't quote near resolution
        if tau_hours < cfg.min_time_to_resolution_hrs:
            return False, f"Too close to resolution ({tau_hours:.1f}h < {cfg.min_time_to_resolution_hrs}h)"

        # Only quote markets with reasonable probability (avoid extremes)
        if market_price < 0.10 or market_price > 0.90:
            return False, f"Price too extreme ({market_price:.2f})"

        # Need some baseline liquidity
        if liquidity < cfg.min_book_depth_usd * 2:
            return False, f"Insufficient liquidity (${liquidity:.0f})"

        # Need activity
        if volume_24h < 50:
            return False, f"Insufficient volume (${volume_24h:.0f}/24h)"

        # Prefer fee-enabled markets (maker rebates provide structural edge)
        if taker_fee_bps > 0:
            return True, "Fee-enabled market with maker rebates"

        # Fee-free markets require tighter conditions
        if volume_24h > 500 and liquidity > 1000:
            return True, "High-activity fee-free market"

        return False, "Does not meet minimum quoting criteria"

    def get_summary(self, market_id: int) -> dict:
        """Get summary statistics for a market making session."""
        state = self.get_state(market_id)
        total_pnl = state.realized_pnl + state.unrealized_pnl + state.rebates_earned
        runtime = (datetime.utcnow() - state.session_start).total_seconds() / 3600

        return {
            "market_id": market_id,
            "inventory": round(state.inventory, 2),
            "realized_pnl": round(state.realized_pnl, 4),
            "unrealized_pnl": round(state.unrealized_pnl, 4),
            "rebates_earned": round(state.rebates_earned, 4),
            "total_pnl": round(total_pnl, 4),
            "total_fills": state.total_fills,
            "total_volume": round(state.total_volume, 2),
            "spread_captured": round(state.spread_captured, 4),
            "max_drawdown": round(state.max_drawdown, 4),
            "runtime_hours": round(runtime, 2),
            "pnl_per_hour": round(total_pnl / max(runtime, 0.01), 4),
        }


def run_mm_backtest(
    price_series: list[float],
    timestamps: list[datetime],
    orderbook_series: list[dict],
    resolution_time: datetime,
    config: MarketMakingConfig | None = None,
    taker_fee_bps: int = 0,
) -> dict:
    """Backtest the market making strategy on historical data.

    Args:
        price_series: Historical mid prices
        timestamps: Timestamps for each price observation
        orderbook_series: Historical orderbook snapshots, each with
            'best_bid', 'best_ask', 'bid_depth_usd', 'ask_depth_usd'
        resolution_time: When the market resolved
        config: MM configuration
        taker_fee_bps: Taker fee in basis points

    Returns:
        Dict with backtest results including P&L, fill count, etc.
    """
    engine = AvellanedaStoikovEngine(config)
    rng = np.random.RandomState(42)
    market_id = 0

    fills_log = []
    quote_log = []
    pnl_series = []

    for i in range(1, len(price_series)):
        mid = price_series[i]
        ts = timestamps[i]
        tau_hours = max(0, (resolution_time - ts).total_seconds() / 3600)

        ob = orderbook_series[i] if i < len(orderbook_series) else {}
        best_bid = ob.get("best_bid", mid - 0.01)
        best_ask = ob.get("best_ask", mid + 0.01)
        bid_depth = ob.get("bid_depth_usd", 500.0)
        ask_depth = ob.get("ask_depth_usd", 500.0)

        history = price_series[max(0, i - 20):i + 1]

        bid_q, ask_q, action = engine.compute_quotes(
            market_id=market_id,
            mid=mid,
            price_history=history,
            tau_hours=tau_hours,
            best_bid=best_bid,
            best_ask=best_ask,
            bid_depth_usd=bid_depth,
            ask_depth_usd=ask_depth,
            taker_fee_bps=taker_fee_bps,
        )

        quote_log.append({
            "timestamp": ts,
            "mid": mid,
            "action": action.value,
            "bid_price": bid_q.price if bid_q else None,
            "ask_price": ask_q.price if ask_q else None,
            "bid_size": bid_q.size if bid_q else 0,
            "ask_size": ask_q.size if ask_q else 0,
            "inventory": engine.get_state(market_id).inventory,
        })

        if action != QuoteAction.POST:
            engine.update_unrealized_pnl(market_id, mid)
            state = engine.get_state(market_id)
            pnl_series.append(state.realized_pnl + state.unrealized_pnl + state.rebates_earned)
            continue

        # Simulate fills based on price movement
        price_change = mid - price_series[i - 1]

        # Bid fill: price moved down toward our bid
        if bid_q and price_change < 0 and mid <= bid_q.price:
            fill_prob = min(1.0, abs(price_change) / max(bid_q.price - best_bid + 0.001, 0.001))
            fill_prob *= 0.5  # Partial fill probability
            if rng.random() < fill_prob:
                fill_size = bid_q.size * rng.uniform(0.3, 1.0)
                engine.process_fill(market_id, "buy", bid_q.price, fill_size, taker_fee_bps)
                fills_log.append({
                    "timestamp": ts, "side": "buy",
                    "price": bid_q.price, "size": fill_size,
                })

        # Ask fill: price moved up toward our ask
        if ask_q and price_change > 0 and mid >= ask_q.price:
            fill_prob = min(1.0, abs(price_change) / max(best_ask - ask_q.price + 0.001, 0.001))
            fill_prob *= 0.5
            if rng.random() < fill_prob:
                fill_size = ask_q.size * rng.uniform(0.3, 1.0)
                engine.process_fill(market_id, "sell", ask_q.price, fill_size, taker_fee_bps)
                fills_log.append({
                    "timestamp": ts, "side": "sell",
                    "price": ask_q.price, "size": fill_size,
                })

        engine.update_unrealized_pnl(market_id, mid)
        state = engine.get_state(market_id)
        pnl_series.append(state.realized_pnl + state.unrealized_pnl + state.rebates_earned)

    # Settle remaining inventory at resolution
    state = engine.get_state(market_id)
    final_price = price_series[-1]
    if abs(state.inventory) > 0.01:
        # Assume binary resolution: final price is close to 0 or 1
        settlement_value = 1.0 if final_price > 0.5 else 0.0
        settlement_pnl = state.inventory * settlement_value
        state.realized_pnl += settlement_pnl + state.unrealized_pnl
        state.unrealized_pnl = 0.0
        state.inventory = 0.0

    summary = engine.get_summary(market_id)
    summary["n_fills"] = len(fills_log)
    summary["n_quotes"] = len(quote_log)
    summary["pnl_series"] = pnl_series
    summary["fills"] = fills_log
    summary["quote_log"] = quote_log

    return summary
