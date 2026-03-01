"""ML-Informed Market Making: use ensemble predictions to skew AS quotes.

Instead of using the ML model as a directional taker signal, this module
uses model predictions to adjust the Avellaneda-Stoikov reservation price.
If the model says a market at 0.50 should be 0.60, we skew our quotes
to be net-long YES — buying at 0.49 and selling at 0.62 — capturing spread
while leaning into the predicted direction.

This converts a taker edge into a maker edge: same signal, better execution,
structural +1.12% advantage instead of -1.12%.

The skewing formula:
    reservation = mid + alpha * (model_fair_value - mid)

Where alpha in [0, 1] controls how aggressively we lean on the model.
alpha=0 => pure AS (no model influence), alpha=1 => fully trust the model.
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime

from ml.strategies.market_making import (
    AvellanedaStoikovEngine,
    MarketMakingConfig,
    Quote,
    QuoteAction,
)

logger = logging.getLogger(__name__)


@dataclass
class MLMMConfig(MarketMakingConfig):
    """Extended config with ML-skewing parameters."""

    alpha: float = 0.3
    min_edge_to_skew: float = 0.02
    max_skew_cents: float = 0.05
    confidence_decay_hours: float = 6.0


class MLInformedMarketMaker:
    """Avellaneda-Stoikov engine with ML-based quote skewing.

    The model prediction adjusts the reservation price, biasing our
    inventory toward the predicted direction while still capturing spread.
    """

    def __init__(self, config: MLMMConfig | None = None):
        self.config = config or MLMMConfig()
        self.engine = AvellanedaStoikovEngine(self.config)

    def compute_skewed_quotes(
        self,
        market_id: int,
        mid: float,
        price_history: list[float],
        tau_hours: float,
        model_fair_value: float,
        model_confidence: float = 1.0,
        prediction_age_hours: float = 0.0,
        best_bid: float | None = None,
        best_ask: float | None = None,
        bid_depth_usd: float = 0.0,
        ask_depth_usd: float = 0.0,
        taker_fee_bps: int = 0,
    ) -> tuple[Quote | None, Quote | None, QuoteAction, dict]:
        """Compute ML-skewed bid/ask quotes.

        Args:
            market_id: Market identifier
            mid: Current book midpoint
            price_history: Recent price observations for volatility estimation
            tau_hours: Hours until market resolution
            model_fair_value: The ML model's predicted probability
            model_confidence: How confident we are in the prediction [0, 1]
            prediction_age_hours: How old the prediction is (for decay)
            best_bid, best_ask: Current best quotes on the book
            bid_depth_usd, ask_depth_usd: Book depth
            taker_fee_bps: Fee rate for this market

        Returns:
            (bid_quote, ask_quote, action, debug_info)
        """
        cfg = self.config
        state = self.engine.get_state(market_id)

        edge = model_fair_value - mid
        abs_edge = abs(edge)

        # Decay confidence based on prediction staleness
        if prediction_age_hours > 0 and cfg.confidence_decay_hours > 0:
            decay = math.exp(-prediction_age_hours / cfg.confidence_decay_hours)
            effective_confidence = model_confidence * decay
        else:
            effective_confidence = model_confidence

        # Only skew if edge exceeds minimum
        if abs_edge < cfg.min_edge_to_skew:
            effective_alpha = 0.0
        else:
            effective_alpha = cfg.alpha * effective_confidence

        # Compute the skewed midpoint
        skew = effective_alpha * edge
        skew = max(-cfg.max_skew_cents, min(cfg.max_skew_cents, skew))
        skewed_mid = mid + skew

        debug_info = {
            "raw_edge": round(edge, 4),
            "effective_alpha": round(effective_alpha, 4),
            "skew_applied": round(skew, 4),
            "skewed_mid": round(skewed_mid, 4),
            "original_mid": round(mid, 4),
            "model_fair_value": round(model_fair_value, 4),
            "prediction_age_hours": round(prediction_age_hours, 2),
            "effective_confidence": round(effective_confidence, 3),
        }

        # Use the AS engine with the skewed mid
        sigma = self.engine.estimate_volatility(price_history)
        reservation = self.engine.compute_reservation_price(
            skewed_mid, state.inventory, sigma, tau_hours
        )
        spread = self.engine.compute_optimal_spread(sigma, tau_hours)

        half_spread = spread / 2.0
        bid_price = reservation - half_spread
        ask_price = reservation + half_spread

        if taker_fee_bps > 0 and cfg.maker_rebate_frac > 0:
            fee_adj = (taker_fee_bps / 10000.0) * cfg.maker_rebate_frac
            bid_price += fee_adj / 2.0
            ask_price -= fee_adj / 2.0

        bid_price = math.floor(bid_price * 100) / 100.0
        ask_price = math.ceil(ask_price * 100) / 100.0

        if ask_price <= bid_price:
            ask_price = bid_price + 0.01

        bid_price = max(0.01, min(0.99, bid_price))
        ask_price = max(0.01, min(0.99, ask_price))

        # Safety: pass through the AS engine's safety checks
        if tau_hours < cfg.no_quote_near_resolution_hrs:
            return None, None, QuoteAction.CANCEL, debug_info
        if tau_hours < cfg.min_time_to_resolution_hrs:
            return None, None, QuoteAction.CANCEL, debug_info

        current_pnl = state.realized_pnl + state.unrealized_pnl
        state.peak_pnl = max(state.peak_pnl, current_pnl)
        dd = state.peak_pnl - current_pnl
        if dd > cfg.max_loss_usd:
            return None, None, QuoteAction.CANCEL, debug_info

        if bid_depth_usd < cfg.min_book_depth_usd or ask_depth_usd < cfg.min_book_depth_usd:
            return None, None, QuoteAction.HOLD, debug_info

        # Inventory-based size skewing
        import numpy as np
        inv_ratio = state.inventory / cfg.max_inventory if cfg.max_inventory > 0 else 0
        inv_ratio = float(np.clip(inv_ratio, -1.0, 1.0))

        bid_size = cfg.quote_size * (1.0 - 0.5 * inv_ratio)
        ask_size = cfg.quote_size * (1.0 + 0.5 * inv_ratio)

        if state.inventory >= cfg.max_inventory:
            bid_size = 0.0
        if state.inventory <= -cfg.max_inventory:
            ask_size = 0.0

        bid_q = Quote(side="buy", price=bid_price, size=bid_size) if bid_size > 0 else None
        ask_q = Quote(side="sell", price=ask_price, size=ask_size) if ask_size > 0 else None

        state.last_mid = mid
        state.last_quote_time = datetime.utcnow()

        debug_info["bid_price"] = bid_price if bid_q else None
        debug_info["ask_price"] = ask_price if ask_q else None
        debug_info["spread"] = round(spread, 4)
        debug_info["sigma"] = round(sigma, 4)
        debug_info["reservation"] = round(reservation, 4)

        return bid_q, ask_q, QuoteAction.POST, debug_info

    def process_fill(self, market_id: int, side: str, price: float, size: float, fee_bps: int = 0):
        """Delegate fill processing to the AS engine."""
        self.engine.process_fill(market_id, side, price, size, fee_bps)

    def update_unrealized_pnl(self, market_id: int, current_mid: float):
        self.engine.update_unrealized_pnl(market_id, current_mid)

    def get_state(self, market_id: int):
        return self.engine.get_state(market_id)

    def get_summary(self, market_id: int) -> dict:
        return self.engine.get_summary(market_id)


def select_markets_for_mm(
    markets: list[dict],
    engine: AvellanedaStoikovEngine | None = None,
) -> list[dict]:
    """Filter a list of markets to those suitable for market making.

    Each market dict should have: price_yes, tau_hours (or end_date),
    liquidity, volume_24h, taker_fee_bps (optional).

    Returns the subset of markets that pass should_quote_market().
    """
    if engine is None:
        engine = AvellanedaStoikovEngine()

    selected = []
    for m in markets:
        price = m.get("price_yes", 0.5)
        tau = m.get("tau_hours", 100.0)
        liq = m.get("liquidity", 0.0)
        vol = m.get("volume_24h", 0.0)
        fee = m.get("taker_fee_bps", 0)

        should, reason = engine.should_quote_market(price, tau, liq, vol, fee)
        if should:
            m["mm_reason"] = reason
            selected.append(m)

    return selected
