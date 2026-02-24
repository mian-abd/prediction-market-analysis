"""Resolution Convergence (Time-Decay) Strategy — profit from near-certain
outcomes approaching resolution.

Research basis:
- "Toward Black-Scholes for Prediction Markets" (arxiv, 2025): Develops
  options-like Greeks (theta, delta) for prediction market contracts.
- Price convergence: as resolution approaches, prices converge to 0 or 1.
  Any remaining spread represents theta capture opportunity.
- Theta accelerates dramatically in final 24-72 hours.

Strategy:
- Find markets within 24-72 hours of resolution where the outcome is nearly
  certain (price > 90% or < 10%).
- Buy the near-certain outcome.
- Edge is small per trade (~3-5%) but win rate is very high (>90%).
- High frequency of trades with high win rate = steady compound growth.
- This is analogous to selling deep out-of-the-money options near expiry.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Market
from ml.strategies.ensemble_edge_detector import (
    POLYMARKET_FEE_RATE, SLIPPAGE_BUFFER, compute_kelly,
)

logger = logging.getLogger(__name__)

MIN_VOLUME_TOTAL = 20_000
MIN_LIQUIDITY = 5_000
MIN_NET_EDGE = 0.015  # Lower threshold since win rate is very high

# Resolution time windows (hours)
MAX_HOURS_TO_RESOLUTION = 72
MIN_HOURS_TO_RESOLUTION = 1  # Avoid markets about to resolve (execution risk)

# Price thresholds for near-certain outcomes (tiered)
HIGH_CERTAINTY_THRESHOLD = 0.92  # Buy YES when price > 92%
LOW_CERTAINTY_THRESHOLD = 0.08   # Buy NO when price < 8%

VERY_HIGH_CERTAINTY = 0.95       # Tighter window (48h) + higher Kelly
VERY_LOW_CERTAINTY = 0.05

# Ultra-high certainty tier: 97%+ probability within 48h
# Research: "endgame arbitrage" at 95-99% yields 548% annualized (PolyCue, 2025)
# These are near-certain outcomes — essentially selling deep OTM options near expiry
ULTRA_HIGH_CERTAINTY = 0.97
ULTRA_LOW_CERTAINTY = 0.03
ULTRA_MAX_HOURS = 48             # Tighter window for the ultra tier
ULTRA_MAX_KELLY = 0.05           # Higher Kelly: near-certain outcomes justify more size

MAX_KELLY = 0.03
ULTRA_KELLY = ULTRA_MAX_KELLY
KELLY_FRACTION = 0.35  # Higher fraction since win rate is very high


def _hours_to_resolution(market: Market) -> Optional[float]:
    """Compute hours until market resolution. Returns None if unknown."""
    if not market.end_date:
        return None
    now = datetime.utcnow()
    end = market.end_date.replace(tzinfo=None) if market.end_date.tzinfo else market.end_date
    delta = end - now
    if delta.total_seconds() < 0:
        return 0.0
    return delta.total_seconds() / 3600


def compute_convergence_edge(
    market_price: float,
    hours_to_resolution: float,
) -> dict:
    """Compute the resolution convergence edge.

    Three tiers of certainty (each with its own time window and Kelly cap):
    - Standard (92-97%): MAX_HOURS_TO_RESOLUTION=72h, MAX_KELLY=0.03
    - Very high (95-97%): same window, tier="high" for better position sizing
    - Ultra (97%+): ULTRA_MAX_HOURS=48h, ULTRA_MAX_KELLY=0.05

    Research: "endgame arbitrage" at 95-99% probability yields 548% annualized
    (PolyCue, 2025). The key risk is incorrect resolution — mitigated by requiring
    high volume and stopping at 97%+ threshold rather than chasing 99.9%.
    """
    if 0.08 <= market_price <= 0.92:
        return {"direction": None, "net_ev": 0.0}

    # Determine which tier applies and validate time window
    is_ultra = (market_price >= ULTRA_HIGH_CERTAINTY or market_price <= ULTRA_LOW_CERTAINTY)
    if is_ultra and hours_to_resolution > ULTRA_MAX_HOURS:
        # Ultra tier requires tighter time window
        is_ultra = False  # Downgrade to standard high-certainty tier

    # Estimate true probability based on price and time to resolution
    time_window = ULTRA_MAX_HOURS if is_ultra else MAX_HOURS_TO_RESOLUTION
    time_factor = max(0.0, 1.0 - hours_to_resolution / time_window)

    if market_price >= HIGH_CERTAINTY_THRESHOLD:
        true_prob = market_price + (1.0 - market_price) * time_factor * 0.7
        true_prob = min(0.99, true_prob)
        direction = "buy_yes"
        p, q = true_prob, market_price
        fee = p * POLYMARKET_FEE_RATE * (1 - q) + SLIPPAGE_BUFFER
        net_ev = p * (1 - q) - (1 - p) * q - fee
    elif market_price <= LOW_CERTAINTY_THRESHOLD:
        true_prob = market_price * (1.0 - time_factor * 0.7)
        true_prob = max(0.01, true_prob)
        direction = "buy_no"
        p, q = true_prob, market_price
        fee = (1 - p) * POLYMARKET_FEE_RATE * q + SLIPPAGE_BUFFER
        net_ev = (1 - p) * q - p * (1 - q) - fee
    else:
        return {"direction": None, "net_ev": 0.0}

    if net_ev <= 0:
        return {"direction": None, "net_ev": 0.0}

    kelly_cap = ULTRA_MAX_KELLY if is_ultra else MAX_KELLY
    kelly_raw = compute_kelly(direction, true_prob, market_price, fee)
    kelly = min(kelly_raw, kelly_cap)

    # Confidence: higher for ultra tier, extreme prices, and short time windows
    price_extremity = max(0, (abs(market_price - 0.5) - 0.42) / 0.08)  # 0 at 92%, 1 at 100%
    time_urgency = time_factor
    confidence = 0.5 + 0.3 * price_extremity + 0.2 * time_urgency
    if is_ultra:
        confidence = min(0.95, confidence + 0.10)  # Bonus for ultra certainty

    # Tier classification
    if is_ultra:
        tier = "high"
    elif market_price >= VERY_HIGH_CERTAINTY or market_price <= VERY_LOW_CERTAINTY:
        tier = "high"
    else:
        tier = "medium"

    return {
        "direction": direction,
        "true_prob": round(true_prob, 4),
        "market_price": round(market_price, 4),
        "raw_edge": round(abs(true_prob - market_price), 4),
        "net_ev": round(net_ev, 4),
        "fee_cost": round(fee, 4),
        "kelly_fraction": round(kelly, 4),
        "confidence": round(min(0.95, confidence), 3),
        "quality_tier": tier,
        "certainty_tier": "ultra" if is_ultra else ("very_high" if market_price >= VERY_HIGH_CERTAINTY or market_price <= VERY_LOW_CERTAINTY else "standard"),
        "time_factor": round(time_factor, 3),
        "hours_to_resolution": round(hours_to_resolution, 1),
    }


async def scan_resolution_convergence(
    session: AsyncSession,
) -> list[dict]:
    """Scan for markets approaching resolution with near-certain outcomes.

    The scan window covers the largest time window (72h standard). Ultra-tier
    markets outside the 48h window are handled by compute_convergence_edge()
    by downgrading them to the standard tier.
    """
    now = datetime.utcnow()
    min_end = now + timedelta(hours=MIN_HOURS_TO_RESOLUTION)
    max_end = now + timedelta(hours=MAX_HOURS_TO_RESOLUTION)

    result = await session.execute(
        select(Market).where(
            Market.is_active == True,  # noqa: E711
            Market.price_yes != None,  # noqa: E711
            Market.end_date != None,  # noqa: E711
            Market.end_date >= min_end,
            Market.end_date <= max_end,
            Market.volume_total >= MIN_VOLUME_TOTAL,
        ).order_by(Market.end_date.asc())
    )
    markets = result.scalars().all()

    if not markets:
        logger.debug("No markets approaching resolution in 1-72h window")
        return []

    edges_found = []

    for market in markets:
        price = market.price_yes or 0.5
        liquidity = float(market.liquidity or 0)

        if liquidity < MIN_LIQUIDITY:
            continue

        hours = _hours_to_resolution(market)
        if hours is None or hours < MIN_HOURS_TO_RESOLUTION or hours > MAX_HOURS_TO_RESOLUTION:
            continue

        edge = compute_convergence_edge(price, hours)
        if not edge.get("direction") or edge["net_ev"] < MIN_NET_EDGE:
            continue

        signal = {
            "market_id": market.id,
            "strategy": "resolution_convergence",
            "question": market.question,
            "category": market.normalized_category or market.category,
            **edge,
        }
        edges_found.append(signal)

        logger.info(
            f"Resolution convergence: {market.question[:50]}... | "
            f"Price: {price:.1%} | {hours:.0f}h to resolution | "
            f"EV: {edge['net_ev']:.1%}"
        )

    logger.info(f"Resolution convergence scan: {len(markets)} candidates, {len(edges_found)} edges")
    return edges_found
