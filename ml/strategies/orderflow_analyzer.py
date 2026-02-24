"""Order Flow Analysis — detect informed trading from orderbook dynamics.

Research basis:
- VPIN literature (Easley, Lopez de Prado, O'Hara): order flow imbalance
  signals informed trading and predicts short-term price moves.
- Signed VPIN generates annualized 11-17% alpha (Borochin & Rush, 2015).
- Polymarket CLOB orderbook provides real-time depth at every price level.

Strategy:
- Analyze orderbook snapshots for sudden depth changes, one-sided pressure
- Compute Order Book Imbalance (OBI) velocity — how fast the imbalance is shifting
- Detect large directional pressure that hasn't moved the price yet
- Generate signals when orderbook shape predicts imminent price movement

This strategy uses data you're ALREADY collecting (orderbook snapshots every 2 min)
but aren't using to generate trading signals beyond simple features.
"""

import logging
import math
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Market, OrderbookSnapshot
from ml.features.orderbook_features import compute_orderbook_features
from ml.strategies.ensemble_edge_detector import (
    POLYMARKET_FEE_RATE, SLIPPAGE_BUFFER, compute_kelly,
)

logger = logging.getLogger(__name__)

MIN_VOLUME_TOTAL = 15_000
MIN_LIQUIDITY = 5_000
MIN_NET_EDGE = 0.025
MIN_OBI_MAGNITUDE = 0.3  # Strong imbalance threshold
OBI_VELOCITY_THRESHOLD = 0.15  # OBI change per snapshot that signals momentum
DEPTH_RATIO_EXTREME = 2.5  # One side has 2.5x the depth of the other
SNAPSHOTS_LOOKBACK = 5  # Compare last 5 snapshots for velocity


async def _get_recent_orderbooks(
    session: AsyncSession,
    market_id: int,
    n_snapshots: int = SNAPSHOTS_LOOKBACK,
) -> list[OrderbookSnapshot]:
    """Fetch N most recent orderbook snapshots for a market."""
    result = await session.execute(
        select(OrderbookSnapshot)
        .where(OrderbookSnapshot.market_id == market_id)
        .order_by(OrderbookSnapshot.timestamp.desc())
        .limit(n_snapshots)
    )
    snapshots = result.scalars().all()
    return list(reversed(snapshots))  # Chronological order


def compute_obi_velocity(snapshots: list[OrderbookSnapshot]) -> dict:
    """Compute Order Book Imbalance velocity from a series of snapshots.

    OBI velocity = rate of change of the orderbook imbalance over time.
    Positive velocity → increasing buy pressure (bullish)
    Negative velocity → increasing sell pressure (bearish)
    """
    if len(snapshots) < 2:
        return {"velocity": 0.0, "acceleration": 0.0, "current_obi": 0.0, "trend": "neutral"}

    obi_values = []
    depth_ratios = []

    for snap in snapshots:
        bids = snap.bids_json or []
        asks = snap.asks_json or []
        if bids and asks:
            features = compute_orderbook_features(bids, asks)
            obi_values.append(features["obi_weighted_5"])
            depth_ratios.append(features["depth_ratio"])
        else:
            obi_values.append(0.0)
            depth_ratios.append(1.0)

    if len(obi_values) < 2:
        return {"velocity": 0.0, "acceleration": 0.0, "current_obi": 0.0, "trend": "neutral"}

    # OBI velocity (first derivative) — average change per step
    deltas = [obi_values[i] - obi_values[i-1] for i in range(1, len(obi_values))]
    velocity = sum(deltas) / len(deltas)

    # OBI acceleration (second derivative)
    if len(deltas) >= 2:
        accel_deltas = [deltas[i] - deltas[i-1] for i in range(1, len(deltas))]
        acceleration = sum(accel_deltas) / len(accel_deltas)
    else:
        acceleration = 0.0

    current_obi = obi_values[-1]
    avg_depth_ratio = sum(depth_ratios) / len(depth_ratios)

    # Determine trend
    if velocity > OBI_VELOCITY_THRESHOLD and current_obi > MIN_OBI_MAGNITUDE:
        trend = "strong_bullish"
    elif velocity > OBI_VELOCITY_THRESHOLD / 2:
        trend = "bullish"
    elif velocity < -OBI_VELOCITY_THRESHOLD and current_obi < -MIN_OBI_MAGNITUDE:
        trend = "strong_bearish"
    elif velocity < -OBI_VELOCITY_THRESHOLD / 2:
        trend = "bearish"
    else:
        trend = "neutral"

    return {
        "velocity": round(velocity, 4),
        "acceleration": round(acceleration, 4),
        "current_obi": round(current_obi, 4),
        "avg_depth_ratio": round(avg_depth_ratio, 3),
        "trend": trend,
        "n_snapshots": len(snapshots),
    }


def compute_orderflow_signal(
    market_price: float,
    flow_analysis: dict,
) -> dict:
    """Compute trading signal from order flow analysis.

    Strong order flow in one direction suggests the price will move that way.
    We trade in the direction of informed flow, before the price adjusts.
    """
    trend = flow_analysis.get("trend", "neutral")
    velocity = flow_analysis.get("velocity", 0.0)
    current_obi = flow_analysis.get("current_obi", 0.0)
    depth_ratio = flow_analysis.get("avg_depth_ratio", 1.0)

    if trend == "neutral":
        return {"direction": None, "net_ev": 0.0}

    # Estimate price impact based on flow strength
    # OBI velocity of 0.3 per step historically moves price ~2-5%
    flow_strength = abs(velocity) * 10  # Scale to meaningful percentage
    depth_pressure = max(0, (abs(math.log(max(depth_ratio, 0.1))) - 0.5) * 0.1)
    estimated_move = min(0.10, flow_strength * 0.05 + depth_pressure)

    if trend in ("strong_bullish", "bullish"):
        direction = "buy_yes"
        implied_prob = min(0.95, market_price + estimated_move)
    else:
        direction = "buy_no"
        implied_prob = max(0.05, market_price - estimated_move)

    p = implied_prob
    q = market_price
    if direction == "buy_yes":
        fee = p * POLYMARKET_FEE_RATE * (1 - q) + SLIPPAGE_BUFFER
        net_ev = p * (1 - q) - (1 - p) * q - fee
    else:
        fee = (1 - p) * POLYMARKET_FEE_RATE * q + SLIPPAGE_BUFFER
        net_ev = (1 - p) * q - p * (1 - q) - fee

    if net_ev <= 0:
        return {"direction": None, "net_ev": 0.0}

    kelly = compute_kelly(direction, implied_prob, market_price, fee)

    # Confidence based on flow strength and consistency
    strength_score = min(1.0, abs(velocity) / 0.5)
    consistency_score = 1.0 if flow_analysis.get("acceleration", 0) * velocity > 0 else 0.5
    confidence = 0.3 + 0.4 * strength_score + 0.3 * consistency_score

    return {
        "direction": direction,
        "implied_prob": round(implied_prob, 4),
        "market_price": round(market_price, 4),
        "raw_edge": round(abs(implied_prob - market_price), 4),
        "net_ev": round(net_ev, 4),
        "fee_cost": round(fee, 4),
        "kelly_fraction": round(kelly, 4),
        "confidence": round(min(0.9, confidence), 3),
        "flow_analysis": flow_analysis,
    }


async def scan_orderflow_signals(
    session: AsyncSession,
) -> list[dict]:
    """Scan active markets for order flow-driven trading signals."""
    result = await session.execute(
        select(Market).where(
            Market.is_active == True,  # noqa: E711
            Market.price_yes != None,  # noqa: E711
            Market.volume_total >= MIN_VOLUME_TOTAL,
            Market.liquidity >= MIN_LIQUIDITY,
        ).order_by(Market.volume_24h.desc()).limit(200)
    )
    markets = result.scalars().all()

    if not markets:
        return []

    edges_found = []

    for market in markets:
        price = market.price_yes or 0.5
        if price < 0.05 or price > 0.95:
            continue  # Skip near-resolved

        snapshots = await _get_recent_orderbooks(session, market.id)
        if len(snapshots) < 3:
            continue

        flow = compute_obi_velocity(snapshots)
        if flow["trend"] == "neutral":
            continue

        signal = compute_orderflow_signal(price, flow)
        if not signal.get("direction") or signal["net_ev"] < MIN_NET_EDGE:
            continue

        signal.update({
            "market_id": market.id,
            "strategy": "orderflow",
            "question": market.question,
            "category": market.normalized_category or market.category,
        })
        edges_found.append(signal)

        logger.info(
            f"Orderflow: {market.question[:50]}... | "
            f"Trend: {flow['trend']} | OBI velocity: {flow['velocity']:.3f} | "
            f"EV: {signal['net_ev']:.1%}"
        )

    logger.info(f"Orderflow scan: {len(markets)} markets, {len(edges_found)} signals")
    return edges_found
