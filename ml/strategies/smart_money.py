"""Smart Money / Whale Tracking — follow proven-profitable Polymarket wallets.

Research basis:
- Polymarket on Polygon: ALL transactions are publicly visible on-chain.
- Top traders achieve 65-75% win rates vs 45-50% average (PolyTrack research).
- Analysis of 46,945 wallets: top 1% capture enormous profits while majority break even.
- "Zombie trade" bias: published win rates are inflated by unclosed positions,
  so we filter strictly for realised P&L, not open position count.

Strategy (real on-chain implementation):
1. Query Polymarket PNL subgraph for wallets with >$5K realised profit.
2. Every 5 minutes, query Activity subgraph for recent trades by these wallets.
3. When ≥2 qualified wallets take the same side in the same market within 1 hour,
   generate a directional signal weighted by whale quality scores.

This replaces the previous volume-surge heuristic with actual on-chain evidence.
"""

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Market
from ml.strategies.ensemble_edge_detector import (
    POLYMARKET_FEE_RATE, SLIPPAGE_BUFFER, compute_kelly,
)

logger = logging.getLogger(__name__)

MIN_VOLUME_TOTAL = 10_000
MIN_NET_EDGE = 0.03
MIN_WHALE_USD_SIZE = 500    # Minimum trade size to count ($500 minimum bet)
MAX_EDGE_ADJUSTMENT = 0.08  # Maximum price adjustment implied by whale consensus


async def _build_token_to_market_map(session: AsyncSession) -> dict[str, tuple[int, str]]:
    """Build a mapping from Polymarket token_id -> (market_id, 'yes'|'no').

    This lets us match on-chain token trades to markets in our DB.
    """
    result = await session.execute(
        select(Market.id, Market.token_id_yes, Market.token_id_no).where(
            Market.is_active == True,  # noqa: E711
        )
    )
    token_map: dict[str, tuple[int, str]] = {}
    for market_id, token_yes, token_no in result.all():
        if token_yes:
            token_map[str(token_yes).lower()] = (market_id, "yes")
        if token_no:
            token_map[str(token_no).lower()] = (market_id, "no")
    return token_map


async def analyze_smart_money_positioning(
    session: AsyncSession,
) -> list[dict]:
    """Generate signals from real on-chain whale activity via Polymarket subgraphs.

    Falls back to the volume-surge heuristic if the subgraph is unavailable.
    """
    try:
        from data_pipeline.collectors.polymarket_subgraph import (
            get_whale_market_signals,
            get_cached_smart_wallets,
        )

        # Build token → market mapping for on-chain trade resolution
        token_to_market = await _build_token_to_market_map(session)

        if not token_to_market:
            logger.info("Smart money: no token IDs in DB, falling back to heuristic")
            return await _heuristic_smart_money(session)

        # Get on-chain whale signals
        whale_signals = await get_whale_market_signals(token_to_market)

        if not whale_signals:
            logger.info("Smart money: no on-chain signals found, trying heuristic fallback")
            return await _heuristic_smart_money(session)

        return await _convert_whale_signals_to_edges(session, whale_signals)

    except Exception as e:
        logger.warning(f"Smart money on-chain analysis failed ({e}), using heuristic fallback")
        return await _heuristic_smart_money(session)


async def _convert_whale_signals_to_edges(
    session: AsyncSession,
    whale_signals: list[dict],
) -> list[dict]:
    """Convert raw whale signals into tradeable edges with EV/Kelly computation."""
    result = await session.execute(
        select(Market).where(
            Market.is_active == True,  # noqa: E711
            Market.price_yes.isnot(None),
            Market.volume_total >= MIN_VOLUME_TOTAL,
        )
    )
    market_map = {m.id: m for m in result.scalars().all()}

    edges_found = []

    for ws in whale_signals:
        market_id = ws["market_id"]
        market = market_map.get(market_id)
        if not market:
            continue

        price = market.price_yes or 0.5
        direction = ws["direction"]
        whale_count = ws["whale_count"]
        avg_entry = ws["avg_entry_price"]
        confidence = ws["confidence"]

        # Estimate implied probability from whale entry prices
        # Whales paid avg_entry_price; if avg > current price, they expect YES
        # Edge estimate: whales have insider-like information advantage
        fee_rate = (getattr(market, 'taker_fee_bps', 0) or 0) / 10000.0
        if direction == "buy_yes":
            implied_prob = min(0.95, price + _estimate_whale_edge(
                avg_entry, price, whale_count, direction="up"
            ))
            p, q = implied_prob, price
            fee = p * fee_rate * (1 - q) + SLIPPAGE_BUFFER
            net_ev = p * (1 - q) - (1 - p) * q - fee
        else:  # buy_no
            implied_prob = max(0.05, price - _estimate_whale_edge(
                avg_entry, price, whale_count, direction="down"
            ))
            p, q = implied_prob, price
            fee = (1 - p) * fee_rate * q + SLIPPAGE_BUFFER
            net_ev = (1 - p) * q - p * (1 - q) - fee

        if net_ev < MIN_NET_EDGE:
            continue

        kelly = compute_kelly(direction, implied_prob, price, fee)

        edges_found.append({
            "market_id": market_id,
            "strategy": "smart_money",
            "question": market.question,
            "category": market.normalized_category or market.category,
            "direction": direction,
            "implied_prob": round(implied_prob, 4),
            "market_price": round(price, 4),
            "raw_edge": round(abs(implied_prob - price), 4),
            "net_ev": round(net_ev, 4),
            "fee_cost": round(fee, 4),
            "kelly_fraction": round(kelly, 4),
            "confidence": round(confidence, 3),
            "whale_count": whale_count,
            "avg_whale_entry": round(avg_entry, 4),
            "total_whale_size_usd": ws.get("total_size_usd", 0),
            "data_source": "on_chain",
        })

        logger.info(
            f"Smart money (on-chain): {market.question[:50]}... | "
            f"{whale_count} whales → {direction} | "
            f"EV: {net_ev:.1%} | confidence: {confidence:.2f}"
        )

    logger.info(
        f"Smart money on-chain scan: {len(whale_signals)} signals → "
        f"{len(edges_found)} tradeable edges"
    )
    return edges_found


def _estimate_whale_edge(
    avg_entry: float,
    current_price: float,
    whale_count: int,
    direction: str,
) -> float:
    """Estimate edge implied by whale consensus.

    Higher whale count and larger divergence from current price → stronger edge.
    Capped at MAX_EDGE_ADJUSTMENT to prevent overestimation.
    """
    price_divergence = abs(avg_entry - current_price)
    count_multiplier = min(2.0, 1.0 + (whale_count - 2) * 0.15)  # Scales with whale count
    edge = min(MAX_EDGE_ADJUSTMENT, price_divergence * count_multiplier * 0.5)
    return max(0.01, edge)


async def _heuristic_smart_money(session: AsyncSession) -> list[dict]:
    """Fallback: volume-surge heuristic when on-chain data unavailable.

    This is the original implementation, retained as a safety net.
    """
    result = await session.execute(
        select(Market).where(
            Market.is_active == True,  # noqa: E711
            Market.price_yes.isnot(None),
            Market.volume_total >= MIN_VOLUME_TOTAL,
        ).order_by(Market.volume_24h.desc()).limit(200)
    )
    markets = result.scalars().all()

    edges_found = []

    for market in markets:
        price = market.price_yes or 0.5
        vol_24h = float(market.volume_24h or 0)
        vol_total = float(market.volume_total or 0)

        if vol_24h < 5000:
            continue

        expected_daily = vol_total / max(30, 1)
        volume_surge = vol_24h / max(expected_daily, 1)

        if volume_surge < 2.0:
            continue
        if 0.20 <= price <= 0.80 and volume_surge < 3.0:
            continue

        if price > 0.70 and volume_surge > 2.0:
            direction = "buy_yes"
            implied_prob = min(0.95, price + min(0.06, (volume_surge - 1) * 0.015))
        elif price < 0.30 and volume_surge > 2.0:
            direction = "buy_no"
            implied_prob = max(0.05, price - min(0.06, (volume_surge - 1) * 0.015))
        else:
            continue

        p, q = implied_prob, price
        mkt_fee_rate = (getattr(market, 'taker_fee_bps', 0) or 0) / 10000.0
        if direction == "buy_yes":
            fee = p * mkt_fee_rate * (1 - q) + SLIPPAGE_BUFFER
            net_ev = p * (1 - q) - (1 - p) * q - fee
        else:
            fee = (1 - p) * mkt_fee_rate * q + SLIPPAGE_BUFFER
            net_ev = (1 - p) * q - p * (1 - q) - fee

        if net_ev < MIN_NET_EDGE:
            continue

        kelly = compute_kelly(direction, implied_prob, price, fee)
        surge_confidence = min(0.4, (volume_surge - 1.5) / 12)

        edges_found.append({
            "market_id": market.id,
            "strategy": "smart_money",
            "question": market.question,
            "category": market.normalized_category or market.category,
            "direction": direction,
            "implied_prob": round(implied_prob, 4),
            "market_price": round(price, 4),
            "raw_edge": round(abs(implied_prob - price), 4),
            "net_ev": round(net_ev, 4),
            "fee_cost": round(fee, 4),
            "kelly_fraction": round(kelly, 4),
            "confidence": round(0.25 + surge_confidence, 3),  # Lower confidence for heuristic
            "volume_surge": round(volume_surge, 2),
            "data_source": "heuristic",
        })

    logger.info(f"Smart money heuristic: {len(markets)} markets, {len(edges_found)} signals")
    return edges_found
