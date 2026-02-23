"""Elo-based edge detector — finds mispriced sports markets.

Same pattern as arbitrage/engine.py: scans active sports markets,
compares Glicko-2 win probabilities against market prices, and persists
edge signals when the gap exceeds fees + slippage.

Fee-aware threshold: edge = |elo_prob - market_price| - (taker_fee_bps/10000) - slippage
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Market, EloEdgeSignal
from ml.models.elo_sports import Glicko2Engine
from ml.strategies.ensemble_edge_detector import compute_kelly, POLYMARKET_FEE_RATE
from data_pipeline.transformers.sports_matchup_parser import (
    parse_matchup,
    fuzzy_match_player,
)

logger = logging.getLogger(__name__)

# Default configuration
MIN_NET_EDGE = 0.03  # 3% minimum net edge after fees
MIN_CONFIDENCE = 0.5  # Minimum Elo confidence
MAX_KELLY = 0.02  # Maximum Kelly fraction (2%)
SLIPPAGE_BUFFER = 0.01  # 1% default slippage estimate
SIGNAL_EXPIRY_MINUTES = 60  # Signals expire after 1 hour
MAX_RD_THRESHOLD = 300  # Skip players with rating deviation > 300 (too uncertain)


async def scan_for_edges(
    session: AsyncSession,
    engine: Glicko2Engine,
    min_net_edge: float = MIN_NET_EDGE,
    min_confidence: float = MIN_CONFIDENCE,
) -> list[dict]:
    """Scan active sports markets for Elo-based edge signals.

    Args:
        session: Active database session
        engine: Loaded Glicko-2 engine with player ratings
        min_net_edge: Minimum net edge after fees (default 3%)
        min_confidence: Minimum Elo confidence (default 0.5)

    Returns:
        List of detected edge signal dicts
    """
    # Get active sports markets with "vs" in the question
    result = await session.execute(
        select(Market).where(
            Market.is_active == True,  # noqa: E711
            Market.normalized_category == "sports",
            Market.price_yes != None,  # noqa: E711
        ).limit(500)
    )
    markets = result.scalars().all()

    if not markets:
        return []

    # Get list of known player names from engine
    known_players = list(engine.ratings.keys())
    if not known_players:
        logger.warning("No player ratings loaded. Run build_elo_ratings.py first.")
        return []

    # Expire old signals
    await _expire_old_signals(session)

    edges_found = []
    markets_parsed = 0
    markets_skipped = 0

    for market in markets:
        # Try to parse as tennis matchup
        matchup = parse_matchup(
            question=market.question or "",
            category=market.normalized_category or market.category or "",
            market_id=market.id,
        )

        if matchup is None:
            markets_skipped += 1
            continue

        markets_parsed += 1

        # Match player names to Elo database
        matched_a = fuzzy_match_player(matchup.player_a, known_players)
        matched_b = fuzzy_match_player(matchup.player_b, known_players)

        if not matched_a or not matched_b:
            continue

        # Check RD threshold (skip if too uncertain)
        rating_a = engine.ratings.get(matched_a, {}).get("overall")
        rating_b = engine.ratings.get(matched_b, {}).get("overall")
        if not rating_a or not rating_b:
            continue
        if rating_a.phi > 300 or rating_b.phi > 300:
            continue

        # Compute Elo win probability
        surface = matchup.surface if matchup.surface != "unknown" else "hard"
        elo_prob_a, confidence = engine.win_probability(matched_a, matched_b, surface)

        if confidence < min_confidence:
            continue

        # Determine market price for the Elo-favored player
        market_price = market.price_yes or 0.5

        # Map Elo probability to same direction as market price
        if matchup.yes_side_player == matchup.player_a:
            elo_prob_yes = elo_prob_a
        else:
            elo_prob_yes = 1.0 - elo_prob_a

        # Calculate edge (fee-aware)
        # Polymarket charges 2% on winnings only when you win
        raw_edge = abs(elo_prob_yes - market_price)
        direction = "buy_yes" if elo_prob_yes > market_price else "buy_no"
        extra_fee = (market.taker_fee_bps or 0) / 10000.0

        if direction == "buy_yes":
            fee_cost = elo_prob_yes * POLYMARKET_FEE_RATE * (1 - market_price) + SLIPPAGE_BUFFER + extra_fee * (1 - market_price)
            net_edge = elo_prob_yes * (1 - market_price) - (1 - elo_prob_yes) * market_price - fee_cost
        else:
            fee_cost = (1 - elo_prob_yes) * POLYMARKET_FEE_RATE * market_price + SLIPPAGE_BUFFER + extra_fee * market_price
            net_edge = (1 - elo_prob_yes) * market_price - elo_prob_yes * (1 - market_price) - fee_cost

        if net_edge < min_net_edge:
            continue

        # Kelly criterion sizing — reuse canonical implementation
        kelly = compute_kelly(direction, elo_prob_yes, market_price, fee_cost)

        # Check if we already have an active signal for this market
        existing = await session.execute(
            select(EloEdgeSignal).where(
                EloEdgeSignal.market_id == market.id,
                EloEdgeSignal.expired_at == None,  # noqa: E711
            )
        )
        if existing.scalar_one_or_none():
            continue  # Already have an active signal

        # Create edge signal
        signal = EloEdgeSignal(
            market_id=market.id,
            sport="tennis",
            detected_at=datetime.utcnow(),
            player_a=matched_a,
            player_b=matched_b,
            surface=surface,
            elo_prob_a=elo_prob_a,
            elo_confidence=confidence,
            market_price_yes=market_price,
            yes_side_player=matchup.yes_side_player,
            raw_edge=raw_edge,
            fee_cost=fee_cost,
            net_edge=net_edge,
            kelly_fraction=kelly,
        )
        session.add(signal)

        edge_dict = {
            "market_id": market.id,
            "question": market.question,
            "player_a": matched_a,
            "player_b": matched_b,
            "surface": surface,
            "elo_prob_a": round(elo_prob_a, 4),
            "confidence": round(confidence, 4),
            "market_price": round(market_price, 4),
            "raw_edge": round(raw_edge, 4),
            "net_edge": round(net_edge, 4),
            "kelly_fraction": round(kelly, 4),
            "direction": "buy_yes" if elo_prob_yes > market_price else "buy_no",
        }
        edges_found.append(edge_dict)

        logger.info(
            f"Edge found: {matched_a} vs {matched_b} | "
            f"Elo: {elo_prob_a:.1%} | Market: {market_price:.1%} | "
            f"Net edge: {net_edge:.1%} | Kelly: {kelly:.2%}"
        )

    await session.commit()

    logger.info(
        f"Elo edge scan: {markets_parsed} parsed, {markets_skipped} skipped, "
        f"{len(edges_found)} edges found"
    )
    return edges_found


async def _expire_old_signals(session: AsyncSession) -> int:
    """Expire signals older than SIGNAL_EXPIRY_MINUTES."""
    cutoff = datetime.utcnow() - timedelta(minutes=SIGNAL_EXPIRY_MINUTES)
    result = await session.execute(
        select(EloEdgeSignal).where(
            EloEdgeSignal.expired_at == None,  # noqa: E711
            EloEdgeSignal.detected_at < cutoff,
        )
    )
    expired = result.scalars().all()
    for signal in expired:
        signal.expired_at = datetime.utcnow()

    if expired:
        await session.commit()
        logger.debug(f"Expired {len(expired)} old edge signals")
    return len(expired)


async def get_active_edges(session: AsyncSession) -> list[dict]:
    """Get all currently active (non-expired) edge signals."""
    result = await session.execute(
        select(EloEdgeSignal).where(
            EloEdgeSignal.expired_at == None,  # noqa: E711
        ).order_by(EloEdgeSignal.net_edge.desc())
    )
    signals = result.scalars().all()

    return [
        {
            "id": s.id,
            "market_id": s.market_id,
            "sport": s.sport,
            "player_a": s.player_a,
            "player_b": s.player_b,
            "surface": s.surface,
            "elo_prob_a": s.elo_prob_a,
            "elo_confidence": s.elo_confidence,
            "market_price_yes": s.market_price_yes,
            "yes_side_player": s.yes_side_player,
            "raw_edge": s.raw_edge,
            "net_edge": s.net_edge,
            "fee_cost": s.fee_cost,
            "kelly_fraction": s.kelly_fraction,
            "detected_at": s.detected_at.isoformat(),
        }
        for s in signals
    ]


async def scan_ufc_edges(
    session: AsyncSession,
    engine: Glicko2Engine,
    min_net_edge: float = MIN_NET_EDGE,
    min_confidence: float = MIN_CONFIDENCE,
) -> list[dict]:
    """Scan active UFC/MMA markets for Elo-based edge signals."""
    from data_pipeline.collectors.ufc_results import parse_ufc_matchup

    result = await session.execute(
        select(Market).where(
            Market.is_active == True,  # noqa: E711
            Market.price_yes != None,  # noqa: E711
        ).limit(500)
    )
    markets = result.scalars().all()

    if not markets:
        return []

    known_fighters = list(engine.ratings.keys())
    if not known_fighters:
        return []

    edges_found = []

    for market in markets:
        matchup = parse_ufc_matchup(
            question=market.question or "",
            category=market.normalized_category or market.category or "",
            market_id=market.id,
        )

        if matchup is None:
            continue

        matched_a = fuzzy_match_player(matchup["fighter_a"], known_fighters, threshold=0.80)
        matched_b = fuzzy_match_player(matchup["fighter_b"], known_fighters, threshold=0.80)

        if not matched_a or not matched_b:
            continue

        rating_a = engine.ratings.get(matched_a, {}).get("cage") or engine.ratings.get(matched_a, {}).get("overall")
        rating_b = engine.ratings.get(matched_b, {}).get("cage") or engine.ratings.get(matched_b, {}).get("overall")
        if not rating_a or not rating_b:
            continue
        if rating_a.phi > MAX_RD_THRESHOLD or rating_b.phi > MAX_RD_THRESHOLD:
            continue
        if rating_a.match_count < 5 or rating_b.match_count < 5:
            continue

        elo_prob_a, confidence = engine.win_probability(matched_a, matched_b, "cage")

        if confidence < min_confidence:
            continue

        market_price = market.price_yes or 0.5

        if matchup["yes_side_fighter"] == matchup["fighter_a"]:
            elo_prob_yes = elo_prob_a
        else:
            elo_prob_yes = 1.0 - elo_prob_a

        raw_edge = abs(elo_prob_yes - market_price)
        direction = "buy_yes" if elo_prob_yes > market_price else "buy_no"
        extra_fee = (market.taker_fee_bps or 0) / 10000.0

        if direction == "buy_yes":
            fee_cost = elo_prob_yes * POLYMARKET_FEE_RATE * (1 - market_price) + SLIPPAGE_BUFFER + extra_fee * (1 - market_price)
            net_edge = elo_prob_yes * (1 - market_price) - (1 - elo_prob_yes) * market_price - fee_cost
        else:
            fee_cost = (1 - elo_prob_yes) * POLYMARKET_FEE_RATE * market_price + SLIPPAGE_BUFFER + extra_fee * market_price
            net_edge = (1 - elo_prob_yes) * market_price - elo_prob_yes * (1 - market_price) - fee_cost

        if net_edge < min_net_edge:
            continue

        vol_total = float(market.volume_total or 0)
        if vol_total < 10000:
            continue

        kelly = compute_kelly(direction, elo_prob_yes, market_price, fee_cost)

        existing = await session.execute(
            select(EloEdgeSignal).where(
                EloEdgeSignal.market_id == market.id,
                EloEdgeSignal.expired_at == None,  # noqa: E711
            )
        )
        if existing.scalar_one_or_none():
            continue

        signal = EloEdgeSignal(
            market_id=market.id,
            sport="mma",
            detected_at=datetime.utcnow(),
            player_a=matched_a,
            player_b=matched_b,
            surface="cage",
            elo_prob_a=elo_prob_a,
            elo_confidence=confidence,
            market_price_yes=market_price,
            yes_side_player=matchup["yes_side_fighter"],
            raw_edge=raw_edge,
            fee_cost=fee_cost,
            net_edge=net_edge,
            kelly_fraction=kelly,
        )
        session.add(signal)
        edges_found.append({
            "market_id": market.id,
            "question": market.question,
            "fighter_a": matched_a,
            "fighter_b": matched_b,
            "elo_prob_a": round(elo_prob_a, 4),
            "confidence": round(confidence, 4),
            "market_price": round(market_price, 4),
            "net_edge": round(net_edge, 4),
            "kelly_fraction": round(kelly, 4),
            "direction": direction,
        })

        logger.info(
            f"UFC Edge: {matched_a} vs {matched_b} | "
            f"Elo: {elo_prob_a:.1%} | Market: {market_price:.1%} | "
            f"Net edge: {net_edge:.1%} | Kelly: {kelly:.2%}"
        )

    if edges_found:
        await session.commit()
    logger.info(f"UFC edge scan: {len(edges_found)} edges found")
    return edges_found
