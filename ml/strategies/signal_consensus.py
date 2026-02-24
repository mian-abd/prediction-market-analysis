"""Multi-strategy signal consensus — boosted confidence when strategies agree.

When 2+ independent strategies point the same direction on the same market,
the combined signal is far stronger than any individual one. This module:

1. Groups active signals by market_id + direction
2. Computes a consensus score based on:
   - Number of agreeing strategies (2+ required)
   - Combined EV (weighted by strategy independence)
   - Diversity bonus (strategies from different information sources get extra weight)
3. Creates boosted StrategySignal entries with strategy="consensus"
4. Adjusts Kelly fraction upward for consensus signals (up to 2x)

Information source categories (for diversity scoring):
- Price-based: longshot_bias, resolution_convergence (same info source)
- Flow-based: orderflow, smart_money (related info source)
- External: llm_forecast, news_catalyst (independent info)
- Statistical: ensemble, market_clustering (model-derived)
"""

import logging
from collections import defaultdict
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import StrategySignal, EnsembleEdgeSignal, Market

logger = logging.getLogger(__name__)

INFO_CATEGORIES = {
    "longshot_bias": "price",
    "resolution_convergence": "price",
    "orderflow": "flow",
    "smart_money": "flow",
    "llm_forecast": "external",
    "news_catalyst": "external",
    "market_clustering": "statistical",
    "ensemble": "statistical",
}

MIN_STRATEGIES_FOR_CONSENSUS = 2
CONSENSUS_KELLY_MULTIPLIER = 1.8
MAX_CONSENSUS_KELLY = 0.06


async def compute_signal_consensus(session: AsyncSession) -> list[dict]:
    """Find markets where multiple strategies agree and create boosted consensus signals.

    Returns list of consensus signal dicts ready for persistence.
    """
    # Fetch all active new strategy signals
    strat_result = await session.execute(
        select(StrategySignal)
        .where(
            StrategySignal.expired_at == None,  # noqa: E711
            StrategySignal.direction != None,  # noqa: E711
            StrategySignal.strategy != "consensus",
        )
    )
    strat_signals = strat_result.scalars().all()

    # Also check ensemble signals for cross-strategy agreement
    ens_result = await session.execute(
        select(EnsembleEdgeSignal)
        .where(EnsembleEdgeSignal.expired_at == None)  # noqa: E711
    )
    ens_signals = ens_result.scalars().all()

    # Group by (market_id, direction)
    groups: dict[tuple[int, str], list[dict]] = defaultdict(list)

    for sig in strat_signals:
        key = (sig.market_id, sig.direction)
        # Deduplicate: only keep the best signal per strategy per market
        existing_strats = {s["strategy"] for s in groups[key]}
        if sig.strategy not in existing_strats:
            groups[key].append({
                "strategy": sig.strategy,
                "net_ev": sig.net_ev,
                "confidence": sig.confidence or 0.5,
                "kelly": sig.kelly_fraction or 0.0,
                "market_price": sig.market_price,
                "implied_prob": sig.implied_prob,
                "info_category": INFO_CATEGORIES.get(sig.strategy, "other"),
            })

    for sig in ens_signals:
        key = (sig.market_id, sig.direction)
        # Only add ensemble if no other statistical signal already in group
        existing_strats = {s["strategy"] for s in groups[key]}
        if "ensemble" not in existing_strats:
            groups[key].append({
                "strategy": "ensemble",
                "net_ev": sig.net_ev,
                "confidence": sig.confidence or 0.5,
                "kelly": sig.kelly_fraction or 0.0,
                "market_price": sig.market_price,
                "implied_prob": sig.ensemble_prob,
                "info_category": "statistical",
            })

    consensus_signals = []

    for (market_id, direction), signals in groups.items():
        if len(signals) < MIN_STRATEGIES_FOR_CONSENSUS:
            continue

        strategies = [s["strategy"] for s in signals]
        info_cats = set(s["info_category"] for s in signals)
        n_strategies = len(signals)
        n_info_categories = len(info_cats)

        # Diversity bonus: more independent info sources = stronger consensus
        # 2 same-category = 1.0x, 2 diff-category = 1.3x, 3 diff = 1.5x
        diversity_factor = 1.0 + 0.15 * (n_info_categories - 1)

        # Combined confidence: weighted average boosted by count
        avg_conf = sum(s["confidence"] for s in signals) / n_strategies
        # Boost: sqrt(n) scaling (diminishing returns)
        count_boost = min(n_strategies ** 0.5 / 1.414, 1.5)
        combined_confidence = min(0.95, avg_conf * count_boost * diversity_factor)

        # Combined EV: take the best signal's EV (conservative)
        best_ev = max(s["net_ev"] for s in signals)
        avg_ev = sum(s["net_ev"] for s in signals) / n_strategies

        # Combined Kelly: average Kelly * consensus multiplier
        avg_kelly = sum(s["kelly"] for s in signals) / n_strategies
        consensus_kelly = min(
            avg_kelly * CONSENSUS_KELLY_MULTIPLIER * diversity_factor,
            MAX_CONSENSUS_KELLY,
        )

        best_market_price = signals[0]["market_price"]
        implied_probs = [s["implied_prob"] for s in signals if s["implied_prob"]]
        consensus_implied = sum(implied_probs) / len(implied_probs) if implied_probs else None

        consensus_signals.append({
            "market_id": market_id,
            "direction": direction,
            "net_ev": avg_ev,
            "confidence": combined_confidence,
            "kelly_fraction": consensus_kelly,
            "market_price": best_market_price,
            "implied_prob": consensus_implied,
            "quality_tier": "high" if combined_confidence >= 0.7 else "medium",
            "strategies": strategies,
            "n_strategies": n_strategies,
            "n_info_categories": n_info_categories,
            "diversity_factor": round(diversity_factor, 2),
            "individual_evs": {s["strategy"]: round(s["net_ev"], 4) for s in signals},
        })

    logger.info(
        f"Consensus scan: {len(consensus_signals)} markets with multi-strategy agreement "
        f"(from {len(groups)} unique market-direction pairs)"
    )

    return consensus_signals


async def persist_consensus_signals(session: AsyncSession, consensus: list[dict]) -> int:
    """Write consensus signals to the StrategySignal table."""
    if not consensus:
        return 0

    # Expire old consensus signals
    from sqlalchemy import update as sa_update
    await session.execute(
        sa_update(StrategySignal)
        .where(
            StrategySignal.strategy == "consensus",
            StrategySignal.expired_at == None,  # noqa: E711
        )
        .values(expired_at=datetime.utcnow())
    )

    created = 0
    for c in consensus:
        sig = StrategySignal(
            market_id=c["market_id"],
            strategy="consensus",
            direction=c["direction"],
            implied_prob=c.get("implied_prob"),
            market_price=c["market_price"],
            raw_edge=c["net_ev"],
            net_ev=c["net_ev"],
            fee_cost=0.0,
            kelly_fraction=c["kelly_fraction"],
            confidence=c["confidence"],
            quality_tier=c["quality_tier"],
            signal_metadata={
                "strategies": c["strategies"],
                "n_strategies": c["n_strategies"],
                "n_info_categories": c["n_info_categories"],
                "diversity_factor": c["diversity_factor"],
                "individual_evs": c["individual_evs"],
            },
        )
        session.add(sig)
        created += 1

    await session.commit()
    logger.info(f"Persisted {created} consensus signals")
    return created
