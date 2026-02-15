"""Score historical edge signals against actual market resolutions.

Run periodically (or on demand) to fill in was_correct and actual_pnl
on EnsembleEdgeSignal rows for resolved markets.
"""

import logging
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Market, EnsembleEdgeSignal, EloEdgeSignal

logger = logging.getLogger(__name__)


async def score_resolved_signals(session: AsyncSession) -> dict:
    """Score all unscored edge signals for resolved markets.

    Updates was_correct and actual_pnl on each signal row.
    Returns count of newly scored signals.
    """
    scored_ensemble = 0
    scored_elo = 0

    # --- Ensemble Edge Signals ---
    result = await session.execute(
        select(EnsembleEdgeSignal, Market)
        .join(Market, EnsembleEdgeSignal.market_id == Market.id)
        .where(
            Market.is_resolved == True,  # noqa
            Market.resolution_value != None,  # noqa
            EnsembleEdgeSignal.was_correct == None,  # noqa — not yet scored
        )
    )
    for signal, market in result.all():
        resolution = market.resolution_value
        mp = signal.market_price

        # P&L: Polymarket charges 2% on net winnings ONLY when you win
        if signal.direction == "buy_yes":
            signal.was_correct = resolution == 1.0
            if signal.was_correct:
                winnings = 1.0 - mp
                signal.actual_pnl = winnings - 0.02 * winnings  # = winnings * 0.98
            else:
                signal.actual_pnl = -mp  # lost cost basis, no fee
        elif signal.direction == "buy_no":
            signal.was_correct = resolution == 0.0
            if signal.was_correct:
                winnings = mp  # = 1.0 - (1 - mp)
                signal.actual_pnl = winnings - 0.02 * winnings
            else:
                signal.actual_pnl = -(1.0 - mp)  # lost cost basis, no fee
        else:
            signal.was_correct = False
            signal.actual_pnl = 0.0

        scored_ensemble += 1

    # --- Elo Edge Signals ---
    elo_result = await session.execute(
        select(EloEdgeSignal, Market)
        .join(Market, EloEdgeSignal.market_id == Market.id)
        .where(
            Market.is_resolved == True,  # noqa
            Market.resolution_value != None,  # noqa
            EloEdgeSignal.was_correct == None,  # noqa
        )
    )
    for signal, market in elo_result.all():
        resolution = market.resolution_value
        mp = signal.market_price_yes
        # Determine direction from Elo probability vs market price
        elo_prob_yes = signal.elo_prob_a if signal.yes_side_player == signal.player_a else (1 - signal.elo_prob_a)
        if elo_prob_yes > mp:
            signal.was_correct = resolution == 1.0
            if signal.was_correct:
                winnings = 1.0 - mp
                signal.actual_pnl = winnings * 0.98  # 2% fee on winnings
            else:
                signal.actual_pnl = -mp
        else:
            signal.was_correct = resolution == 0.0
            if signal.was_correct:
                winnings = mp
                signal.actual_pnl = winnings * 0.98
            else:
                signal.actual_pnl = -(1.0 - mp)

        scored_elo += 1

    if scored_ensemble or scored_elo:
        await session.commit()
        logger.info(f"Scored signals: {scored_ensemble} ensemble, {scored_elo} Elo")

    return {
        "scored_ensemble": scored_ensemble,
        "scored_elo": scored_elo,
    }
