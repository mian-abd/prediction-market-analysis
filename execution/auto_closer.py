"""Auto-close lifecycle for auto-trading positions.

Only touches positions where portfolio_type == "auto" AND exit_time IS NULL.
Manual/copy positions are never modified.

Close conditions (checked in order):
1. Market resolved -> close at resolution value (1.0 or 0.0)
1b. Effectively resolved -> price near 0/1 (lock in profits immediately)
1c. Market deactivated -> close at last price
2. Edge invalidation -> price moved >10% away from entry AND losing (prediction markets)
2a. Stale unprofitable -> >24h old and not profitable
2b. Trailing stop -> position had >8% gain but gave back >50% of peak
2c. Time-decay exit -> within 1-4 hours of market end date
3. Stop loss hit -> close at current market price (tighter: 12/8/5% by price tier)
4. Signal expired -> close at current market price (if close_on_signal_expiry=True)
"""

import logging
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import (
    PortfolioPosition, Market, AutoTradingConfig,
    EnsembleEdgeSignal, EloEdgeSignal, StrategySignal,
)

logger = logging.getLogger(__name__)


async def auto_close_positions(session: AsyncSession) -> list[int]:
    """Check all open auto positions for close conditions. Returns closed IDs."""
    # Load configs for stop-loss and signal-expiry settings
    configs = {}
    config_rows = (await session.execute(select(AutoTradingConfig))).scalars().all()
    for c in config_rows:
        configs[c.strategy] = c

    # Get all open auto positions
    result = await session.execute(
        select(PortfolioPosition)
        .where(
            PortfolioPosition.portfolio_type == "auto",
            PortfolioPosition.exit_time == None,  # noqa: E711
        )
    )
    positions = result.scalars().all()

    if not positions:
        return []

    closed_ids = []

    for pos in positions:
        market = await session.get(Market, pos.market_id)
        if not market:
            continue

        # Determine which config applies
        if pos.strategy == "auto_ensemble":
            strategy_key = "ensemble"
        elif pos.strategy == "auto_elo":
            strategy_key = "elo"
        else:
            strategy_key = "new_strategies"
        config = configs.get(strategy_key) or configs.get("ensemble")

        exit_price = None
        close_reason = None

        # 1. Market resolved — highest priority
        if market.is_resolved and market.resolution_value is not None:
            exit_price = market.resolution_value
            close_reason = "market_resolved"

        # 1b. Effectively resolved — price near 0 or 1 means market is decided
        # Lock in profits/losses rather than waiting for official resolution
        if exit_price is None and market.price_yes is not None:
            if market.price_yes <= 0.02 or market.price_yes >= 0.98:
                exit_price = market.price_yes
                close_reason = "effectively_resolved"

        # 1c. Market deactivated (expired or dead) — close at last known price
        if exit_price is None and not market.is_active and market.price_yes is not None:
            exit_price = market.price_yes
            close_reason = "market_deactivated"

        # 2. Edge invalidation — CRITICAL for prediction markets
        # If price moved significantly away from entry AND we're losing, edge is gone
        # This is NOT volatility (stocks) — it's new information invalidating our thesis
        if exit_price is None and market.price_yes is not None:
            price_deviation = abs(market.price_yes - pos.entry_price)

            if pos.side == "yes":
                unrealized = (market.price_yes - pos.entry_price) * pos.quantity
            else:
                unrealized = (pos.entry_price - market.price_yes) * pos.quantity

            # If price moved >10 percentage points AND we're losing, exit
            if price_deviation > 0.10 and unrealized < 0:
                exit_price = market.price_yes
                close_reason = "edge_invalidation"

            # Time-based exit: if position is >24h old and not profitable, cut it
            if exit_price is None and pos.entry_time:
                age_hours = (datetime.utcnow() - pos.entry_time).total_seconds() / 3600
                if age_hours > 24 and unrealized <= 0:
                    exit_price = market.price_yes
                    close_reason = "stale_unprofitable"

            # Trailing stop: protect gains once we're in profit
            # If position has >8% gain, lock in at least 50% of gains
            if exit_price is None and unrealized > 0:
                cost_per_share = pos.entry_price if pos.side == "yes" else (1.0 - pos.entry_price)
                position_cost = cost_per_share * pos.quantity
                pnl_pct = unrealized / position_cost if position_cost > 0 else 0
                if pnl_pct > 0.08:
                    locked_pnl = unrealized * 0.50
                    if unrealized < locked_pnl:
                        exit_price = market.price_yes
                        close_reason = "trailing_stop"

        # 2b. Time-decay exit: close positions approaching market end date
        # Markets in their final hours are binary — no time for recovery
        if exit_price is None and market.end_date and market.price_yes is not None:
            hours_until_close = (market.end_date - datetime.utcnow()).total_seconds() / 3600
            if hours_until_close < 1 and hours_until_close > 0:
                exit_price = market.price_yes
                close_reason = "time_decay_expiry"
            elif hours_until_close < 4:
                if pos.side == "yes":
                    unrealized = (market.price_yes - pos.entry_price) * pos.quantity
                else:
                    unrealized = (pos.entry_price - market.price_yes) * pos.quantity
                if unrealized <= 0:
                    exit_price = market.price_yes
                    close_reason = "time_decay_losing"

        # 3. Stop loss — check AFTER edge invalidation to prevent unbounded losses
        # Scale stop-loss by cost basis: cheap contracts need more room (they're volatile)
        if exit_price is None and config and config.stop_loss_pct > 0:
            if market.price_yes is not None:
                if pos.side == "yes":
                    unrealized = (market.price_yes - pos.entry_price) * pos.quantity
                    position_cost = pos.entry_price * pos.quantity
                else:
                    unrealized = (pos.entry_price - market.price_yes) * pos.quantity
                    position_cost = (1.0 - pos.entry_price) * pos.quantity

                # Tighter stop-losses to cut losses faster
                cost_per_share = pos.entry_price if pos.side == "yes" else (1.0 - pos.entry_price)
                if cost_per_share < 0.30:
                    effective_stop = 0.12  # 12% for cheap contracts
                elif cost_per_share < 0.50:
                    effective_stop = 0.08  # 8% for mid-range
                else:
                    effective_stop = min(config.stop_loss_pct, 0.05)  # 5% max for expensive

                if position_cost > 0 and unrealized / position_cost < -effective_stop:
                    exit_price = market.price_yes
                    close_reason = "stop_loss"
                # Momentum-based close: prediction markets trending toward extremes (0/1) are unlikely to recover
                elif (position_cost > 0 and
                      unrealized / position_cost < -0.05 and
                      (market.price_yes <= 0.05 or market.price_yes >= 0.95)):
                    exit_price = market.price_yes
                    close_reason = "momentum_downtrend"

        # 4. Signal expired (only if config says so)
        if exit_price is None and config and config.close_on_signal_expiry:
            has_active_signal = await _has_active_signal(session, pos.market_id, pos.strategy)
            if not has_active_signal:
                if market.price_yes is not None:
                    # Cap signal-expiry close at stop-loss level to prevent
                    # signal expiry from realizing worse losses than stop-loss
                    if config.stop_loss_pct > 0:
                        if pos.side == "yes":
                            unrealized = (market.price_yes - pos.entry_price) * pos.quantity
                            position_cost = pos.entry_price * pos.quantity
                        else:
                            unrealized = (pos.entry_price - market.price_yes) * pos.quantity
                            position_cost = (1.0 - pos.entry_price) * pos.quantity
                        loss_pct = abs(unrealized / position_cost) if position_cost > 0 else 0
                        if unrealized < 0 and loss_pct > config.stop_loss_pct:
                            close_reason = "signal_expired_stop_loss_cap"
                        else:
                            close_reason = "signal_expired"
                    else:
                        close_reason = "signal_expired"
                    exit_price = market.price_yes

        # Close the position
        if exit_price is not None:
            pos.exit_price = exit_price
            pos.exit_time = datetime.utcnow()
            if pos.side == "yes":
                gross_pnl = (exit_price - pos.entry_price) * pos.quantity
            else:
                gross_pnl = (pos.entry_price - exit_price) * pos.quantity
            # Polymarket charges 2% fee on net winnings only (0% on losses)
            fee = 0.02 * max(gross_pnl, 0.0)
            pos.realized_pnl = gross_pnl - fee

            closed_ids.append(pos.id)
            logger.info(
                f"Auto-closed position {pos.id} ({close_reason}): "
                f"market {pos.market_id} | {pos.side} | P&L=${pos.realized_pnl:.2f}"
            )

    if closed_ids:
        await session.commit()
        logger.info(f"Auto-closer: {len(closed_ids)} positions closed")

    return closed_ids


async def _has_active_signal(session: AsyncSession, market_id: int, strategy: str) -> bool:
    """Check if there's still an active (non-expired) signal for this market."""
    if strategy == "auto_ensemble":
        result = await session.execute(
            select(EnsembleEdgeSignal.id)
            .where(
                EnsembleEdgeSignal.market_id == market_id,
                EnsembleEdgeSignal.expired_at == None,  # noqa: E711
            )
            .limit(1)
        )
    elif strategy == "auto_elo":
        result = await session.execute(
            select(EloEdgeSignal.id)
            .where(
                EloEdgeSignal.market_id == market_id,
                EloEdgeSignal.expired_at == None,  # noqa: E711
            )
            .limit(1)
        )
    elif strategy.startswith("auto_"):
        # New strategy positions (auto_longshot_bias, auto_llm_forecast, etc.)
        strategy_name = strategy.removeprefix("auto_")
        result = await session.execute(
            select(StrategySignal.id)
            .where(
                StrategySignal.market_id == market_id,
                StrategySignal.strategy == strategy_name,
                StrategySignal.expired_at == None,  # noqa: E711
            )
            .limit(1)
        )
    else:
        return True  # Unknown strategy — don't close

    return result.scalar_one_or_none() is not None
