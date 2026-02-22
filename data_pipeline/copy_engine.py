"""Copy trading engine - creates positions from ensemble ML signals.

Copy trading follows the ensemble model's best edge signals, NOT random
markets. When a user follows a "trader" with auto_copy enabled:
1. seed_copy_positions() creates positions from the top current edge signals
2. sync_copy_positions() runs periodically via scheduler to add new signals
   and close stale/invalidated positions

The "trader" acts as a portfolio allocation wrapper — the actual alpha comes
from the ensemble model.
"""

import logging
import math
from datetime import datetime

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import (
    PortfolioPosition, FollowedTrader, CopyTrade, TraderActivity,
    TraderProfile, Market, EnsembleEdgeSignal,
)

logger = logging.getLogger(__name__)

# Copy positions are based on ensemble signals — same edge source as auto-trading
# but with independent allocation budgets and the "manual" portfolio type.
MAX_COPY_POSITIONS_PER_FOLLOW = 5
EDGE_DECAY_HALF_LIFE_HOURS = 2.0


async def seed_copy_positions(
    follow: FollowedTrader,
    session: AsyncSession,
) -> list[int]:
    """Create initial copy positions from top ensemble edge signals.

    Uses the ML model's highest-confidence signals to create positions,
    giving every copy trade the ensemble's proven statistical edge.
    Only creates positions for signals that pass quality gates.
    """
    if not follow.auto_copy:
        return []

    # Get top ensemble edge signals: non-expired, medium+ quality, positive EV
    signals_result = await session.execute(
        select(EnsembleEdgeSignal, Market)
        .join(Market, EnsembleEdgeSignal.market_id == Market.id)
        .where(
            EnsembleEdgeSignal.expired_at == None,  # noqa
            EnsembleEdgeSignal.quality_tier.in_(["medium", "high"]),
            EnsembleEdgeSignal.net_ev > 0.03,  # 3% minimum edge
            EnsembleEdgeSignal.confidence >= 0.5,
            Market.is_active == True,  # noqa
            Market.price_yes > 0.05,
            Market.price_yes < 0.95,
        )
        .order_by(EnsembleEdgeSignal.net_ev.desc())
        .limit(MAX_COPY_POSITIONS_PER_FOLLOW * 2)  # Fetch extra for filtering
    )
    signal_pairs = signals_result.all()

    if not signal_pairs:
        logger.info(f"No ensemble signals available for copy seeding (follower {follow.follower_id})")
        return []

    # Check which markets already have open copy positions for this follower
    existing_result = await session.execute(
        select(PortfolioPosition.market_id)
        .where(
            PortfolioPosition.user_id == follow.follower_id,
            PortfolioPosition.strategy == "copy_trade",
            PortfolioPosition.exit_time == None,  # noqa
        )
    )
    existing_markets = {row[0] for row in existing_result.all()}

    now = datetime.utcnow()
    created_ids = []
    n_signals = min(MAX_COPY_POSITIONS_PER_FOLLOW, len(signal_pairs))
    allocation_per_signal = follow.allocation_amount / max(n_signals, 1)

    for signal, market in signal_pairs:
        if len(created_ids) >= MAX_COPY_POSITIONS_PER_FOLLOW:
            break

        if market.id in existing_markets:
            continue

        # Skip markets too close to resolution
        if market.end_date:
            hours_until = (market.end_date - now).total_seconds() / 3600
            if hours_until < 6:
                continue

        # Apply edge decay — skip stale signals
        if signal.detected_at:
            signal_age_hours = (now - signal.detected_at).total_seconds() / 3600
            decay = math.exp(-0.693 * signal_age_hours / EDGE_DECAY_HALF_LIFE_HOURS)
            decayed_ev = signal.net_ev * decay
            if decayed_ev < 0.02:
                continue
        else:
            decayed_ev = signal.net_ev

        direction = signal.direction
        entry_price = market.price_yes  # Always store YES price
        side = "yes" if direction == "buy_yes" else "no"

        cost_per_share = entry_price if side == "yes" else (1.0 - entry_price)
        quantity = allocation_per_signal / max(cost_per_share, 0.01)

        if follow.max_position_size:
            max_qty = follow.max_position_size / max(cost_per_share, 0.01)
            quantity = min(quantity, max_qty)

        if quantity <= 0:
            continue

        position = PortfolioPosition(
            user_id=follow.follower_id,
            market_id=market.id,
            platform_id=market.platform_id,
            side=side,
            entry_price=entry_price,
            quantity=round(quantity, 2),
            entry_time=now,
            strategy="copy_trade",
            portfolio_type="manual",
            is_simulated=True,
        )
        session.add(position)
        await session.flush()

        # Record in copy_trades (trader_position_id=0 since signal-based, not position-based)
        copy_trade = CopyTrade(
            follower_position_id=position.id,
            trader_position_id=position.id,  # Self-referential for signal-based
            follower_id=follow.follower_id,
            trader_id=follow.trader_id,
            copy_ratio=follow.copy_percentage,
        )
        session.add(copy_trade)
        created_ids.append(position.id)
        existing_markets.add(market.id)

        logger.info(
            f"Copy position {position.id} for {follow.follower_id}: "
            f"market {market.id} | {side} @ {entry_price:.3f} | "
            f"EV={signal.net_ev:.1%} | tier={signal.quality_tier} | qty={quantity:.2f}"
        )

    if created_ids:
        activity = TraderActivity(
            trader_id=follow.trader_id,
            activity_type="copy_seed",
            activity_data={
                "follower_id": follow.follower_id,
                "positions_created": len(created_ids),
                "source": "ensemble_signals",
            },
        )
        session.add(activity)

    return created_ids


async def sync_copy_positions(session: AsyncSession) -> dict:
    """Periodic sync: open new copy positions from fresh signals, close stale ones.

    Called by the scheduler every ~15 min (same cadence as auto-trading).
    For each active follower with auto_copy:
    1. Close copy positions whose underlying signal expired or was invalidated
    2. Open new positions from fresh ensemble signals (up to max per follow)
    """
    opened = 0
    closed = 0

    # Get all active auto_copy followers
    follows_result = await session.execute(
        select(FollowedTrader).where(
            FollowedTrader.is_active == True,  # noqa
            FollowedTrader.auto_copy == True,
        )
    )
    followers = follows_result.scalars().all()

    if not followers:
        return {"opened": 0, "closed": 0}

    now = datetime.utcnow()

    for follow in followers:
        # --- 1. Close stale copy positions ---
        open_copies_result = await session.execute(
            select(PortfolioPosition, Market)
            .join(Market, PortfolioPosition.market_id == Market.id)
            .where(
                PortfolioPosition.user_id == follow.follower_id,
                PortfolioPosition.strategy == "copy_trade",
                PortfolioPosition.exit_time == None,  # noqa
            )
        )
        open_copies = open_copies_result.all()

        for pos, market in open_copies:
            should_close = False
            close_reason = ""

            # Check if underlying ensemble signal still exists and is valid
            signal_result = await session.execute(
                select(EnsembleEdgeSignal).where(
                    EnsembleEdgeSignal.market_id == pos.market_id,
                    EnsembleEdgeSignal.expired_at == None,  # noqa
                )
                .order_by(EnsembleEdgeSignal.detected_at.desc())
                .limit(1)
            )
            active_signal = signal_result.scalar_one_or_none()

            if active_signal is None:
                should_close = True
                close_reason = "signal_expired"

            # Close if market resolved or became inactive
            if not market.is_active or market.is_resolved:
                should_close = True
                close_reason = "market_resolved"

            # Close if market is very near resolution
            if market.end_date:
                hours_until = (market.end_date - now).total_seconds() / 3600
                if hours_until < 2:
                    should_close = True
                    close_reason = "near_expiry"

            if should_close:
                exit_price = market.price_yes or pos.entry_price
                pos.exit_price = exit_price
                pos.exit_time = now

                if pos.side == "yes":
                    gross_pnl = (exit_price - pos.entry_price) * pos.quantity
                else:
                    gross_pnl = (pos.entry_price - exit_price) * pos.quantity

                # Polymarket 2% fee on net winnings only
                fee = 0.02 * max(gross_pnl, 0.0)
                pos.realized_pnl = gross_pnl - fee

                closed += 1
                logger.info(
                    f"Closed copy position {pos.id} ({close_reason}): "
                    f"P&L=${pos.realized_pnl:.2f}"
                )

        # --- 2. Open new positions from fresh signals ---
        # Count current open copy positions for this follower
        open_count_result = await session.execute(
            select(func.count(PortfolioPosition.id))
            .where(
                PortfolioPosition.user_id == follow.follower_id,
                PortfolioPosition.strategy == "copy_trade",
                PortfolioPosition.exit_time == None,  # noqa
            )
        )
        current_open = open_count_result.scalar() or 0

        slots = MAX_COPY_POSITIONS_PER_FOLLOW - current_open
        if slots <= 0:
            continue

        # Get current open market IDs to avoid duplicates
        existing_result = await session.execute(
            select(PortfolioPosition.market_id)
            .where(
                PortfolioPosition.user_id == follow.follower_id,
                PortfolioPosition.strategy == "copy_trade",
                PortfolioPosition.exit_time == None,  # noqa
            )
        )
        existing_markets = {row[0] for row in existing_result.all()}

        # Find fresh signals not already in portfolio
        signal_query = (
            select(EnsembleEdgeSignal, Market)
            .join(Market, EnsembleEdgeSignal.market_id == Market.id)
            .where(
                EnsembleEdgeSignal.expired_at == None,  # noqa
                EnsembleEdgeSignal.quality_tier.in_(["medium", "high"]),
                EnsembleEdgeSignal.net_ev > 0.03,
                EnsembleEdgeSignal.confidence >= 0.5,
                Market.is_active == True,  # noqa
                Market.price_yes > 0.05,
                Market.price_yes < 0.95,
            )
            .order_by(EnsembleEdgeSignal.net_ev.desc())
            .limit(slots * 2)
        )
        if existing_markets:
            signal_query = signal_query.where(
                EnsembleEdgeSignal.market_id.notin_(existing_markets)
            )
        signals_result = await session.execute(signal_query)
        new_signals = signals_result.all()

        allocation_per = follow.allocation_amount / max(MAX_COPY_POSITIONS_PER_FOLLOW, 1)
        created_this_follow = 0

        for signal, market in new_signals:
            if created_this_follow >= slots:
                break

            if market.id in existing_markets:
                continue

            # Skip near-resolution
            if market.end_date:
                hours_until = (market.end_date - now).total_seconds() / 3600
                if hours_until < 6:
                    continue

            # Edge decay check
            if signal.detected_at:
                age_hours = (now - signal.detected_at).total_seconds() / 3600
                decay = math.exp(-0.693 * age_hours / EDGE_DECAY_HALF_LIFE_HOURS)
                if signal.net_ev * decay < 0.02:
                    continue

            direction = signal.direction
            side = "yes" if direction == "buy_yes" else "no"
            entry_price = market.price_yes

            cost_per_share = entry_price if side == "yes" else (1.0 - entry_price)
            quantity = allocation_per / max(cost_per_share, 0.01)

            if follow.max_position_size:
                max_qty = follow.max_position_size / max(cost_per_share, 0.01)
                quantity = min(quantity, max_qty)

            if quantity <= 0:
                continue

            position = PortfolioPosition(
                user_id=follow.follower_id,
                market_id=market.id,
                platform_id=market.platform_id,
                side=side,
                entry_price=entry_price,
                quantity=round(quantity, 2),
                entry_time=now,
                strategy="copy_trade",
                portfolio_type="manual",
                is_simulated=True,
            )
            session.add(position)
            await session.flush()

            copy_trade = CopyTrade(
                follower_position_id=position.id,
                trader_position_id=position.id,
                follower_id=follow.follower_id,
                trader_id=follow.trader_id,
                copy_ratio=follow.copy_percentage,
            )
            session.add(copy_trade)

            existing_markets.add(market.id)
            created_this_follow += 1
            opened += 1

            logger.info(
                f"New copy position {position.id} for {follow.follower_id}: "
                f"market {market.id} | {side} @ {entry_price:.3f} | "
                f"EV={signal.net_ev:.1%} | tier={signal.quality_tier}"
            )

    if opened or closed:
        await session.commit()
        logger.info(f"Copy sync: {opened} opened, {closed} closed")

    return {"opened": opened, "closed": closed}


async def on_position_opened(
    position: PortfolioPosition,
    session: AsyncSession,
) -> list[int]:
    """Event handler when an internal system trader opens a position.

    NOTE: This is only called for positions created within our system.
    External Polymarket traders never trigger this — their positions are
    tracked via ensemble signals in seed_copy_positions/sync_copy_positions.
    """
    trader_id = position.user_id
    copied_ids = []

    result = await session.execute(
        select(FollowedTrader).where(
            FollowedTrader.trader_id == trader_id,
            FollowedTrader.is_active == True,
            FollowedTrader.auto_copy == True,
        )
    )
    followers = result.scalars().all()

    if not followers:
        return copied_ids

    logger.info(f"Auto-copying position {position.id} from {trader_id} to {len(followers)} followers")

    for follow in followers:
        try:
            copy_ratio = follow.copy_percentage
            copied_quantity = position.quantity * copy_ratio

            cost_per_share = position.entry_price if position.side == "yes" else (1.0 - position.entry_price)
            if follow.max_position_size:
                max_qty_by_size = follow.max_position_size / max(cost_per_share, 0.01)
                copied_quantity = min(copied_quantity, max_qty_by_size)

            if position.side == "yes":
                position_cost = copied_quantity * position.entry_price
            else:
                position_cost = copied_quantity * (1.0 - position.entry_price)
            if position_cost > follow.allocation_amount:
                copied_quantity = follow.allocation_amount / max(cost_per_share, 0.01)

            if copied_quantity <= 0:
                continue

            copied_position = PortfolioPosition(
                user_id=follow.follower_id,
                market_id=position.market_id,
                platform_id=position.platform_id,
                side=position.side,
                entry_price=position.entry_price,
                quantity=copied_quantity,
                entry_time=datetime.utcnow(),
                strategy="copy_trade",
                portfolio_type="manual",
                is_simulated=True,
            )
            session.add(copied_position)
            await session.flush()

            copy_trade = CopyTrade(
                follower_position_id=copied_position.id,
                trader_position_id=position.id,
                follower_id=follow.follower_id,
                trader_id=trader_id,
                copy_ratio=copy_ratio,
            )
            session.add(copy_trade)
            copied_ids.append(copied_position.id)

        except Exception as e:
            logger.error(f"Failed to copy for follower {follow.follower_id}: {e}")
            continue

    if copied_ids:
        activity = TraderActivity(
            trader_id=trader_id,
            activity_type="open_position",
            activity_data={
                "position_id": position.id,
                "market_id": position.market_id,
                "side": position.side,
                "copies_created": len(copied_ids),
            },
        )
        session.add(activity)

    return copied_ids


async def on_position_closed(
    position: PortfolioPosition,
    session: AsyncSession,
) -> list[int]:
    """Event handler when an internal system position is closed.
    Automatically closes all linked copy positions with fee-adjusted P&L.
    """
    closed_ids = []

    result = await session.execute(
        select(CopyTrade).where(
            CopyTrade.trader_position_id == position.id,
        )
    )
    copy_trades = result.scalars().all()

    if not copy_trades:
        return closed_ids

    for ct in copy_trades:
        try:
            copied_pos = await session.get(PortfolioPosition, ct.follower_position_id)
            if not copied_pos or copied_pos.exit_time is not None:
                continue

            copied_pos.exit_price = position.exit_price
            copied_pos.exit_time = datetime.utcnow()

            if copied_pos.side == "yes":
                gross_pnl = (position.exit_price - copied_pos.entry_price) * copied_pos.quantity
            else:
                gross_pnl = (copied_pos.entry_price - position.exit_price) * copied_pos.quantity

            # Polymarket 2% fee on net winnings only
            fee = 0.02 * max(gross_pnl, 0.0)
            copied_pos.realized_pnl = gross_pnl - fee

            closed_ids.append(copied_pos.id)

        except Exception as e:
            logger.error(f"Failed to close copy {ct.follower_position_id}: {e}")
            continue

    return closed_ids


async def log_trader_activity(
    trader_id: str,
    activity_type: str,
    data: dict,
    session: AsyncSession,
):
    """Helper to log any trader activity."""
    activity = TraderActivity(
        trader_id=trader_id,
        activity_type=activity_type,
        activity_data=data,
    )
    session.add(activity)
