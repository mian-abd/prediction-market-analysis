"""Auto-copy engine - replicates trades from followed traders to followers.

When a trader opens/closes a position, this engine:
1. Finds all followers with auto_copy enabled
2. Creates proportional copied positions for each follower
3. Records copy trades and activity feed entries
"""

import logging
from datetime import datetime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import (
    PortfolioPosition, FollowedTrader, CopyTrade, TraderActivity, TraderProfile,
)

logger = logging.getLogger(__name__)


async def on_position_opened(
    position: PortfolioPosition,
    session: AsyncSession,
) -> list[int]:
    """
    Event handler when a trader opens a new position.
    Creates copied positions for all auto-copy followers.

    Args:
        position: The newly created position
        session: Active database session

    Returns:
        List of created copied position IDs
    """
    trader_id = position.user_id
    copied_ids = []

    # Find all active followers with auto_copy enabled
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
            # Calculate copy size
            copy_ratio = follow.copy_percentage
            copied_quantity = position.quantity * copy_ratio

            # Apply max position size limit
            if follow.max_position_size:
                max_qty_by_size = follow.max_position_size / position.entry_price
                copied_quantity = min(copied_quantity, max_qty_by_size)

            # Check allocation budget
            position_cost = copied_quantity * position.entry_price
            if position_cost > follow.allocation_amount:
                copied_quantity = follow.allocation_amount / position.entry_price

            if copied_quantity <= 0:
                logger.debug(f"Skipping copy for {follow.follower_id}: insufficient allocation")
                continue

            # Create copied position
            copied_position = PortfolioPosition(
                user_id=follow.follower_id,
                market_id=position.market_id,
                platform_id=position.platform_id,
                side=position.side,
                entry_price=position.entry_price,
                quantity=copied_quantity,
                entry_time=datetime.utcnow(),
                strategy="copy_trade",
                is_simulated=True,
            )
            session.add(copied_position)
            await session.flush()  # Get the ID

            # Record in copy_trades
            copy_trade = CopyTrade(
                follower_position_id=copied_position.id,
                trader_position_id=position.id,
                follower_id=follow.follower_id,
                trader_id=trader_id,
                copy_ratio=copy_ratio,
            )
            session.add(copy_trade)

            copied_ids.append(copied_position.id)

            logger.info(
                f"Copied position {position.id} -> {copied_position.id} "
                f"for follower {follow.follower_id} "
                f"(qty: {position.quantity:.2f} -> {copied_quantity:.2f})"
            )

        except Exception as e:
            logger.error(f"Failed to copy for follower {follow.follower_id}: {e}")
            continue

    # Log activity
    if copied_ids:
        activity = TraderActivity(
            trader_id=trader_id,
            activity_type="open_position",
            activity_data={
                "position_id": position.id,
                "market_id": position.market_id,
                "side": position.side,
                "entry_price": position.entry_price,
                "quantity": position.quantity,
                "strategy": position.strategy,
                "copies_created": len(copied_ids),
            },
        )
        session.add(activity)

    return copied_ids


async def on_position_closed(
    position: PortfolioPosition,
    session: AsyncSession,
) -> list[int]:
    """
    Event handler when a trader closes a position.
    Automatically closes all linked copy positions.

    Args:
        position: The position being closed
        session: Active database session

    Returns:
        List of closed copied position IDs
    """
    trader_id = position.user_id
    closed_ids = []

    # Find all copy trades linked to this position
    result = await session.execute(
        select(CopyTrade).where(
            CopyTrade.trader_position_id == position.id,
        )
    )
    copy_trades = result.scalars().all()

    if not copy_trades:
        return closed_ids

    logger.info(f"Auto-closing {len(copy_trades)} copied positions for trader {trader_id}")

    for ct in copy_trades:
        try:
            copied_pos = await session.get(PortfolioPosition, ct.follower_position_id)
            if not copied_pos or copied_pos.exit_time is not None:
                continue

            # Close the copied position at the same exit price
            copied_pos.exit_price = position.exit_price
            copied_pos.exit_time = datetime.utcnow()
            copied_pos.realized_pnl = (
                (position.exit_price - copied_pos.entry_price) * copied_pos.quantity
            )

            closed_ids.append(copied_pos.id)

            logger.info(
                f"Closed copied position {copied_pos.id} "
                f"P&L: ${copied_pos.realized_pnl:.2f}"
            )

        except Exception as e:
            logger.error(f"Failed to close copy {ct.follower_position_id}: {e}")
            continue

    # Log activity
    if closed_ids:
        activity = TraderActivity(
            trader_id=trader_id,
            activity_type="close_position",
            activity_data={
                "position_id": position.id,
                "market_id": position.market_id,
                "exit_price": position.exit_price,
                "realized_pnl": position.realized_pnl,
                "copies_closed": len(closed_ids),
            },
        )
        session.add(activity)

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
