"""Copy Trading API routes - trader discovery, following, and performance tracking."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, desc, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta
from typing import List, Optional
from pydantic import BaseModel

from db.database import get_session
from db.models import (
    TraderProfile, FollowedTrader, CopyTrade, TraderActivity,
    PortfolioPosition
)

router = APIRouter(prefix="/copy-trading", tags=["copy_trading"])


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class TraderSummary(BaseModel):
    user_id: str
    display_name: str
    bio: Optional[str]
    total_pnl: float
    roi_pct: float
    win_rate: float
    total_trades: int
    risk_score: int
    follower_count: int
    is_following: bool = False


class TraderDetail(BaseModel):
    user_id: str
    display_name: str
    bio: Optional[str]
    total_pnl: float
    roi_pct: float
    win_rate: float
    total_trades: int
    winning_trades: int
    avg_trade_duration_hrs: float
    risk_score: int
    max_drawdown: float
    follower_count: int
    is_following: bool = False
    created_at: datetime


class FollowRequest(BaseModel):
    trader_id: str
    allocation_amount: float = 1000.0
    copy_percentage: float = 1.0
    max_position_size: Optional[float] = None
    auto_copy: bool = True


class Following(BaseModel):
    trader_id: str
    display_name: str
    allocation_amount: float
    copy_percentage: float
    auto_copy: bool
    followed_at: datetime
    copied_trades: int = 0
    copy_pnl: float = 0.0


# ============================================================================
# TRADER DISCOVERY & LEADERBOARD
# ============================================================================

@router.get("/leaderboard", response_model=List[TraderSummary])
async def get_trader_leaderboard(
    sort_by: str = Query("total_pnl", regex="^(total_pnl|roi_pct|win_rate|total_trades)$"),
    limit: int = Query(50, ge=1, le=100),
    current_user: str = "user_1",  # TODO: Extract from auth
    session: AsyncSession = Depends(get_session),
):
    """
    Get ranked list of traders for copy trading discovery.

    - **sort_by**: total_pnl, roi_pct, win_rate, or total_trades
    - **limit**: max traders to return (1-100)
    """
    # Build query with optional filters
    query = select(TraderProfile).where(
        TraderProfile.is_public == True,
        TraderProfile.accepts_copiers == True,
        TraderProfile.total_trades > 0,  # Only traders with trading history
    )

    # Sort by requested field
    if sort_by == "total_pnl":
        query = query.order_by(desc(TraderProfile.total_pnl))
    elif sort_by == "roi_pct":
        query = query.order_by(desc(TraderProfile.roi_pct))
    elif sort_by == "win_rate":
        query = query.order_by(desc(TraderProfile.win_rate))
    elif sort_by == "total_trades":
        query = query.order_by(desc(TraderProfile.total_trades))

    query = query.limit(limit)
    result = await session.execute(query)
    traders = result.scalars().all()

    # Check which traders current user is following
    following_query = select(FollowedTrader.trader_id).where(
        FollowedTrader.follower_id == current_user,
        FollowedTrader.is_active == True,
    )
    following_result = await session.execute(following_query)
    following_ids = set(following_result.scalars().all())

    return [
        TraderSummary(
            user_id=trader.user_id,
            display_name=trader.display_name,
            bio=trader.bio,
            total_pnl=trader.total_pnl,
            roi_pct=trader.roi_pct,
            win_rate=trader.win_rate,
            total_trades=trader.total_trades,
            risk_score=trader.risk_score,
            follower_count=trader.follower_count,
            is_following=(trader.user_id in following_ids),
        )
        for trader in traders
    ]


@router.get("/traders/{trader_id}", response_model=TraderDetail)
async def get_trader_profile(
    trader_id: str,
    current_user: str = "user_1",  # TODO: Extract from auth
    session: AsyncSession = Depends(get_session),
):
    """Get detailed trader profile with full stats."""
    result = await session.execute(
        select(TraderProfile).where(TraderProfile.user_id == trader_id)
    )
    trader = result.scalar_one_or_none()

    if not trader:
        raise HTTPException(status_code=404, detail="Trader not found")

    # Check if current user is following
    following_result = await session.execute(
        select(FollowedTrader).where(
            FollowedTrader.follower_id == current_user,
            FollowedTrader.trader_id == trader_id,
            FollowedTrader.is_active == True,
        )
    )
    is_following = following_result.scalar_one_or_none() is not None

    return TraderDetail(
        user_id=trader.user_id,
        display_name=trader.display_name,
        bio=trader.bio,
        total_pnl=trader.total_pnl,
        roi_pct=trader.roi_pct,
        win_rate=trader.win_rate,
        total_trades=trader.total_trades,
        winning_trades=trader.winning_trades,
        avg_trade_duration_hrs=trader.avg_trade_duration_hrs,
        risk_score=trader.risk_score,
        max_drawdown=trader.max_drawdown,
        follower_count=trader.follower_count,
        is_following=is_following,
        created_at=trader.created_at,
    )


# ============================================================================
# FOLLOWING / UNFOLLOWING
# ============================================================================

@router.post("/follow")
async def follow_trader(
    request: FollowRequest,
    current_user: str = "user_1",  # TODO: Extract from auth
    session: AsyncSession = Depends(get_session),
):
    """Start following a trader and copy their trades."""
    # Check if trader exists
    trader_result = await session.execute(
        select(TraderProfile).where(TraderProfile.user_id == request.trader_id)
    )
    trader = trader_result.scalar_one_or_none()

    if not trader:
        raise HTTPException(status_code=404, detail="Trader not found")

    if not trader.accepts_copiers:
        raise HTTPException(status_code=400, detail="Trader does not accept copiers")

    # Check if already following
    existing = await session.execute(
        select(FollowedTrader).where(
            FollowedTrader.follower_id == current_user,
            FollowedTrader.trader_id == request.trader_id,
        )
    )
    existing_follow = existing.scalar_one_or_none()

    if existing_follow and existing_follow.is_active:
        raise HTTPException(status_code=400, detail="Already following this trader")

    # Create or reactivate follow relationship
    if existing_follow:
        existing_follow.is_active = True
        existing_follow.allocation_amount = request.allocation_amount
        existing_follow.copy_percentage = request.copy_percentage
        existing_follow.max_position_size = request.max_position_size
        existing_follow.auto_copy = request.auto_copy
        existing_follow.followed_at = datetime.utcnow()
        existing_follow.unfollowed_at = None
    else:
        new_follow = FollowedTrader(
            follower_id=current_user,
            trader_id=request.trader_id,
            allocation_amount=request.allocation_amount,
            copy_percentage=request.copy_percentage,
            max_position_size=request.max_position_size,
            auto_copy=request.auto_copy,
        )
        session.add(new_follow)

    # Update follower count
    trader.follower_count += 1

    await session.commit()

    return {"message": "Successfully following trader", "trader_id": request.trader_id}


@router.delete("/follow/{trader_id}")
async def unfollow_trader(
    trader_id: str,
    current_user: str = "user_1",  # TODO: Extract from auth
    session: AsyncSession = Depends(get_session),
):
    """Stop following a trader (does not close existing copied positions)."""
    result = await session.execute(
        select(FollowedTrader).where(
            FollowedTrader.follower_id == current_user,
            FollowedTrader.trader_id == trader_id,
            FollowedTrader.is_active == True,
        )
    )
    follow = result.scalar_one_or_none()

    if not follow:
        raise HTTPException(status_code=404, detail="Not following this trader")

    follow.is_active = False
    follow.unfollowed_at = datetime.utcnow()

    # Update follower count
    trader_result = await session.execute(
        select(TraderProfile).where(TraderProfile.user_id == trader_id)
    )
    trader = trader_result.scalar_one_or_none()
    if trader and trader.follower_count > 0:
        trader.follower_count -= 1

    await session.commit()

    return {"message": "Successfully unfollowed trader", "trader_id": trader_id}


# ============================================================================
# USER'S FOLLOWING LIST & PERFORMANCE
# ============================================================================

@router.get("/following", response_model=List[Following])
async def get_following_traders(
    current_user: str = "user_1",  # TODO: Extract from auth
    session: AsyncSession = Depends(get_session),
):
    """Get list of traders the current user is following."""
    query = (
        select(FollowedTrader, TraderProfile)
        .join(TraderProfile, FollowedTrader.trader_id == TraderProfile.user_id)
        .where(
            FollowedTrader.follower_id == current_user,
            FollowedTrader.is_active == True,
        )
    )
    result = await session.execute(query)
    following = result.all()

    # Calculate copied trade stats for each trader
    following_list = []
    for follow, trader in following:
        # Count copied trades
        copy_count_query = select(func.count(CopyTrade.id)).where(
            CopyTrade.follower_id == current_user,
            CopyTrade.trader_id == trader.user_id,
        )
        copy_count = await session.scalar(copy_count_query) or 0

        # Calculate total P&L from copied trades
        copy_pnl_query = (
            select(func.sum(PortfolioPosition.realized_pnl))
            .select_from(CopyTrade)
            .join(PortfolioPosition, CopyTrade.follower_position_id == PortfolioPosition.id)
            .where(
                CopyTrade.follower_id == current_user,
                CopyTrade.trader_id == trader.user_id,
                PortfolioPosition.realized_pnl.isnot(None),
            )
        )
        copy_pnl = await session.scalar(copy_pnl_query) or 0.0

        following_list.append(
            Following(
                trader_id=trader.user_id,
                display_name=trader.display_name,
                allocation_amount=follow.allocation_amount,
                copy_percentage=follow.copy_percentage,
                auto_copy=follow.auto_copy,
                followed_at=follow.followed_at,
                copied_trades=copy_count,
                copy_pnl=copy_pnl,
            )
        )

    return following_list


# ============================================================================
# COPY TRADING PERFORMANCE
# ============================================================================

@router.get("/performance")
async def get_copy_performance(
    current_user: str = "user_1",  # TODO: Extract from auth
    session: AsyncSession = Depends(get_session),
):
    """Get overall copy trading performance for current user."""
    # Total copied trades
    total_trades_query = select(func.count(CopyTrade.id)).where(
        CopyTrade.follower_id == current_user
    )
    total_trades = await session.scalar(total_trades_query) or 0

    # Total realized P&L from copied trades
    total_pnl_query = (
        select(func.sum(PortfolioPosition.realized_pnl))
        .select_from(CopyTrade)
        .join(PortfolioPosition, CopyTrade.follower_position_id == PortfolioPosition.id)
        .where(
            CopyTrade.follower_id == current_user,
            PortfolioPosition.realized_pnl.isnot(None),
        )
    )
    total_pnl = await session.scalar(total_pnl_query) or 0.0

    # Winning trades
    winning_trades_query = (
        select(func.count(CopyTrade.id))
        .select_from(CopyTrade)
        .join(PortfolioPosition, CopyTrade.follower_position_id == PortfolioPosition.id)
        .where(
            CopyTrade.follower_id == current_user,
            PortfolioPosition.realized_pnl > 0,
        )
    )
    winning_trades = await session.scalar(winning_trades_query) or 0

    # Active copied positions
    active_positions_query = (
        select(func.count(CopyTrade.id))
        .select_from(CopyTrade)
        .join(PortfolioPosition, CopyTrade.follower_position_id == PortfolioPosition.id)
        .where(
            CopyTrade.follower_id == current_user,
            PortfolioPosition.exit_time.is_(None),
        )
    )
    active_positions = await session.scalar(active_positions_query) or 0

    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

    return {
        "total_copied_trades": total_trades,
        "active_positions": active_positions,
        "total_pnl": total_pnl,
        "winning_trades": winning_trades,
        "win_rate": win_rate,
    }
