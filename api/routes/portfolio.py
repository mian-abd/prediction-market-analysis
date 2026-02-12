"""Portfolio and paper trading endpoints.
Supports simulated positions for strategy backtesting on prop accounts."""

from datetime import datetime
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_session
from db.models import PortfolioPosition, Market, Platform

router = APIRouter(tags=["portfolio"])


class OpenPositionRequest(BaseModel):
    market_id: int
    side: str  # "yes" or "no"
    entry_price: float
    quantity: float
    strategy: str = "manual"  # manual | single_market_arb | cross_platform_arb | calibration


class ClosePositionRequest(BaseModel):
    exit_price: float


@router.get("/portfolio/positions")
async def list_positions(
    status: str = Query(default="open", regex="^(open|closed|all)$"),
    strategy: str | None = None,
    limit: int = Query(default=50, le=200),
    session: AsyncSession = Depends(get_session),
):
    """List portfolio positions."""
    query = select(PortfolioPosition)

    if status == "open":
        query = query.where(PortfolioPosition.exit_time == None)  # noqa: E711
    elif status == "closed":
        query = query.where(PortfolioPosition.exit_time != None)  # noqa: E711

    if strategy:
        query = query.where(PortfolioPosition.strategy == strategy)

    query = query.order_by(PortfolioPosition.entry_time.desc()).limit(limit)

    result = await session.execute(query)
    positions = result.scalars().all()

    # Enrich with market details
    platform_map = {}
    platforms_result = await session.execute(select(Platform))
    for p in platforms_result.scalars().all():
        platform_map[p.id] = p.name

    enriched = []
    for pos in positions:
        market = await session.get(Market, pos.market_id)
        current_price = None
        unrealized_pnl = None

        if market and pos.exit_time is None:
            # Calculate unrealized P&L for open positions
            if pos.side == "yes":
                current_price = market.price_yes
            else:
                current_price = market.price_no

            if current_price is not None:
                unrealized_pnl = (current_price - pos.entry_price) * pos.quantity

        enriched.append({
            "id": pos.id,
            "market_id": pos.market_id,
            "question": market.question if market else "Unknown",
            "platform": platform_map.get(pos.platform_id, "unknown"),
            "side": pos.side,
            "entry_price": pos.entry_price,
            "quantity": pos.quantity,
            "entry_time": pos.entry_time.isoformat(),
            "exit_price": pos.exit_price,
            "exit_time": pos.exit_time.isoformat() if pos.exit_time else None,
            "realized_pnl": pos.realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "current_price": current_price,
            "strategy": pos.strategy,
            "is_simulated": pos.is_simulated,
        })

    return {"positions": enriched, "count": len(enriched)}


@router.post("/portfolio/positions")
async def open_position(
    req: OpenPositionRequest,
    session: AsyncSession = Depends(get_session),
):
    """Open a new paper trading position."""
    market = await session.get(Market, req.market_id)
    if not market:
        return {"error": "Market not found"}

    position = PortfolioPosition(
        market_id=req.market_id,
        platform_id=market.platform_id,
        side=req.side,
        entry_price=req.entry_price,
        quantity=req.quantity,
        entry_time=datetime.utcnow(),
        strategy=req.strategy,
        is_simulated=True,
    )
    session.add(position)
    await session.commit()
    await session.refresh(position)

    return {
        "id": position.id,
        "market_id": position.market_id,
        "side": position.side,
        "entry_price": position.entry_price,
        "quantity": position.quantity,
        "strategy": position.strategy,
        "message": "Paper position opened",
    }


@router.post("/portfolio/positions/{position_id}/close")
async def close_position(
    position_id: int,
    req: ClosePositionRequest,
    session: AsyncSession = Depends(get_session),
):
    """Close an open paper trading position."""
    position = await session.get(PortfolioPosition, position_id)
    if not position:
        return {"error": "Position not found"}
    if position.exit_time is not None:
        return {"error": "Position already closed"}

    position.exit_price = req.exit_price
    position.exit_time = datetime.utcnow()
    position.realized_pnl = (req.exit_price - position.entry_price) * position.quantity

    await session.commit()

    return {
        "id": position.id,
        "realized_pnl": position.realized_pnl,
        "exit_price": position.exit_price,
        "message": "Position closed",
    }


@router.get("/portfolio/summary")
async def portfolio_summary(
    session: AsyncSession = Depends(get_session),
):
    """Overall portfolio performance summary."""
    # Open positions
    open_count = (await session.execute(
        select(func.count(PortfolioPosition.id))
        .where(PortfolioPosition.exit_time == None)  # noqa: E711
    )).scalar() or 0

    # Closed positions
    closed_count = (await session.execute(
        select(func.count(PortfolioPosition.id))
        .where(PortfolioPosition.exit_time != None)  # noqa: E711
    )).scalar() or 0

    # Total realized P&L
    total_realized = (await session.execute(
        select(func.sum(PortfolioPosition.realized_pnl))
        .where(PortfolioPosition.exit_time != None)  # noqa: E711
    )).scalar() or 0.0

    # Win rate
    winning = (await session.execute(
        select(func.count(PortfolioPosition.id))
        .where(
            PortfolioPosition.exit_time != None,  # noqa: E711
            PortfolioPosition.realized_pnl > 0,
        )
    )).scalar() or 0

    win_rate = (winning / closed_count * 100) if closed_count > 0 else 0.0

    # P&L by strategy
    strategy_pnl_result = await session.execute(
        select(
            PortfolioPosition.strategy,
            func.count(PortfolioPosition.id),
            func.sum(PortfolioPosition.realized_pnl),
        )
        .where(PortfolioPosition.exit_time != None)  # noqa: E711
        .group_by(PortfolioPosition.strategy)
    )

    by_strategy = [
        {
            "strategy": row[0] or "manual",
            "trades": row[1],
            "total_pnl": row[2] or 0.0,
        }
        for row in strategy_pnl_result.all()
    ]

    # Total exposure (open positions)
    total_exposure = (await session.execute(
        select(func.sum(PortfolioPosition.entry_price * PortfolioPosition.quantity))
        .where(PortfolioPosition.exit_time == None)  # noqa: E711
    )).scalar() or 0.0

    return {
        "open_positions": open_count,
        "closed_positions": closed_count,
        "total_realized_pnl": total_realized,
        "win_rate": win_rate,
        "total_exposure": total_exposure,
        "by_strategy": by_strategy,
    }


@router.get("/portfolio/equity-curve")
async def get_equity_curve(
    time_range: str = Query(default="30d", regex="^(7d|30d|90d|all)$"),
    session: AsyncSession = Depends(get_session),
):
    """Get cumulative P&L over time for equity curve visualization.

    Returns time series data with cumulative P&L by strategy.
    """
    from datetime import timedelta
    from collections import defaultdict

    # Determine cutoff date
    cutoff_map = {
        "7d": timedelta(days=7),
        "30d": timedelta(days=30),
        "90d": timedelta(days=90),
        "all": None,
    }
    cutoff_delta = cutoff_map[time_range]

    query = select(PortfolioPosition).where(
        PortfolioPosition.exit_time != None  # noqa: E711
    ).order_by(PortfolioPosition.exit_time)

    if cutoff_delta:
        cutoff_time = datetime.utcnow() - cutoff_delta
        query = query.where(PortfolioPosition.exit_time >= cutoff_time)

    result = await session.execute(query)
    positions = result.scalars().all()

    if not positions:
        return {"data": [], "strategies": [], "total_pnl": 0.0}

    # Build time series with cumulative P&L
    # Group by strategy
    strategy_data = defaultdict(list)
    strategy_cumulative = defaultdict(float)
    all_cumulative = 0.0

    for pos in positions:
        strategy = pos.strategy or "manual"
        pnl = pos.realized_pnl or 0.0

        # Update cumulative P&L
        strategy_cumulative[strategy] += pnl
        all_cumulative += pnl

        # Record data point
        timestamp = pos.exit_time.isoformat()
        strategy_data[strategy].append({
            "timestamp": timestamp,
            "cumulative_pnl": strategy_cumulative[strategy],
        })

        # Also record for "all" strategy
        if "all" not in strategy_data or strategy_data["all"][-1]["timestamp"] != timestamp:
            strategy_data["all"].append({
                "timestamp": timestamp,
                "cumulative_pnl": all_cumulative,
            })

    # Build response
    strategies = []
    for strategy_name, points in strategy_data.items():
        strategies.append({
            "name": strategy_name,
            "data": points,
            "final_pnl": points[-1]["cumulative_pnl"] if points else 0.0,
        })

    return {
        "data": strategies,
        "strategies": list(strategy_data.keys()),
        "total_pnl": all_cumulative,
    }


@router.get("/portfolio/win-rate")
async def get_win_rate_by_strategy(
    min_trades: int = Query(default=1, ge=1),
    session: AsyncSession = Depends(get_session),
):
    """Get win rate breakdown by strategy.

    Returns win rate, trade count, avg profit, max loss per strategy.
    """
    # Get all closed positions
    result = await session.execute(
        select(PortfolioPosition).where(
            PortfolioPosition.exit_time != None  # noqa: E711
        )
    )
    positions = result.scalars().all()

    if not positions:
        return {"strategies": [], "overall_win_rate": 0.0}

    # Group by strategy
    from collections import defaultdict

    strategy_stats = defaultdict(lambda: {
        "wins": 0,
        "losses": 0,
        "total_trades": 0,
        "total_pnl": 0.0,
        "winning_pnl": [],
        "losing_pnl": [],
    })

    for pos in positions:
        strategy = pos.strategy or "manual"
        pnl = pos.realized_pnl or 0.0

        stats = strategy_stats[strategy]
        stats["total_trades"] += 1
        stats["total_pnl"] += pnl

        if pnl > 0:
            stats["wins"] += 1
            stats["winning_pnl"].append(pnl)
        else:
            stats["losses"] += 1
            stats["losing_pnl"].append(pnl)

    # Calculate metrics
    strategies = []
    total_wins = 0
    total_trades = 0

    for strategy_name, stats in strategy_stats.items():
        if stats["total_trades"] < min_trades:
            continue  # Skip strategies with too few trades

        win_rate = (stats["wins"] / stats["total_trades"] * 100) if stats["total_trades"] > 0 else 0.0
        avg_win = sum(stats["winning_pnl"]) / len(stats["winning_pnl"]) if stats["winning_pnl"] else 0.0
        avg_loss = sum(stats["losing_pnl"]) / len(stats["losing_pnl"]) if stats["losing_pnl"] else 0.0
        max_loss = min(stats["losing_pnl"]) if stats["losing_pnl"] else 0.0

        strategies.append({
            "strategy": strategy_name,
            "win_rate": win_rate,
            "total_trades": stats["total_trades"],
            "wins": stats["wins"],
            "losses": stats["losses"],
            "total_pnl": stats["total_pnl"],
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_loss": max_loss,
        })

        total_wins += stats["wins"]
        total_trades += stats["total_trades"]

    # Overall win rate
    overall_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0.0

    return {
        "strategies": strategies,
        "overall_win_rate": overall_win_rate,
        "total_trades": total_trades,
    }
