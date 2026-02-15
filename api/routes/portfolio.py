"""Portfolio and paper trading endpoints.
Supports simulated positions for strategy backtesting on prop accounts.
Supports portfolio_type filtering: 'manual' (manual + copy) vs 'auto' (ensemble + elo)."""

from datetime import datetime
from fastapi import APIRouter, Depends, Header, Query
from pydantic import BaseModel
from sqlalchemy import select, func, case
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_session
from db.models import PortfolioPosition, Market, Platform
from data_pipeline.copy_engine import on_position_opened, on_position_closed
from risk.risk_manager import check_risk_limits, get_risk_status

router = APIRouter(tags=["portfolio"])


async def get_current_user(x_user_id: str = Header(default="anonymous")) -> str:
    """Extract user ID from X-User-Id header. Defaults to 'anonymous'."""
    return x_user_id


class OpenPositionRequest(BaseModel):
    market_id: int
    side: str  # "yes" or "no"
    entry_price: float
    quantity: float
    strategy: str = "manual"  # manual | single_market_arb | cross_platform_arb | calibration


class ClosePositionRequest(BaseModel):
    exit_price: float


def _apply_portfolio_filter(query, portfolio_type: str | None):
    """Add portfolio_type WHERE clause if specified."""
    if portfolio_type is not None:
        return query.where(PortfolioPosition.portfolio_type == portfolio_type)
    return query


@router.get("/portfolio/positions")
async def list_positions(
    status: str = Query(default="open", regex="^(open|closed|all)$"),
    strategy: str | None = None,
    portfolio_type: str | None = Query(default=None, pattern="^(manual|auto)$"),
    limit: int = Query(default=50, le=200),
    current_user: str = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """List portfolio positions with optional portfolio_type filter (filtered by current user)."""
    query = select(PortfolioPosition).where(PortfolioPosition.user_id == current_user)

    if status == "open":
        query = query.where(PortfolioPosition.exit_time == None)  # noqa: E711
    elif status == "closed":
        query = query.where(PortfolioPosition.exit_time != None)  # noqa: E711

    if strategy:
        query = query.where(PortfolioPosition.strategy == strategy)

    query = _apply_portfolio_filter(query, portfolio_type)
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
            current_price = market.price_yes if pos.side == "yes" else market.price_no

            if market.price_yes is not None:
                if pos.side == "yes":
                    unrealized_pnl = (market.price_yes - pos.entry_price) * pos.quantity
                else:
                    unrealized_pnl = (pos.entry_price - market.price_yes) * pos.quantity

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
            "portfolio_type": pos.portfolio_type,
            "is_simulated": pos.is_simulated,
        })

    return {"positions": enriched, "count": len(enriched)}


@router.post("/portfolio/positions")
async def open_position(
    req: OpenPositionRequest,
    current_user: str = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Open a new paper trading position (always manual portfolio)."""
    market = await session.get(Market, req.market_id)
    if not market:
        return {"error": "Market not found"}

    # Side-aware position cost
    if req.side == "yes":
        position_cost = req.entry_price * req.quantity
    else:
        position_cost = (1.0 - req.entry_price) * req.quantity

    risk_check = await check_risk_limits(session, position_cost, current_user, portfolio_type="manual")
    if not risk_check.allowed:
        return {"error": risk_check.reason, "risk_check": {
            "allowed": False,
            "current_exposure": risk_check.current_exposure,
            "daily_pnl": risk_check.daily_pnl,
            "daily_trades": risk_check.daily_trades,
            "circuit_breaker_active": risk_check.circuit_breaker_active,
        }}

    position = PortfolioPosition(
        market_id=req.market_id,
        platform_id=market.platform_id,
        user_id=current_user,
        side=req.side,
        entry_price=req.entry_price,
        quantity=req.quantity,
        entry_time=datetime.utcnow(),
        strategy=req.strategy,
        portfolio_type="manual",
        is_simulated=True,
    )
    session.add(position)
    await session.flush()
    await session.refresh(position)

    # Auto-copy to followers
    copied_ids = await on_position_opened(position, session)

    await session.commit()

    return {
        "id": position.id,
        "market_id": position.market_id,
        "side": position.side,
        "entry_price": position.entry_price,
        "quantity": position.quantity,
        "strategy": position.strategy,
        "copies_created": len(copied_ids),
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
    if position.side == "yes":
        position.realized_pnl = (req.exit_price - position.entry_price) * position.quantity
    else:
        position.realized_pnl = (position.entry_price - req.exit_price) * position.quantity

    # Auto-close copied positions
    closed_ids = await on_position_closed(position, session)

    await session.commit()

    return {
        "id": position.id,
        "realized_pnl": position.realized_pnl,
        "exit_price": position.exit_price,
        "copies_closed": len(closed_ids),
        "message": "Position closed",
    }


@router.get("/portfolio/summary")
async def portfolio_summary(
    portfolio_type: str | None = Query(default=None, pattern="^(manual|auto)$"),
    current_user: str = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Overall portfolio performance summary with optional portfolio_type filter (filtered by current user)."""
    # Open positions
    open_q = select(func.count(PortfolioPosition.id)).where(
        PortfolioPosition.user_id == current_user,
        PortfolioPosition.exit_time == None,  # noqa: E711
    )
    open_q = _apply_portfolio_filter(open_q, portfolio_type)
    open_count = (await session.execute(open_q)).scalar() or 0

    # Closed positions
    closed_q = select(func.count(PortfolioPosition.id)).where(
        PortfolioPosition.user_id == current_user,
        PortfolioPosition.exit_time != None,  # noqa: E711
    )
    closed_q = _apply_portfolio_filter(closed_q, portfolio_type)
    closed_count = (await session.execute(closed_q)).scalar() or 0

    # Total realized P&L
    pnl_q = select(func.sum(PortfolioPosition.realized_pnl)).where(
        PortfolioPosition.user_id == current_user,
        PortfolioPosition.exit_time != None,  # noqa: E711
    )
    pnl_q = _apply_portfolio_filter(pnl_q, portfolio_type)
    total_realized = (await session.execute(pnl_q)).scalar() or 0.0

    # Win rate
    win_q = select(func.count(PortfolioPosition.id)).where(
        PortfolioPosition.user_id == current_user,
        PortfolioPosition.exit_time != None,  # noqa: E711
        PortfolioPosition.realized_pnl > 0,
    )
    win_q = _apply_portfolio_filter(win_q, portfolio_type)
    winning = (await session.execute(win_q)).scalar() or 0

    win_rate = (winning / closed_count * 100) if closed_count > 0 else 0.0

    # P&L by strategy
    strategy_q = select(
        PortfolioPosition.strategy,
        func.count(PortfolioPosition.id),
        func.sum(PortfolioPosition.realized_pnl),
    ).where(
        PortfolioPosition.user_id == current_user,
        PortfolioPosition.exit_time != None,  # noqa: E711
    ).group_by(PortfolioPosition.strategy)
    strategy_q = _apply_portfolio_filter(strategy_q, portfolio_type)
    strategy_pnl_result = await session.execute(strategy_q)

    by_strategy = [
        {
            "strategy": row[0] or "manual",
            "trades": row[1],
            "total_pnl": row[2] or 0.0,
        }
        for row in strategy_pnl_result.all()
    ]

    # Total exposure
    cost_expr = case(
        (PortfolioPosition.side == "yes", PortfolioPosition.entry_price * PortfolioPosition.quantity),
        else_=(1.0 - PortfolioPosition.entry_price) * PortfolioPosition.quantity,
    )
    exposure_q = select(func.sum(cost_expr)).where(
        PortfolioPosition.user_id == current_user,
        PortfolioPosition.exit_time == None,  # noqa: E711
    )
    exposure_q = _apply_portfolio_filter(exposure_q, portfolio_type)
    total_exposure = (await session.execute(exposure_q)).scalar() or 0.0

    # Unrealized P&L for open positions
    open_pos_q = select(PortfolioPosition).where(
        PortfolioPosition.user_id == current_user,
        PortfolioPosition.exit_time == None,  # noqa: E711
    )
    open_pos_q = _apply_portfolio_filter(open_pos_q, portfolio_type)
    open_pos_result = await session.execute(open_pos_q)
    open_positions_list = open_pos_result.scalars().all()

    total_unrealized = 0.0
    for pos in open_positions_list:
        market = await session.get(Market, pos.market_id)
        if market and market.price_yes is not None:
            if pos.side == "yes":
                total_unrealized += (market.price_yes - pos.entry_price) * pos.quantity
            else:
                total_unrealized += (pos.entry_price - market.price_yes) * pos.quantity

    # Sharpe ratio from daily realized P&L
    sharpe_ratio = None
    if closed_count >= 2:
        from collections import defaultdict as _dd
        daily_pnl_map = _dd(float)
        for pos in (await session.execute(
            _apply_portfolio_filter(
                select(PortfolioPosition).where(PortfolioPosition.exit_time != None),  # noqa: E711
                portfolio_type,
            )
        )).scalars().all():
            day_key = pos.exit_time.strftime("%Y-%m-%d")
            daily_pnl_map[day_key] += pos.realized_pnl or 0.0
        if len(daily_pnl_map) >= 2:
            import numpy as np
            daily_returns = list(daily_pnl_map.values())
            mean_r = np.mean(daily_returns)
            std_r = np.std(daily_returns, ddof=1)
            if std_r > 0:
                sharpe_ratio = round(float(mean_r / std_r * (252 ** 0.5)), 2)  # annualized

    return {
        "open_positions": open_count,
        "closed_positions": closed_count,
        "total_realized_pnl": round(total_realized, 2),
        "total_unrealized_pnl": round(total_unrealized, 2),
        "total_pnl": round(total_realized + total_unrealized, 2),
        "win_rate": round(win_rate, 1),
        "total_exposure": round(total_exposure, 2),
        "by_strategy": by_strategy,
        "sharpe_ratio": sharpe_ratio,
    }


@router.get("/portfolio/equity-curve")
async def get_equity_curve(
    time_range: str = Query(default="30d", regex="^(7d|30d|90d|all)$"),
    portfolio_type: str | None = Query(default=None, pattern="^(manual|auto)$"),
    current_user: str = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Get cumulative P&L over time for equity curve visualization (filtered by current user).

    Includes both closed positions (realized P&L) and open positions (unrealized P&L).
    """
    from datetime import timedelta
    from collections import defaultdict

    cutoff_map = {
        "7d": timedelta(days=7),
        "30d": timedelta(days=30),
        "90d": timedelta(days=90),
        "all": None,
    }
    cutoff_delta = cutoff_map[time_range]

    # Get ALL positions (both open and closed) for current user
    query = select(PortfolioPosition).where(PortfolioPosition.user_id == current_user).order_by(PortfolioPosition.entry_time)

    if cutoff_delta:
        cutoff_time = datetime.utcnow() - cutoff_delta
        query = query.where(PortfolioPosition.entry_time >= cutoff_time)

    query = _apply_portfolio_filter(query, portfolio_type)

    result = await session.execute(query)
    positions = result.scalars().all()

    if not positions:
        return {"data": [], "strategies": [], "total_pnl": 0.0}

    # Fetch current market prices for unrealized P&L
    market_prices = {}
    for pos in positions:
        if pos.exit_time is None and pos.market_id not in market_prices:
            market = await session.get(Market, pos.market_id)
            if market and market.price_yes is not None:
                market_prices[pos.market_id] = market.price_yes

    # Build events: entries (timeline anchors) + exits (realized P&L)
    # NO per-position unrealized events - those create the sloped-line bug
    events = []
    trade_events = []  # Discrete trade markers for equity curve overlay
    # Pre-fetch market questions for trade markers
    market_names = {}
    for pos in positions:
        if pos.market_id not in market_names:
            market = await session.get(Market, pos.market_id)
            if market:
                q = market.question or ""
                market_names[pos.market_id] = q[:60] + ("..." if len(q) > 60 else "")

    for pos in positions:
        strategy = pos.strategy or "manual"
        events.append({
            "timestamp": pos.entry_time,
            "strategy": strategy,
            "pnl": 0.0,
            "event_type": "entry",
        })
        # Entry trade marker
        trade_events.append({
            "timestamp": pos.entry_time.isoformat(),
            "type": "entry",
            "side": pos.side,
            "strategy": strategy,
            "market": market_names.get(pos.market_id, ""),
            "price": round(pos.entry_price, 4),
            "quantity": round(pos.quantity, 2),
            "pnl": None,
        })
        if pos.exit_time:
            pnl = pos.realized_pnl or 0.0
            events.append({
                "timestamp": pos.exit_time,
                "strategy": strategy,
                "pnl": pnl,
                "event_type": "exit",
            })
            # Exit trade marker with realized P&L
            trade_events.append({
                "timestamp": pos.exit_time.isoformat(),
                "type": "exit",
                "side": pos.side,
                "strategy": strategy,
                "market": market_names.get(pos.market_id, ""),
                "price": round(pos.exit_price or pos.entry_price, 4),
                "quantity": round(pos.quantity, 2),
                "pnl": round(pnl, 2),
            })

    # Compute total unrealized as SINGLE aggregate (not per-position)
    total_unrealized = 0.0
    unrealized_by_strategy = defaultdict(float)
    for pos in positions:
        if pos.exit_time is None:
            px = market_prices.get(pos.market_id)
            if px is not None:
                if pos.side == "yes":
                    u = (px - pos.entry_price) * pos.quantity
                else:
                    u = (pos.entry_price - px) * pos.quantity
                total_unrealized += u
                unrealized_by_strategy[pos.strategy or "manual"] += u

    # Sort events chronologically
    events.sort(key=lambda e: e["timestamp"])

    # Build cumulative P&L timeline (step function)
    strategy_data = defaultdict(list)
    strategy_cumulative = defaultdict(float)
    all_cumulative = 0.0

    for event in events:
        strategy = event["strategy"]
        timestamp = event["timestamp"].isoformat()

        if event["event_type"] == "exit":
            # Exit events increment cumulative P&L (step up/down)
            strategy_cumulative[strategy] += event["pnl"]
            all_cumulative += event["pnl"]

        # Emit data point for BOTH entry and exit events
        # Entry events anchor the timeline at current cumulative (flat step)
        # Exit events show the new cumulative after realized P&L
        strategy_data[strategy].append({
            "timestamp": timestamp,
            "cumulative_pnl": round(strategy_cumulative[strategy], 2),
        })

        # Add to "all" timeline (deduplicate same timestamp)
        if not strategy_data.get("all") or strategy_data["all"][-1]["timestamp"] != timestamp:
            strategy_data["all"].append({
                "timestamp": timestamp,
                "cumulative_pnl": round(all_cumulative, 2),
            })
        else:
            strategy_data["all"][-1]["cumulative_pnl"] = round(all_cumulative, 2)

    # Build intermediate unrealized P&L points from price snapshots.
    # This makes the equity curve show live P&L movement between trades.
    from db.models import PriceSnapshot
    open_positions = [p for p in positions if p.exit_time is None]
    if open_positions:
        # Get hourly price snapshots for open positions since their entry
        earliest_entry = min(p.entry_time for p in open_positions)
        open_market_ids = list({p.market_id for p in open_positions})
        snap_result = await session.execute(
            select(PriceSnapshot)
            .where(
                PriceSnapshot.market_id.in_(open_market_ids),
                PriceSnapshot.timestamp >= earliest_entry,
                PriceSnapshot.price_yes != None,  # noqa
            )
            .order_by(PriceSnapshot.timestamp)
        )
        snapshots = snap_result.scalars().all()

        # Group snapshots by hour for reasonable granularity
        hourly_prices: dict[str, dict[int, float]] = defaultdict(dict)
        for snap in snapshots:
            hour_key = snap.timestamp.strftime("%Y-%m-%dT%H:00:00")
            hourly_prices[hour_key][snap.market_id] = snap.price_yes

        # For each hourly bucket, compute aggregate unrealized P&L
        realized_at_point = all_cumulative  # realized P&L from closed positions
        for hour_key in sorted(hourly_prices.keys()):
            hour_unrealized = 0.0
            hour_unrealized_by_strat = defaultdict(float)
            price_map = hourly_prices[hour_key]

            for pos in open_positions:
                px = price_map.get(pos.market_id)
                if px is None:
                    continue
                # Only count if the position was already open at this hour
                pos_entry_hour = pos.entry_time.strftime("%Y-%m-%dT%H:00:00")
                if hour_key < pos_entry_hour:
                    continue
                if pos.side == "yes":
                    u = (px - pos.entry_price) * pos.quantity
                else:
                    u = (pos.entry_price - px) * pos.quantity
                hour_unrealized += u
                hour_unrealized_by_strat[pos.strategy or "manual"] += u

            if hour_unrealized != 0.0:
                ts = hour_key
                total_at_hour = realized_at_point + hour_unrealized
                # Add to "all" timeline
                if not strategy_data.get("all") or strategy_data["all"][-1].get("timestamp") != ts:
                    strategy_data["all"].append({
                        "timestamp": ts,
                        "cumulative_pnl": round(total_at_hour, 2),
                    })

    # Append final "now" point with current unrealized P&L
    now_iso = datetime.utcnow().isoformat()
    for strat_name, unrealized in unrealized_by_strategy.items():
        strategy_cumulative[strat_name] += unrealized
        strategy_data[strat_name].append({
            "timestamp": now_iso,
            "cumulative_pnl": round(strategy_cumulative[strat_name], 2),
        })

    all_cumulative += total_unrealized
    if strategy_data.get("all"):
        strategy_data["all"].append({
            "timestamp": now_iso,
            "cumulative_pnl": round(all_cumulative, 2),
        })
    else:
        strategy_data["all"] = [{
            "timestamp": now_iso,
            "cumulative_pnl": round(all_cumulative, 2),
        }]

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
        "total_pnl": round(all_cumulative, 2),
        "trade_events": trade_events,
    }


@router.get("/portfolio/win-rate")
async def get_win_rate_by_strategy(
    min_trades: int = Query(default=1, ge=1),
    portfolio_type: str | None = Query(default=None, pattern="^(manual|auto)$"),
    current_user: str = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Get win rate breakdown by strategy (filtered by current user)."""
    query = select(PortfolioPosition).where(
        PortfolioPosition.user_id == current_user,
        PortfolioPosition.exit_time != None,  # noqa: E711
    )
    query = _apply_portfolio_filter(query, portfolio_type)

    result = await session.execute(query)
    positions = result.scalars().all()

    if not positions:
        return {"strategies": [], "overall_win_rate": 0.0}

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

    strategies = []
    total_wins = 0
    total_trades = 0

    for strategy_name, stats in strategy_stats.items():
        if stats["total_trades"] < min_trades:
            continue

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

    overall_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0.0

    return {
        "strategies": strategies,
        "overall_win_rate": overall_win_rate,
        "total_trades": total_trades,
    }


@router.delete("/portfolio/reset")
async def reset_portfolio(
    session: AsyncSession = Depends(get_session),
):
    """Reset portfolio by deleting ALL positions (all users, open and closed).

    WARNING: This is destructive and cannot be undone.
    """
    result = await session.execute(select(PortfolioPosition))
    positions = result.scalars().all()

    count = len(positions)
    for pos in positions:
        await session.delete(pos)

    await session.commit()

    return {
        "deleted_count": count,
        "message": f"Portfolio reset complete. Deleted {count} positions.",
    }


@router.get("/portfolio/risk-status")
async def get_portfolio_risk_status(
    portfolio_type: str | None = Query(default=None, pattern="^(manual|auto)$"),
    current_user: str = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Get current risk limit utilization for dashboard display (filtered by current user).

    When portfolio_type is omitted, returns both manual and auto risk status.
    """
    return await get_risk_status(session, user_id=current_user, portfolio_type=portfolio_type)
