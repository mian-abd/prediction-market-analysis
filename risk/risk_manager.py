"""Pre-trade risk limit enforcement.

Checks portfolio-level constraints before allowing new positions:
- Max position size (single trade)
- Max total exposure (all open positions)
- Max daily loss (circuit breaker)
- Max daily trade count

Supports per-portfolio isolation (manual vs auto) via portfolio_type parameter.
"""

import logging
from datetime import datetime
from dataclasses import dataclass

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy import case

from config.settings import settings
from db.models import PortfolioPosition, AutoTradingConfig, Market

logger = logging.getLogger(__name__)


@dataclass
class RiskCheckResult:
    allowed: bool
    reason: str = ""
    current_exposure: float = 0.0
    daily_pnl: float = 0.0
    daily_trades: int = 0
    circuit_breaker_active: bool = False


async def _get_portfolio_limits(session, portfolio_type: str | None, strategy: str | None = None) -> dict:
    """Load risk limits for a portfolio type from AutoTradingConfig or global settings.

    When strategy is provided (e.g., "auto_ensemble"), uses that strategy's specific limits.
    When strategy is None and portfolio_type is "auto", uses the most restrictive position limit
    and each strategy's own exposure cap (summed for total portfolio budget).
    """
    if portfolio_type == "auto":
        configs = (await session.execute(select(AutoTradingConfig))).scalars().all()
        if configs:
            # If a specific strategy is requested, use its limits directly
            if strategy:
                strat_name = strategy.replace("auto_", "")
                for c in configs:
                    if c.strategy == strat_name:
                        return {
                            "max_position_usd": c.max_position_usd or 100.0,
                            "max_total_exposure_usd": c.max_total_exposure_usd or 500.0,
                            "max_loss_per_day_usd": c.max_loss_per_day_usd or 25.0,
                            "max_daily_trades": c.max_daily_trades or 20,
                        }
            # Aggregate: use min for position size (most conservative), sum for budgets
            return {
                "max_position_usd": min((c.max_position_usd or 100.0) for c in configs),
                "max_total_exposure_usd": sum((c.max_total_exposure_usd or 500.0) for c in configs),
                "max_loss_per_day_usd": sum((c.max_loss_per_day_usd or 25.0) for c in configs),
                "max_daily_trades": sum((c.max_daily_trades or 20) for c in configs),
            }
    return {
        "max_position_usd": settings.max_position_usd,
        "max_total_exposure_usd": settings.max_total_exposure_usd,
        "max_loss_per_day_usd": settings.max_loss_per_day_usd,
        "max_daily_trades": settings.max_daily_trades,
    }


def _cost_expr():
    """SQL expression for actual position cost (side-aware)."""
    return case(
        (PortfolioPosition.side == "yes", PortfolioPosition.entry_price * PortfolioPosition.quantity),
        else_=(1.0 - PortfolioPosition.entry_price) * PortfolioPosition.quantity,
    )


def _apply_portfolio_filter(query, portfolio_type: str | None):
    """Add portfolio_type WHERE clause if specified."""
    if portfolio_type is not None:
        return query.where(PortfolioPosition.portfolio_type == portfolio_type)
    return query


async def check_risk_limits(
    session: AsyncSession,
    position_cost: float,
    user_id: str = "anonymous",
    portfolio_type: str | None = None,
    strategy: str | None = None,
) -> RiskCheckResult:
    """Check all risk limits before opening a new position.

    Args:
        session: Active database session
        position_cost: Actual capital deployed for the proposed trade
        user_id: User ID for per-user limits
        portfolio_type: "manual", "auto", or None (global)
        strategy: Optional strategy name (e.g., "auto_ensemble") for per-strategy limits

    Returns:
        RiskCheckResult with allowed=True if all limits pass
    """
    limits = await _get_portfolio_limits(session, portfolio_type, strategy=strategy)

    # 1. Single position size check
    if position_cost > limits["max_position_usd"]:
        return RiskCheckResult(
            allowed=False,
            reason=f"Position size ${position_cost:.2f} exceeds limit ${limits['max_position_usd']:.2f}",
        )

    # 2. Total exposure check — mark-to-market (uses current prices, not entry prices)
    open_pos_query = (
        select(PortfolioPosition)
        .where(PortfolioPosition.exit_time == None)  # noqa: E711
    )
    open_pos_query = _apply_portfolio_filter(open_pos_query, portfolio_type)
    open_positions = (await session.execute(open_pos_query)).scalars().all()

    total_exposure = 0.0
    for pos in open_positions:
        market = await session.get(Market, pos.market_id)
        if market and market.price_yes is not None:
            # Mark-to-market: current value of position
            if pos.side == "yes":
                total_exposure += market.price_yes * pos.quantity
            else:
                total_exposure += (1.0 - market.price_yes) * pos.quantity
        else:
            # Fallback to entry cost if market price unavailable
            if pos.side == "yes":
                total_exposure += pos.entry_price * pos.quantity
            else:
                total_exposure += (1.0 - pos.entry_price) * pos.quantity

    if total_exposure + position_cost > limits["max_total_exposure_usd"]:
        return RiskCheckResult(
            allowed=False,
            reason=(
                f"Total exposure ${total_exposure + position_cost:.2f} "
                f"would exceed limit ${limits['max_total_exposure_usd']:.2f}"
            ),
            current_exposure=total_exposure,
        )

    # 3. Daily P&L check (circuit breaker)
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    pnl_query = (
        select(func.sum(PortfolioPosition.realized_pnl))
        .where(
            PortfolioPosition.exit_time != None,  # noqa: E711
            PortfolioPosition.exit_time >= today_start,
        )
    )
    pnl_query = _apply_portfolio_filter(pnl_query, portfolio_type)
    daily_pnl = (await session.execute(pnl_query)).scalar() or 0.0

    if daily_pnl < -limits["max_loss_per_day_usd"]:
        logger.warning(
            f"CIRCUIT BREAKER: daily P&L ${daily_pnl:.2f} "
            f"exceeds max loss ${limits['max_loss_per_day_usd']:.2f}"
        )
        return RiskCheckResult(
            allowed=False,
            reason=(
                f"Daily loss circuit breaker: P&L ${daily_pnl:.2f} "
                f"exceeds limit -${limits['max_loss_per_day_usd']:.2f}"
            ),
            current_exposure=total_exposure,
            daily_pnl=daily_pnl,
            circuit_breaker_active=True,
        )

    # 4. Portfolio-level stop-loss (prediction markets version: -5% of exposure)
    # If total unrealized loss is > 5% of open exposure, stop opening new positions
    # Reuse open_positions from mark-to-market check above
    total_unrealized = 0.0
    for pos in open_positions:
        market = await session.get(Market, pos.market_id)
        if market and market.price_yes is not None:
            if pos.side == "yes":
                unrealized = (market.price_yes - pos.entry_price) * pos.quantity
            else:
                unrealized = (pos.entry_price - market.price_yes) * pos.quantity
            total_unrealized += unrealized

    if total_exposure > 0 and total_unrealized < -0.05 * total_exposure:
        logger.warning(
            f"PORTFOLIO STOP-LOSS: unrealized loss ${total_unrealized:.2f} "
            f"exceeds -5% of exposure ${total_exposure:.2f}"
        )
        return RiskCheckResult(
            allowed=False,
            reason=(
                f"Portfolio stop-loss triggered: unrealized loss ${total_unrealized:.2f} "
                f"exceeds -5% threshold"
            ),
            current_exposure=total_exposure,
            daily_pnl=daily_pnl,
            circuit_breaker_active=True,
        )

    # 5. Daily trade count check
    trades_query = (
        select(func.count(PortfolioPosition.id))
        .where(PortfolioPosition.entry_time >= today_start)
    )
    trades_query = _apply_portfolio_filter(trades_query, portfolio_type)
    daily_trades = (await session.execute(trades_query)).scalar() or 0

    if daily_trades >= limits["max_daily_trades"]:
        return RiskCheckResult(
            allowed=False,
            reason=f"Daily trade limit reached: {daily_trades}/{limits['max_daily_trades']}",
            current_exposure=total_exposure,
            daily_pnl=daily_pnl,
            daily_trades=daily_trades,
        )

    return RiskCheckResult(
        allowed=True,
        current_exposure=total_exposure,
        daily_pnl=daily_pnl,
        daily_trades=daily_trades,
    )


async def get_risk_status(session: AsyncSession, portfolio_type: str | None = None) -> dict:
    """Get current risk status for dashboard display.

    When portfolio_type is None, returns both portfolios in a combined response.
    When specified, returns single-portfolio format (backward compatible).
    """
    if portfolio_type is None:
        # Return both portfolios
        manual_status = await _get_single_risk_status(session, "manual")
        auto_status = await _get_single_risk_status(session, "auto")
        return {"manual": manual_status, "auto": auto_status}

    return await _get_single_risk_status(session, portfolio_type)


async def _get_single_risk_status(session: AsyncSession, portfolio_type: str) -> dict:
    """Get risk status for a single portfolio type."""
    limits = await _get_portfolio_limits(session, portfolio_type)
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    exposure_query = (
        select(func.sum(_cost_expr()))
        .where(PortfolioPosition.exit_time == None)  # noqa: E711
    )
    exposure_query = _apply_portfolio_filter(exposure_query, portfolio_type)
    total_exposure = (await session.execute(exposure_query)).scalar() or 0.0

    pnl_query = (
        select(func.sum(PortfolioPosition.realized_pnl))
        .where(
            PortfolioPosition.exit_time != None,  # noqa: E711
            PortfolioPosition.exit_time >= today_start,
        )
    )
    pnl_query = _apply_portfolio_filter(pnl_query, portfolio_type)
    daily_pnl = (await session.execute(pnl_query)).scalar() or 0.0

    trades_query = (
        select(func.count(PortfolioPosition.id))
        .where(PortfolioPosition.entry_time >= today_start)
    )
    trades_query = _apply_portfolio_filter(trades_query, portfolio_type)
    daily_trades = (await session.execute(trades_query)).scalar() or 0

    open_query = (
        select(func.count(PortfolioPosition.id))
        .where(PortfolioPosition.exit_time == None)  # noqa: E711
    )
    open_query = _apply_portfolio_filter(open_query, portfolio_type)
    open_positions = (await session.execute(open_query)).scalar() or 0

    max_exposure = limits["max_total_exposure_usd"]
    max_loss = limits["max_loss_per_day_usd"]
    max_trades = limits["max_daily_trades"]

    exposure_pct = (total_exposure / max_exposure * 100) if max_exposure > 0 else 0
    loss_pct = (abs(daily_pnl) / max_loss * 100) if daily_pnl < 0 and max_loss > 0 else 0
    trades_pct = (daily_trades / max_trades * 100) if max_trades > 0 else 0

    circuit_breaker = daily_pnl < -max_loss

    return {
        "exposure": {
            "current": total_exposure,
            "limit": max_exposure,
            "utilization_pct": round(exposure_pct, 1),
        },
        "daily_pnl": {
            "current": daily_pnl,
            "limit": -max_loss,
            "utilization_pct": round(loss_pct, 1),
        },
        "daily_trades": {
            "current": daily_trades,
            "limit": max_trades,
            "utilization_pct": round(trades_pct, 1),
        },
        "max_position_usd": limits["max_position_usd"],
        "open_positions": open_positions,
        "circuit_breaker_active": circuit_breaker,
    }
