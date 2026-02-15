"""Auto-trading configuration endpoints.

CRUD for per-strategy auto-trading configs + live status.
"""

from datetime import datetime
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, func, case
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_session
from db.models import AutoTradingConfig, PortfolioPosition

router = APIRouter(tags=["auto-trading"])


class AutoTradingConfigUpdate(BaseModel):
    """Pydantic model for config updates with validation."""
    is_enabled: bool | None = None
    min_quality_tier: str | None = None
    min_confidence: float | None = Field(default=None, ge=0, le=1)
    min_net_ev: float | None = Field(default=None, ge=0)
    bankroll: float | None = Field(default=None, gt=0)
    max_kelly_fraction: float | None = Field(default=None, gt=0, le=1)
    max_position_usd: float | None = Field(default=None, gt=0)
    max_total_exposure_usd: float | None = Field(default=None, gt=0)
    max_loss_per_day_usd: float | None = Field(default=None, gt=0)
    max_daily_trades: int | None = Field(default=None, ge=0)
    stop_loss_pct: float | None = Field(default=None, gt=0, le=1)
    close_on_signal_expiry: bool | None = None


def _config_to_dict(config: AutoTradingConfig) -> dict:
    return {
        "strategy": config.strategy,
        "is_enabled": config.is_enabled,
        "min_quality_tier": config.min_quality_tier,
        "min_confidence": config.min_confidence,
        "min_net_ev": config.min_net_ev,
        "bankroll": config.bankroll,
        "max_kelly_fraction": config.max_kelly_fraction,
        "max_position_usd": config.max_position_usd,
        "max_total_exposure_usd": config.max_total_exposure_usd,
        "max_loss_per_day_usd": config.max_loss_per_day_usd,
        "max_daily_trades": config.max_daily_trades,
        "stop_loss_pct": config.stop_loss_pct,
        "close_on_signal_expiry": config.close_on_signal_expiry,
        "updated_at": config.updated_at.isoformat() if config.updated_at else None,
    }


@router.get("/auto-trading/config")
async def get_all_configs(session: AsyncSession = Depends(get_session)):
    """Return all strategy configs."""
    result = await session.execute(
        select(AutoTradingConfig).order_by(AutoTradingConfig.strategy)
    )
    configs = result.scalars().all()
    return {"configs": [_config_to_dict(c) for c in configs]}


@router.put("/auto-trading/config/{strategy}")
async def update_config(
    strategy: str,
    update: AutoTradingConfigUpdate,
    session: AsyncSession = Depends(get_session),
):
    """Update config for a specific strategy."""
    result = await session.execute(
        select(AutoTradingConfig).where(AutoTradingConfig.strategy == strategy)
    )
    config = result.scalar_one_or_none()
    if not config:
        return {"error": f"Strategy '{strategy}' not found"}

    update_data = update.model_dump(exclude_none=True)
    for field, value in update_data.items():
        setattr(config, field, value)
    config.updated_at = datetime.utcnow()

    await session.commit()
    await session.refresh(config)

    return {"config": _config_to_dict(config), "message": f"Config for '{strategy}' updated"}


@router.post("/auto-trading/toggle/{strategy}")
async def toggle_strategy(
    strategy: str,
    enabled: bool = Query(...),
    session: AsyncSession = Depends(get_session),
):
    """Quick enable/disable toggle for a strategy."""
    result = await session.execute(
        select(AutoTradingConfig).where(AutoTradingConfig.strategy == strategy)
    )
    config = result.scalar_one_or_none()
    if not config:
        return {"error": f"Strategy '{strategy}' not found"}

    config.is_enabled = enabled
    config.updated_at = datetime.utcnow()
    await session.commit()

    return {"strategy": strategy, "is_enabled": enabled, "message": f"{'Enabled' if enabled else 'Disabled'} {strategy}"}


@router.get("/auto-trading/status")
async def get_auto_trading_status(session: AsyncSession = Depends(get_session)):
    """Live status: enabled strategies, open auto positions, today's auto P&L, recent trades."""
    # Configs
    configs = (await session.execute(
        select(AutoTradingConfig).order_by(AutoTradingConfig.strategy)
    )).scalars().all()

    enabled_strategies = [c.strategy for c in configs if c.is_enabled]

    # Open auto positions count by strategy
    open_by_strategy = {}
    for strat in ["auto_ensemble", "auto_elo"]:
        count = (await session.execute(
            select(func.count(PortfolioPosition.id))
            .where(
                PortfolioPosition.portfolio_type == "auto",
                PortfolioPosition.strategy == strat,
                PortfolioPosition.exit_time == None,  # noqa: E711
            )
        )).scalar() or 0
        open_by_strategy[strat] = count

    # Per-strategy P&L and exposure
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    cost_expr = case(
        (PortfolioPosition.side == "yes", PortfolioPosition.entry_price * PortfolioPosition.quantity),
        else_=(1.0 - PortfolioPosition.entry_price) * PortfolioPosition.quantity,
    )

    pnl_by_strategy = {}
    exposure_by_strategy = {}
    for strat in ["auto_ensemble", "auto_elo"]:
        pnl = (await session.execute(
            select(func.sum(PortfolioPosition.realized_pnl))
            .where(
                PortfolioPosition.portfolio_type == "auto",
                PortfolioPosition.strategy == strat,
                PortfolioPosition.exit_time != None,  # noqa: E711
                PortfolioPosition.exit_time >= today_start,
            )
        )).scalar() or 0.0
        pnl_by_strategy[strat] = pnl

        exp = (await session.execute(
            select(func.sum(cost_expr))
            .where(
                PortfolioPosition.portfolio_type == "auto",
                PortfolioPosition.strategy == strat,
                PortfolioPosition.exit_time == None,  # noqa: E711
            )
        )).scalar() or 0.0
        exposure_by_strategy[strat] = exp

    today_pnl = sum(pnl_by_strategy.values())
    total_exposure = sum(exposure_by_strategy.values())

    # Recent auto trades (last 10)
    recent = (await session.execute(
        select(PortfolioPosition)
        .where(PortfolioPosition.portfolio_type == "auto")
        .order_by(PortfolioPosition.entry_time.desc())
        .limit(10)
    )).scalars().all()

    recent_trades = [{
        "id": p.id,
        "market_id": p.market_id,
        "strategy": p.strategy,
        "side": p.side,
        "entry_price": p.entry_price,
        "quantity": p.quantity,
        "entry_time": p.entry_time.isoformat(),
        "exit_time": p.exit_time.isoformat() if p.exit_time else None,
        "realized_pnl": p.realized_pnl,
    } for p in recent]

    return {
        "enabled_strategies": enabled_strategies,
        "configs": [_config_to_dict(c) for c in configs],
        "open_positions": open_by_strategy,
        "total_exposure": total_exposure,
        "today_pnl": today_pnl,
        "pnl_by_strategy": pnl_by_strategy,
        "exposure_by_strategy": exposure_by_strategy,
        "recent_trades": recent_trades,
    }
