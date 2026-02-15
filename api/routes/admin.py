"""Admin endpoints for database management (USE WITH CAUTION)."""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy import update, text, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_session
from db.models import (
    PortfolioPosition, EnsembleEdgeSignal, EloEdgeSignal, AutoTradingConfig
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/reset-auto-portfolio")
async def reset_auto_portfolio(session: AsyncSession = Depends(get_session)):
    """Reset all auto trading positions and signals (fresh start).

    WARNING: This will close all open auto positions and archive all signals.
    Only use this when you want to start completely fresh.
    """
    try:
        # Close all open auto positions
        closed_count = 0
        result = await session.execute(
            text("""
                SELECT pp.id, pp.market_id, pp.side, pp.entry_price, pp.quantity, m.price_yes
                FROM portfolio_positions pp
                JOIN markets m ON m.id = pp.market_id
                WHERE pp.portfolio_type = 'auto'
                  AND pp.exit_time IS NULL
                  AND m.price_yes IS NOT NULL
            """)
        )
        positions = result.fetchall()

        for pos in positions:
            pos_id, market_id, side, entry_price, quantity, price_yes = pos

            if side == 'yes':
                realized_pnl = (price_yes - entry_price) * quantity
            else:
                realized_pnl = (entry_price - price_yes) * quantity

            await session.execute(
                update(PortfolioPosition)
                .where(PortfolioPosition.id == pos_id)
                .values(
                    exit_time=datetime.utcnow(),
                    exit_price=price_yes,
                    realized_pnl=realized_pnl
                )
            )
            closed_count += 1

        # Archive old ensemble signals
        ensemble_result = await session.execute(
            update(EnsembleEdgeSignal)
            .where(EnsembleEdgeSignal.expired_at == None)
            .values(expired_at=datetime.utcnow())
        )
        ensemble_archived = ensemble_result.rowcount

        # Archive old elo signals
        elo_result = await session.execute(
            update(EloEdgeSignal)
            .where(EloEdgeSignal.expired_at == None)
            .values(expired_at=datetime.utcnow())
        )
        elo_archived = elo_result.rowcount

        await session.commit()

        logger.info(f"Portfolio reset: {closed_count} positions closed, {ensemble_archived} ensemble signals archived, {elo_archived} elo signals archived")

        return {
            "success": True,
            "positions_closed": closed_count,
            "ensemble_signals_archived": ensemble_archived,
            "elo_signals_archived": elo_archived,
            "message": "Auto portfolio reset complete. All positions closed and signals archived."
        }

    except Exception as e:
        await session.rollback()
        logger.error(f"Portfolio reset failed: {e}")
        return {"success": False, "error": str(e)}


@router.post("/init-auto-trading")
async def init_auto_trading(session: AsyncSession = Depends(get_session)):
    """Initialize auto-trading configs if they don't exist.

    Creates default ensemble and elo configs with demo-optimized parameters.
    Safe to call multiple times (idempotent).
    """
    try:
        created = []

        # Check if ensemble config exists
        result = await session.execute(
            select(AutoTradingConfig).where(AutoTradingConfig.strategy == "ensemble")
        )
        ensemble_config = result.scalar_one_or_none()

        if not ensemble_config:
            ensemble_config = AutoTradingConfig(
                strategy="ensemble",
                is_enabled=True,
                bankroll=1000.0,
                min_confidence=0.5,
                min_net_ev=0.05,
                max_kelly_fraction=0.02,
                stop_loss_pct=0.25,
                min_quality_tier="medium",
                close_on_signal_expiry=True,
            )
            session.add(ensemble_config)
            created.append("ensemble")
            logger.info("Created ensemble auto-trading config")
        else:
            # Update existing to demo-optimized values
            ensemble_config.is_enabled = True
            ensemble_config.min_confidence = 0.5
            ensemble_config.stop_loss_pct = 0.25
            ensemble_config.min_quality_tier = "medium"
            created.append("ensemble (updated)")

        # Check if elo config exists
        result = await session.execute(
            select(AutoTradingConfig).where(AutoTradingConfig.strategy == "elo")
        )
        elo_config = result.scalar_one_or_none()

        if not elo_config:
            elo_config = AutoTradingConfig(
                strategy="elo",
                is_enabled=False,
                bankroll=500.0,
                min_confidence=0.5,
                min_net_ev=0.03,
                max_kelly_fraction=0.02,
                stop_loss_pct=0.15,
                min_quality_tier="medium",
                close_on_signal_expiry=True,
            )
            session.add(elo_config)
            created.append("elo")
            logger.info("Created elo auto-trading config")

        await session.commit()

        # Return current state
        all_configs = await session.execute(select(AutoTradingConfig))
        configs = all_configs.scalars().all()

        return {
            "success": True,
            "created_or_updated": created,
            "configs": [
                {
                    "strategy": c.strategy,
                    "is_enabled": c.is_enabled,
                    "bankroll": c.bankroll,
                    "min_confidence": c.min_confidence,
                    "min_net_ev": c.min_net_ev,
                    "max_kelly_fraction": c.max_kelly_fraction,
                    "stop_loss_pct": c.stop_loss_pct,
                    "min_quality_tier": c.min_quality_tier,
                }
                for c in configs
            ]
        }

    except Exception as e:
        await session.rollback()
        logger.error(f"Init auto-trading failed: {e}")
        return {"success": False, "error": str(e)}
