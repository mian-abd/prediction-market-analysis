"""Admin endpoints for database management (USE WITH CAUTION)."""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy import update, text
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

            # Calculate realized P&L
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
        return {
            "success": False,
            "error": str(e)
        }


@router.post("/update-risk-params")
async def update_risk_params(session: AsyncSession = Depends(get_session)):
    """Update risk parameters for demo (wider stops, lower confidence).

    Changes:
    - Stop-loss: 15% → 25%
    - Min confidence: 0.7 → 0.5
    - Min quality tier: 'high' → 'medium'
    """
    try:
        result = await session.execute(
            update(AutoTradingConfig)
            .where(AutoTradingConfig.strategy == 'ensemble')
            .values(
                stop_loss_pct=0.25,
                min_confidence=0.5,
                min_quality_tier='medium',
                close_on_signal_expiry=True
            )
        )

        await session.commit()

        # Verify
        verify_result = await session.execute(
            text("SELECT stop_loss_pct, min_confidence, min_quality_tier FROM auto_trading_configs WHERE strategy = 'ensemble'")
        )
        config = verify_result.fetchone()

        logger.info(f"Risk params updated: stop_loss={config[0]}, min_confidence={config[1]}, quality_tier={config[2]}")

        return {
            "success": True,
            "updated_rows": result.rowcount,
            "new_config": {
                "stop_loss_pct": config[0],
                "min_confidence": config[1],
                "min_quality_tier": config[2]
            },
            "message": "Risk parameters updated successfully"
        }

    except Exception as e:
        await session.rollback()
        logger.error(f"Risk params update failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }
