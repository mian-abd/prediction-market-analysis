"""Admin endpoints for database management (USE WITH CAUTION)."""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy import update, text, select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_session, async_session
from db.models import (
    PortfolioPosition, EnsembleEdgeSignal, EloEdgeSignal, FavoriteLongshotEdgeSignal,
    AutoTradingConfig, TraderProfile,
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

        # Archive old favorite-longshot signals
        fl_result = await session.execute(
            update(FavoriteLongshotEdgeSignal)
            .where(FavoriteLongshotEdgeSignal.expired_at == None)
            .values(expired_at=datetime.utcnow())
        )
        fl_archived = fl_result.rowcount

        await session.commit()

        logger.info(
            f"Portfolio reset: {closed_count} positions closed, "
            f"{ensemble_archived} ensemble, {elo_archived} elo, {fl_archived} favorite-longshot signals archived"
        )

        return {
            "success": True,
            "positions_closed": closed_count,
            "ensemble_signals_archived": ensemble_archived,
            "elo_signals_archived": elo_archived,
            "favorite_longshot_signals_archived": fl_archived,
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


@router.post("/backfill-traders")
async def backfill_traders(replace: bool = False):
    """Fetch real trader profiles from Polymarket leaderboard and populate DB.

    This enables the Copy Trading page. Safe to call multiple times.
    Use ?replace=true to refresh all trader data.
    """
    try:
        from data_pipeline.collectors.trader_data import fetch_polymarket_leaderboard

        async with async_session() as session:
            # Check existing traders
            result = await session.execute(select(TraderProfile))
            existing = result.scalars().all()

            if existing and not replace:
                return {
                    "success": True,
                    "message": f"Already have {len(existing)} traders. Use ?replace=true to refresh.",
                    "trader_count": len(existing),
                }

            if replace and existing:
                await session.execute(delete(TraderProfile))
                await session.commit()
                logger.info(f"Deleted {len(existing)} existing traders")

            # Fetch from multiple Polymarket leaderboard windows
            all_traders_data = []

            for time_period, order_by, limit in [
                ("MONTH", "PNL", 50),
                ("MONTH", "VOL", 30),
                ("WEEK", "PNL", 20),
            ]:
                try:
                    traders = await fetch_polymarket_leaderboard(
                        time_period=time_period,
                        limit=limit,
                        order_by=order_by,
                        category="OVERALL",
                    )
                    all_traders_data.extend(traders)
                except Exception as e:
                    logger.warning(f"Failed to fetch {time_period}/{order_by}: {e}")

            # Deduplicate by wallet
            seen_wallets = set()
            unique_traders = []
            for trader in all_traders_data:
                wallet = trader.get("proxyWallet")
                if wallet and wallet not in seen_wallets:
                    seen_wallets.add(wallet)
                    unique_traders.append(trader)

            # Use real trade data for stats
            from data_pipeline.collectors.trader_data import (
                fetch_trader_positions, calculate_trader_stats, generate_trader_bio,
                clean_display_name,
            )
            import asyncio as _asyncio

            created_count = 0
            for trader_data in unique_traders:
                wallet = trader_data.get("proxyWallet")
                if not wallet:
                    continue

                display_name = clean_display_name(trader_data)

                # Fetch real position history
                positions = await fetch_trader_positions(wallet, limit=100)
                await _asyncio.sleep(0.2)  # Rate limit

                if positions:
                    stats = calculate_trader_stats(trader_data, positions)
                else:
                    pnl = float(trader_data.get("pnl", 0))
                    volume = float(trader_data.get("vol", 0))
                    stats = {
                        "total_pnl": pnl,
                        "roi_pct": (pnl / max(volume * 0.3, 1)) * 100 if volume > 0 else 0,
                        "win_rate": 0.0, "total_trades": 0, "winning_trades": 0,
                        "avg_trade_duration_hrs": 0.0, "risk_score": 5, "max_drawdown": 0.0,
                    }

                bio = generate_trader_bio(trader_data, stats)

                profile = TraderProfile(
                    user_id=wallet,
                    display_name=display_name,
                    bio=bio,
                    total_pnl=stats["total_pnl"],
                    roi_pct=stats["roi_pct"],
                    win_rate=stats["win_rate"],
                    total_trades=stats["total_trades"],
                    winning_trades=stats["winning_trades"],
                    avg_trade_duration_hrs=stats["avg_trade_duration_hrs"],
                    risk_score=stats["risk_score"],
                    max_drawdown=stats["max_drawdown"],
                    follower_count=0,
                    is_public=True,
                    accepts_copiers=True,
                )
                session.add(profile)
                created_count += 1

                if created_count % 10 == 0:
                    await session.commit()

            await session.commit()
            logger.info(f"Backfilled {created_count} trader profiles (real trade data)")

            return {
                "success": True,
                "traders_created": created_count,
                "total_fetched": len(all_traders_data),
                "unique_wallets": len(unique_traders),
                "message": f"Successfully created {created_count} trader profiles from Polymarket leaderboard.",
            }

    except Exception as e:
        logger.error(f"Trader backfill failed: {e}")
        return {"success": False, "error": str(e)}
