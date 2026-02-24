"""FastAPI application factory."""

import asyncio
import logging
import warnings
from contextlib import asynccontextmanager

# Silence sklearn deprecation warnings (force_all_finite -> ensure_all_finite)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend

from config.settings import settings
from db.database import init_db, close_db
from data_pipeline.scheduler import run_pipeline_loop, cleanup_realtime_streams

logger = logging.getLogger(__name__)

# Background task reference
_pipeline_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    global _pipeline_task

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Initialize database tables
    logger.info("Initializing database...")
    await init_db()

    # Initialize in-memory cache
    logger.info("Initializing API cache...")
    FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache")

    # Auto-initialize and sync auto-trading configs (survives Railway redeployments)
    logger.info("Initializing default configs...")
    try:
        from db.database import async_session
        from sqlalchemy import select
        from db.models import AutoTradingConfig

        # Profitability-tuned defaults (2026-02-22):
        # - Lower min_confidence/min_net_ev to allow favorable-asymmetry trades through
        # - min_quality_tier="medium" allows more signals (asymmetry filter handles risk)
        # - Kelly stays conservative at 0.02 until edge is proven
        DESIRED_CONFIGS = {
            "ensemble": dict(
                is_enabled=True, bankroll=1000.0,
                min_confidence=0.5, min_net_ev=0.03, max_kelly_fraction=0.02,
                stop_loss_pct=0.05, min_quality_tier="medium", close_on_signal_expiry=True,
                max_position_usd=100.0, max_total_exposure_usd=500.0,
                max_loss_per_day_usd=25.0, max_daily_trades=20,
            ),
            "elo": dict(
                is_enabled=False, bankroll=500.0,
                min_confidence=0.5, min_net_ev=0.03, max_kelly_fraction=0.02,
                stop_loss_pct=0.05, min_quality_tier="medium", close_on_signal_expiry=True,
                max_position_usd=100.0, max_total_exposure_usd=500.0,
                max_loss_per_day_usd=25.0, max_daily_trades=20,
            ),
            "favorite_longshot": dict(
                is_enabled=False, bankroll=500.0,
                min_confidence=0.5, min_net_ev=0.03, max_kelly_fraction=0.03,
                stop_loss_pct=0.05, min_quality_tier="medium", close_on_signal_expiry=True,
                max_position_usd=100.0, max_total_exposure_usd=500.0,
                max_loss_per_day_usd=25.0, max_daily_trades=20,
            ),
        }

        # Migration fields: pushed to DB on every restart so code is source of truth.
        # min_net_ev lowered 0.05→0.03: calibration-only model generates 3-5% edges;
        # 5% threshold was blocking ALL signals since scanner creates at >3%.
        MIGRATION_FIELDS = {"min_confidence", "min_net_ev", "min_quality_tier"}

        async with async_session() as session:
            result = await session.execute(select(AutoTradingConfig))
            existing = {c.strategy: c for c in result.scalars().all()}

            for strategy, defaults in DESIRED_CONFIGS.items():
                if strategy not in existing:
                    session.add(AutoTradingConfig(strategy=strategy, **defaults))
                    logger.info(f"Created auto-trading config: {strategy}")
                else:
                    # Force-sync MIGRATION_FIELDS on every restart so code is the source of truth.
                    # Other fields (stop_loss_pct, bankroll, etc.) remain user-editable via UI.
                    cfg = existing[strategy]
                    for field in MIGRATION_FIELDS:
                        if field in defaults:
                            setattr(cfg, field, defaults[field])
                    logger.info(f"Synced profitability fields for: {strategy}")

            await session.commit()
            logger.info("Auto-trading configs initialized/synced")
    except Exception as e:
        logger.error(f"Failed to init auto-trading configs: {e}")

    # Start background data pipeline
    logger.info("Starting background data pipeline...")
    _pipeline_task = asyncio.create_task(run_pipeline_loop())

    yield

    # Shutdown
    logger.info("Shutting down...")

    # Close WebSocket streams
    try:
        await cleanup_realtime_streams()
    except Exception as e:
        logger.error(f"Stream cleanup error: {e}")

    # Cancel pipeline task
    if _pipeline_task:
        _pipeline_task.cancel()
        try:
            await _pipeline_task
        except asyncio.CancelledError:
            pass

    await close_db()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Prediction Market Analysis Platform",
        description="Math-first prediction market analysis with arbitrage detection, ML models, and AI analysis",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_url, "http://localhost:5173", "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    from api.routes import markets, arbitrage, system, ml_predictions, ai_analysis, portfolio, analytics, copy_trading, elo_sports, strategy_signals, auto_trading, admin
    app.include_router(markets.router, prefix="/api/v1")
    app.include_router(arbitrage.router, prefix="/api/v1")
    app.include_router(system.router, prefix="/api/v1")
    app.include_router(ml_predictions.router, prefix="/api/v1")
    app.include_router(ai_analysis.router, prefix="/api/v1")
    app.include_router(portfolio.router, prefix="/api/v1")
    app.include_router(analytics.router, prefix="/api/v1")
    app.include_router(copy_trading.router, prefix="/api/v1")
    app.include_router(elo_sports.router, prefix="/api/v1")
    app.include_router(strategy_signals.router, prefix="/api/v1")
    app.include_router(auto_trading.router, prefix="/api/v1")
    app.include_router(admin.router, prefix="/api/v1")

    return app


app = create_app()
