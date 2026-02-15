"""FastAPI application factory."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend

from config.settings import settings
from db.database import init_db, close_db
from data_pipeline.scheduler import run_pipeline_loop

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

    # Auto-initialize default configs (survives Railway redeployments)
    logger.info("Initializing default configs...")
    try:
        from db.database import async_session
        from sqlalchemy import select
        from db.models import AutoTradingConfig

        async with async_session() as session:
            result = await session.execute(select(AutoTradingConfig))
            existing = result.scalars().all()
            if not existing:
                session.add(AutoTradingConfig(
                    strategy="ensemble", is_enabled=True, bankroll=1000.0,
                    min_confidence=0.5, min_net_ev=0.05, max_kelly_fraction=0.02,
                    stop_loss_pct=0.25, min_quality_tier="medium", close_on_signal_expiry=True,
                ))
                session.add(AutoTradingConfig(
                    strategy="elo", is_enabled=False, bankroll=500.0,
                    min_confidence=0.5, min_net_ev=0.03, max_kelly_fraction=0.02,
                    stop_loss_pct=0.15, min_quality_tier="medium", close_on_signal_expiry=True,
                ))
                await session.commit()
                logger.info("Created default auto-trading configs (ensemble enabled, elo disabled)")
            else:
                logger.info(f"Auto-trading configs already exist ({len(existing)} strategies)")
    except Exception as e:
        logger.error(f"Failed to init auto-trading configs: {e}")

    # Start background data pipeline
    logger.info("Starting background data pipeline...")
    _pipeline_task = asyncio.create_task(run_pipeline_loop())

    yield

    # Shutdown
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
