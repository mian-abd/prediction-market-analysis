"""FastAPI application factory."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    from api.routes import markets, arbitrage, system, ml_predictions, ai_analysis, portfolio
    app.include_router(markets.router, prefix="/api/v1")
    app.include_router(arbitrage.router, prefix="/api/v1")
    app.include_router(system.router, prefix="/api/v1")
    app.include_router(ml_predictions.router, prefix="/api/v1")
    app.include_router(ai_analysis.router, prefix="/api/v1")
    app.include_router(portfolio.router, prefix="/api/v1")

    return app


app = create_app()
