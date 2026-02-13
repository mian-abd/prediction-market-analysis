"""Backfill historical resolved markets for ML training.

Run once to populate database with outcomes for training XGBoost/LightGBM.
Uses Kalshi API which provides clear outcome data.
"""

import sys
from pathlib import Path

# Add project root to path so we can import db, data_pipeline, etc.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import logging
from datetime import datetime

from db.database import async_session, init_db
from data_pipeline.storage import ensure_platforms, upsert_markets
from data_pipeline.collectors.kalshi_resolved import (
    fetch_all_resolved_markets,
    parse_resolved_market,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


async def backfill_kalshi_resolved(max_markets: int = 1000):
    """Fetch and store resolved Kalshi markets (has outcome data!)."""
    async with async_session() as session:
        platforms = await ensure_platforms(session)

        logger.info("Fetching settled markets from Kalshi...")
        raw_markets = await fetch_all_resolved_markets(max_markets=max_markets)

        logger.info(f"Parsing {len(raw_markets)} settled markets...")
        parsed_markets = [parse_resolved_market(m) for m in raw_markets]

        # Filter out markets without resolution values
        valid_markets = [m for m in parsed_markets if m.get("resolution_value") is not None]
        logger.info(f"Found {len(valid_markets)} markets with valid outcomes")

        logger.info("Storing in database...")
        count = await upsert_markets(session, valid_markets, platforms["kalshi"])

        logger.info(f"✅ Backfilled {count} resolved Kalshi markets")
        return count


async def main():
    """Run backfill process."""
    logger.info("Initializing database...")
    await init_db()

    logger.info("Starting resolved markets backfill...")
    start_time = datetime.now()

    kalshi_count = await backfill_kalshi_resolved(max_markets=1000)

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"✅ Backfill complete in {elapsed:.1f}s")
    logger.info(f"   Kalshi: {kalshi_count} resolved markets")
    logger.info("")
    logger.info("Now you can train XGBoost/LightGBM on this data!")


if __name__ == "__main__":
    asyncio.run(main())
