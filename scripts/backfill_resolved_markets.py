"""Backfill historical resolved markets for ML training.

Run once to populate database with outcomes for training XGBoost/LightGBM.
Fetches from both Kalshi and Polymarket APIs.

Improvement over v1:
- Fetches up to 5000 Kalshi markets (was 1000)
- Adds Polymarket resolved markets
- Filters for volume > 0 post-fetch (only useful training samples)
- Reports usable vs total counts
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


async def backfill_kalshi_resolved(max_markets: int = 5000):
    """Fetch and store resolved Kalshi markets (has outcome data!)."""
    from data_pipeline.collectors.kalshi_resolved import (
        fetch_all_resolved_markets,
        parse_resolved_market,
    )

    async with async_session() as session:
        platforms = await ensure_platforms(session)

        logger.info(f"Fetching up to {max_markets} settled markets from Kalshi...")
        raw_markets = await fetch_all_resolved_markets(max_markets=max_markets)

        logger.info(f"Parsing {len(raw_markets)} settled markets...")
        parsed_markets = [parse_resolved_market(m) for m in raw_markets]

        # Filter out markets without resolution values
        valid_markets = [m for m in parsed_markets if m.get("resolution_value") is not None]

        # Count how many have volume (usable for training)
        with_volume = [m for m in valid_markets if (m.get("volume_total") or 0) > 0]
        logger.info(
            f"Kalshi: {len(valid_markets)} with outcomes, "
            f"{len(with_volume)} with volume (usable for training)"
        )

        logger.info("Storing in database...")
        count = await upsert_markets(session, valid_markets, platforms["kalshi"])

        logger.info(f"Backfilled {count} resolved Kalshi markets")
        return count, len(with_volume)


async def backfill_polymarket_resolved(max_markets: int = 2000):
    """Fetch and store resolved Polymarket markets."""
    from data_pipeline.collectors.polymarket_resolved import (
        fetch_all_resolved_markets,
        parse_resolved_market,
    )

    async with async_session() as session:
        platforms = await ensure_platforms(session)

        logger.info(f"Fetching up to {max_markets} resolved markets from Polymarket...")
        raw_markets = await fetch_all_resolved_markets(max_markets=max_markets)

        logger.info(f"Parsing {len(raw_markets)} resolved markets...")
        parsed_markets = [parse_resolved_market(m) for m in raw_markets]

        # Filter out markets without resolution values
        valid_markets = [
            m for m in parsed_markets
            if m.get("resolution_value") is not None
            and m.get("resolution_value") in (0.0, 1.0)  # Only clear YES/NO outcomes
        ]

        with_volume = [m for m in valid_markets if (m.get("volume_total") or 0) > 0]
        logger.info(
            f"Polymarket: {len(valid_markets)} with outcomes, "
            f"{len(with_volume)} with volume (usable for training)"
        )

        logger.info("Storing in database...")
        count = await upsert_markets(session, valid_markets, platforms["polymarket"])

        logger.info(f"Backfilled {count} resolved Polymarket markets")
        return count, len(with_volume)


async def main():
    """Run backfill process."""
    logger.info("Initializing database...")
    await init_db()

    logger.info("Starting resolved markets backfill...")
    start_time = datetime.now()

    # Backfill from both sources
    kalshi_count, kalshi_usable = await backfill_kalshi_resolved(max_markets=5000)

    poly_count, poly_usable = 0, 0
    try:
        poly_count, poly_usable = await backfill_polymarket_resolved(max_markets=2000)
    except Exception as e:
        logger.warning(f"Polymarket backfill failed (non-fatal): {e}")

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"\nBackfill complete in {elapsed:.1f}s")
    logger.info(f"  Kalshi:     {kalshi_count} total, {kalshi_usable} usable")
    logger.info(f"  Polymarket: {poly_count} total, {poly_usable} usable")
    logger.info(f"  Combined:   {kalshi_count + poly_count} total, {kalshi_usable + poly_usable} usable")
    logger.info("")
    logger.info("Now you can train XGBoost/LightGBM on this data!")


if __name__ == "__main__":
    asyncio.run(main())
