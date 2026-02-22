"""Backfill orderbook snapshots for active markets.

Collects orderbook data for top markets to populate features that currently have
99.95% zero variance. Orderbook imbalance (OBI) is proven predictive in LOB literature.

Usage:
    python scripts/backfill_orderbooks.py [--limit N]
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.database import async_session
from data_pipeline.storage import insert_orderbook_snapshot, get_active_markets
from data_pipeline.collectors import polymarket_clob

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def backfill_orderbooks(limit: int = 500):
    """Backfill orderbook snapshots for top N active markets."""
    logger.info(f"Starting orderbook backfill for top {limit} markets...")

    async with async_session() as session:
        # Get active Polymarket markets (sorted by volume)
        markets = await get_active_markets(session, platform_name="polymarket", limit=limit)
        logger.info(f"Found {len(markets)} active Polymarket markets")

        success_count = 0
        skip_count = 0
        error_count = 0

        for i, market in enumerate(markets, 1):
            token_id = market.token_id_yes
            if not token_id:
                skip_count += 1
                continue

            try:
                # Fetch orderbook from CLOB API
                raw_ob = await polymarket_clob.fetch_orderbook(token_id)
                if raw_ob:
                    # Parse orderbook (computes OBI, spreads, etc.)
                    parsed = polymarket_clob.parse_orderbook(raw_ob)

                    # Insert snapshot in its own transaction to avoid lock conflicts
                    try:
                        await insert_orderbook_snapshot(session, market.id, "yes", parsed)
                        success_count += 1
                    except Exception as db_err:
                        # Rollback failed transaction and continue
                        await session.rollback()
                        error_count += 1
                        logger.debug(f"DB error for market {market.id}: {db_err}")
                        continue

                    if i % 50 == 0:
                        logger.info(f"Progress: {i}/{len(markets)} markets processed, {success_count} orderbooks collected")
                else:
                    skip_count += 1

            except Exception as e:
                error_count += 1
                logger.debug(f"Fetch error for market {market.id}: {e}")

            # Rate limit: avoid hammering the API (2.5 req/sec max)
            await asyncio.sleep(0.4)

        logger.info("=" * 60)
        logger.info(f"✅ Backfill complete!")
        logger.info(f"  Success: {success_count} orderbooks collected")
        logger.info(f"  Skipped: {skip_count} markets (no token_id or empty orderbook)")
        logger.info(f"  Errors:  {error_count} markets failed")
        logger.info(f"  Total:   {len(markets)} markets processed")
        logger.info("=" * 60)
        logger.info("📊 Orderbook features should now have significantly better coverage in next model retrain")

        return {
            "success": success_count,
            "skipped": skip_count,
            "errors": error_count,
            "total": len(markets)
        }


def main():
    parser = argparse.ArgumentParser(description="Backfill orderbook snapshots for active markets")
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Number of top markets to backfill (default: 500)"
    )
    args = parser.parse_args()

    start_time = datetime.utcnow()
    logger.info(f"Starting backfill at {start_time}")

    result = asyncio.run(backfill_orderbooks(limit=args.limit))

    end_time = datetime.utcnow()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"Backfill completed in {duration:.1f} seconds")

    # Return success if we collected at least some orderbooks
    sys.exit(0 if result["success"] > 0 else 1)


if __name__ == "__main__":
    main()
