"""Backfill historical price data for resolved markets using Polymarket CLOB API.

This script fetches hourly price history for the top resolved markets (by volume)
to enable momentum features that currently train on constant zeros.

Usage:
    python scripts/backfill_price_history.py --limit 1000
    python scripts/backfill_price_history.py --limit 500 --days 14
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import asyncio
import logging
from datetime import datetime, timedelta
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from db.database import init_db, async_session
from db.models import Market, PriceSnapshot
from data_pipeline.collectors.polymarket_clob import fetch_price_history

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


async def backfill_market_history(
    market: Market,
    session,
    days: int = 30,
) -> dict:
    """Backfill price history for a single resolved market.

    Args:
        market: Market ORM object
        session: AsyncSession
        days: Number of days to backfill (max 30 for storage safety)

    Returns:
        dict with keys: market_id, snapshots_added, error
    """
    if not market.token_id_yes:
        return {"market_id": market.id, "snapshots_added": 0, "error": "no_token_id"}

    # Calculate time window (last N days before resolution)
    if market.resolved_at:
        end_time = int(market.resolved_at.timestamp())
    else:
        logger.warning(f"Market {market.id} not resolved but in query")
        end_time = int(datetime.utcnow().timestamp())

    start_time = end_time - (days * 86400)  # N days back

    try:
        # API returns: [{"t": unix_timestamp, "p": "0.55"}, ...]
        history = await fetch_price_history(
            token_id=market.token_id_yes,  # CRITICAL: Use token_id_yes (not token_id)
            interval="max",
            fidelity=60,  # 60 minutes = 1-hour intervals (fidelity is in MINUTES, not seconds)
        )

        if not history:
            logger.debug(f"No history returned for market {market.id}")
            return {"market_id": market.id, "snapshots_added": 0, "error": "no_history"}

        # Parse and filter snapshots
        snapshots = []
        for point in history:
            ts = datetime.fromtimestamp(point["t"])

            # Skip if outside our window
            if ts < datetime.fromtimestamp(start_time):
                continue
            if ts > datetime.fromtimestamp(end_time):
                continue

            snapshots.append(
                PriceSnapshot(
                    market_id=market.id,
                    timestamp=ts,
                    price_yes=float(point["p"]),  # p is string in API
                    price_no=1.0 - float(point["p"]),  # Assume binary market
                    midpoint=(float(point["p"]) + (1.0 - float(point["p"]))) / 2,
                    spread=abs(float(point["p"]) - (1.0 - float(point["p"]))),
                    volume=0,  # Use 0 (not None) to match column default
                )
            )

        if not snapshots:
            return {"market_id": market.id, "snapshots_added": 0, "error": "no_snapshots_in_window"}

        # IDEMPOTENCY: Batch insert with IntegrityError handling
        # (Assumes unique constraint on (market_id, timestamp) exists)
        try:
            session.add_all(snapshots)
            await session.commit()
            logger.info(f"Market {market.id}: added {len(snapshots)} snapshots")
            return {"market_id": market.id, "snapshots_added": len(snapshots), "error": None}
        except IntegrityError as e:
            await session.rollback()
            logger.debug(f"Market {market.id}: snapshots already exist (IntegrityError)")
            return {"market_id": market.id, "snapshots_added": 0, "error": "already_exists"}

    except Exception as e:
        logger.error(f"Market {market.id}: backfill failed: {e}")
        return {"market_id": market.id, "snapshots_added": 0, "error": str(e)}


async def backfill_all_markets(limit: int = 1000, days: int = 30):
    """Backfill price history for top N resolved markets by volume.

    Args:
        limit: Maximum number of markets to backfill (default 1000)
        days: Number of days to backfill per market (default 30)
    """
    await init_db()

    # Query markets first (separate session)
    async with async_session() as query_session:
        result = await query_session.execute(
            select(Market)
            .where(
                Market.resolution_value.isnot(None),  # Only resolved markets
                Market.token_id_yes.isnot(None),  # Must have token_id
            )
            .order_by(Market.volume_total.desc())
            .limit(limit)
        )
        markets = result.scalars().all()

    logger.info(f"Found {len(markets)} resolved markets to backfill")
    logger.info(f"Backfilling last {days} days of history (hourly)")

    # Process markets sequentially (to respect rate limits)
    results = {
        "success": 0,
        "no_history": 0,
        "no_snapshots_in_window": 0,
        "already_exists": 0,
        "errors": 0,
        "total_snapshots": 0,
    }

    for i, market in enumerate(markets):
        if i > 0 and i % 100 == 0:
            logger.info(
                f"Progress: {i}/{len(markets)} markets processed "
                f"({results['success']} successful, {results['total_snapshots']} snapshots added)"
            )

        # Create fresh session for each market (isolation from IntegrityError rollbacks)
        async with async_session() as session:
            result = await backfill_market_history(market, session, days=days)

        if result["error"] is None:
            results["success"] += 1
            results["total_snapshots"] += result["snapshots_added"]
        elif result["error"] == "no_history":
            results["no_history"] += 1
        elif result["error"] == "no_snapshots_in_window":
            results["no_snapshots_in_window"] += 1
        elif result["error"] == "already_exists":
            results["already_exists"] += 1
        else:
            results["errors"] += 1

        # Rate limiting: ~1 request per second (conservative)
        await asyncio.sleep(1.0)

    # Final summary
    logger.info("=" * 60)
    logger.info("Backfill Complete!")
    logger.info(f"Total markets processed: {len(markets)}")
    logger.info(f"Successful: {results['success']}")
    logger.info(f"Total snapshots added: {results['total_snapshots']}")
    logger.info(f"No history available: {results['no_history']}")
    logger.info(f"No snapshots in time window: {results['no_snapshots_in_window']}")
    logger.info(f"Already exists (skipped): {results['already_exists']}")
    logger.info(f"Errors: {results['errors']}")
    logger.info("=" * 60)

    coverage_pct = (results["success"] / len(markets) * 100) if markets else 0
    logger.info(f"Coverage: {coverage_pct:.1f}% of markets have history")

    if results["total_snapshots"] > 0:
        logger.info("\nNext steps:")
        logger.info("1. Run: python scripts/train_ensemble.py")
        logger.info("2. Verify momentum features no longer pruned")
        logger.info("3. Check Brier improvement (target: 0.058-0.060)")


def main():
    parser = argparse.ArgumentParser(description="Backfill historical price data")
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of markets to backfill (default: 1000)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to backfill per market (default: 30)",
    )
    args = parser.parse_args()

    logger.info("Starting price history backfill...")
    logger.info(f"Config: limit={args.limit}, days={args.days}")

    asyncio.run(backfill_all_markets(limit=args.limit, days=args.days))


if __name__ == "__main__":
    main()
