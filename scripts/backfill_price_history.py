"""Backfill historical price data for resolved markets using Polymarket CLOB API.

This script fetches hourly price history for the top resolved markets (by volume)
to enable momentum features that currently train on constant zeros.

Usage:
    python scripts/backfill_price_history.py --limit 1000
    python scripts/backfill_price_history.py --limit 500 --days 14
    python scripts/backfill_price_history.py --market-ids data/backfill_priority_list.json --limit 2000 --days 60
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


async def backfill_all_markets(
    limit: int = 1000,
    days: int = 30,
    market_ids_path: str | None = None,
):
    """Backfill price history for pipeline-tracked resolved markets.

    This ensures backfilled historical data aligns with markets the pipeline
    is actively collecting live data for, fixing the coverage gap issue.

    Args:
        limit: Maximum number of markets to backfill (default 1000)
        days: Number of days to backfill per market (default 30)
        market_ids_path: Optional path to JSON file of market IDs to backfill.
            When provided, overrides the default volume-ordered query — only the
            listed markets are backfilled (up to `limit`). This enables targeted
            backfill from the output of prioritize_backfill.py.
    """
    await init_db()

    # Query markets first (separate session)
    async with async_session() as query_session:
        if market_ids_path:
            import json as _json
            ids_file = Path(market_ids_path)
            if not ids_file.exists():
                logger.error(f"--market-ids file not found: {ids_file}")
                return
            with open(ids_file) as f:
                target_ids: list[int] = _json.load(f)
            if not target_ids:
                logger.info("--market-ids file is empty, nothing to backfill.")
                return
            # Apply limit (first N from priority-sorted list)
            target_ids = target_ids[:limit]
            logger.info(
                f"Using targeted backfill from {ids_file.name}: "
                f"{len(target_ids)} markets (limit={limit})"
            )
            result = await query_session.execute(
                select(Market)
                .where(
                    Market.id.in_(target_ids),
                    Market.token_id_yes.isnot(None),  # Must have token for API
                )
            )
            markets = result.scalars().all()
            # Re-sort to match original priority order
            id_order = {mid: i for i, mid in enumerate(target_ids)}
            markets = sorted(markets, key=lambda m: id_order.get(m.id, 999999))
        else:
            # Get ALL resolved markets with token IDs (not just tracked ones)
            # This ensures we backfill historical data for training, not just live markets
            logger.info("Finding resolved markets with token IDs...")
            result = await query_session.execute(
                select(Market)
                .where(
                    Market.resolution_value.isnot(None),  # Resolved markets
                    Market.token_id_yes.isnot(None),  # With token IDs for API
                )
                .order_by(Market.volume_total.desc())  # Prioritize high-volume
                .limit(limit)
            )
            markets = result.scalars().all()

    logger.info("=" * 60)
    logger.info("BACKFILL SUMMARY:")
    logger.info(f"Resolved markets with token_id: {len(markets)}")
    logger.info(f"Will backfill: {len(markets)} markets")
    logger.info(f"Days per market: {days} days (hourly snapshots)")
    logger.info(f"Expected snapshots: ~{len(markets) * days * 24:,}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Starting backfill...")

    # Process markets in parallel batches (to respect rate limits while maximizing throughput)
    BATCH_SIZE = 10  # Process 10 markets concurrently
    SLEEP_PER_BATCH = 2.0  # 2 seconds for 10 requests = 5 req/sec average

    results = {
        "success": 0,
        "no_history": 0,
        "no_snapshots_in_window": 0,
        "already_exists": 0,
        "errors": 0,
        "total_snapshots": 0,
    }

    # Process in batches
    for batch_start in range(0, len(markets), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(markets))
        batch = markets[batch_start:batch_end]

        if batch_start > 0 and batch_start % 100 == 0:
            logger.info(
                f"Progress: {batch_start}/{len(markets)} markets processed "
                f"({results['success']} successful, {results['total_snapshots']} snapshots added)"
            )

        # Process batch in parallel
        async def process_market(market):
            async with async_session() as session:
                return await backfill_market_history(market, session, days=days)

        batch_results = await asyncio.gather(*[process_market(m) for m in batch])

        # Aggregate results
        for result in batch_results:
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

        # Rate limiting: 2 seconds per batch of 10 = ~5 req/sec average
        await asyncio.sleep(SLEEP_PER_BATCH)

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
    parser.add_argument(
        "--market-ids",
        type=str,
        default=None,
        dest="market_ids",
        help=(
            "Path to JSON file containing list of market IDs to backfill "
            "(overrides default volume-ordered query). Typically produced by "
            "scripts/prioritize_backfill.py."
        ),
    )
    args = parser.parse_args()

    logger.info("Starting price history backfill...")
    logger.info(
        f"Config: limit={args.limit}, days={args.days}"
        + (f", market-ids={args.market_ids}" if args.market_ids else "")
    )

    asyncio.run(
        backfill_all_markets(
            limit=args.limit,
            days=args.days,
            market_ids_path=args.market_ids,
        )
    )


if __name__ == "__main__":
    main()
