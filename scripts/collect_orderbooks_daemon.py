"""Continuous orderbook snapshot collector — runs indefinitely in background.

Cycles through ALL active Polymarket markets, collecting orderbook snapshots
at configurable intervals. Higher-volume markets get polled more frequently.

Crash-safe: each snapshot is an independent DB commit. Restarting just
begins a new collection cycle with zero data loss.

Usage:
  python scripts/collect_orderbooks_daemon.py                    # Default settings
  python scripts/collect_orderbooks_daemon.py --cycle-delay 60   # 1 min between cycles
  python scripts/collect_orderbooks_daemon.py --request-delay 0.3
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import asyncio
import json
import logging
import signal
import time
from datetime import datetime

import httpx
from sqlalchemy import select, func

from db.database import init_db, async_session
from db.models import Market, OrderbookSnapshot
from data_pipeline.collectors.polymarket_clob import fetch_orderbook, parse_orderbook
from data_pipeline.storage import insert_orderbook_snapshot

LOG_FILE = project_root / "data" / "logs" / "orderbook_collector.log"
STATS_FILE = project_root / "data" / "checkpoints" / "orderbook_collector_stats.json"

shutdown_requested = False


def setup_logging():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(fh)
    root.addHandler(ch)
    return logging.getLogger("ob_collector")


logger = setup_logging()


def handle_signal(signum, frame):
    global shutdown_requested
    shutdown_requested = True
    logger.warning("Shutdown signal %d received. Finishing current market...", signum)


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


class CycleStats:
    """Track statistics across collection cycles."""

    def __init__(self):
        self.total_cycles = 0
        self.total_snapshots = 0
        self.total_errors = 0
        self.started_at = datetime.utcnow().isoformat()
        self.cycle_snapshots = 0
        self.cycle_errors = 0
        self.cycle_skipped = 0

    def start_cycle(self):
        self.total_cycles += 1
        self.cycle_snapshots = 0
        self.cycle_errors = 0
        self.cycle_skipped = 0

    def record_success(self):
        self.cycle_snapshots += 1
        self.total_snapshots += 1

    def record_error(self):
        self.cycle_errors += 1
        self.total_errors += 1

    def record_skip(self):
        self.cycle_skipped += 1

    def save(self):
        STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "started_at": self.started_at,
            "last_updated": datetime.utcnow().isoformat(),
            "total_cycles": self.total_cycles,
            "total_snapshots": self.total_snapshots,
            "total_errors": self.total_errors,
            "avg_snapshots_per_cycle": round(
                self.total_snapshots / max(self.total_cycles, 1), 1
            ),
        }
        with open(STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


async def get_active_markets_tiered() -> list[tuple[Market, str]]:
    """Get active markets in priority tiers.

    Returns list of (market, tier) where tier is 'high', 'medium', or 'low'.
    High-volume markets are collected every cycle, medium every 2nd, low every 5th.

    Only includes markets that are genuinely live (unresolved, end_date in
    the future or NULL). This avoids wasting ~85% of requests on 404s from
    resolved markets whose is_active flag is stale.
    """
    from sqlalchemy import or_

    now = datetime.utcnow()

    async with async_session() as session:
        result = await session.execute(
            select(Market)
            .where(
                Market.is_active == True,  # noqa: E712
                Market.token_id_yes.isnot(None),
                Market.resolved_at.is_(None),
                or_(
                    Market.end_date.is_(None),
                    Market.end_date > now,
                ),
            )
            .order_by(Market.volume_total.desc().nullslast())
        )
        markets = result.scalars().all()

    tiered = []
    for i, m in enumerate(markets):
        if i < 200:
            tiered.append((m, "high"))
        elif i < 500:
            tiered.append((m, "medium"))
        else:
            tiered.append((m, "low"))

    logger.info(
        "Filtered to %d genuinely live markets (was 6638 before fix)",
        len(markets),
    )
    return tiered


async def collect_single_orderbook(
    market: Market,
    request_delay: float,
) -> bool:
    """Collect and store orderbook for a single market. Returns True on success."""
    token_id = market.token_id_yes
    if not token_id:
        return False

    try:
        raw_ob = await fetch_orderbook(token_id)
        if not raw_ob:
            return False

        bids = raw_ob.get("bids", [])
        asks = raw_ob.get("asks", [])
        if not bids and not asks:
            return False

        parsed = parse_orderbook(raw_ob)

        async with async_session() as session:
            await insert_orderbook_snapshot(session, market.id, "yes", parsed)

        return True
    except Exception as e:
        logger.debug("Error collecting OB for market %d: %s", market.id, str(e)[:80])
        return False
    finally:
        await asyncio.sleep(request_delay)


async def run_collection_cycle(
    tiered_markets: list[tuple[Market, str]],
    cycle_num: int,
    request_delay: float,
    stats: CycleStats,
):
    """Run one full collection cycle."""
    stats.start_cycle()
    cycle_start = time.time()

    for market, tier in tiered_markets:
        if shutdown_requested:
            break

        if tier == "medium" and cycle_num % 2 != 0:
            stats.record_skip()
            continue
        if tier == "low" and cycle_num % 5 != 0:
            stats.record_skip()
            continue

        success = await collect_single_orderbook(market, request_delay)
        if success:
            stats.record_success()
        else:
            stats.record_error()

    cycle_duration = time.time() - cycle_start
    logger.info(
        "Cycle %d complete: %d snapshots, %d errors, %d skipped (%.1fs)",
        cycle_num, stats.cycle_snapshots, stats.cycle_errors,
        stats.cycle_skipped, cycle_duration,
    )
    stats.save()


async def run_daemon(args):
    await init_db()

    logger.info("=" * 60)
    logger.info("ORDERBOOK COLLECTOR DAEMON")
    logger.info("=" * 60)
    logger.info("Cycle delay: %ds | Request delay: %.2fs",
                args.cycle_delay, args.request_delay)

    stats = CycleStats()
    cycle_num = 0

    while not shutdown_requested:
        tiered_markets = await get_active_markets_tiered()
        total = len(tiered_markets)
        high = sum(1 for _, t in tiered_markets if t == "high")
        med = sum(1 for _, t in tiered_markets if t == "medium")
        low = total - high - med

        logger.info(
            "Active markets: %d (high=%d, medium=%d, low=%d)",
            total, high, med, low,
        )

        cycle_num += 1
        await run_collection_cycle(
            tiered_markets, cycle_num, args.request_delay, stats,
        )

        if not shutdown_requested:
            logger.info("Sleeping %ds before next cycle...", args.cycle_delay)
            for _ in range(args.cycle_delay):
                if shutdown_requested:
                    break
                await asyncio.sleep(1)

    stats.save()
    logger.info("Daemon stopped. Total: %d cycles, %d snapshots",
                stats.total_cycles, stats.total_snapshots)


def main():
    parser = argparse.ArgumentParser(
        description="Continuous orderbook snapshot collector (daemon)",
    )
    parser.add_argument("--cycle-delay", type=int, default=120,
                        help="Seconds between collection cycles (default: 120)")
    parser.add_argument("--request-delay", type=float, default=0.25,
                        help="Seconds between API requests (default: 0.25)")
    args = parser.parse_args()

    asyncio.run(run_daemon(args))


if __name__ == "__main__":
    main()
