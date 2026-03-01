"""Massive price history backfill with crash-safe checkpointing.

Fetches full price history for ALL markets (resolved + active) from
Polymarket's /prices-history endpoint. Designed to run for hours/days
in the background without losing progress on crash.

Features:
  - JSONL checkpoint file: append-only, crash-safe, instant resume
  - Multiple fidelity fallbacks (60m → 360m → 720m → 1440m)
  - Chunked time ranges (15-day windows) for resolved market reliability
  - Per-snapshot UPSERT: no batch failures
  - Graceful SIGINT handling: saves stats before exit
  - Dual logging: console + file
  - Progress bar with ETA

Usage:
  python scripts/backfill_all_prices.py                    # All markets
  python scripts/backfill_all_prices.py --resolved-only    # Only resolved
  python scripts/backfill_all_prices.py --active-only      # Only active
  python scripts/backfill_all_prices.py --resume           # Resume from checkpoint
  python scripts/backfill_all_prices.py --max-days 90      # 90 days of history
  python scripts/backfill_all_prices.py --delay 0.5        # Slower rate limit
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
from datetime import datetime, timedelta
from typing import Optional

import httpx
from sqlalchemy import select, func, text
from sqlalchemy.exc import IntegrityError

from db.database import init_db, async_session, engine
from db.models import Market, PriceSnapshot
from config.settings import settings

CLOB_BASE = settings.polymarket_clob_url
CHECKPOINT_DIR = project_root / "data" / "checkpoints"
CHECKPOINT_FILE = CHECKPOINT_DIR / "price_backfill_progress.jsonl"
STATS_FILE = CHECKPOINT_DIR / "price_backfill_stats.json"
LOG_FILE = project_root / "data" / "logs" / "price_backfill.log"

FIDELITY_LADDER = [60, 360, 720, 1440]
CHUNK_DAYS = 15
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0

shutdown_requested = False


def setup_logging():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    return logging.getLogger("backfill")


logger = setup_logging()


def handle_signal(signum, frame):
    global shutdown_requested
    shutdown_requested = True
    logger.warning("Shutdown requested (signal %d). Finishing current market...", signum)


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


class CheckpointManager:
    """Append-only JSONL checkpoint with crash-safe resume."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.processed: dict[int, dict] = {}
        self._file = None

    def load(self) -> int:
        """Load checkpoint. Returns count of previously processed markets."""
        if not self.filepath.exists():
            return 0
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    self.processed[entry["id"]] = entry
                except (json.JSONDecodeError, KeyError):
                    continue
        return len(self.processed)

    def open(self):
        self._file = open(self.filepath, "a", encoding="utf-8", buffering=1)

    def close(self):
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None

    def is_processed(self, market_id: int) -> bool:
        return market_id in self.processed

    def record(self, market_id: int, status: str, snapshots: int = 0,
               fidelity_used: int = 0, error: str = ""):
        entry = {
            "id": market_id,
            "status": status,
            "snapshots": snapshots,
            "fidelity": fidelity_used,
            "ts": datetime.utcnow().isoformat(),
        }
        if error:
            entry["error"] = error[:200]
        self.processed[market_id] = entry
        if self._file:
            self._file.write(json.dumps(entry) + "\n")
            self._file.flush()


class StatsTracker:
    """Tracks and persists run statistics."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.started_at = datetime.utcnow().isoformat()
        self.success = 0
        self.no_data = 0
        self.already_covered = 0
        self.errors = 0
        self.total_snapshots = 0
        self.processed = 0
        self.total_markets = 0
        self._last_save = time.time()

    def save(self):
        data = {
            "started_at": self.started_at,
            "last_updated": datetime.utcnow().isoformat(),
            "total_markets": self.total_markets,
            "processed": self.processed,
            "success": self.success,
            "no_data": self.no_data,
            "already_covered": self.already_covered,
            "errors": self.errors,
            "total_snapshots": self.total_snapshots,
            "success_rate": f"{self.success / max(self.processed, 1) * 100:.1f}%",
            "avg_snapshots_per_market": round(
                self.total_snapshots / max(self.success, 1), 1
            ),
        }
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self._last_save = time.time()

    def maybe_save(self, interval: float = 30.0):
        if time.time() - self._last_save > interval:
            self.save()


async def fetch_price_history_chunk(
    client: httpx.AsyncClient,
    token_id: str,
    start_ts: int,
    end_ts: int,
    fidelity: int,
) -> list[dict]:
    """Fetch a single chunk of price history with retry."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.get(
                f"{CLOB_BASE}/prices-history",
                params={
                    "market": token_id,
                    "startTs": start_ts,
                    "endTs": end_ts,
                    "fidelity": fidelity,
                },
                timeout=30,
            )
            if resp.status_code == 429:
                wait = RETRY_BACKOFF * (2 ** attempt)
                logger.warning("Rate limited (429). Sleeping %.1fs...", wait)
                await asyncio.sleep(wait)
                continue
            if resp.status_code != 200:
                logger.debug(
                    "HTTP %d for token %s (fidelity=%d)",
                    resp.status_code, token_id[:16], fidelity,
                )
                return []
            data = resp.json()
            return data.get("history", [])
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            wait = RETRY_BACKOFF * (2 ** attempt)
            logger.debug("Request error (attempt %d): %s. Retrying in %.1fs", attempt + 1, e, wait)
            await asyncio.sleep(wait)
        except Exception as e:
            logger.error("Unexpected error fetching history: %s", e)
            return []
    return []


async def fetch_full_history(
    client: httpx.AsyncClient,
    token_id: str,
    start_dt: datetime,
    end_dt: datetime,
    delay: float,
) -> tuple[list[dict], int]:
    """Fetch full price history using chunked time ranges with fidelity fallback.

    Returns (history_points, fidelity_used).
    """
    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())
    total_days = (end_dt - start_dt).days

    for fidelity in FIDELITY_LADDER:
        all_points = []
        chunk_start = start_ts

        while chunk_start < end_ts:
            if shutdown_requested:
                return all_points, fidelity

            chunk_end = min(chunk_start + CHUNK_DAYS * 86400, end_ts)
            points = await fetch_price_history_chunk(
                client, token_id, chunk_start, chunk_end, fidelity,
            )
            if points:
                all_points.extend(points)
            chunk_start = chunk_end
            await asyncio.sleep(delay)

        if all_points:
            logger.debug(
                "Got %d points with fidelity=%d for %s (%d days)",
                len(all_points), fidelity, token_id[:16], total_days,
            )
            return all_points, fidelity

    return [], 0


async def try_interval_max_first(
    client: httpx.AsyncClient,
    token_id: str,
    delay: float,
) -> tuple[list[dict], int]:
    """Quick attempt with interval=max before falling back to chunked approach."""
    try:
        resp = await client.get(
            f"{CLOB_BASE}/prices-history",
            params={
                "market": token_id,
                "interval": "max",
                "fidelity": 60,
            },
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            history = data.get("history", [])
            if history and len(history) > 5:
                return history, 60
    except Exception:
        pass
    await asyncio.sleep(delay)
    return [], 0


async def insert_snapshots_safe(
    session,
    market_id: int,
    points: list[dict],
) -> int:
    """Insert snapshots one-by-one, skipping duplicates. Returns count inserted."""
    inserted = 0
    for point in points:
        ts_raw = point.get("t", 0)
        price_raw = point.get("p", 0)

        if not ts_raw:
            continue

        try:
            price = float(price_raw)
        except (ValueError, TypeError):
            continue

        if price <= 0 or price >= 1:
            continue

        try:
            snapshot_time = datetime.utcfromtimestamp(int(ts_raw))
        except (ValueError, OSError):
            continue

        snapshot = PriceSnapshot(
            market_id=market_id,
            timestamp=snapshot_time,
            price_yes=price,
            price_no=round(1.0 - price, 6),
            midpoint=0.5,
            spread=round(abs(2 * price - 1.0), 6),
            volume=0,
        )
        session.add(snapshot)
        try:
            await session.flush()
            inserted += 1
        except IntegrityError:
            await session.rollback()

    if inserted > 0:
        await session.commit()

    return inserted


async def count_existing_snapshots(session, market_id: int) -> int:
    """Count how many price snapshots already exist for a market."""
    result = await session.execute(
        select(func.count(PriceSnapshot.id)).where(
            PriceSnapshot.market_id == market_id
        )
    )
    return result.scalar() or 0


async def process_single_market(
    client: httpx.AsyncClient,
    market: Market,
    max_days: int,
    delay: float,
    min_existing_threshold: int,
) -> tuple[str, int, int]:
    """Process one market. Returns (status, snapshots_added, fidelity_used)."""
    token_id = market.token_id_yes or market.condition_id
    if not token_id:
        return "no_token", 0, 0

    async with async_session() as session:
        existing = await count_existing_snapshots(session, market.id)
        if existing >= min_existing_threshold:
            return "already_covered", 0, 0

    end_dt = market.resolved_at or datetime.utcnow()
    start_dt = end_dt - timedelta(days=max_days)
    if market.created_at and market.created_at > start_dt:
        start_dt = market.created_at

    if (end_dt - start_dt).total_seconds() < 3600:
        return "too_short", 0, 0

    points, fidelity = await try_interval_max_first(client, token_id, delay)

    if not points:
        points, fidelity = await fetch_full_history(
            client, token_id, start_dt, end_dt, delay,
        )

    if not points:
        if market.token_id_yes and market.condition_id and token_id == market.token_id_yes:
            points, fidelity = await try_interval_max_first(
                client, market.condition_id, delay,
            )
            if not points:
                points, fidelity = await fetch_full_history(
                    client, market.condition_id, start_dt, end_dt, delay,
                )

    if not points:
        return "no_data", 0, 0

    async with async_session() as session:
        inserted = await insert_snapshots_safe(session, market.id, points)

    return "success" if inserted > 0 else "duplicate", inserted, fidelity


async def get_all_markets(
    resolved_only: bool = False,
    active_only: bool = False,
) -> list[Market]:
    """Fetch all target markets from DB."""
    async with async_session() as session:
        query = select(Market).where(
            Market.token_id_yes.isnot(None)
            | Market.condition_id.isnot(None)
        )

        if resolved_only:
            query = query.where(
                Market.resolution_value.isnot(None),
                Market.resolved_at.isnot(None),
            )
        elif active_only:
            query = query.where(Market.is_active == True)  # noqa: E712

        query = query.order_by(Market.volume_total.desc().nullslast())
        result = await session.execute(query)
        return list(result.scalars().all())


async def run_backfill(args):
    await init_db()

    logger.info("=" * 70)
    logger.info("MASSIVE PRICE HISTORY BACKFILL")
    logger.info("=" * 70)

    ckpt = CheckpointManager(CHECKPOINT_FILE)
    previously_processed = ckpt.load()
    if previously_processed > 0:
        logger.info("Resuming: %d markets already processed from checkpoint", previously_processed)
    ckpt.open()

    stats = StatsTracker(STATS_FILE)

    mode = "all"
    if args.resolved_only:
        mode = "resolved"
    elif args.active_only:
        mode = "active"

    logger.info("Mode: %s | Max days: %d | Delay: %.2fs | Min existing: %d",
                mode, args.max_days, args.delay, args.min_existing)

    markets = await get_all_markets(
        resolved_only=args.resolved_only,
        active_only=args.active_only,
    )
    stats.total_markets = len(markets)

    logger.info("Total markets to process: %d", len(markets))
    logger.info("Already in checkpoint: %d", previously_processed)
    remaining = sum(1 for m in markets if not ckpt.is_processed(m.id))
    logger.info("Remaining: %d", remaining)
    logger.info("-" * 70)

    if remaining == 0:
        logger.info("All markets already processed! Nothing to do.")
        ckpt.close()
        return

    start_time = time.time()
    processed_this_run = 0

    async with httpx.AsyncClient() as client:
        for i, market in enumerate(markets):
            if shutdown_requested:
                logger.info("Shutdown requested. Saving progress...")
                break

            if ckpt.is_processed(market.id):
                continue

            try:
                status, snapshots, fidelity = await process_single_market(
                    client, market, args.max_days, args.delay, args.min_existing,
                )
            except Exception as e:
                status, snapshots, fidelity = "error", 0, 0
                logger.error("Market %d failed: %s", market.id, str(e)[:100])

            ckpt.record(market.id, status, snapshots, fidelity,
                        error="" if status != "error" else "see log")

            if status == "success":
                stats.success += 1
                stats.total_snapshots += snapshots
            elif status == "no_data":
                stats.no_data += 1
            elif status == "already_covered":
                stats.already_covered += 1
            elif status == "error":
                stats.errors += 1

            stats.processed += 1
            processed_this_run += 1
            stats.maybe_save()

            if processed_this_run % 50 == 0 and processed_this_run > 0:
                elapsed = time.time() - start_time
                rate = processed_this_run / elapsed
                eta_sec = (remaining - processed_this_run) / rate if rate > 0 else 0
                eta_h = eta_sec / 3600

                logger.info(
                    "Progress: %d/%d (%.1f%%) | Success: %d | Snapshots: %d | "
                    "Rate: %.1f/min | ETA: %.1fh",
                    processed_this_run, remaining,
                    processed_this_run / remaining * 100,
                    stats.success, stats.total_snapshots,
                    rate * 60, eta_h,
                )

    ckpt.close()
    stats.save()

    elapsed_total = time.time() - start_time
    logger.info("=" * 70)
    logger.info("BACKFILL COMPLETE")
    logger.info("=" * 70)
    logger.info("Runtime: %.1f minutes (%.1f hours)", elapsed_total / 60, elapsed_total / 3600)
    logger.info("Markets processed this run: %d", processed_this_run)
    logger.info("Success (new data): %d", stats.success)
    logger.info("No data available: %d", stats.no_data)
    logger.info("Already covered: %d", stats.already_covered)
    logger.info("Errors: %d", stats.errors)
    logger.info("Total snapshots inserted: %d", stats.total_snapshots)
    logger.info("Avg snapshots per successful market: %.1f",
                stats.total_snapshots / max(stats.success, 1))
    logger.info("-" * 70)
    logger.info("Checkpoint: %s", CHECKPOINT_FILE)
    logger.info("Stats: %s", STATS_FILE)
    logger.info("Log: %s", LOG_FILE)
    if stats.total_snapshots > 0:
        logger.info("")
        logger.info("Next: python scripts/train_ensemble.py")


def main():
    parser = argparse.ArgumentParser(
        description="Massive price history backfill with checkpointing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/backfill_all_prices.py                     # All markets, default settings
  python scripts/backfill_all_prices.py --resolved-only     # Only resolved (training data)
  python scripts/backfill_all_prices.py --active-only       # Only active (inference data)
  python scripts/backfill_all_prices.py --max-days 180      # 6 months of history
  python scripts/backfill_all_prices.py --delay 0.5         # Slower (safer) rate limit
  python scripts/backfill_all_prices.py --min-existing 48   # Skip if >=48 snapshots exist
        """,
    )
    parser.add_argument("--resolved-only", action="store_true",
                        help="Only backfill resolved markets (training data)")
    parser.add_argument("--active-only", action="store_true",
                        help="Only backfill active markets (inference data)")
    parser.add_argument("--max-days", type=int, default=90,
                        help="Max days of history per market (default: 90)")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Seconds between API requests (default: 0.3)")
    parser.add_argument("--min-existing", type=int, default=24,
                        help="Skip markets with >= N existing snapshots (default: 24)")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from checkpoint (default: True)")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore checkpoint, start fresh")
    args = parser.parse_args()

    if args.fresh and CHECKPOINT_FILE.exists():
        backup = CHECKPOINT_FILE.with_suffix(".jsonl.bak")
        CHECKPOINT_FILE.rename(backup)
        logger.info("Moved old checkpoint to %s", backup)

    asyncio.run(run_backfill(args))


if __name__ == "__main__":
    main()
