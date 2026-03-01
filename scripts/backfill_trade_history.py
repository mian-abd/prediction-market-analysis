"""Backfill trade history for all markets from Polymarket Data API.

Fetches actual trade executions (buy/sell, size, price, timestamp) per market.
This data enables order flow features: volume profiles, buy/sell imbalance,
large-trade detection, trade intensity curves.

Features:
  - JSONL checkpoint file: crash-safe resume
  - Rate limiting with exponential backoff on 429
  - Paginated fetching (up to 10K trades per market)
  - Graceful shutdown on SIGINT

Usage:
  python scripts/backfill_trade_history.py                     # All markets
  python scripts/backfill_trade_history.py --resolved-only     # Training data only
  python scripts/backfill_trade_history.py --min-volume 1000   # High-volume only
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
from sqlalchemy.exc import IntegrityError

from db.database import init_db, async_session
from db.models import Market, Trade

POLYMARKET_DATA_API = "https://data-api.polymarket.com"

CHECKPOINT_DIR = project_root / "data" / "checkpoints"
CHECKPOINT_FILE = CHECKPOINT_DIR / "trade_backfill_progress.jsonl"
STATS_FILE = CHECKPOINT_DIR / "trade_backfill_stats.json"
LOG_FILE = project_root / "data" / "logs" / "trade_backfill.log"

MAX_PER_PAGE = 1000
MAX_OFFSET = 10000
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0

shutdown_requested = False


def setup_logging():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(fh)
    root.addHandler(ch)
    return logging.getLogger("trade_backfill")


logger = setup_logging()


def handle_signal(signum, frame):
    global shutdown_requested
    shutdown_requested = True
    logger.warning("Shutdown requested (signal %d).", signum)


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


class CheckpointManager:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.processed: dict[int, dict] = {}
        self._file = None

    def load(self) -> int:
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

    def record(self, market_id: int, status: str, trades: int = 0, error: str = ""):
        entry = {
            "id": market_id,
            "status": status,
            "trades": trades,
            "ts": datetime.utcnow().isoformat(),
        }
        if error:
            entry["error"] = error[:200]
        self.processed[market_id] = entry
        if self._file:
            self._file.write(json.dumps(entry) + "\n")
            self._file.flush()


async def fetch_trades_page(
    client: httpx.AsyncClient,
    condition_id: str,
    offset: int,
    delay: float,
) -> list[dict]:
    """Fetch one page of trades for a market. Returns list of trade dicts."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.get(
                f"{POLYMARKET_DATA_API}/trades",
                params={
                    "market": condition_id,
                    "limit": MAX_PER_PAGE,
                    "offset": offset,
                },
                timeout=30,
            )
            if resp.status_code == 429:
                wait = RETRY_BACKOFF * (2 ** attempt)
                logger.warning("Rate limited. Sleeping %.1fs...", wait)
                await asyncio.sleep(wait)
                continue
            if resp.status_code != 200:
                logger.debug("HTTP %d for trades (cid=%s)", resp.status_code, condition_id[:16])
                return []
            data = resp.json()
            return data if isinstance(data, list) else []
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            wait = RETRY_BACKOFF * (2 ** attempt)
            logger.debug("Request error (attempt %d): %s", attempt + 1, e)
            await asyncio.sleep(wait)
        except Exception as e:
            logger.error("Unexpected error: %s", e)
            return []
    return []


async def fetch_all_trades(
    client: httpx.AsyncClient,
    condition_id: str,
    delay: float,
) -> list[dict]:
    """Fetch all available trades for a market via pagination."""
    all_trades = []
    offset = 0

    while offset <= MAX_OFFSET:
        if shutdown_requested:
            break
        page = await fetch_trades_page(client, condition_id, offset, delay)
        if not page:
            break
        all_trades.extend(page)
        if len(page) < MAX_PER_PAGE:
            break
        offset += MAX_PER_PAGE
        await asyncio.sleep(delay)

    return all_trades


async def store_trades(market_id: int, raw_trades: list[dict]) -> int:
    """Parse and store trades in DB. Returns count of new trades inserted."""
    if not raw_trades:
        return 0

    inserted = 0
    async with async_session() as session:
        existing_count = await session.execute(
            select(func.count(Trade.id)).where(Trade.market_id == market_id)
        )
        existing = existing_count.scalar() or 0

        if existing >= len(raw_trades):
            return 0

        for raw in raw_trades:
            try:
                ts_raw = raw.get("timestamp")
                if not ts_raw:
                    continue

                if isinstance(ts_raw, (int, float)):
                    ts = datetime.utcfromtimestamp(ts_raw)
                elif isinstance(ts_raw, str):
                    ts_str = ts_raw.replace("Z", "+00:00")
                    try:
                        ts = datetime.fromisoformat(ts_str).replace(tzinfo=None)
                    except ValueError:
                        ts = datetime.utcfromtimestamp(float(ts_raw))
                else:
                    continue

                side = raw.get("side", "BUY")
                outcome = raw.get("outcome", "Yes")
                price_raw = raw.get("price", 0)
                size_raw = raw.get("size", 0)

                try:
                    price = float(price_raw)
                    size = float(size_raw)
                except (ValueError, TypeError):
                    continue

                if price <= 0 or size <= 0:
                    continue

                tx_hash = raw.get("transactionHash", "")

                trade = Trade(
                    market_id=market_id,
                    external_trade_id=tx_hash[:255] if tx_hash else None,
                    timestamp=ts,
                    side=side[:4],
                    outcome=outcome[:3],
                    price=price,
                    size=size,
                )
                session.add(trade)
                try:
                    await session.flush()
                    inserted += 1
                except IntegrityError:
                    await session.rollback()

            except Exception as e:
                logger.debug("Skipping trade: %s", str(e)[:60])
                continue

        if inserted > 0:
            await session.commit()

    return inserted


async def process_market(
    client: httpx.AsyncClient,
    market: Market,
    delay: float,
) -> tuple[str, int]:
    """Process one market. Returns (status, trades_stored)."""
    condition_id = market.condition_id
    if not condition_id:
        return "no_condition_id", 0

    raw_trades = await fetch_all_trades(client, condition_id, delay)
    if not raw_trades:
        return "no_trades", 0

    stored = await store_trades(market.id, raw_trades)
    return ("success" if stored > 0 else "already_exists"), stored


async def get_target_markets(
    resolved_only: bool,
    active_only: bool,
    min_volume: float,
) -> list[Market]:
    async with async_session() as session:
        query = select(Market).where(Market.condition_id.isnot(None))

        if resolved_only:
            query = query.where(
                Market.resolution_value.isnot(None),
                Market.resolved_at.isnot(None),
            )
        elif active_only:
            query = query.where(Market.is_active == True)  # noqa: E712

        if min_volume > 0:
            query = query.where(Market.volume_total >= min_volume)

        query = query.order_by(Market.volume_total.desc().nullslast())
        result = await session.execute(query)
        return list(result.scalars().all())


async def run_backfill(args):
    await init_db()

    logger.info("=" * 60)
    logger.info("TRADE HISTORY BACKFILL")
    logger.info("=" * 60)

    ckpt = CheckpointManager(CHECKPOINT_FILE)
    prev = ckpt.load()
    if prev > 0:
        logger.info("Resuming: %d markets in checkpoint", prev)
    ckpt.open()

    markets = await get_target_markets(args.resolved_only, args.active_only, args.min_volume)
    logger.info("Target markets: %d | Min volume: %.0f | Delay: %.2fs",
                len(markets), args.min_volume, args.delay)

    remaining = sum(1 for m in markets if not ckpt.is_processed(m.id))
    logger.info("Remaining: %d", remaining)

    stats = {"success": 0, "no_trades": 0, "errors": 0, "total_trades": 0}
    start_time = time.time()
    processed = 0

    async with httpx.AsyncClient() as client:
        for market in markets:
            if shutdown_requested:
                break
            if ckpt.is_processed(market.id):
                continue

            try:
                status, trades = await process_market(client, market, args.delay)
            except Exception as e:
                status, trades = "error", 0
                logger.error("Market %d failed: %s", market.id, str(e)[:80])

            ckpt.record(market.id, status, trades)
            if status == "success":
                stats["success"] += 1
                stats["total_trades"] += trades
            elif status == "no_trades":
                stats["no_trades"] += 1
            elif status == "error":
                stats["errors"] += 1

            processed += 1

            if processed % 50 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta_h = (remaining - processed) / rate / 3600 if rate > 0 else 0
                logger.info(
                    "Progress: %d/%d | Success: %d | Trades: %d | %.1f/min | ETA: %.1fh",
                    processed, remaining, stats["success"],
                    stats["total_trades"], rate * 60, eta_h,
                )

    ckpt.close()

    STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATS_FILE, "w") as f:
        json.dump({**stats, "processed": processed, "total_markets": len(markets),
                    "runtime_sec": time.time() - start_time}, f, indent=2)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("COMPLETE in %.1f min", elapsed / 60)
    logger.info("Success: %d | No trades: %d | Errors: %d",
                stats["success"], stats["no_trades"], stats["errors"])
    logger.info("Total trades stored: %d", stats["total_trades"])
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Backfill trade history from Polymarket")
    parser.add_argument("--resolved-only", action="store_true")
    parser.add_argument("--active-only", action="store_true")
    parser.add_argument("--min-volume", type=float, default=0,
                        help="Minimum total volume to include (default: 0)")
    parser.add_argument("--delay", type=float, default=0.4,
                        help="Seconds between API requests (default: 0.4)")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore checkpoint, start fresh")
    args = parser.parse_args()

    if args.fresh and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.rename(CHECKPOINT_FILE.with_suffix(".jsonl.bak"))

    asyncio.run(run_backfill(args))


if __name__ == "__main__":
    main()
