"""High-frequency orderbook collector — 30-second snapshots for top markets.

Designed to run alongside the existing orderbook daemon. While the daemon
covers ~2,150 markets at ~5-minute intervals, this script hammers the top
N most liquid markets at 30-second intervals, building the dense orderbook
time series needed for realistic market-making backtests.

Features:
  - 30-second polling interval (configurable)
  - Top N markets by volume/liquidity, refreshed hourly
  - JSONL checkpoint for crash-safe resume and stats tracking
  - SIGINT handling for clean shutdown
  - Writes to the same orderbook_snapshots table as the daemon
  - Rate limiting to stay within Polymarket API limits

Usage:
  python scripts/collect_hf_orderbooks.py                     # Default: top 100, 30s
  python scripts/collect_hf_orderbooks.py --top 50 --interval 15
  python scripts/collect_hf_orderbooks.py --top 200 --interval 60
"""

import argparse
import json
import logging
import signal
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from data_pipeline.collectors.polymarket_clob import parse_orderbook

LOG_FILE = project_root / "data" / "logs" / "hf_orderbook_collector.log"
STATS_FILE = project_root / "data" / "checkpoints" / "hf_orderbook_stats.json"
PROGRESS_FILE = project_root / "data" / "checkpoints" / "hf_orderbook_progress.jsonl"
DB_PATH = project_root / "data" / "markets.db"
CLOB_URL = "https://clob.polymarket.com"

shutdown_requested = False


def handle_signal(signum, frame):
    global shutdown_requested
    shutdown_requested = True
    logging.getLogger(__name__).info("Shutdown requested (signal %d), finishing current cycle...", signum)


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

logger = logging.getLogger("hf_ob")


def setup_logging():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

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


class StatsTracker:
    def __init__(self):
        self.cycles = 0
        self.total_snapshots = 0
        self.total_errors = 0
        self.start_time = time.time()
        self._load()

    def _load(self):
        if STATS_FILE.exists():
            try:
                with open(STATS_FILE) as f:
                    d = json.load(f)
                self.cycles = d.get("cycles", 0)
                self.total_snapshots = d.get("total_snapshots", 0)
                self.total_errors = d.get("total_errors", 0)
                logger.info(
                    "Resumed: %d cycles, %d snapshots, %d errors",
                    self.cycles, self.total_snapshots, self.total_errors,
                )
            except Exception:
                pass

    def save(self):
        STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        elapsed = time.time() - self.start_time
        data = {
            "cycles": self.cycles,
            "total_snapshots": self.total_snapshots,
            "total_errors": self.total_errors,
            "runtime_seconds": round(elapsed, 1),
            "snapshots_per_hour": round(self.total_snapshots / max(elapsed / 3600, 0.01), 1),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        with open(STATS_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def log_progress(self, cycle_snapshots: int, cycle_errors: int, market_count: int):
        PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "cycle": self.cycles,
            "snapshots": cycle_snapshots,
            "errors": cycle_errors,
            "markets": market_count,
        }
        with open(PROGRESS_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")


def get_top_markets(n: int) -> list[dict]:
    """Get top N active markets by volume, with token_id and metadata."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, token_id_yes, question, volume_total, liquidity, taker_fee_bps
        FROM markets
        WHERE is_active = 1
          AND token_id_yes IS NOT NULL
          AND resolved_at IS NULL
          AND (end_date IS NULL OR end_date > datetime('now'))
        ORDER BY volume_total DESC
        LIMIT ?
    """, (n,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def fetch_orderbook_sync(token_id: str, client: httpx.Client) -> dict | None:
    """Fetch orderbook via REST API (synchronous)."""
    try:
        resp = client.get(f"{CLOB_URL}/book", params={"token_id": token_id}, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
    except Exception:
        return None


def insert_snapshot_sync(conn: sqlite3.Connection, market_id: int, parsed: dict):
    """Insert orderbook snapshot using synchronous sqlite3 (avoids async lock contention)."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO orderbook_snapshots
           (market_id, side, timestamp, best_bid, best_ask, bid_ask_spread,
            bid_depth_total, ask_depth_total, obi_level1, obi_weighted,
            depth_ratio, bids_json, asks_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            market_id,
            "yes",
            now,
            parsed["best_bid"],
            parsed["best_ask"],
            parsed["bid_ask_spread"],
            parsed["bid_depth_total"],
            parsed["ask_depth_total"],
            parsed["obi_level1"],
            parsed["obi_weighted"],
            parsed["depth_ratio"],
            json.dumps(parsed["bids"]),
            json.dumps(parsed["asks"]),
        ),
    )


def run_cycle(
    markets: list[dict],
    http_client: httpx.Client,
    db_conn: sqlite3.Connection,
    request_delay: float,
) -> tuple[int, int]:
    """Collect one snapshot per market. Returns (successes, errors)."""
    ok = 0
    err = 0
    for m in markets:
        if shutdown_requested:
            break
        token_id = m["token_id_yes"]
        raw = fetch_orderbook_sync(token_id, http_client)
        if raw is None:
            err += 1
            continue
        try:
            parsed = parse_orderbook(raw)
            insert_snapshot_sync(db_conn, m["id"], parsed)
            db_conn.commit()  # commit immediately — holds write lock for <1ms per market
            ok += 1
        except Exception as e:
            logger.debug("Insert error for market %d: %s", m["id"], str(e)[:60])
            try:
                db_conn.rollback()
            except Exception:
                pass
            err += 1
        time.sleep(request_delay)
    return ok, err


def main():
    parser = argparse.ArgumentParser(description="High-frequency orderbook collector")
    parser.add_argument("--top", type=int, default=100, help="Number of top markets to collect")
    parser.add_argument("--interval", type=int, default=30, help="Seconds between collection cycles")
    parser.add_argument("--request-delay", type=float, default=0.15, help="Delay between API requests (seconds)")
    parser.add_argument("--refresh-interval", type=int, default=3600, help="Seconds between market list refreshes")
    args = parser.parse_args()

    setup_logging()
    stats = StatsTracker()

    logger.info("=" * 60)
    logger.info("HIGH-FREQUENCY ORDERBOOK COLLECTOR")
    logger.info("  Top markets: %d", args.top)
    logger.info("  Collection interval: %ds", args.interval)
    logger.info("  Request delay: %.2fs", args.request_delay)
    logger.info("  Market refresh: every %ds", args.refresh_interval)
    logger.info("=" * 60)

    markets = get_top_markets(args.top)
    logger.info("Loaded %d markets for HF collection", len(markets))
    last_refresh = time.time()

    db_conn = sqlite3.connect(str(DB_PATH), timeout=60)
    db_conn.execute("PRAGMA journal_mode=WAL")
    db_conn.execute("PRAGMA synchronous=NORMAL")

    http_client = httpx.Client(http2=True, timeout=10)

    try:
        while not shutdown_requested:
            cycle_start = time.time()

            if time.time() - last_refresh > args.refresh_interval:
                markets = get_top_markets(args.top)
                logger.info("Refreshed market list: %d markets", len(markets))
                last_refresh = time.time()

            ok, err = run_cycle(markets, http_client, db_conn, args.request_delay)
            stats.cycles += 1
            stats.total_snapshots += ok
            stats.total_errors += err
            stats.log_progress(ok, err, len(markets))

            if stats.cycles % 10 == 0:
                stats.save()

            elapsed = time.time() - cycle_start
            logger.info(
                "Cycle %d: %d/%d ok (%d err) in %.1fs | Total: %d snapshots",
                stats.cycles, ok, len(markets), err, elapsed, stats.total_snapshots,
            )

            sleep_time = max(0, args.interval - elapsed)
            if sleep_time > 0 and not shutdown_requested:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received")
    finally:
        stats.save()
        db_conn.close()
        http_client.close()
        logger.info("Shutdown complete. %d total snapshots collected.", stats.total_snapshots)


if __name__ == "__main__":
    main()
