"""Live trade stream collector via Polymarket WebSocket.

Connects to the Polymarket CLOB WebSocket and captures trade executions
in real time. Inserts into the `trades` table with precise timestamps.

Features:
  - WebSocket connection to wss://ws-subscriptions-clob.polymarket.com/ws/market
  - Auto-reconnect with exponential backoff
  - PING/PONG keepalive (every 10 seconds)
  - Subscribes to top N markets by volume
  - Refreshes subscription list periodically (adds new markets, drops resolved)
  - JSONL checkpoint for stats tracking
  - SIGINT handling for clean shutdown
  - Deduplication via external_trade_id

Usage:
  python scripts/collect_live_trades.py                      # Default: top 200
  python scripts/collect_live_trades.py --top 500            # Max per connection
  python scripts/collect_live_trades.py --top 100 --refresh 1800
"""

import argparse
import asyncio
import json
import logging
import signal
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    import websockets
except ImportError:
    print("ERROR: websockets not installed. Run: pip install websockets")
    sys.exit(1)

LOG_FILE = project_root / "data" / "logs" / "live_trade_collector.log"
STATS_FILE = project_root / "data" / "checkpoints" / "live_trade_stats.json"
PROGRESS_FILE = project_root / "data" / "checkpoints" / "live_trade_progress.jsonl"
DB_PATH = project_root / "data" / "markets.db"

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
MAX_ASSETS_PER_CONNECTION = 500
PING_INTERVAL = 10
RECONNECT_BASE_DELAY = 2
RECONNECT_MAX_DELAY = 120

shutdown_requested = False
logger = logging.getLogger("live_trades")


def handle_signal(signum, frame):
    global shutdown_requested
    shutdown_requested = True
    logger.info("Shutdown requested (signal %d)", signum)


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


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


class TradeStats:
    def __init__(self):
        self.trades_inserted = 0
        self.trades_duplicate = 0
        self.messages_received = 0
        self.connections = 0
        self.errors = 0
        self.start_time = time.time()
        self._load()

    def _load(self):
        if STATS_FILE.exists():
            try:
                with open(STATS_FILE) as f:
                    d = json.load(f)
                self.trades_inserted = d.get("trades_inserted", 0)
                self.trades_duplicate = d.get("trades_duplicate", 0)
                self.messages_received = d.get("messages_received", 0)
                self.connections = d.get("connections", 0)
                self.errors = d.get("errors", 0)
                logger.info(
                    "Resumed: %d trades, %d messages, %d connections",
                    self.trades_inserted, self.messages_received, self.connections,
                )
            except Exception:
                pass

    def save(self):
        STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        elapsed = time.time() - self.start_time
        data = {
            "trades_inserted": self.trades_inserted,
            "trades_duplicate": self.trades_duplicate,
            "messages_received": self.messages_received,
            "connections": self.connections,
            "errors": self.errors,
            "runtime_seconds": round(elapsed, 1),
            "trades_per_hour": round(self.trades_inserted / max(elapsed / 3600, 0.01), 1),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        with open(STATS_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def log_progress(self):
        PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "trades": self.trades_inserted,
            "dupes": self.trades_duplicate,
            "msgs": self.messages_received,
            "errors": self.errors,
        }
        with open(PROGRESS_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")


def get_top_market_tokens(n: int) -> dict[str, int]:
    """Get token_id -> market_id mapping for top N active markets."""
    conn = sqlite3.connect(str(DB_PATH), timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    rows = conn.execute("""
        SELECT id, token_id_yes
        FROM markets
        WHERE is_active = 1
          AND token_id_yes IS NOT NULL
          AND resolved_at IS NULL
          AND (end_date IS NULL OR end_date > datetime('now'))
        ORDER BY volume_total DESC
        LIMIT ?
    """, (n,)).fetchall()
    conn.close()
    return {row[1]: row[0] for row in rows}


def insert_trade(conn: sqlite3.Connection, market_id: int, trade_data: dict) -> bool:
    """Insert a trade, returning True if new, False if duplicate."""
    trade_id = trade_data.get("id", "") or trade_data.get("tradeId", "")
    timestamp = trade_data.get("timestamp") or trade_data.get("t")
    price = trade_data.get("price") or trade_data.get("p")
    size = trade_data.get("size") or trade_data.get("s")
    side = trade_data.get("side", "").upper()

    if not all([price, size]):
        return False

    try:
        price = float(price)
        size = float(size)
    except (ValueError, TypeError):
        return False

    if isinstance(timestamp, (int, float)):
        ts = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
    elif isinstance(timestamp, str):
        ts = timestamp
    else:
        ts = datetime.now(timezone.utc).isoformat()

    if not side or side not in ("BUY", "SELL"):
        side = "BUY"

    outcome = "Yes"

    ext_id = str(trade_id) if trade_id else f"ws_{ts}_{price}_{size}"

    try:
        conn.execute(
            """INSERT OR IGNORE INTO trades
               (market_id, external_trade_id, timestamp, side, outcome, price, size)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (market_id, ext_id, ts, side, outcome, price, size),
        )
        return True
    except sqlite3.IntegrityError:
        return False


def parse_ws_message(msg: str, token_to_market: dict[str, int]) -> list[tuple[int, dict]]:
    """Parse a WebSocket message into (market_id, trade_data) pairs."""
    try:
        data = json.loads(msg)
    except json.JSONDecodeError:
        return []

    trades = []

    if isinstance(data, list):
        for item in data:
            trades.extend(_extract_trades_from_event(item, token_to_market))
    elif isinstance(data, dict):
        trades.extend(_extract_trades_from_event(data, token_to_market))

    return trades


def _extract_trades_from_event(event: dict, token_to_market: dict[str, int]) -> list[tuple[int, dict]]:
    """Extract trade events from a single WS event."""
    event_type = event.get("event_type", "") or event.get("type", "")
    asset_id = event.get("asset_id", "") or event.get("market", "")

    market_id = token_to_market.get(asset_id)
    if market_id is None:
        return []

    if event_type in ("last_trade_price", "trade"):
        return [(market_id, event)]

    if event_type == "book" and "trades" in event:
        return [(market_id, t) for t in event["trades"]]

    return []


async def run_ws_collector(top_n: int, refresh_interval: int, stats: TradeStats):
    """Main WebSocket collection loop with auto-reconnect."""
    reconnect_delay = RECONNECT_BASE_DELAY

    while not shutdown_requested:
        token_to_market = get_top_market_tokens(min(top_n, MAX_ASSETS_PER_CONNECTION))
        token_ids = list(token_to_market.keys())

        if not token_ids:
            logger.warning("No active markets found, retrying in 60s...")
            await asyncio.sleep(60)
            continue

        logger.info("Connecting to WebSocket with %d markets...", len(token_ids))

        try:
            async with websockets.connect(WS_URL, ping_interval=None) as ws:
                stats.connections += 1
                reconnect_delay = RECONNECT_BASE_DELAY
                logger.info("Connected (#%d)", stats.connections)

                sub_msg = json.dumps({
                    "assets_ids": token_ids,
                    "type": "market",
                    "custom_feature_enabled": True,
                })
                await ws.send(sub_msg)
                logger.info("Subscribed to %d assets", len(token_ids))

                db_conn = sqlite3.connect(str(DB_PATH), timeout=60)
                db_conn.execute("PRAGMA journal_mode=WAL")

                last_ping = time.time()
                last_refresh = time.time()
                last_save = time.time()
                batch_count = 0

                try:
                    while not shutdown_requested:
                        now = time.time()

                        if now - last_ping > PING_INTERVAL:
                            try:
                                await ws.send("PING")
                                last_ping = now
                            except Exception:
                                break

                        if now - last_refresh > refresh_interval:
                            new_map = get_top_market_tokens(min(top_n, MAX_ASSETS_PER_CONNECTION))
                            added = set(new_map.keys()) - set(token_to_market.keys())
                            removed = set(token_to_market.keys()) - set(new_map.keys())

                            if added:
                                await ws.send(json.dumps({
                                    "assets_ids": list(added),
                                    "type": "market",
                                    "operation": "subscribe",
                                }))
                            if removed:
                                await ws.send(json.dumps({
                                    "assets_ids": list(removed),
                                    "type": "market",
                                    "operation": "unsubscribe",
                                }))

                            if added or removed:
                                logger.info("Refreshed subs: +%d -%d (total %d)", len(added), len(removed), len(new_map))
                            token_to_market = new_map
                            last_refresh = now

                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=15)
                        except asyncio.TimeoutError:
                            continue

                        if raw == "PONG":
                            continue

                        stats.messages_received += 1
                        trade_events = parse_ws_message(raw, token_to_market)

                        for market_id, trade_data in trade_events:
                            try:
                                inserted = insert_trade(db_conn, market_id, trade_data)
                            except sqlite3.OperationalError:
                                continue  # brief lock contention — skip single trade
                            if inserted:
                                stats.trades_inserted += 1
                                batch_count += 1
                            else:
                                stats.trades_duplicate += 1

                        if batch_count >= 50:
                            try:
                                db_conn.commit()
                            except sqlite3.OperationalError:
                                pass  # busy/locked — will retry on next batch
                            batch_count = 0

                        if now - last_save > 60:
                            try:
                                db_conn.commit()
                            except sqlite3.OperationalError as e:
                                logger.debug("Commit deferred (locked): %s", e)
                            stats.save()
                            stats.log_progress()
                            last_save = now
                            batch_count = 0
                            logger.info(
                                "Stats: %d trades, %d msgs, %d dupes",
                                stats.trades_inserted, stats.messages_received, stats.trades_duplicate,
                            )

                finally:
                    try:
                        db_conn.commit()
                    except sqlite3.OperationalError:
                        pass  # lock contention on cleanup — data in current batch may be lost
                    db_conn.close()

        except websockets.exceptions.ConnectionClosed as e:
            logger.warning("WebSocket closed: %s", e)
            stats.errors += 1
        except Exception as e:
            logger.error("WebSocket error: %s", e)
            stats.errors += 1

        if not shutdown_requested:
            logger.info("Reconnecting in %ds...", reconnect_delay)
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, RECONNECT_MAX_DELAY)

    stats.save()
    logger.info("Collector stopped. Total: %d trades from %d messages.", stats.trades_inserted, stats.messages_received)


def main():
    parser = argparse.ArgumentParser(description="Live trade stream collector (WebSocket)")
    parser.add_argument("--top", type=int, default=200, help="Number of top markets to subscribe")
    parser.add_argument("--refresh", type=int, default=3600, help="Subscription refresh interval (seconds)")
    args = parser.parse_args()

    setup_logging()
    stats = TradeStats()

    logger.info("=" * 60)
    logger.info("LIVE TRADE STREAM COLLECTOR")
    logger.info("  Markets: top %d", args.top)
    logger.info("  Refresh interval: %ds", args.refresh)
    logger.info("  WebSocket: %s", WS_URL)
    logger.info("=" * 60)

    asyncio.run(run_ws_collector(args.top, args.refresh, stats))


if __name__ == "__main__":
    main()
