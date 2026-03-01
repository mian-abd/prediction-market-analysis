"""Market status refresher — keeps DB in sync with live Polymarket state.

Polls the Polymarket Gamma API for market metadata updates, fixing stale
is_active flags, updating end_date, resolution_value, and fee rates.

This prevents the orderbook daemon from wasting requests on resolved markets
and ensures accurate fee calculations for the market making engine.

Features:
  - Polls Gamma API for all markets (paginated, 100 per page)
  - Updates: is_active, is_resolved, resolved_at, resolution_value, end_date,
    taker_fee_bps, price_yes, volume_total, liquidity
  - JSONL checkpoint with per-run stats
  - Can run on a loop (every 30 min) or as a one-shot
  - SIGINT handling for clean shutdown

Usage:
  python scripts/refresh_market_status.py                    # One-shot refresh
  python scripts/refresh_market_status.py --loop 1800        # Every 30 min
  python scripts/refresh_market_status.py --loop 900 --limit 5000
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

LOG_FILE = project_root / "data" / "logs" / "market_refresh.log"
STATS_FILE = project_root / "data" / "checkpoints" / "market_refresh_stats.json"
PROGRESS_FILE = project_root / "data" / "checkpoints" / "market_refresh_progress.jsonl"
DB_PATH = project_root / "data" / "markets.db"

GAMMA_URL = "https://gamma-api.polymarket.com"

shutdown_requested = False
logger = logging.getLogger("market_refresh")


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


def fetch_gamma_markets(client: httpx.Client, offset: int = 0, limit: int = 100) -> list[dict]:
    """Fetch a page of active+open markets from Gamma API.

    Filtering active=true&closed=false returns current markets (IDs 500K+)
    which are the ones actually in our DB, rather than the default ascending-
    ID sort that starts with 2020-era resolved markets we never imported.
    """
    try:
        resp = client.get(
            f"{GAMMA_URL}/markets",
            params={"offset": offset, "limit": limit, "active": "true", "closed": "false"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning("Gamma API error at offset %d: %s", offset, str(e)[:80])
        return []


def map_gamma_to_db(gm: dict) -> dict | None:
    """Extract relevant fields from a Gamma market object."""
    condition_id = gm.get("conditionId") or gm.get("condition_id")
    slug = gm.get("slug", "") or ""
    if not condition_id and not slug:
        return None

    active = gm.get("active", False)
    closed = gm.get("closed", False)
    resolved = gm.get("resolved", False) or gm.get("is_resolved", False)

    end_date = gm.get("endDate") or gm.get("end_date_iso")
    resolved_at = gm.get("resolvedAt") or gm.get("resolved_at")

    tokens = gm.get("tokens", []) or gm.get("clobTokenIds", [])
    token_yes = None
    token_no = None
    if isinstance(tokens, list) and len(tokens) >= 2:
        if isinstance(tokens[0], dict):
            token_yes = tokens[0].get("token_id")
            token_no = tokens[1].get("token_id") if len(tokens) > 1 else None
        else:
            token_yes = str(tokens[0])
            token_no = str(tokens[1]) if len(tokens) > 1 else None

    outcome_prices = gm.get("outcomePrices")
    price_yes = None
    if outcome_prices:
        try:
            if isinstance(outcome_prices, str):
                prices = json.loads(outcome_prices)
            else:
                prices = outcome_prices
            if isinstance(prices, list) and len(prices) >= 1:
                price_yes = float(prices[0])
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    volume = None
    for key in ("volume", "volumeNum", "volume_num"):
        if key in gm and gm[key] is not None:
            try:
                volume = float(gm[key])
                break
            except (ValueError, TypeError):
                pass

    liquidity = None
    for key in ("liquidity", "liquidityNum"):
        if key in gm and gm[key] is not None:
            try:
                liquidity = float(gm[key])
                break
            except (ValueError, TypeError):
                pass

    resolution_value = None
    if resolved:
        outcome = gm.get("outcome", "")
        if outcome in ("Yes", "yes", "YES", "1"):
            resolution_value = 1.0
        elif outcome in ("No", "no", "NO", "0"):
            resolution_value = 0.0

    return {
        "condition_id": condition_id,
        "slug": slug,
        "is_active": active and not closed,
        "is_resolved": resolved,
        "resolved_at": resolved_at,
        "resolution_value": resolution_value,
        "end_date": end_date,
        "token_id_yes": token_yes,
        "token_id_no": token_no,
        "price_yes": price_yes,
        "volume_total": volume,
        "liquidity": liquidity,
    }


def refresh_db(markets_data: list[dict]) -> dict:
    """Apply updates to the database. Returns stats.

    Matches first by condition_id (indexed), then falls back to slug (also
    indexed) for markets that were loaded without a condition_id. Commits
    every BATCH_SIZE rows so the write lock is released frequently.
    """
    BATCH_SIZE = 200

    conn = sqlite3.connect(str(DB_PATH), timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")

    updated = 0
    not_found = 0
    errors = 0
    pending = 0

    for md in markets_data:
        if shutdown_requested:
            break
        try:
            row = None

            # Primary: match by condition_id (indexed)
            cid = md.get("condition_id")
            if cid:
                row = conn.execute(
                    "SELECT id FROM markets WHERE condition_id = ?",
                    (cid,),
                ).fetchone()

            # Fallback: match by slug (indexed) — covers the 330K markets
            # that were loaded from Gamma without a condition_id
            if row is None and md.get("slug"):
                row = conn.execute(
                    "SELECT id FROM markets WHERE slug = ?",
                    (md["slug"],),
                ).fetchone()

            if row is None:
                not_found += 1
                continue

            market_id = row[0]
            set_parts = []
            params = []

            for col in ("is_active", "is_resolved", "resolved_at", "resolution_value",
                        "end_date", "price_yes", "volume_total", "liquidity"):
                val = md.get(col)
                if val is not None:
                    set_parts.append(f"{col} = ?")
                    params.append(val)

            # Back-fill condition_id for markets that had it NULL
            if cid and md.get("slug"):
                set_parts.append("condition_id = ?")
                params.append(cid)

            if md.get("token_id_yes"):
                set_parts.append("token_id_yes = ?")
                params.append(md["token_id_yes"])
            if md.get("token_id_no"):
                set_parts.append("token_id_no = ?")
                params.append(md["token_id_no"])

            set_parts.append("last_fetched_at = ?")
            params.append(datetime.now(timezone.utc).isoformat())
            params.append(market_id)

            if set_parts:
                sql = f"UPDATE markets SET {', '.join(set_parts)} WHERE id = ?"
                conn.execute(sql, params)
                updated += 1
                pending += 1

            # Release write lock frequently so collectors aren't blocked
            if pending >= BATCH_SIZE:
                try:
                    conn.commit()
                except sqlite3.OperationalError as e:
                    logger.debug("Batch commit deferred (locked): %s", e)
                pending = 0

        except Exception as e:
            logger.debug("Update error for %s: %s", md.get("condition_id", "?")[:12], str(e)[:60])
            errors += 1

    try:
        conn.commit()
    except sqlite3.OperationalError as e:
        logger.warning("Final commit failed (locked): %s", e)
    conn.close()
    return {"updated": updated, "not_found": not_found, "errors": errors}


def run_refresh(max_markets: int) -> dict:
    """Run a single refresh cycle."""
    logger.info("Fetching markets from Gamma API...")
    client = httpx.Client(timeout=30)

    all_markets = []
    offset = 0
    page_size = 100

    while len(all_markets) < max_markets and not shutdown_requested:
        page = fetch_gamma_markets(client, offset, page_size)
        if not page:
            break
        all_markets.extend(page)
        offset += page_size
        if len(page) < page_size:
            break
        time.sleep(0.2)

    client.close()
    logger.info("Fetched %d markets from Gamma API", len(all_markets))

    parsed = []
    for gm in all_markets:
        md = map_gamma_to_db(gm)
        if md:
            parsed.append(md)

    logger.info("Parsed %d markets for DB update", len(parsed))

    if not parsed:
        return {"fetched": len(all_markets), "parsed": 0, "updated": 0}

    result = refresh_db(parsed)
    result["fetched"] = len(all_markets)
    result["parsed"] = len(parsed)

    logger.info(
        "Refresh complete: %d fetched, %d parsed, %d updated, %d not found, %d errors",
        result["fetched"], result["parsed"], result["updated"],
        result.get("not_found", 0), result.get("errors", 0),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Market status refresher")
    parser.add_argument("--loop", type=int, default=0, help="Loop interval in seconds (0=one-shot)")
    parser.add_argument("--limit", type=int, default=36000, help="Max markets to fetch per cycle")
    args = parser.parse_args()

    setup_logging()

    logger.info("=" * 60)
    logger.info("MARKET STATUS REFRESHER")
    logger.info("  Mode: %s", f"Loop every {args.loop}s" if args.loop else "One-shot")
    logger.info("  Max markets: %d", args.limit)
    logger.info("=" * 60)

    if args.loop > 0:
        cycle = 0
        while not shutdown_requested:
            cycle += 1
            logger.info("--- Refresh cycle %d ---", cycle)
            result = run_refresh(args.limit)

            PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(PROGRESS_FILE, "a") as f:
                entry = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "cycle": cycle,
                    **result,
                }
                f.write(json.dumps(entry) + "\n")

            if not shutdown_requested:
                logger.info("Next refresh in %ds...", args.loop)
                time.sleep(args.loop)
    else:
        result = run_refresh(args.limit)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
