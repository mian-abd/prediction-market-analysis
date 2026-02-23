"""Backfill historical resolved markets for ML training.

Run once to populate database with outcomes for training XGBoost/LightGBM.
Fetches from both Kalshi and Polymarket APIs, then bulk-writes via raw
sqlite3 (synchronous) to avoid all async/ORM lock issues.

Improvements:
- Fetches up to 10000 Kalshi + 5000 Polymarket markets
- Bulk write with direct sqlite3 — zero "database is locked" issues
- Filters for volume > 0 post-fetch (only useful training samples)
- Reports usable vs total counts
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import logging
import sqlite3
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


def _get_db_path() -> str:
    """Resolve the SQLite database path from settings."""
    from config.settings import settings
    url = settings.database_url
    # sqlite+aiosqlite:///./data/markets.db  →  data/markets.db
    path = url.replace("sqlite+aiosqlite:///", "").replace("sqlite:///", "")
    db_path = Path(path)
    if not db_path.is_absolute():
        db_path = project_root / db_path
    return str(db_path)


def _get_platform_ids_sync(conn: sqlite3.Connection) -> dict[str, int]:
    """Ensure platform rows exist and return {name: id}."""
    platforms = {}
    for name, url in [("polymarket", "https://polymarket.com"), ("kalshi", "https://kalshi.com")]:
        row = conn.execute("SELECT id FROM platforms WHERE name=?", (name,)).fetchone()
        if row:
            platforms[name] = row[0]
        else:
            cursor = conn.execute(
                "INSERT INTO platforms (name, base_url, is_active) VALUES (?, ?, 1)",
                (name, url),
            )
            conn.commit()
            platforms[name] = cursor.lastrowid
    return platforms


def _upsert_markets_sync(
    conn: sqlite3.Connection,
    parsed_markets: list[dict],
    platform_id: int,
) -> int:
    """Bulk upsert markets using raw sqlite3 INSERT OR REPLACE.

    Uses a single transaction for the entire batch — fastest possible
    approach and eliminates all lock contention.
    """
    count = 0
    errors = 0
    now = datetime.utcnow().isoformat()

    # Use executemany for speed; build the full params list first
    INSERT_SQL = """INSERT OR REPLACE INTO markets (
        platform_id, external_id, condition_id, token_id_yes, token_id_no,
        question, description, category, normalized_category, slug,
        price_yes, price_no, volume_24h, volume_total, liquidity, open_interest,
        is_active, is_resolved, resolution_outcome, resolution_value,
        end_date, resolved_at, created_at, updated_at, last_fetched_at,
        taker_fee_bps, maker_fee_bps, is_neg_risk
    ) VALUES (?,?,?,?,?, ?,?,?,?,?, ?,?,?,?,?,?, ?,?,?,?, ?,?,?,?,?, ?,?,?)"""

    rows = []
    for m in parsed_markets:
        try:
            rows.append((
                platform_id,
                m["external_id"],
                m.get("condition_id"),
                m.get("token_id_yes"),
                m.get("token_id_no"),
                m.get("question", ""),
                m.get("description", ""),
                m.get("category", ""),
                _normalize_cat(m.get("category", ""), m.get("question", ""), m.get("description", "")),
                m.get("slug", ""),
                m.get("price_yes"),
                m.get("price_no"),
                m.get("volume_24h", 0),
                m.get("volume_total", 0),
                m.get("liquidity", 0),
                m.get("open_interest", 0),
                1 if m.get("is_active") else 0,
                1 if m.get("is_resolved") else 0,
                m.get("resolution_outcome"),
                m.get("resolution_value"),
                _dt(m.get("end_date")),
                _dt(m.get("resolved_at")),
                now,
                now,
                now,
                m.get("taker_fee_bps", 0),
                m.get("maker_fee_bps", 0),
                1 if m.get("is_neg_risk") else 0,
            ))
        except Exception as e:
            errors += 1
            if errors <= 3:
                logger.warning(f"Param build error for {m.get('external_id','?')[:30]}: {e}")

    if not rows:
        logger.warning("No rows to insert!")
        return 0

    try:
        conn.executemany(INSERT_SQL, rows)
        conn.commit()
        count = len(rows) - errors
        logger.info(f"executemany complete: {count} rows inserted/replaced")
    except Exception as e:
        logger.error(f"Bulk insert failed: {type(e).__name__}: {e}")
        conn.rollback()

    return count


def _dt(val) -> str | None:
    """Convert datetime to ISO string, strip timezone."""
    if val is None:
        return None
    if hasattr(val, "isoformat"):
        if hasattr(val, "tzinfo") and val.tzinfo:
            val = val.replace(tzinfo=None)
        return val.isoformat()
    return str(val)


def _normalize_cat(category: str, question: str, description: str) -> str:
    """Simple category normalization."""
    try:
        from data_pipeline.category_normalizer import normalize_category
        return normalize_category(category, question, description) or "other"
    except Exception:
        return (category or "other").lower()


async def backfill_kalshi_resolved(db_path: str, max_markets: int = 10000):
    """Fetch Kalshi resolved markets (async HTTP), then write via sqlite3."""
    from data_pipeline.collectors.kalshi_resolved import (
        fetch_all_resolved_markets,
        parse_resolved_market,
    )

    logger.info(f"Fetching up to {max_markets} settled markets from Kalshi...")
    raw_markets = await fetch_all_resolved_markets(max_markets=max_markets)

    logger.info(f"Parsing {len(raw_markets)} settled markets...")
    parsed_markets = [parse_resolved_market(m) for m in raw_markets]

    valid_markets = [m for m in parsed_markets if m.get("resolution_value") is not None]
    with_volume = [m for m in valid_markets if (m.get("volume_total") or 0) > 0]
    logger.info(
        f"Kalshi: {len(valid_markets)} with outcomes, "
        f"{len(with_volume)} with volume (usable for training)"
    )

    logger.info("Storing in database...")
    with sqlite3.connect(db_path, timeout=30) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        platforms = _get_platform_ids_sync(conn)
        count = _upsert_markets_sync(conn, valid_markets, platforms["kalshi"])

    logger.info(f"Backfilled {count} resolved Kalshi markets")
    return count, len(with_volume)


async def backfill_polymarket_resolved(db_path: str, max_markets: int = 2000):
    """Fetch Polymarket resolved markets (async HTTP), then write via sqlite3."""
    from data_pipeline.collectors.polymarket_resolved import (
        fetch_all_resolved_markets,
        parse_resolved_market,
    )

    logger.info(f"Fetching up to {max_markets} resolved markets from Polymarket...")
    raw_markets = await fetch_all_resolved_markets(max_markets=max_markets)

    logger.info(f"Parsing {len(raw_markets)} resolved markets...")
    parsed_markets = [parse_resolved_market(m) for m in raw_markets]

    valid_markets = [
        m for m in parsed_markets
        if m.get("resolution_value") is not None
        and m.get("resolution_value") in (0.0, 1.0)
    ]
    with_volume = [m for m in valid_markets if (m.get("volume_total") or 0) > 0]
    logger.info(
        f"Polymarket: {len(valid_markets)} with outcomes, "
        f"{len(with_volume)} with volume (usable for training)"
    )

    logger.info("Storing in database...")
    with sqlite3.connect(db_path, timeout=30) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        platforms = _get_platform_ids_sync(conn)
        count = _upsert_markets_sync(conn, valid_markets, platforms["polymarket"])

    logger.info(f"Backfilled {count} resolved Polymarket markets")
    return count, len(with_volume)


async def main():
    """Run backfill process."""
    # Ensure DB tables exist via SQLAlchemy (creates schema if needed)
    logger.info("Initializing database schema...")
    from db.database import init_db, engine
    await init_db()
    await engine.dispose()  # release all async connections before sqlite3 writes
    logger.info("Schema ready.")

    db_path = _get_db_path()
    logger.info(f"Database: {db_path}")
    logger.info("Starting resolved markets backfill...")
    start_time = datetime.now()

    kalshi_count, kalshi_usable = await backfill_kalshi_resolved(db_path, max_markets=10000)

    poly_count, poly_usable = 0, 0
    try:
        poly_count, poly_usable = await backfill_polymarket_resolved(db_path, max_markets=5000)
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
