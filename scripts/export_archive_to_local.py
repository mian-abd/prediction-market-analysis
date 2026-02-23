"""Export old time-series rows from Postgres to local Parquet archive.

Run this BEFORE enabling cleanup (CLEANUP_ENABLED=true) so that no data is
ever deleted from Postgres without first being safely archived on disk.

Usage:
    python scripts/export_archive_to_local.py
    python scripts/export_archive_to_local.py --archive-dir data/archive --older-than-days 7
    python scripts/export_archive_to_local.py --older-than-days 14 --include-arb
    python scripts/export_archive_to_local.py --older-than-days 7 --force   # re-export existing days

After export, inspect data/archive/manifest_<run_id>.json to confirm row counts,
then enable cleanup in your environment:
    CLEANUP_ENABLED=true  RETENTION_DAYS=7

Flow:
    1. Determine cutoff  = now() - older_than_days
    2. For each table (price_snapshots, orderbook_snapshots, optionally arb):
       a. Query rows grouped by date (UTC day)
       b. Write one Parquet file per day to <archive_dir>/<table>/YYYY-MM-DD.parquet
       c. Skip days that already have a file (unless --force)
    3. Write manifest JSON with cutoff, row counts, file list.
    4. Print summary and exit with code 0.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import pandas as pd
from sqlalchemy import select, text

from db.database import async_session, init_db
from db.models import PriceSnapshot, OrderbookSnapshot, ArbitrageOpportunity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

CHUNK_SIZE = 10_000  # rows per DB fetch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _day_str(ts) -> str:
    """Return 'YYYY-MM-DD' from a datetime (naïve or aware)."""
    if hasattr(ts, "date"):
        return ts.date().isoformat()
    return str(ts)[:10]


async def _fetch_table_rows(session, model, timestamp_col, cutoff) -> list[dict]:
    """Stream all rows older than cutoff from `model` in chunks."""
    rows: list[dict] = []
    offset = 0
    while True:
        result = await session.execute(
            select(model)
            .where(timestamp_col < cutoff)
            .order_by(timestamp_col)
            .limit(CHUNK_SIZE)
            .offset(offset)
        )
        batch = result.scalars().all()
        if not batch:
            break
        for row in batch:
            rows.append({c.key: getattr(row, c.key) for c in row.__table__.columns})
        logger.info(f"  fetched {offset + len(batch)} rows so far ...")
        offset += len(batch)
        if len(batch) < CHUNK_SIZE:
            break
    return rows


def _write_day_parquet(
    df_day: pd.DataFrame,
    table_dir: Path,
    day: str,
    force: bool,
) -> tuple[Path, int]:
    """Write a single-day DataFrame to <table_dir>/YYYY-MM-DD.parquet.

    Returns (path, rows_written).  Skips if file exists and not --force.
    """
    out_path = table_dir / f"{day}.parquet"
    if out_path.exists() and not force:
        logger.info(f"    {out_path.name} already exists, skipping (use --force to re-export)")
        return out_path, 0
    table_dir.mkdir(parents=True, exist_ok=True)
    df_day.to_parquet(out_path, index=False, compression="gzip", engine="pyarrow")
    return out_path, len(df_day)


def _export_rows_to_parquet(
    rows: list[dict],
    timestamp_field: str,
    table_dir: Path,
    force: bool,
) -> tuple[dict[str, int], list[str]]:
    """Group rows by UTC day and write one Parquet file per day.

    Returns:
        day_counts: {day_str: rows_written}
        files_written: list of file paths written (newly written or existing)
    """
    if not rows:
        return {}, []

    df = pd.DataFrame(rows)
    # Normalise timestamp column to UTC-naive for consistent day bucketing
    ts_col = df[timestamp_field]
    if pd.api.types.is_datetime64_any_dtype(ts_col):
        df["_day"] = ts_col.dt.tz_localize(None).dt.date.astype(str)
    else:
        df["_day"] = ts_col.astype(str).str[:10]

    day_counts: dict[str, int] = {}
    files_written: list[str] = []

    for day, group in df.groupby("_day"):
        day_df = group.drop(columns=["_day"])
        path, n = _write_day_parquet(day_df, table_dir, str(day), force)
        files_written.append(str(path))
        if n > 0:
            day_counts[str(day)] = n
            logger.info(f"    wrote {n:,} rows → {path.name}")
        else:
            day_counts[str(day)] = 0

    return day_counts, files_written


# ---------------------------------------------------------------------------
# Main export logic
# ---------------------------------------------------------------------------

async def export_archive(
    archive_dir: Path,
    older_than_days: int,
    include_arb: bool,
    force: bool,
) -> dict:
    """Run the full export and return the manifest dict."""
    await init_db()

    cutoff = datetime.utcnow() - timedelta(days=older_than_days)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    logger.info("=" * 60)
    logger.info("EXPORT ARCHIVE TO LOCAL PARQUET")
    logger.info("=" * 60)
    logger.info(f"Cutoff      : {cutoff.isoformat()} UTC  (older_than_days={older_than_days})")
    logger.info(f"Archive dir : {archive_dir.resolve()}")
    logger.info(f"Run ID      : {run_id}")
    logger.info(f"Include arb : {include_arb}")
    logger.info(f"Force       : {force}")
    logger.info("")

    manifest: dict = {
        "run_id": run_id,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "cutoff": cutoff.isoformat(),
        "older_than_days": older_than_days,
        "archive_dir": str(archive_dir.resolve()),
        "tables": {},
        "files": [],
        "total_rows_exported": 0,
    }

    async with async_session() as session:

        # ── price_snapshots ────────────────────────────────────────────
        logger.info("Exporting price_snapshots ...")
        price_rows = await _fetch_table_rows(
            session, PriceSnapshot, PriceSnapshot.timestamp, cutoff
        )
        logger.info(f"  total rows: {len(price_rows):,}")

        price_dir = archive_dir / "price_snapshots"
        price_day_counts, price_files = _export_rows_to_parquet(
            price_rows, "timestamp", price_dir, force
        )
        total_price = sum(price_day_counts.values())
        manifest["tables"]["price_snapshots"] = {
            "total_rows": len(price_rows),
            "rows_written": total_price,
            "days": price_day_counts,
        }
        manifest["files"].extend(price_files)
        logger.info(f"  price_snapshots: {total_price:,} rows written across {len(price_day_counts)} days")

        # ── orderbook_snapshots ────────────────────────────────────────
        logger.info("Exporting orderbook_snapshots ...")
        ob_rows = await _fetch_table_rows(
            session, OrderbookSnapshot, OrderbookSnapshot.timestamp, cutoff
        )
        logger.info(f"  total rows: {len(ob_rows):,}")

        ob_dir = archive_dir / "orderbook_snapshots"
        ob_day_counts, ob_files = _export_rows_to_parquet(
            ob_rows, "timestamp", ob_dir, force
        )
        total_ob = sum(ob_day_counts.values())
        manifest["tables"]["orderbook_snapshots"] = {
            "total_rows": len(ob_rows),
            "rows_written": total_ob,
            "days": ob_day_counts,
        }
        manifest["files"].extend(ob_files)
        logger.info(f"  orderbook_snapshots: {total_ob:,} rows written across {len(ob_day_counts)} days")

        # ── arbitrage_opportunities (optional) ─────────────────────────
        if include_arb:
            logger.info("Exporting arbitrage_opportunities ...")
            arb_rows = await _fetch_table_rows(
                session, ArbitrageOpportunity, ArbitrageOpportunity.detected_at, cutoff
            )
            logger.info(f"  total rows: {len(arb_rows):,}")

            arb_dir = archive_dir / "arbitrage_opportunities"
            arb_day_counts, arb_files = _export_rows_to_parquet(
                arb_rows, "detected_at", arb_dir, force
            )
            total_arb = sum(arb_day_counts.values())
            manifest["tables"]["arbitrage_opportunities"] = {
                "total_rows": len(arb_rows),
                "rows_written": total_arb,
                "days": arb_day_counts,
            }
            manifest["files"].extend(arb_files)
            logger.info(f"  arbitrage_opportunities: {total_arb:,} rows written across {len(arb_day_counts)} days")

    manifest["total_rows_exported"] = sum(
        t["rows_written"] for t in manifest["tables"].values()
    )

    # ── Write manifest ─────────────────────────────────────────────────
    archive_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = archive_dir / f"manifest_{run_id}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info("")
    logger.info(f"Manifest written: {manifest_path}")

    # ── Summary ────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("EXPORT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Total rows exported : {manifest['total_rows_exported']:,}")
    for table, info in manifest["tables"].items():
        logger.info(f"  {table:<35}: {info['rows_written']:>8,} rows  ({len(info['days'])} days)")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Verify archive files and manifest look correct.")
    logger.info("  2. To enable batched cleanup in the live app, set:")
    logger.info("       CLEANUP_ENABLED=true")
    logger.info(f"       RETENTION_DAYS={older_than_days}")
    logger.info("  3. Re-run export periodically (e.g. weekly) before each cleanup cycle.")
    logger.info("  4. To train with full history:")
    logger.info(f"       python scripts/train_ensemble.py --archive-dir {archive_dir}")
    logger.info("")

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Export old DB rows to local Parquet archive before running cleanup.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--archive-dir",
        default="data/archive",
        help="Root directory for archive files.",
    )
    parser.add_argument(
        "--older-than-days",
        type=int,
        default=7,
        help="Export rows older than this many days (same cutoff as retention_days).",
    )
    parser.add_argument(
        "--include-arb",
        action="store_true",
        default=False,
        help="Also export arbitrage_opportunities table.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-export days that already have a Parquet file.",
    )
    args = parser.parse_args()

    archive_dir = Path(args.archive_dir)
    if not archive_dir.is_absolute():
        archive_dir = project_root / archive_dir

    manifest = asyncio.run(
        export_archive(
            archive_dir=archive_dir,
            older_than_days=args.older_than_days,
            include_arb=args.include_arb,
            force=args.force,
        )
    )

    if manifest["total_rows_exported"] == 0 and not any(
        t["total_rows"] > 0 for t in manifest["tables"].values()
    ):
        logger.info("No rows older than cutoff found — nothing to export.")


if __name__ == "__main__":
    main()
