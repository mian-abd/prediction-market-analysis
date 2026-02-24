"""Analyze training universe — identify snapshot coverage gaps for targeted backfill.

Mirrors the exact market filter used by train_ensemble.py / build_training_matrix()
so results map 1:1 to the actual training set.

Outputs:
    data/training_universe_analysis.json  — coverage stats by platform & category
    data/missing_snapshot_markets.json    — list of market IDs needing backfill

Usage:
    python scripts/analyze_training_universe.py
"""

import sys
import json
import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import select, func
from db.database import init_db, async_session
from db.models import Market, Platform, PriceSnapshot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = project_root / "data"


async def analyze_training_universe() -> dict:
    """Query exactly the markets train_ensemble.py would use and check coverage."""
    await init_db()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    async with async_session() as session:
        # ── 1. Build platform name map ─────────────────────────────────────────
        platforms_result = await session.execute(select(Platform))
        platform_map: dict[int, str] = {p.id: p.name for p in platforms_result.scalars().all()}

        # ── 2. Query resolved markets (same filters as load_resolved_markets) ──
        logger.info("Querying resolved markets...")
        markets_result = await session.execute(
            select(Market).where(
                Market.is_resolved == True,  # noqa: E712
                Market.resolution_value != None,  # noqa: E711
            ).order_by(Market.resolved_at.asc())
        )
        all_resolved = markets_result.scalars().all()
        logger.info(f"Total resolved markets: {len(all_resolved)}")

        # Filter: build_training_matrix also drops zero-volume
        usable_markets = [m for m in all_resolved if (m.volume_total or 0) > 0]
        logger.info(f"Usable (volume > 0):   {len(usable_markets)}")

        market_ids = [m.id for m in usable_markets]
        if not market_ids:
            logger.warning("No usable markets found — is the DB populated?")
            return {}

        # ── 3. Load all PriceSnapshots for usable markets ─────────────────────
        logger.info("Querying price snapshots...")
        CHUNK = 900  # SQLite 999-variable limit
        snapshot_counts: dict[int, int] = defaultdict(int)
        snapshot_min_ts: dict[int, datetime] = {}
        snapshot_max_ts: dict[int, datetime] = {}

        for i in range(0, len(market_ids), CHUNK):
            chunk = market_ids[i : i + CHUNK]
            rows = (
                await session.execute(
                    select(
                        PriceSnapshot.market_id,
                        func.count(PriceSnapshot.id).label("cnt"),
                        func.min(PriceSnapshot.timestamp).label("min_ts"),
                        func.max(PriceSnapshot.timestamp).label("max_ts"),
                    )
                    .where(PriceSnapshot.market_id.in_(chunk))
                    .group_by(PriceSnapshot.market_id)
                )
            ).all()
            for row in rows:
                snapshot_counts[row.market_id] = row.cnt
                snapshot_min_ts[row.market_id] = row.min_ts
                snapshot_max_ts[row.market_id] = row.max_ts

        # ── 4. Determine as_of coverage per market ────────────────────────────
        # Coverage = has at least one PriceSnapshot with timestamp <= (resolved_at - 24h)
        # (same logic as load_resolved_markets in train_ensemble.py)
        has_asof_snapshot: set[int] = set()
        missing_snapshot_markets: list[dict] = []

        for m in usable_markets:
            if m.resolved_at:
                as_of = m.resolved_at - timedelta(hours=24)
            elif m.end_date:
                as_of = m.end_date - timedelta(hours=24)
            else:
                continue  # No time reference — skip

            # We have preloaded snapshot min/max timestamps; do a rough check
            # Markets that actually have snapshots before as_of will have max_ts <= as_of
            # or min_ts <= as_of. For exactness we'd need a per-row query, but this
            # approximation is fast and accurate when snapshots span the full window.
            if m.id not in snapshot_counts:
                missing_snapshot_markets.append({
                    "market_id": m.id,
                    "platform": platform_map.get(m.platform_id, "unknown"),
                    "category": m.normalized_category or m.category or "other",
                    "volume_total": m.volume_total or 0,
                    "resolved_at": m.resolved_at.isoformat() if m.resolved_at else None,
                    "token_id_yes": m.token_id_yes,
                    "has_token_id": bool(m.token_id_yes),
                })
            else:
                min_ts = snapshot_min_ts.get(m.id)
                if min_ts and min_ts <= as_of:
                    has_asof_snapshot.add(m.id)
                else:
                    # Snapshots exist but none before as_of
                    missing_snapshot_markets.append({
                        "market_id": m.id,
                        "platform": platform_map.get(m.platform_id, "unknown"),
                        "category": m.normalized_category or m.category or "other",
                        "volume_total": m.volume_total or 0,
                        "resolved_at": m.resolved_at.isoformat() if m.resolved_at else None,
                        "token_id_yes": m.token_id_yes,
                        "has_token_id": bool(m.token_id_yes),
                    })

        # ── 5. Coverage breakdown by platform & category ──────────────────────
        by_platform: dict[str, dict] = defaultdict(lambda: {"total": 0, "with_coverage": 0})
        by_category: dict[str, dict] = defaultdict(lambda: {"total": 0, "with_coverage": 0})
        by_platform_category: dict[str, dict] = defaultdict(lambda: {"total": 0, "with_coverage": 0})

        for m in usable_markets:
            pname = platform_map.get(m.platform_id, "unknown")
            cat = m.normalized_category or m.category or "other"
            has_cov = m.id in has_asof_snapshot

            by_platform[pname]["total"] += 1
            if has_cov:
                by_platform[pname]["with_coverage"] += 1

            by_category[cat]["total"] += 1
            if has_cov:
                by_category[cat]["with_coverage"] += 1

            key = f"{pname}/{cat}"
            by_platform_category[key]["total"] += 1
            if has_cov:
                by_platform_category[key]["with_coverage"] += 1

        def add_pct(d: dict) -> dict:
            return {
                k: {**v, "coverage_pct": round(v["with_coverage"] / v["total"] * 100, 1) if v["total"] else 0}
                for k, v in d.items()
            }

        n_total = len(usable_markets)
        n_covered = len(has_asof_snapshot)
        coverage_pct = round(n_covered / n_total * 100, 1) if n_total else 0

        analysis = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_resolved_markets": len(all_resolved),
                "usable_markets": n_total,
                "markets_with_asof_snapshot": n_covered,
                "markets_missing_snapshot": len(missing_snapshot_markets),
                "coverage_pct": coverage_pct,
                "target_pct": 50.0,
            },
            "by_platform": add_pct(by_platform),
            "by_category": add_pct(by_category),
            "by_platform_category": add_pct(by_platform_category),
        }

        # ── 6. Save outputs ───────────────────────────────────────────────────
        analysis_path = DATA_DIR / "training_universe_analysis.json"
        missing_path = DATA_DIR / "missing_snapshot_markets.json"

        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2)

        with open(missing_path, "w") as f:
            json.dump(missing_snapshot_markets, f, indent=2)

        # ── 7. Print summary ──────────────────────────────────────────────────
        logger.info("")
        logger.info("=" * 60)
        logger.info("TRAINING UNIVERSE ANALYSIS")
        logger.info("=" * 60)
        logger.info(f"Total resolved markets:      {len(all_resolved):,}")
        logger.info(f"Usable (volume > 0):         {n_total:,}")
        logger.info(f"With as_of snapshot:         {n_covered:,}  ({coverage_pct}%)")
        logger.info(f"Missing snapshot:            {len(missing_snapshot_markets):,}")
        logger.info(f"Target coverage:             50.0%")
        logger.info("")
        logger.info("By Platform:")
        for pname, stats in sorted(add_pct(by_platform).items()):
            logger.info(f"  {pname:20s}  {stats['with_coverage']:4d}/{stats['total']:4d}  ({stats['coverage_pct']}%)")
        logger.info("")
        logger.info("By Category (top 10 by total):")
        sorted_cats = sorted(add_pct(by_category).items(), key=lambda x: x[1]["total"], reverse=True)[:10]
        for cat, stats in sorted_cats:
            logger.info(f"  {cat:20s}  {stats['with_coverage']:4d}/{stats['total']:4d}  ({stats['coverage_pct']}%)")
        logger.info("")
        logger.info(f"Saved: {analysis_path}")
        logger.info(f"Saved: {missing_path}")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Next step: python scripts/prioritize_backfill.py")

        return analysis


if __name__ == "__main__":
    asyncio.run(analyze_training_universe())
