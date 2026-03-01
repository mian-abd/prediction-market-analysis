"""Run all Elo-based edge scanners against current active markets.

Loads pre-built Glicko-2 ratings for Tennis (ATP+WTA), UFC, and NBA,
then scans all active Polymarket markets for mispriced games where
our Elo probability diverges meaningfully from the market price.

Run this script periodically (e.g. every 30-60 minutes) to monitor
for live edge opportunities.

Usage:
    python scripts/scan_elo_edges.py
    python scripts/scan_elo_edges.py --min-edge 0.04
    python scripts/scan_elo_edges.py --dry-run     # print only, no DB writes
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import logging
import argparse
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

SPORT_CONFIGS = [
    ("tennis_atp", "ml/saved_models/elo_atp_ratings.joblib",  "tennis", "atp"),
    ("tennis_wta", "ml/saved_models/elo_wta_ratings.joblib",  "tennis", "wta"),
    ("ufc",        "ml/saved_models/elo_ufc_ratings.joblib",  "ufc",    "ufc"),
    ("nba",        "ml/saved_models/elo_nba_ratings.joblib",  "nba",    "nba"),
]


def _load_engine(path_str: str):
    """Load a Glicko2Engine from a joblib file. Returns None if missing."""
    import joblib
    path = project_root / path_str
    if not path.exists():
        logger.warning(f"Ratings file not found: {path_str}")
        return None
    try:
        from ml.models.elo_sports import Glicko2Engine
        data = joblib.load(str(path))
        engine = Glicko2Engine(config=data.get("config"))
        engine.ratings = data.get("ratings", {})
        logger.info(
            f"Loaded {path.name}: {len(engine.ratings)} players/teams"
        )
        return engine
    except Exception as e:
        logger.error(f"Failed to load {path_str}: {e}")
        return None


async def scan_all(min_edge: float = 0.03, dry_run: bool = False) -> dict:
    """Scan all sports for Elo-based edges in active markets."""
    from db.database import init_db, async_session
    from ml.strategies.elo_edge_detector import (
        scan_for_edges,
        scan_ufc_edges,
        scan_nba_edges,
        get_active_edges,
    )

    await init_db()

    all_edges: dict[str, list] = {}
    total_found = 0

    async with async_session() as session:

        # ── Tennis ATP ───────────────────────────────────────────────────
        atp_engine = _load_engine("ml/saved_models/elo_atp_ratings.joblib")
        if atp_engine and atp_engine.ratings:
            logger.info("Scanning Tennis ATP markets...")
            try:
                edges = await scan_for_edges(
                    session, atp_engine, min_net_edge=min_edge
                )
                if dry_run and edges:
                    await session.rollback()
                all_edges["tennis_atp"] = edges
                total_found += len(edges)
                logger.info(f"Tennis ATP: {len(edges)} edges found")
            except Exception as e:
                logger.error(f"ATP scan failed: {e}")

        # ── Tennis WTA ───────────────────────────────────────────────────
        wta_engine = _load_engine("ml/saved_models/elo_wta_ratings.joblib")
        if wta_engine and wta_engine.ratings:
            logger.info("Scanning Tennis WTA markets...")
            try:
                edges = await scan_for_edges(
                    session, wta_engine, min_net_edge=min_edge
                )
                if dry_run and edges:
                    await session.rollback()
                all_edges["tennis_wta"] = edges
                total_found += len(edges)
                logger.info(f"Tennis WTA: {len(edges)} edges found")
            except Exception as e:
                logger.error(f"WTA scan failed: {e}")

        # ── UFC / MMA ────────────────────────────────────────────────────
        ufc_engine = _load_engine("ml/saved_models/elo_ufc_ratings.joblib")
        if ufc_engine and ufc_engine.ratings:
            logger.info("Scanning UFC markets...")
            try:
                edges = await scan_ufc_edges(
                    session, ufc_engine, min_net_edge=min_edge
                )
                if dry_run and edges:
                    await session.rollback()
                all_edges["ufc"] = edges
                total_found += len(edges)
                logger.info(f"UFC: {len(edges)} edges found")
            except Exception as e:
                logger.error(f"UFC scan failed: {e}")

        # ── NBA ──────────────────────────────────────────────────────────
        nba_engine = _load_engine("ml/saved_models/elo_nba_ratings.joblib")
        if nba_engine and nba_engine.ratings:
            logger.info("Scanning NBA markets...")
            try:
                edges = await scan_nba_edges(
                    session, nba_engine, min_net_edge=min_edge
                )
                if dry_run and edges:
                    await session.rollback()
                all_edges["nba"] = edges
                total_found += len(edges)
                logger.info(f"NBA: {len(edges)} edges found")
            except Exception as e:
                logger.error(f"NBA scan failed: {e}")

        # ── Active signals summary ───────────────────────────────────────
        try:
            active = await get_active_edges(session)
        except Exception:
            active = []

    # ── Pretty-print results ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"ELO EDGE SCAN RESULTS  (min_edge={min_edge:.0%})")
    print(f"{'='*60}")

    for sport, edges in all_edges.items():
        if not edges:
            print(f"\n{sport.upper():15s} — no edges found")
            continue
        print(f"\n{sport.upper()} — {len(edges)} edge(s):")
        for e in sorted(edges, key=lambda x: x.get("net_edge", 0), reverse=True):
            pa = e.get("player_a") or e.get("team_a") or e.get("fighter_a") or ""
            pb = e.get("player_b") or e.get("team_b") or e.get("fighter_b") or ""
            print(
                f"  {pa} vs {pb}  |  "
                f"Elo={e.get('elo_prob_a', 0):.1%}  "
                f"Mkt={e.get('market_price', 0):.1%}  "
                f"Edge={e.get('net_edge', 0):.1%}  "
                f"Kelly={e.get('kelly_fraction', 0):.2%}"
            )
            if e.get("question"):
                print(f"    Q: {e['question'][:80]}")

    print(f"\nTotal edges found : {total_found}")
    print(f"Active signals DB : {len(active)}")
    print(f"Dry-run mode      : {dry_run}")
    print(f"{'='*60}\n")

    # Save to JSON
    out = {
        "scan_time": __import__("datetime").datetime.utcnow().isoformat(),
        "min_edge": min_edge,
        "edges": all_edges,
        "total_found": total_found,
        "active_signals": len(active),
    }
    out_path = project_root / "data" / "elo_edge_scan.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info(f"Results saved to {out_path}")

    return all_edges


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan all sports for Elo edges")
    parser.add_argument("--min-edge", type=float, default=0.03,
                        help="Minimum net edge to report (default 3%%)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print edges but do not write to DB")
    args = parser.parse_args()

    asyncio.run(scan_all(min_edge=args.min_edge, dry_run=args.dry_run))
