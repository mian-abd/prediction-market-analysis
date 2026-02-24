"""Rank missing markets by training value for efficient targeted backfill.

Reads data/missing_snapshot_markets.json (produced by analyze_training_universe.py)
and outputs data/backfill_priority_list.json — market IDs sorted by score descending.

Scoring (higher = more valuable to backfill):
    +100  Polymarket market with token_id_yes (CLOB API supports historical fetch)
    +10   Kalshi market (no historical API yet — lower priority but still tracked)
    +0-50 Volume score: min(volume_total / 1000, 50)
    +0-50 Recency score: max(0, 50 - days_since_resolution)

Only markets with token_id_yes and platform == "polymarket" can actually be
backfilled via the CLOB API. Kalshi/others are included in the list but will be
skipped by backfill_price_history.py since they lack token_id_yes.

Usage:
    python scripts/prioritize_backfill.py
    python scripts/prioritize_backfill.py --input data/missing_snapshot_markets.json
"""

import sys
import json
import argparse
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = project_root / "data"


def score_market(market: dict, now: datetime) -> float:
    """Compute training-value score for a missing market.

    Higher score = higher priority for backfill.
    """
    score = 0.0
    platform = (market.get("platform") or "").lower()

    # Platform availability via CLOB API
    if platform == "polymarket" and market.get("has_token_id"):
        score += 100.0
    elif platform in ("kalshi", "predictit"):
        score += 10.0
    else:
        score += 5.0  # Unknown platform — some may work

    # Volume score (more traded = more signal, caps at +50)
    vol = float(market.get("volume_total") or 0)
    score += min(vol / 1000.0, 50.0)

    # Recency score (recently resolved markets are more relevant, caps at +50)
    resolved_at_str = market.get("resolved_at")
    if resolved_at_str:
        try:
            resolved_dt = datetime.fromisoformat(resolved_at_str)
            # Make both aware or both naive
            if resolved_dt.tzinfo is not None:
                now_aware = now.replace(tzinfo=timezone.utc)
                days_old = (now_aware - resolved_dt).total_seconds() / 86400
            else:
                days_old = (now - resolved_dt).total_seconds() / 86400
            score += max(0.0, 50.0 - days_old)
        except (ValueError, TypeError):
            pass

    return score


def prioritize_backfill(input_path: Path, output_path: Path) -> list[dict]:
    """Load missing markets, score, sort, and save priority list."""
    if not input_path.exists():
        logger.error(
            f"Input not found: {input_path}\n"
            "Run: python scripts/analyze_training_universe.py first."
        )
        return []

    with open(input_path) as f:
        missing_markets: list[dict] = json.load(f)

    if not missing_markets:
        logger.info("No missing markets — nothing to prioritize.")
        return []

    now = datetime.now(timezone.utc).replace(tzinfo=None)  # naive UTC for arithmetic
    scored = []
    for m in missing_markets:
        s = score_market(m, now)
        scored.append({**m, "priority_score": round(s, 1)})

    scored.sort(key=lambda x: x["priority_score"], reverse=True)

    # Extract pure list of market IDs for backfill_price_history.py --market-ids
    priority_ids = [m["market_id"] for m in scored]

    # Save both the full scored list (for debugging) and the ID list
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(priority_ids, f, indent=2)

    # Save full scored details alongside
    details_path = output_path.with_name("backfill_priority_details.json")
    with open(details_path, "w") as f:
        json.dump(scored, f, indent=2)

    # Stats
    polymarket_backfillable = sum(
        1 for m in scored
        if m.get("platform") == "polymarket" and m.get("has_token_id")
    )
    top10 = scored[:10]

    logger.info("")
    logger.info("=" * 60)
    logger.info("BACKFILL PRIORITY LIST")
    logger.info("=" * 60)
    logger.info(f"Total missing markets:        {len(scored):,}")
    logger.info(f"Polymarket (backfillable):    {polymarket_backfillable:,}")
    logger.info(f"Other platforms:              {len(scored) - polymarket_backfillable:,}")
    logger.info("")
    logger.info("Top 10 by priority score:")
    for m in top10:
        logger.info(
            f"  id={m['market_id']:6d}  score={m['priority_score']:6.1f}  "
            f"platform={m.get('platform','?'):12s}  "
            f"cat={m.get('category','?'):15s}  "
            f"vol=${m.get('volume_total', 0):,.0f}"
        )
    logger.info("")
    logger.info(f"Saved priority IDs:    {output_path}")
    logger.info(f"Saved priority details: {details_path}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next step:")
    logger.info(
        f"  python scripts/backfill_price_history.py "
        f"--market-ids {output_path} --limit 2000 --days 60"
    )

    return scored


def main():
    parser = argparse.ArgumentParser(
        description="Rank missing markets by training value for targeted backfill."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_DIR / "missing_snapshot_markets.json",
        help="Path to missing_snapshot_markets.json (default: data/missing_snapshot_markets.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "backfill_priority_list.json",
        help="Output path for priority ID list (default: data/backfill_priority_list.json)",
    )
    args = parser.parse_args()
    prioritize_backfill(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
