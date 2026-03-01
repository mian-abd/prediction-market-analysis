"""Export pre-built Elo rating .joblib files to the elo_ratings DB table.

Run this after build_elo_ratings.py has already saved the .joblib files, to
populate the database without re-downloading/processing all the match data.

Usage:
    python scripts/export_elo_to_db.py
    python scripts/export_elo_to_db.py --sports tennis ufc nba
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import logging
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

SPORT_FILES = {
    "tennis_atp": ("ml/saved_models/elo_atp_ratings.joblib", "tennis"),
    "tennis_wta": ("ml/saved_models/elo_wta_ratings.joblib", "tennis_wta"),
    "ufc":        ("ml/saved_models/elo_ufc_ratings.joblib", "ufc"),
    "nba":        ("ml/saved_models/elo_nba_ratings.joblib", "nba"),
}


async def export_one(path_str: str, sport_label: str) -> int:
    """Load a joblib file and upsert its ratings into the DB."""
    path = project_root / path_str
    if not path.exists():
        logger.warning(f"Skipping {sport_label}: file not found at {path}")
        return 0

    import joblib
    from db.database import init_db, async_session as db_session
    from db.models import EloRating
    from sqlalchemy import delete

    logger.info(f"Loading {path.name} ...")
    data = joblib.load(str(path))
    ratings: dict = data.get("ratings", {})

    if not ratings:
        logger.warning(f"No ratings found in {path.name}")
        return 0

    await init_db()

    async with db_session() as session:
        # Clear existing ratings for this sport label
        await session.execute(
            delete(EloRating).where(EloRating.sport == sport_label)
        )

        count = 0
        for player_name, surfaces in ratings.items():
            for surface_name, rating in surfaces.items():
                elo_row = EloRating(
                    sport=sport_label,
                    player_name=player_name,
                    surface=surface_name,
                    mu=rating.mu,
                    phi=rating.phi,
                    sigma=rating.sigma,
                    match_count=rating.match_count,
                    last_match_date=(
                        str(rating.last_match_date) if rating.last_match_date else None
                    ),
                    updated_at=datetime.now(timezone.utc),
                )
                session.add(elo_row)
                count += 1

        await session.commit()
        logger.info(f"Exported {count} {sport_label} ratings to database")
        return count


async def main(sports: list[str]) -> None:
    total = 0
    for key in sports:
        if key not in SPORT_FILES:
            logger.warning(f"Unknown sport key: {key}. Choose from: {list(SPORT_FILES)}")
            continue
        path_str, sport_label = SPORT_FILES[key]
        n = await export_one(path_str, sport_label)
        total += n
    logger.info(f"Total rows exported: {total}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export Elo .joblib files to DB")
    parser.add_argument(
        "--sports",
        nargs="+",
        default=["tennis_atp", "tennis_wta", "ufc"],
        choices=list(SPORT_FILES),
        help="Which sports to export (default: tennis_atp tennis_wta ufc)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.sports))
