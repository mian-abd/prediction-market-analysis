"""Build Glicko-2 NBA team ratings from historical game results.

Data source: NBA Stats API via nba_api Python package (free, no key needed).
Falls back to Basketball Reference CSV if nba_api is unavailable.

Usage:
    pip install nba_api
    python scripts/build_elo_ratings_nba.py
    python scripts/build_elo_ratings_nba.py --start-season 2010 --export-db
    python scripts/build_elo_ratings_nba.py --csv-path /path/to/games.csv --export-db

Temporal integrity:
- Games processed chronologically (sorted by date)
- Ratings at time T only use game data before T
- Backtest uses last N months as held-out test set
- No future data leaks into past ratings
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import logging
from datetime import timedelta, timezone
from datetime import datetime

from ml.models.elo_sports import Glicko2Engine, nba_sport_config, MatchResult
from data_pipeline.collectors.nba_results import fetch_all_nba_matches

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


async def build_and_backtest_nba(
    start_season: int = 2000,
    end_season: int = 2025,
    backtest_months: int = 12,
    csv_path: str | None = None,
    export_db: bool = False,
    delay: float = 0.6,
) -> Glicko2Engine | None:
    """Build NBA Glicko-2 team ratings and validate with temporal backtest.

    Args:
        start_season: First season year (e.g. 2000 = 2000-01 season)
        end_season: Last season year (e.g. 2025 = 2024-25 season)
        backtest_months: Number of months for held-out backtest period
        csv_path: Optional path to Basketball Reference CSV (overrides nba_api)
        export_db: Whether to write ratings to the elo_ratings DB table
        delay: Seconds between nba_api calls (rate limit avoidance)

    Returns:
        Fitted Glicko2Engine, or None if no data was loaded.
    """
    logger.info("Loading NBA game results...")
    all_matches = await fetch_all_nba_matches(
        start_season=start_season,
        end_season=end_season,
        csv_path=csv_path,
        delay=delay,
    )

    if not all_matches:
        logger.error(
            "No NBA game data loaded. "
            "Install nba_api (pip install nba_api) or provide --csv-path."
        )
        return None

    unique_teams = len(
        set(m.winner for m in all_matches) | set(m.loser for m in all_matches)
    )
    logger.info(f"Total NBA games: {len(all_matches)}")
    logger.info(f"Date range:      {all_matches[0].match_date} to {all_matches[-1].match_date}")
    logger.info(f"Teams:           {unique_teams}")

    # Temporal split
    cutoff_date = all_matches[-1].match_date - timedelta(days=backtest_months * 30)
    train_matches = [m for m in all_matches if m.match_date < cutoff_date]
    test_matches = [m for m in all_matches if m.match_date >= cutoff_date]

    logger.info(f"\n{'='*55}")
    logger.info("TEMPORAL SPLIT")
    logger.info(f"  Training:  {len(train_matches)} games (before {cutoff_date})")
    logger.info(f"  Backtest:  {len(test_matches)} games (after {cutoff_date})")
    logger.info(f"{'='*55}")

    # Build ratings on training data
    logger.info("\nBuilding NBA team ratings on training data...")
    config = nba_sport_config()
    engine = Glicko2Engine(config)
    train_stats = engine.process_matches(train_matches)

    logger.info(f"\nTraining stats:")
    logger.info(f"  Accuracy: {train_stats['accuracy']:.1%}")
    logger.info(f"  Brier:    {train_stats['brier_score']:.4f}")
    logger.info(f"  Teams:    {train_stats['unique_players']}")

    # Backtest on held-out period
    logger.info("\nRunning temporal backtest on held-out period...")
    correct = 0
    total = 0
    brier_sum = 0.0
    confident_correct = 0
    confident_total = 0

    playoff_correct = 0
    playoff_total = 0

    for match in test_matches:
        prob_winner, confidence = engine.win_probability(
            match.winner, match.loser, "court",
        )
        is_correct = prob_winner > 0.5
        brier = (1.0 - prob_winner) ** 2

        total += 1
        brier_sum += brier
        if is_correct:
            correct += 1

        if confidence > 0.5:
            confident_total += 1
            if is_correct:
                confident_correct += 1

        if match.tourney_level == "playoff":
            playoff_total += 1
            if is_correct:
                playoff_correct += 1

        engine.update_ratings(match)

    accuracy = correct / max(1, total)
    brier_score = brier_sum / max(1, total)
    confident_accuracy = confident_correct / max(1, confident_total) if confident_total else 0.0
    playoff_accuracy = playoff_correct / max(1, playoff_total) if playoff_total else 0.0

    logger.info(f"\n{'='*55}")
    logger.info(f"BACKTEST RESULTS ({cutoff_date} to {all_matches[-1].match_date})")
    logger.info(f"{'='*55}")
    logger.info(f"  Total games:       {total}")
    logger.info(f"  Overall accuracy:  {accuracy:.1%}")
    logger.info(f"  Brier score:       {brier_score:.4f}")
    logger.info(f"  Confident preds:   {confident_total} ({confident_accuracy:.1%} accuracy)")
    if playoff_total:
        logger.info(f"  Playoff accuracy:  {playoff_accuracy:.1%} ({playoff_total} games)")

    # Top teams
    logger.info(f"\nTop 15 NBA teams by current rating:")
    for i, t in enumerate(engine.get_top_players(15, "court"), 1):
        logger.info(
            f"  {i:3d}. {t['name']:30s} "
            f"Rating: {t['mu']:7.1f} (RD: {t['phi']:.0f}) "
            f"Games: {t['matches']}"
        )

    # Save ratings
    out_path = project_root / "ml" / "saved_models" / "elo_nba_ratings.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    engine.save(str(out_path))
    logger.info(f"\nRatings saved to {out_path}")

    if export_db:
        await _export_nba_ratings_to_db(engine)

    return engine


async def _export_nba_ratings_to_db(engine: Glicko2Engine) -> int:
    """Export NBA team Glicko-2 ratings to the elo_ratings DB table."""
    from db.database import init_db, async_session as db_session
    from db.models import EloRating
    from sqlalchemy import delete

    await init_db()

    async with db_session() as session:
        await session.execute(
            delete(EloRating).where(EloRating.sport == "nba")
        )

        count = 0
        for team_name, surfaces in engine.ratings.items():
            for surface_name, rating in surfaces.items():
                elo_row = EloRating(
                    sport="nba",
                    player_name=team_name,
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
        logger.info(f"Exported {count} NBA Elo ratings to database")

    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Glicko-2 NBA team ratings")
    parser.add_argument("--start-season", type=int, default=2000,
                        help="First season year (2000 = 2000-01 season)")
    parser.add_argument("--end-season", type=int, default=2025,
                        help="Last season year (2025 = 2024-25 season)")
    parser.add_argument("--backtest-months", type=int, default=12)
    parser.add_argument("--csv-path", type=str, default=None,
                        help="Basketball Reference CSV path (overrides nba_api)")
    parser.add_argument("--export-db", action="store_true",
                        help="Export ratings to database after building")
    parser.add_argument("--delay", type=float, default=0.6,
                        help="Seconds between nba_api calls (default 0.6)")
    args = parser.parse_args()

    asyncio.run(build_and_backtest_nba(
        start_season=args.start_season,
        end_season=args.end_season,
        backtest_months=args.backtest_months,
        csv_path=args.csv_path,
        export_db=args.export_db,
        delay=args.delay,
    ))
