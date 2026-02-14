"""Build Glicko-2 tennis ratings from historical ATP match data.

Fetches match data from Jeff Sackmann's GitHub CSVs, processes chronologically,
and validates with a held-out backtest on the last 12 months.

Usage:
    python scripts/build_elo_ratings.py
    python scripts/build_elo_ratings.py --start-year 2010 --end-year 2026
    python scripts/build_elo_ratings.py --backtest-months 12

Temporal integrity:
- Matches are processed chronologically (sorted by date)
- Ratings at time T only use match data before T
- Backtest uses last N months as held-out test set
- No future data leaks into past ratings
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import logging
from datetime import date, timedelta

from ml.models.elo_sports import Glicko2Engine, SportConfig, MatchResult
from data_pipeline.collectors.sports_results import fetch_all_tennis_matches

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


async def build_and_backtest(
    start_year: int = 2015,
    end_year: int = 2026,
    backtest_months: int = 12,
    tour: str = "atp",
):
    """Build ratings and validate with temporal backtest."""

    # 1. Fetch all matches
    logger.info(f"Fetching {tour.upper()} matches from {start_year} to {end_year}...")
    all_matches = await fetch_all_tennis_matches(
        start_year=start_year, end_year=end_year, tour=tour,
    )

    if not all_matches:
        logger.error("No matches fetched. Check network connectivity.")
        return

    logger.info(f"Total matches: {len(all_matches)}")
    logger.info(f"Date range: {all_matches[0].match_date} to {all_matches[-1].match_date}")

    # 2. Split into training and backtest periods
    cutoff_date = all_matches[-1].match_date - timedelta(days=backtest_months * 30)
    train_matches = [m for m in all_matches if m.match_date < cutoff_date]
    test_matches = [m for m in all_matches if m.match_date >= cutoff_date]

    logger.info(f"\n{'='*55}")
    logger.info(f"TEMPORAL SPLIT")
    logger.info(f"  Training:  {len(train_matches)} matches (before {cutoff_date})")
    logger.info(f"  Backtest:  {len(test_matches)} matches (after {cutoff_date})")
    logger.info(f"{'='*55}")

    # 3. Build ratings on training data
    logger.info("\nBuilding ratings on training data...")
    engine = Glicko2Engine(SportConfig(sport="tennis"))
    train_stats = engine.process_matches(train_matches)

    logger.info(f"\nTraining stats:")
    logger.info(f"  Accuracy: {train_stats['accuracy']:.1%}")
    logger.info(f"  Brier:    {train_stats['brier_score']:.4f}")
    logger.info(f"  Players:  {train_stats['unique_players']}")

    # 4. Backtest on held-out test set
    # IMPORTANT: We use the ratings as they stand after training.
    # As each test match is processed, ratings are updated, but predictions
    # are always made BEFORE the update (forward-only).
    logger.info("\nRunning temporal backtest on held-out period...")

    correct = 0
    total = 0
    brier_sum = 0.0
    confident_correct = 0
    confident_total = 0

    # Surface-specific tracking
    surface_stats: dict[str, dict] = {}

    for match in test_matches:
        # Predict BEFORE updating
        prob_winner, confidence = engine.win_probability(
            match.winner, match.loser, match.surface,
        )

        predicted_winner = match.winner if prob_winner > 0.5 else match.loser
        is_correct = predicted_winner == match.winner
        brier = (1.0 - prob_winner) ** 2

        total += 1
        brier_sum += brier
        if is_correct:
            correct += 1

        # Track confident predictions (confidence > 0.5)
        if confidence > 0.5:
            confident_total += 1
            if is_correct:
                confident_correct += 1

        # Track by surface
        if match.surface not in surface_stats:
            surface_stats[match.surface] = {"correct": 0, "total": 0, "brier": 0.0}
        surface_stats[match.surface]["total"] += 1
        surface_stats[match.surface]["brier"] += brier
        if is_correct:
            surface_stats[match.surface]["correct"] += 1

        # Update ratings with this match result (for future predictions)
        engine.update_ratings(match)

    # 5. Report results
    accuracy = correct / max(1, total)
    brier_score = brier_sum / max(1, total)
    confident_accuracy = confident_correct / max(1, confident_total)

    logger.info(f"\n{'='*55}")
    logger.info(f"BACKTEST RESULTS ({cutoff_date} to {all_matches[-1].match_date})")
    logger.info(f"{'='*55}")
    logger.info(f"  Total matches:     {total}")
    logger.info(f"  Overall accuracy:  {accuracy:.1%}")
    logger.info(f"  Brier score:       {brier_score:.4f}")
    logger.info(f"  Confident preds:   {confident_total} ({confident_accuracy:.1%} accuracy)")

    logger.info(f"\n  By surface:")
    for surface, stats in sorted(surface_stats.items()):
        s_acc = stats["correct"] / max(1, stats["total"])
        s_brier = stats["brier"] / max(1, stats["total"])
        logger.info(
            f"    {surface:8s}: {s_acc:.1%} accuracy, "
            f"{s_brier:.4f} Brier, {stats['total']} matches"
        )

    # 6. Show top players
    logger.info(f"\nTop 20 players (overall):")
    for i, p in enumerate(engine.get_top_players(20, "overall"), 1):
        logger.info(
            f"  {i:3d}. {p['name']:25s} "
            f"Rating: {p['mu']:7.1f} (RD: {p['phi']:.0f}) "
            f"Matches: {p['matches']}"
        )

    # 7. Save ratings
    save_path = f"ml/saved_models/elo_{tour}_ratings.joblib"
    engine.save(save_path)

    logger.info(f"\nRatings saved to {save_path}")
    logger.info(f"Total unique players: {len(engine.ratings)}")

    return engine


async def export_ratings_to_db(engine: Glicko2Engine, sport: str = "tennis"):
    """Export Glicko-2 ratings from engine to EloRating database table.

    This populates the DB for API serving (GET /elo/ratings, /elo/player/{name}).
    """
    from db.database import init_db, async_session as db_session
    from db.models import EloRating
    from sqlalchemy import delete
    from datetime import datetime

    await init_db()

    async with db_session() as session:
        # Clear old ratings for this sport
        await session.execute(
            delete(EloRating).where(EloRating.sport == sport)
        )

        count = 0
        for player_name, surfaces in engine.ratings.items():
            for surface_name, rating in surfaces.items():
                elo_row = EloRating(
                    sport=sport,
                    player_name=player_name,
                    surface=surface_name,
                    mu=rating.mu,
                    phi=rating.phi,
                    sigma=rating.sigma,
                    match_count=rating.match_count,
                    last_match_date=str(rating.last_match_date) if rating.last_match_date else None,
                    updated_at=datetime.utcnow(),
                )
                session.add(elo_row)
                count += 1

        await session.commit()
        logger.info(f"Exported {count} Elo ratings to database ({sport})")

    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Glicko-2 tennis ratings")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--backtest-months", type=int, default=12)
    parser.add_argument("--tour", choices=["atp", "wta"], default="atp")
    parser.add_argument("--export-db", action="store_true", help="Export ratings to database after building")
    args = parser.parse_args()

    async def _main():
        engine = await build_and_backtest(
            start_year=args.start_year,
            end_year=args.end_year,
            backtest_months=args.backtest_months,
            tour=args.tour,
        )
        if engine and args.export_db:
            await export_ratings_to_db(engine, sport="tennis")

    asyncio.run(_main())
