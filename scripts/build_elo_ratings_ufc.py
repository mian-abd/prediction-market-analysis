"""Build Glicko-2 UFC/MMA ratings from historical fight data.

Data sources:
1. Kagglehub: m0hamedai1/the-ultimate-ufc-archive-1993-present (~8.2k fights)
2. Local CSV: Set UFC_CSV_PATH env or --csv-path
3. Hugging Face: xtinkarpiu/ufc-fight-data (limited sample)

Usage:
    python scripts/build_elo_ratings_ufc.py
    python scripts/build_elo_ratings_ufc.py --csv-path /path/to/ufc.csv
    python scripts/build_elo_ratings_ufc.py --export-db
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import logging
from datetime import timedelta

from ml.models.elo_sports import Glicko2Engine, ufc_sport_config
from data_pipeline.collectors.ufc_results import fetch_all_ufc_matches, load_ufc_from_csv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


async def build_and_backtest_ufc(
    csv_path: str | Path | None = None,
    backtest_months: int = 12,
    export_db: bool = False,
) -> Glicko2Engine | None:
    """Build UFC ratings and validate with temporal backtest."""
    if csv_path:
        all_matches = load_ufc_from_csv(csv_path)
    else:
        all_matches = await fetch_all_ufc_matches(use_kagglehub=True, use_huggingface=False)

    if not all_matches:
        logger.error("No UFC matches loaded. Install kagglehub or set UFC_CSV_PATH.")
        return None

    unique_fighters = len(set(m.winner for m in all_matches) | set(m.loser for m in all_matches))
    logger.info(f"Total UFC fights: {len(all_matches)}")
    logger.info(f"Date range: {all_matches[0].match_date} to {all_matches[-1].match_date}")
    logger.info(f"Fighters: {unique_fighters}")

    cutoff_date = all_matches[-1].match_date - timedelta(days=backtest_months * 30)
    train_matches = [m for m in all_matches if m.match_date < cutoff_date]
    test_matches = [m for m in all_matches if m.match_date >= cutoff_date]

    logger.info(f"\n{'='*55}")
    logger.info("TEMPORAL SPLIT")
    logger.info(f"  Training:  {len(train_matches)} fights (before {cutoff_date})")
    logger.info(f"  Backtest:  {len(test_matches)} fights (after {cutoff_date})")
    logger.info(f"{'='*55}")

    config = ufc_sport_config()
    engine = Glicko2Engine(config)
    train_stats = engine.process_matches(train_matches)

    logger.info(f"\nTraining stats:")
    logger.info(f"  Accuracy: {train_stats['accuracy']:.1%}")
    logger.info(f"  Brier:    {train_stats['brier_score']:.4f}")
    logger.info(f"  Fighters: {train_stats['unique_players']}")

    correct = 0
    total = 0
    brier_sum = 0.0
    confident_correct = 0
    confident_total = 0

    for match in test_matches:
        prob_winner, confidence = engine.win_probability(
            match.winner, match.loser, "cage",
        )
        if prob_winner > 0.5:
            correct += 1
        brier_sum += (1.0 - prob_winner) ** 2
        total += 1

        if confidence > 0.5:
            confident_total += 1
            if prob_winner > 0.5:
                confident_correct += 1

        engine.update_ratings(match)

    if total > 0:
        accuracy = correct / total
        brier = brier_sum / total
        logger.info(f"\nBacktest: {correct}/{total} correct ({accuracy:.1%}), Brier={brier:.4f}")
        if confident_total > 0:
            logger.info(f"  Confident: {confident_correct}/{confident_total} ({confident_correct/confident_total:.1%})")

    out_path = project_root / "ml" / "saved_models" / "elo_ufc_ratings.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    engine.save(str(out_path))

    logger.info(f"\nTop 20 UFC fighters:")
    for i, p in enumerate(engine.get_top_players(20, "cage"), 1):
        logger.info(f"  {i:3d}. {p['name']:25s} Rating: {p['mu']:7.1f} (RD: {p['phi']:.0f})")

    if export_db:
        from scripts.build_elo_ratings import export_ratings_to_db
        await export_ratings_to_db(engine, sport="ufc")

    return engine


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Glicko-2 UFC ratings")
    parser.add_argument("--csv-path", type=str, help="Path to UFC CSV (Kaggle format)")
    parser.add_argument("--backtest-months", type=int, default=12)
    parser.add_argument("--export-db", action="store_true")
    args = parser.parse_args()

    asyncio.run(build_and_backtest_ufc(
        csv_path=args.csv_path,
        backtest_months=args.backtest_months,
        export_db=args.export_db,
    ))
