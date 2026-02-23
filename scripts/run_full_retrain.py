"""Run full data refresh and model retrain pipeline.

Use this to:
- Backfill latest resolved markets (Kalshi + Polymarket)
- Rebuild tennis Elo (ATP + WTA) and UFC Elo
- Retrain the ensemble (XGBoost + LightGBM + calibration)
- Optionally export Elo ratings to DB for API serving

The live app already:
- Collects markets, prices, orderbooks continuously
- Refreshes the confidence adjuster from closed trades every ~30 min (learning from streams)
- Scans edges and runs paper trading on a schedule

This script is for periodic retraining (e.g. weekly). Run manually or via cron/Task Scheduler.

Usage:
    python scripts/run_full_retrain.py
    python scripts/run_full_retrain.py --export-db    # also export Elo to DB (for production)
    python scripts/run_full_retrain.py --skip-elo     # only backfill + train ensemble
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def run(cmd: list[str], step_name: str) -> bool:
    """Run a script; return True on success, False on failure."""
    logger.info("=" * 60)
    logger.info("STEP: %s", step_name)
    logger.info("CMD: %s", " ".join(cmd))
    logger.info("=" * 60)
    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        env=None,  # inherit so DATABASE_URL etc. are available
    )
    if result.returncode != 0:
        logger.error("%s failed with exit code %d", step_name, result.returncode)
        return False
    logger.info("%s completed successfully", step_name)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Full retrain: backfill + Elo + ensemble")
    parser.add_argument(
        "--export-db",
        action="store_true",
        help="Export Elo ratings to DB after building (for production/Railway)",
    )
    parser.add_argument(
        "--skip-elo",
        action="store_true",
        help="Skip Elo builds; only run backfill + train_ensemble",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1990,
        help="Start year for tennis Elo (default 1990)",
    )
    args = parser.parse_args()

    py = sys.executable
    scripts = project_root / "scripts"

    # 1. Backfill resolved markets (more data for ensemble training)
    if not run(
        [py, str(scripts / "backfill_resolved_markets.py")],
        "Backfill resolved markets",
    ):
        return 1

    if not args.skip_elo:
        # 2. Tennis Elo (ATP + WTA)
        tennis_cmd = [
            py,
            str(scripts / "build_elo_ratings.py"),
            "--tour", "all",
            "--start-year", str(args.start_year),
        ]
        if args.export_db:
            tennis_cmd.append("--export-db")
        if not run(tennis_cmd, "Build tennis Elo (ATP + WTA)"):
            return 2

        # 3. UFC Elo
        ufc_cmd = [py, str(scripts / "build_elo_ratings_ufc.py")]
        if args.export_db:
            ufc_cmd.append("--export-db")
        if not run(ufc_cmd, "Build UFC Elo"):
            return 3

    # 4. Train ensemble (uses backfilled resolved markets)
    if not run(
        [py, str(scripts / "train_ensemble.py")],
        "Train ensemble (XGBoost + LightGBM + calibration)",
    ):
        return 4

    logger.info("")
    logger.info("Full retrain pipeline completed. Commit ml/saved_models/ and push to deploy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
