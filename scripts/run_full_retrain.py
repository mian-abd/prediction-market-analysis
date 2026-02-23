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
import os
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

    # Use a project-local temp dir so heavy scripts don't rely on C: free space.
    env = os.environ.copy()
    tmp_dir = project_root / "tmp"
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        tmp_dir = None
    if tmp_dir is not None:
        env.setdefault("TEMP", str(tmp_dir))
        env.setdefault("TMP", str(tmp_dir))

    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        env=env,
    )
    if result.returncode != 0:
        logger.error("%s failed with exit code %d", step_name, result.returncode)
        return False
    logger.info("%s completed successfully", step_name)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Full retrain: export archive + backfill + Elo + ensemble")
    parser.add_argument(
        "--export-db",
        action="store_true",
        help="Export Elo ratings to DB after building (for production/Railway)",
    )
    parser.add_argument(
        "--skip-elo",
        action="store_true",
        help="Skip Elo builds; only run export + backfill + train_ensemble",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1990,
        help="Start year for tennis Elo (default 1990)",
    )
    parser.add_argument(
        "--archive-dir",
        type=str,
        default="data/archive",
        help="Local archive directory for Parquet snapshots (used by export + train_ensemble).",
    )
    parser.add_argument(
        "--older-than-days",
        type=int,
        default=7,
        help="Export rows older than this many days before retrain.",
    )
    args = parser.parse_args()

    py = sys.executable
    scripts = project_root / "scripts"

    archive_dir = Path(args.archive_dir)

    # 0. Export old snapshots to local archive so nothing is lost before cleanup.
    export_cmd = [
        py,
        str(scripts / "export_archive_to_local.py"),
        "--archive-dir",
        str(archive_dir),
        "--older-than-days",
        str(args.older_than_days),
    ]
    if not run(export_cmd, "Export old data to local archive"):
        return 1

    # 1. Backfill resolved markets (more data for ensemble training)
    if not run(
        [py, str(scripts / "backfill_resolved_markets.py")],
        "Backfill resolved markets",
    ):
        return 2

    if not args.skip_elo:
        # 2. Tennis Elo (ATP + WTA)
        tennis_cmd = [
            py,
            str(scripts / "build_elo_ratings.py"),
            "--tour",
            "all",
            "--start-year",
            str(args.start_year),
        ]
        if args.export_db:
            tennis_cmd.append("--export-db")
        if not run(tennis_cmd, "Build tennis Elo (ATP + WTA)"):
            return 3

        # 3. UFC Elo
        ufc_cmd = [py, str(scripts / "build_elo_ratings_ufc.py")]
        if args.export_db:
            ufc_cmd.append("--export-db")
        if not run(ufc_cmd, "Build UFC Elo"):
            return 4

    # 4. Train ensemble (uses backfilled resolved markets + archive snapshots)
    train_cmd = [
        py,
        str(scripts / "train_ensemble.py"),
        "--archive-dir",
        str(archive_dir),
    ]
    if not run(
        train_cmd,
        "Train ensemble (XGBoost + LightGBM + calibration) with archive",
    ):
        return 5

    logger.info("")
    logger.info("Full retrain pipeline completed. Commit ml/saved_models/ and push to deploy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
