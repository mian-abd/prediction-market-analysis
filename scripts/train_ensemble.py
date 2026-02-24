"""Train XGBoost + LightGBM + Calibration ensemble on resolved markets.

Honest methodology:
- Temporal train/test split (no future leakage)
- Walk-forward CV for OOF Brier scores (no test-set peeking)
- Significance-gated model inclusion (Wilcoxon test)
- Feature pruning (zero-variance, near-constant)
- Ablation study + profit simulation
- Model card for auditability

Usage:
    python scripts/train_ensemble.py
    python scripts/train_ensemble.py --archive-dir data/archive

The --archive-dir flag enables loading historical price and orderbook snapshots
from local Parquet files produced by scripts/export_archive_to_local.py.  When
provided, training merges DB snapshots (recent) with archive snapshots (old) so
the model refers to and learns from the complete history.

Prerequisite:
    python scripts/backfill_resolved_markets.py
"""

import argparse
import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set by main() from --archive-dir; used inside load_resolved_markets().
_ARCHIVE_DIR: Path | None = None
_SNAPSHOT_ONLY: bool = False  # Set by --snapshot-only flag: train only on as_of snapshot markets
_TRADEABLE_RANGE: tuple[float, float] | None = None  # Set by --tradeable-range: filter to uncertain markets
_AS_OF_DAYS: int = 1  # Set by --as-of-days: how many days before resolution to use as reference

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta, timezone
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss, precision_recall_curve
from scipy.stats import wilcoxon

from db.database import async_session, init_db
from db.models import Market, PriceSnapshot, OrderbookSnapshot
from sqlalchemy import select, func

from ml.features.training_features import (
    build_training_matrix,
    prune_features,
    ENSEMBLE_FEATURE_NAMES,
    N_FEATURES,
)
from ml.models.calibration_model import CalibrationModel
from ml.models.xgboost_model import XGBoostModel
from ml.models.lightgbm_model import LightGBMModel
from ml.models.ensemble import EnsembleModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _load_archive_price_snapshots(
    archive_dir: Path,
    market_ids: set[int],
    date_start,
    date_end,
) -> dict[int, list[tuple]]:
    """Read price_snapshots Parquet files from the archive for a date range.

    Returns {market_id: [(timestamp, price_yes), ...]} sorted ascending.
    Only rows whose market_id is in `market_ids` are loaded.
    """
    import pandas as pd

    price_dir = archive_dir / "price_snapshots"
    if not price_dir.exists():
        return {}

    rows_by_market: dict[int, list[tuple]] = {}

    # Iterate day-by-day across the requested date range
    current = date_start
    while current <= date_end:
        day_str = current.date().isoformat() if hasattr(current, "date") else str(current)[:10]
        parquet_path = price_dir / f"{day_str}.parquet"
        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path, engine="pyarrow")
                df = df[df["market_id"].isin(market_ids)]
                for _, row in df.iterrows():
                    mid = int(row["market_id"])
                    ts = row["timestamp"]
                    price = float(row["price_yes"])
                    if mid not in rows_by_market:
                        rows_by_market[mid] = []
                    rows_by_market[mid].append((ts, price))
            except Exception as e:
                logger.warning(f"Archive price read failed ({parquet_path.name}): {e}")
        current = current + timedelta(days=1)

    # Sort each market's list by timestamp
    for mid in rows_by_market:
        rows_by_market[mid].sort(key=lambda x: x[0])

    return rows_by_market


def _load_archive_orderbook_snapshots(
    archive_dir: Path,
    market_ids: set[int],
    date_start,
    date_end,
) -> dict[int, list[dict]]:
    """Read orderbook_snapshots Parquet files from the archive for a date range.

    Returns {market_id: [row_dict, ...]} sorted ascending by timestamp.
    """
    import pandas as pd

    ob_dir = archive_dir / "orderbook_snapshots"
    if not ob_dir.exists():
        return {}

    rows_by_market: dict[int, list[dict]] = {}

    current = date_start
    while current <= date_end:
        day_str = current.date().isoformat() if hasattr(current, "date") else str(current)[:10]
        parquet_path = ob_dir / f"{day_str}.parquet"
        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path, engine="pyarrow")
                df = df[df["market_id"].isin(market_ids)]
                for _, row in df.iterrows():
                    mid = int(row["market_id"])
                    if mid not in rows_by_market:
                        rows_by_market[mid] = []
                    rows_by_market[mid].append(row.to_dict())
            except Exception as e:
                logger.warning(f"Archive orderbook read failed ({parquet_path.name}): {e}")
        current = current + timedelta(days=1)

    for mid in rows_by_market:
        rows_by_market[mid].sort(key=lambda x: x.get("timestamp", datetime.min))

    return rows_by_market


async def load_resolved_markets() -> tuple[list[Market], dict[int, list[float]], dict[int, float], dict[int, any]]:
    """Load resolved markets ordered by resolved_at for temporal split."""
    async with async_session() as session:
        result = await session.execute(
            select(Market).where(
                Market.is_resolved == True,
                Market.resolution_value != None,
                # Removed price_yes != None filter: backfilled snapshots provide as_of prices
            ).order_by(Market.resolved_at.asc())  # Temporal ordering
        )
        markets = list(result.scalars().all())
        market_ids = [m.id for m in markets]
        session.expunge_all()

        # Load price snapshots WITH TIMESTAMPS for as_of enforcement
        # Build two maps:
        # 1. price_snapshots_map: filtered snapshots (timestamp <= as_of) for momentum
        # 2. price_at_as_of_map: price 24h before resolution (overrides market.price_yes)
        price_snapshots_map: dict[int, list[float]] = {}
        price_at_as_of_map: dict[int, float] = {}

        if market_ids:
            # SQLite has a 999-variable limit; chunk IN clauses to avoid it
            CHUNK = 900
            all_snapshot_rows = []
            for i in range(0, len(market_ids), CHUNK):
                chunk = market_ids[i:i + CHUNK]
                chunk_result = await session.execute(
                    select(PriceSnapshot.market_id, PriceSnapshot.timestamp, PriceSnapshot.price_yes)
                    .where(PriceSnapshot.market_id.in_(chunk))
                    .order_by(PriceSnapshot.market_id, PriceSnapshot.timestamp)
                )
                all_snapshot_rows.extend(chunk_result.all())
            # Group snapshots by market_id
            raw_snapshots: dict[int, list[tuple]] = {}
            for market_id, timestamp, price in all_snapshot_rows:
                if market_id not in raw_snapshots:
                    raw_snapshots[market_id] = []
                raw_snapshots[market_id].append((timestamp, price))

            # Per-market: filter by as_of and extract "price at as_of"
            markets_by_id = {m.id: m for m in markets}
            skipped_no_as_of = 0
            skipped_no_snapshots = 0

            for market_id, snapshots in raw_snapshots.items():
                market = markets_by_id.get(market_id)
                if not market:
                    continue

                # as_of = resolved_at - N days (configurable via --as-of-days, default 1)
                _as_of_delta = timedelta(hours=_AS_OF_DAYS * 24)
                if market.resolved_at:
                    as_of = market.resolved_at - _as_of_delta
                elif getattr(market, 'end_date', None):
                    as_of = market.end_date - _as_of_delta
                else:
                    # No valid as_of time, skip this market (strict mode)
                    skipped_no_as_of += 1
                    continue

                # Filter snapshots to timestamp <= as_of
                filtered = [(ts, p) for (ts, p) in snapshots if ts <= as_of]
                filtered.sort(key=lambda x: x[0])  # Ensure sorted by timestamp

                if not filtered:
                    # No snapshots before as_of, skip (strict mode)
                    skipped_no_snapshots += 1
                    continue

                # price_snapshots_map: list of prices for momentum features
                price_snapshots_map[market_id] = [p for (ts, p) in filtered]

                # price_at_as_of_map: last price before as_of (overrides market.price_yes)
                price_at_as_of_map[market_id] = filtered[-1][1]

            logger.info(
                f"Price snapshots: {len(price_snapshots_map)} markets have snapshot data at as_of "
                f"(avg {np.mean([len(v) for v in price_snapshots_map.values()]):.0f} snapshots/market)"
                if price_snapshots_map else "No price snapshots found"
            )
            if skipped_no_as_of > 0:
                logger.info(f"  Skipped {skipped_no_as_of} markets (no valid as_of time)")
            if skipped_no_snapshots > 0:
                logger.info(f"  Skipped {skipped_no_snapshots} markets (no snapshots before as_of)")

            # ── Merge from archive (if --archive-dir provided) ─────────────
            if _ARCHIVE_DIR is not None:
                markets_needing_archive = set(market_ids) - set(price_snapshots_map.keys())
                if markets_needing_archive:
                    # Determine date range: oldest resolved_at - 30d to max as_of
                    min_resolved = min(
                        (m.resolved_at or m.end_date)
                        for m in markets if (m.resolved_at or getattr(m, "end_date", None))
                    )
                    max_resolved = max(
                        (m.resolved_at or m.end_date)
                        for m in markets if (m.resolved_at or getattr(m, "end_date", None))
                    )
                    date_start = min_resolved - timedelta(days=32)
                    date_end = max_resolved

                    logger.info(
                        f"  Archive: loading price snapshots for {len(markets_needing_archive)} markets "
                        f"without DB snapshots  ({date_start.date()} → {date_end.date()})"
                    )
                    archive_price = _load_archive_price_snapshots(
                        _ARCHIVE_DIR, markets_needing_archive, date_start, date_end
                    )

                    # Merge: apply same as_of filter used for DB snapshots
                    merged_count = 0
                    _as_of_delta = timedelta(hours=_AS_OF_DAYS * 24)
                    for market_id, snap_list in archive_price.items():
                        market = markets_by_id.get(market_id)
                        if not market:
                            continue
                        if market.resolved_at:
                            as_of = market.resolved_at - _as_of_delta
                        elif getattr(market, "end_date", None):
                            as_of = market.end_date - _as_of_delta
                        else:
                            continue
                        filtered = [(ts, p) for (ts, p) in snap_list if ts <= as_of]
                        if not filtered:
                            continue
                        price_snapshots_map[market_id] = [p for (ts, p) in filtered]
                        price_at_as_of_map[market_id] = filtered[-1][1]
                        merged_count += 1

                    logger.info(
                        f"  Archive: merged {merged_count} additional markets from Parquet "
                        f"(total with snapshots: {len(price_snapshots_map)})"
                    )
                else:
                    logger.info("  Archive: all markets already have DB snapshot data, skipping archive load")

        # Load latest orderbook snapshot BEFORE as_of for each market (temporal integrity)
        orderbook_snapshots_map: dict[int, OrderbookSnapshot] = {}
        if market_ids:
            # Load ALL orderbook snapshots, then filter per-market by as_of
            # Chunk IN clause to respect SQLite's 999-variable limit
            CHUNK = 900
            all_orderbooks_list = []
            for i in range(0, len(market_ids), CHUNK):
                chunk = market_ids[i:i + CHUNK]
                chunk_result = await session.execute(
                    select(OrderbookSnapshot)
                    .where(OrderbookSnapshot.market_id.in_(chunk))
                    .order_by(OrderbookSnapshot.market_id, OrderbookSnapshot.timestamp.desc())
                )
                all_orderbooks_list.extend(chunk_result.scalars().all())
            all_orderbooks = all_orderbooks_list

            # Group by market and filter to as_of
            markets_by_id = {m.id: m for m in markets}
            skipped_no_ob_as_of = 0
            stale_orderbooks = 0  # Count orderbooks >10min old

            for market_id in market_ids:
                market = markets_by_id.get(market_id)
                if not market:
                    continue

                # Compute as_of = resolved_at - N days (same logic as price snapshots)
                _as_of_delta = timedelta(hours=_AS_OF_DAYS * 24)
                if market.resolved_at:
                    as_of = market.resolved_at - _as_of_delta
                elif getattr(market, 'end_date', None):
                    as_of = market.end_date - _as_of_delta
                else:
                    skipped_no_ob_as_of += 1
                    continue

                # Filter orderbooks to timestamp <= as_of, take latest
                valid_obs = [ob for ob in all_orderbooks
                            if ob.market_id == market_id and ob.timestamp <= as_of]

                if valid_obs:
                    # Already sorted by timestamp desc, so first is latest before as_of
                    latest_ob = valid_obs[0]
                    orderbook_snapshots_map[market_id] = latest_ob

                    # Staleness check: warn if orderbook is >10min old relative to as_of
                    staleness_minutes = (as_of - latest_ob.timestamp).total_seconds() / 60
                    if staleness_minutes > 10:
                        stale_orderbooks += 1
                # Else: no orderbook before as_of, skip (no entry in map)

            if skipped_no_ob_as_of > 0:
                logger.info(f"  Orderbook: skipped {skipped_no_ob_as_of} markets (no valid as_of time)")
            if stale_orderbooks > 0:
                logger.warning(
                    f"  ⚠️  {stale_orderbooks}/{len(orderbook_snapshots_map)} orderbooks are >10min stale "
                    f"(increase collection frequency or backfill historical data)"
                )

            # ── Merge orderbooks from archive (if --archive-dir provided) ──
            if _ARCHIVE_DIR is not None:
                ob_markets_needing_archive = set(market_ids) - set(orderbook_snapshots_map.keys())
                if ob_markets_needing_archive:
                    min_resolved = min(
                        (m.resolved_at or m.end_date)
                        for m in markets if (m.resolved_at or getattr(m, "end_date", None))
                    )
                    max_resolved = max(
                        (m.resolved_at or m.end_date)
                        for m in markets if (m.resolved_at or getattr(m, "end_date", None))
                    )
                    date_start = min_resolved - timedelta(days=32)
                    date_end = max_resolved

                    logger.info(
                        f"  Archive: loading orderbook snapshots for {len(ob_markets_needing_archive)} markets "
                        f"without DB snapshots  ({date_start.date()} → {date_end.date()})"
                    )
                    archive_ob = _load_archive_orderbook_snapshots(
                        _ARCHIVE_DIR, ob_markets_needing_archive, date_start, date_end
                    )

                    ob_merged_count = 0
                    _as_of_delta = timedelta(hours=_AS_OF_DAYS * 24)
                    for market_id, ob_list in archive_ob.items():
                        market = markets_by_id.get(market_id)
                        if not market:
                            continue
                        if market.resolved_at:
                            as_of = market.resolved_at - _as_of_delta
                        elif getattr(market, "end_date", None):
                            as_of = market.end_date - _as_of_delta
                        else:
                            continue
                        # Pick latest orderbook row before as_of
                        valid = [row for row in ob_list if row.get("timestamp", datetime.min) <= as_of]
                        if not valid:
                            continue
                        latest_row = valid[-1]

                        # Reconstruct an OrderbookSnapshot-compatible object from the dict
                        ob_obj = OrderbookSnapshot(
                            market_id=market_id,
                            side=latest_row.get("side", "yes"),
                            timestamp=latest_row.get("timestamp"),
                            best_bid=latest_row.get("best_bid"),
                            best_ask=latest_row.get("best_ask"),
                            bid_ask_spread=latest_row.get("bid_ask_spread"),
                            bid_depth_total=latest_row.get("bid_depth_total"),
                            ask_depth_total=latest_row.get("ask_depth_total"),
                            obi_level1=latest_row.get("obi_level1"),
                            obi_weighted=latest_row.get("obi_weighted"),
                            depth_ratio=latest_row.get("depth_ratio"),
                            bids_json=latest_row.get("bids_json"),
                            asks_json=latest_row.get("asks_json"),
                        )
                        orderbook_snapshots_map[market_id] = ob_obj
                        ob_merged_count += 1

                    logger.info(
                        f"  Archive: merged {ob_merged_count} additional orderbooks from Parquet "
                        f"(total with orderbooks: {len(orderbook_snapshots_map)})"
                    )
                else:
                    logger.info("  Archive: all markets already have DB orderbook data, skipping archive load")

            session.expunge_all()

        logger.info(
            f"Orderbook snapshots: {len(orderbook_snapshots_map)} markets have orderbook data"
            if orderbook_snapshots_map else "No orderbook snapshots found"
        )

        return markets, price_snapshots_map, price_at_as_of_map, orderbook_snapshots_map


def temporal_split(
    markets: list, X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Split by resolved_at date, not random index.

    Returns (X_train, X_test, y_train, y_test, cutoff_date_str)
    """
    # Markets are already sorted by resolved_at from the query
    dates = []
    for m in markets:
        if m.resolved_at:
            dates.append(m.resolved_at)
        elif m.end_date:
            dates.append(m.end_date)
        else:
            dates.append(datetime(2020, 1, 1))  # Fallback for missing dates

    # We only have dates for markets that passed the build_training_matrix filter,
    # but markets list may be longer. Use the actual array length.
    n = len(y)
    # The markets list is pre-sorted, but build_training_matrix may skip some.
    # We need to track which markets made it into the matrix.
    # Since we can't easily do that here, use position-based split (data is already
    # temporally ordered because markets were loaded ORDER BY resolved_at ASC).
    cutoff_idx = int(n * train_ratio)

    # Ensure test set has at least 2 YES and 2 NO samples
    # GUARD: lower bound prevents infinite loop when no class diversity exists
    min_cutoff = max(20, int(n * 0.5))
    while cutoff_idx > min_cutoff and cutoff_idx < n - 4:
        test_y = y[cutoff_idx:]
        if test_y.sum() >= 2 and (len(test_y) - test_y.sum()) >= 2:
            break
        cutoff_idx -= 1

    if cutoff_idx <= min_cutoff:
        # Could not find a split with class diversity — fall back to default 80/20
        cutoff_idx = int(n * train_ratio)
        logger.warning(
            f"Could not find split with class diversity in test set. "
            f"Using default {train_ratio:.0%}/{1-train_ratio:.0%} split."
        )

    X_train, X_test = X[:cutoff_idx], X[cutoff_idx:]
    y_train, y_test = y[:cutoff_idx], y[cutoff_idx:]

    # Determine approximate cutoff date
    cutoff_date_str = "unknown"
    if len(dates) >= n:
        # Find the date at the cutoff position
        # Since some markets are skipped, this is approximate
        cutoff_date_str = str(dates[min(cutoff_idx, len(dates) - 1)])[:10]

    return X_train, X_test, y_train, y_test, cutoff_date_str


def walk_forward_oof(
    X_train_full: np.ndarray,
    y_train: np.ndarray,
    full_feature_names: list[str],
    X_train_pruned: np.ndarray,
    pruned_feature_names: list[str],
    n_folds: int = 5,
    min_train_samples: int = 50,
) -> dict:
    """Walk-forward CV: train on past, validate on future.

    Args:
        X_train_full: Full unpruned feature matrix (for calibration)
        y_train: Labels
        full_feature_names: All 25 feature names
        X_train_pruned: Pruned feature matrix (for tree models)
        pruned_feature_names: Active feature names after pruning
        n_folds: Number of walk-forward folds
        min_train_samples: Minimum samples required in training fold

    Returns dict of {model_name: {"oof_preds": array, "oof_mask": array, "oof_brier": float}}
    """
    n = len(y_train)
    fold_size = n // (n_folds + 1)

    if fold_size < 10:
        logger.warning(f"Very small fold size ({fold_size}). Walk-forward may be unreliable.")

    price_col_full = full_feature_names.index("price_yes")

    # Initialize OOF prediction arrays (NaN = not predicted yet)
    oof = {
        "calibration": np.full(n, np.nan),
        "xgboost": np.full(n, np.nan),
        "lightgbm": np.full(n, np.nan),
    }

    valid_folds = 0

    for fold in range(n_folds):
        train_end = (fold + 2) * fold_size  # Expanding window
        val_start = train_end
        val_end = min(val_start + fold_size, n)

        if val_end <= val_start:
            break
        if train_end < min_train_samples:
            logger.info(f"  Fold {fold}: skipped (only {train_end} training samples, need {min_train_samples})")
            continue

        # Full feature matrix (for calibration)
        fold_X_train_full = X_train_full[:train_end]
        fold_X_val_full = X_train_full[val_start:val_end]

        # Pruned feature matrix (for tree models)
        fold_X_train_pruned = X_train_pruned[:train_end]
        fold_X_val_pruned = X_train_pruned[val_start:val_end]

        fold_y_train = y_train[:train_end]
        fold_y_val = y_train[val_start:val_end]

        # Skip if validation has no class diversity
        if fold_y_val.sum() == 0 or fold_y_val.sum() == len(fold_y_val):
            logger.info(f"  Fold {fold}: skipped (no class diversity in validation)")
            continue

        valid_folds += 1
        logger.info(
            f"  Fold {fold}: train [0:{train_end}] ({int(fold_y_train.sum())} YES), "
            f"val [{val_start}:{val_end}] ({int(fold_y_val.sum())} YES)"
        )

        # Calibration (price-only, uses full matrix)
        cal = CalibrationModel()
        cal.train(fold_X_train_full[:, price_col_full], fold_y_train)
        oof["calibration"][val_start:val_end] = cal.predict(fold_X_val_full[:, price_col_full])

        # XGBoost (uses pruned matrix)
        xgb = XGBoostModel()
        xgb.train(fold_X_train_pruned, fold_y_train, feature_names=pruned_feature_names)
        oof["xgboost"][val_start:val_end] = xgb.predict_proba(fold_X_val_pruned)

        # LightGBM (uses pruned matrix)
        lgb = LightGBMModel()
        lgb.train(fold_X_train_pruned, fold_y_train, feature_names=pruned_feature_names)
        oof["lightgbm"][val_start:val_end] = lgb.predict_proba(fold_X_val_pruned)

    logger.info(f"  Walk-forward: {valid_folds}/{n_folds} folds completed")

    # Compute OOF Brier scores (only on samples that have predictions)
    results = {}
    for name, preds in oof.items():
        mask = ~np.isnan(preds)
        if mask.sum() < 10:
            logger.warning(f"  {name}: only {mask.sum()} OOF predictions, skipping")
            results[name] = {"oof_preds": preds, "oof_mask": mask, "oof_brier": 1.0}
            continue
        brier = brier_score_loss(y_train[mask], preds[mask])
        results[name] = {"oof_preds": preds, "oof_mask": mask, "oof_brier": brier}
        logger.info(f"  {name} OOF Brier: {brier:.4f} (n={mask.sum()})")

    return results


def significance_gate(
    y: np.ndarray,
    cal_preds: np.ndarray,
    model_preds: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.10,
) -> tuple[bool, float]:
    """Test if model significantly beats calibration using Wilcoxon signed-rank test.

    Returns (is_significant, p_value)
    """
    cal_sq_errors = (y[mask] - cal_preds[mask]) ** 2
    model_sq_errors = (y[mask] - model_preds[mask]) ** 2

    try:
        _, p_value = wilcoxon(cal_sq_errors, model_sq_errors)
        return p_value < alpha, float(p_value)
    except ValueError as e:
        logger.warning(f"  Wilcoxon test failed: {e}")
        return False, 1.0


def ablation_study(
    y_test: np.ndarray,
    cal_test_preds: np.ndarray,
    xgb_test_preds: np.ndarray | None,
    lgb_test_preds: np.ndarray | None,
    weights: dict[str, float],
) -> dict:
    """Test each model's incremental contribution."""
    results = {}

    # Calibration-only
    cal_brier = brier_score_loss(y_test, cal_test_preds)
    results["calibration_only"] = cal_brier

    # Calibration + XGBoost (50/50)
    if xgb_test_preds is not None:
        cal_xgb = 0.5 * cal_test_preds + 0.5 * xgb_test_preds
        results["cal_plus_xgb"] = brier_score_loss(y_test, cal_xgb)

    # Calibration + LightGBM (50/50)
    if lgb_test_preds is not None:
        cal_lgb = 0.5 * cal_test_preds + 0.5 * lgb_test_preds
        results["cal_plus_lgb"] = brier_score_loss(y_test, cal_lgb)

    # Full ensemble with actual weights
    preds = {"calibration": cal_test_preds}
    if xgb_test_preds is not None and "xgboost" in weights:
        preds["xgboost"] = xgb_test_preds
    if lgb_test_preds is not None and "lightgbm" in weights:
        preds["lightgbm"] = lgb_test_preds

    full_blend = sum(weights.get(name, 0) * p for name, p in preds.items())
    results["full_ensemble"] = brier_score_loss(y_test, full_blend)

    return results


def profit_simulation(
    y_test: np.ndarray,
    ensemble_preds: np.ndarray,
    market_prices: np.ndarray,
    min_net_edge: float = 0.05,
    slippage: float = 0.015,
) -> dict:
    """Walk through test set chronologically, simulate trading with realistic Polymarket fees.

    Fee model: Polymarket charges 2% on NET WINNINGS only (0% on losses).
    This is much more favorable than a flat fee model.

    Returns dict with gated vs ungated P&L at multiple thresholds.
    """
    FEE_RATE = 0.02  # Polymarket: 2% on winnings only

    def compute_trade_pnl(direction: str, actual: float, q: float) -> float:
        """Compute P&L for a single trade with realistic Polymarket fees."""
        if direction == "yes":
            gross_pnl = (actual - q)
        else:
            gross_pnl = ((1 - actual) - (1 - q))

        gross_pnl -= slippage

        # Fee only charged on net winnings (positive gross P&L)
        if gross_pnl > 0:
            fee = FEE_RATE * gross_pnl
            return gross_pnl - fee
        return gross_pnl

    # Ungated: trade everything
    ungated_pnl = 0.0
    ungated_trades = 0
    ungated_wins = 0

    # Gated: only trade when net_ev > min_net_edge
    gated_pnl = 0.0
    gated_trades = 0
    gated_wins = 0

    # Multi-threshold analysis
    thresholds = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]
    threshold_results = {t: {"pnl": 0.0, "trades": 0, "wins": 0} for t in thresholds}

    for i in range(len(y_test)):
        p = ensemble_preds[i]  # Model probability
        q = market_prices[i]   # Market price
        actual = y_test[i]     # 1.0 or 0.0

        # Fee-aware directional EV (Polymarket: fee only on winning)
        fee_yes = p * FEE_RATE * (1 - q) + slippage
        fee_no = (1 - p) * FEE_RATE * q + slippage
        ev_yes = p * (1 - q) - (1 - p) * q - fee_yes
        ev_no = (1 - p) * q - p * (1 - q) - fee_no

        if ev_yes > ev_no:
            direction = "yes"
            net_ev = ev_yes
        else:
            direction = "no"
            net_ev = ev_no

        trade_pnl = compute_trade_pnl(direction, actual, q)

        # Ungated: always trade if any positive EV
        if net_ev > 0:
            ungated_pnl += trade_pnl
            ungated_trades += 1
            if trade_pnl > 0:
                ungated_wins += 1

        # Gated: only trade if edge exceeds threshold
        if net_ev >= min_net_edge:
            gated_pnl += trade_pnl
            gated_trades += 1
            if trade_pnl > 0:
                gated_wins += 1

        # Multi-threshold analysis
        for t in thresholds:
            if net_ev >= t:
                threshold_results[t]["pnl"] += trade_pnl
                threshold_results[t]["trades"] += 1
                if trade_pnl > 0:
                    threshold_results[t]["wins"] += 1

    return {
        "ungated_pnl": round(ungated_pnl, 4),
        "ungated_trades": ungated_trades,
        "ungated_win_rate": round(ungated_wins / max(1, ungated_trades), 3),
        "gated_pnl": round(gated_pnl, 4),
        "gated_trades": gated_trades,
        "gated_win_rate": round(gated_wins / max(1, gated_trades), 3),
        "min_net_edge": min_net_edge,
        "fee_model": "polymarket_2pct_on_winnings",
        "slippage": slippage,
        "threshold_sweep": {
            f"{t:.0%}": {
                "pnl": round(threshold_results[t]["pnl"], 4),
                "trades": threshold_results[t]["trades"],
                "win_rate": round(threshold_results[t]["wins"] / max(1, threshold_results[t]["trades"]), 3),
            }
            for t in thresholds
        },
    }


async def main():
    await init_db()

    logger.info("=" * 60)
    logger.info("ENSEMBLE TRAINING — HONEST METHODOLOGY")
    logger.info("=" * 60)

    # --- Load data ---
    logger.info("\nLoading resolved markets (ordered by resolved_at)...")
    markets, price_snapshots_map, price_at_as_of_map, orderbook_snapshots_map = await load_resolved_markets()
    logger.info(f"Loaded {len(markets)} resolved markets")

    # --- Build feature matrix (with as_of enforcement) ---
    snapshot_only = _SNAPSHOT_ONLY
    tradeable_range = _TRADEABLE_RANGE

    mode_parts = []
    if snapshot_only:
        mode_parts.append("snapshot_only=True")
    if tradeable_range is not None:
        lo, hi = tradeable_range
        mode_parts.append(f"tradeable_range=[{lo:.2f}, {hi:.2f}]")
    as_of_label = f"as_of=resolved_at-{_AS_OF_DAYS}d"
    mode_desc = ", ".join(mode_parts) if mode_parts else f"{as_of_label}, all prices"
    if mode_parts:
        mode_parts.insert(0, as_of_label)
        mode_desc = ", ".join(mode_parts)
    logger.info(f"\nExtracting features ({mode_desc})...")
    if tradeable_range is not None:
        logger.info(
            f"  NOTE: Near-decided markets (price outside [{tradeable_range[0]:.0%}, "
            f"{tradeable_range[1]:.0%}]) excluded — forces learning genuine uncertainty signal."
        )

    X, y = build_training_matrix(
        markets,
        price_snapshots_map=price_snapshots_map,
        price_at_as_of_map=price_at_as_of_map,
        orderbook_snapshots_map=orderbook_snapshots_map,
        snapshot_only=snapshot_only,
        tradeable_range=tradeable_range,
    )
    logger.info(f"Feature matrix: {X.shape} ({N_FEATURES} features)")
    logger.info(f"Features: {ENSEMBLE_FEATURE_NAMES}")

    n_yes = int(y.sum())
    n_no = len(y) - n_yes
    logger.info(f"Class balance: {n_yes} YES / {n_no} NO ({100 * y.mean():.1f}% YES)")

    if len(y) < 20:
        logger.error(f"Only {len(y)} samples with volume > 0. Not enough for training.")
        return

    if n_yes == 0 or n_no == 0:
        logger.error(
            f"FATAL: No class diversity — all {len(y)} samples are {'YES' if n_yes > 0 else 'NO'}. "
            f"Cannot train a meaningful model. Check data pipeline and as_of filtering."
        )
        return

    # --- Tripwire: Price-near-resolution leakage check ---
    price_col_idx = ENSEMBLE_FEATURE_NAMES.index("price_yes")
    prices = X[:, price_col_idx]
    pct_near_0 = float((prices < 0.05).sum() / len(prices))
    pct_near_1 = float((prices > 0.95).sum() / len(prices))
    logger.info(f"\nPrice distribution: {pct_near_0:.1%} near 0, {pct_near_1:.1%} near 1")
    if pct_near_0 + pct_near_1 > 0.30:
        logger.warning(
            "WARNING: >30% of training prices are near 0 or 1. "
            "Brier score may be artificially low (near-decided markets)."
        )

    # --- Tripwire: Volume contamination check ---
    # Volume features should NOT be highly correlated with resolution outcome
    # High correlation suggests post-resolution volume spikes are leaking into training
    volume_corrs = {}
    for vol_feature in ["volume_volatility", "volume_trend_7d", "log_volume_total", "volume_per_day"]:
        if vol_feature in ENSEMBLE_FEATURE_NAMES:
            vol_idx = ENSEMBLE_FEATURE_NAMES.index(vol_feature)
            # Compute Pearson correlation between volume feature and resolution outcome
            corr = np.corrcoef(X[:, vol_idx], y)[0, 1]
            volume_corrs[vol_feature] = corr

    if volume_corrs:
        logger.info(f"\nVolume feature correlations with resolution:")
        for feat, corr in volume_corrs.items():
            logger.info(f"  {feat}: {corr:.3f}")
            if abs(corr) > 0.4:
                logger.warning(
                    f"HIGH CORRELATION WARNING: {feat} has correlation {corr:.3f} with resolution. "
                    f"Suggests post-resolution volume contamination. Consider excluding volume features or "
                    f"implementing clean volume_24h computed from historical snapshots."
                )

    # --- Tripwire: Naive baseline ---
    class_prior = float(y.mean())
    naive_brier = class_prior * (1 - class_prior)
    logger.info(f"Naive baseline (predict {class_prior:.3f} everywhere): Brier = {naive_brier:.4f}")

    # --- Temporal split (BEFORE pruning, so we keep all features for calibration) ---
    logger.info("\n--- Temporal Train/Test Split ---")
    X_train_full, X_test_full, y_train, y_test, cutoff_date = temporal_split(
        markets, X, y, train_ratio=0.8
    )
    logger.info(f"Train: {len(y_train)} ({int(y_train.sum())} YES), "
                f"Test: {len(y_test)} ({int(y_test.sum())} YES)")
    logger.info(f"Temporal cutoff: ~{cutoff_date}")

    # --- Prune features (only for tree models, NOT for calibration) ---
    logger.info("\n--- Feature Pruning ---")
    X_train_pruned, active_features, dropped_features = prune_features(
        X_train_full, y_train, list(ENSEMBLE_FEATURE_NAMES)
    )
    X_test_pruned = X_test_full[:, [ENSEMBLE_FEATURE_NAMES.index(f) for f in active_features]]
    logger.info(f"Active: {len(active_features)} features, Dropped: {len(dropped_features)}")
    if dropped_features:
        for d in dropped_features:
            logger.info(f"  DROPPED: {d}")

    # Get price column from FULL feature set (for calibration)
    price_col_full = ENSEMBLE_FEATURE_NAMES.index("price_yes")

    # --- Baseline ---
    baseline_brier = brier_score_loss(y_test, X_test_full[:, price_col_full])
    logger.info(f"\n{'='*55}")
    logger.info(f"BASELINE (market price as probability)")
    logger.info(f"  Brier: {baseline_brier:.4f}")

    # --- Walk-forward OOF for weight computation ---
    logger.info(f"\n--- Walk-Forward OOF (for ensemble weights) ---")
    oof_results = walk_forward_oof(X_train_full, y_train, list(ENSEMBLE_FEATURE_NAMES), X_train_pruned, active_features)

    # --- Significance-gated model inclusion ---
    logger.info(f"\n--- Significance-Gated Model Inclusion ---")
    ensemble_models: dict[str, float] = {}
    cal_oof = oof_results["calibration"]

    # Calibration always included
    ensemble_models["calibration"] = cal_oof["oof_brier"]
    logger.info(f"  calibration: INCLUDED (baseline, OOF Brier={cal_oof['oof_brier']:.4f})")

    for name in ["xgboost", "lightgbm"]:
        model_oof = oof_results[name]
        if model_oof["oof_brier"] >= cal_oof["oof_brier"]:
            logger.info(f"  {name}: EXCLUDED (OOF {model_oof['oof_brier']:.4f} >= cal {cal_oof['oof_brier']:.4f})")
            continue

        # Use intersection of masks for fair comparison
        shared_mask = cal_oof["oof_mask"] & model_oof["oof_mask"]
        is_sig, p_val = significance_gate(
            y_train, cal_oof["oof_preds"], model_oof["oof_preds"], shared_mask
        )
        if is_sig:
            ensemble_models[name] = model_oof["oof_brier"]
            logger.info(f"  {name}: INCLUDED (OOF {model_oof['oof_brier']:.4f}, p={p_val:.3f})")
        else:
            logger.info(f"  {name}: EXCLUDED (not significant, p={p_val:.3f})")

    # --- Compute weights from OOF Brier scores ---
    logger.info(f"\n--- Computing Ensemble Weights (from OOF only) ---")
    weights = EnsembleModel.compute_weights(ensemble_models)
    logger.info(f"  Models included: {list(ensemble_models.keys())}")
    logger.info(f"  Weights: {weights}")
    logger.info(f"  Source: Walk-forward OOF Brier (NO test-set peeking)")

    # --- Train final models on full training set ---
    logger.info(f"\n--- Training Final Models on Full Train Set ---")

    cal_model = CalibrationModel()
    cal_model.train(X_train_full[:, price_col_full], y_train)
    cal_test_preds = cal_model.predict(X_test_full[:, price_col_full])
    cal_brier = brier_score_loss(y_test, cal_test_preds)
    logger.info(f"  Calibration test Brier: {cal_brier:.4f}")

    xgb_test_preds = None
    lgb_test_preds = None

    if "xgboost" in ensemble_models:
        xgb_model = XGBoostModel()
        xgb_model.train(X_train_pruned, y_train, feature_names=active_features)
        xgb_test_preds = xgb_model.predict_proba(X_test_pruned)
        xgb_brier = brier_score_loss(y_test, xgb_test_preds)
        logger.info(f"  XGBoost test Brier: {xgb_brier:.4f}")
        top_feats = list(xgb_model.get_feature_importance().items())[:5]
        logger.info(f"  Top features: {top_feats}")
    else:
        xgb_model = None

    if "lightgbm" in ensemble_models:
        lgb_model = LightGBMModel()
        lgb_model.train(X_train_pruned, y_train, feature_names=active_features)
        lgb_test_preds = lgb_model.predict_proba(X_test_pruned)
        lgb_brier = brier_score_loss(y_test, lgb_test_preds)
        logger.info(f"  LightGBM test Brier: {lgb_brier:.4f}")
        top_feats = list(lgb_model.get_feature_importance().items())[:5]
        logger.info(f"  Top features: {top_feats}")
    else:
        lgb_model = None

    # --- Compute ensemble predictions on test set ---
    test_preds = {"calibration": cal_test_preds}
    if xgb_test_preds is not None:
        test_preds["xgboost"] = xgb_test_preds
    if lgb_test_preds is not None:
        test_preds["lightgbm"] = lgb_test_preds

    ensemble_test_preds = sum(
        weights.get(name, 0) * p for name, p in test_preds.items()
    )
    ensemble_brier = brier_score_loss(y_test, ensemble_test_preds)

    # --- Brier by Price Bucket (understand real edge in tradeable range) ---
    market_prices_for_bucket = X_test_full[:, price_col_full]
    bucket_edges = [(0.0, 0.20, "0-20% (near NO)"),
                    (0.20, 0.80, "20-80% (tradeable)"),
                    (0.80, 1.01, "80-100% (near YES)")]
    brier_by_bucket = {}
    logger.info(f"\n--- Brier by Price Bucket ---")
    for lo, hi, label in bucket_edges:
        mask = (market_prices_for_bucket >= lo) & (market_prices_for_bucket < hi)
        n_bucket = int(mask.sum())
        if n_bucket >= 5:
            bucket_brier = float(brier_score_loss(y_test[mask], ensemble_test_preds[mask]))
            bucket_baseline = float(brier_score_loss(y_test[mask], market_prices_for_bucket[mask]))
            improvement = (1 - bucket_brier / bucket_baseline) * 100 if bucket_baseline > 0 else 0
            brier_by_bucket[label] = {
                "n": n_bucket, "brier": round(bucket_brier, 4),
                "baseline": round(bucket_baseline, 4),
                "improvement_pct": round(improvement, 1),
            }
            logger.info(
                f"  {label}: n={n_bucket}, Brier={bucket_brier:.4f} "
                f"(baseline {bucket_baseline:.4f}, {improvement:+.1f}%)"
            )
        else:
            brier_by_bucket[label] = {"n": n_bucket, "brier": None, "baseline": None, "improvement_pct": None}
            logger.info(f"  {label}: n={n_bucket} (too few for Brier)")

    # --- Post-Ensemble Calibration ---
    logger.info(f"\n--- Post-Ensemble Calibration ---")
    post_calibrator = CalibrationModel()

    # Compute ensemble predictions on training set
    train_preds = {"calibration": cal_model.predict(X_train_full[:, price_col_full])}
    if xgb_model:
        train_preds["xgboost"] = xgb_model.predict_proba(X_train_pruned)
    if lgb_model:
        train_preds["lightgbm"] = lgb_model.predict_proba(X_train_pruned)

    ensemble_train_preds = sum(
        weights.get(name, 0) * p for name, p in train_preds.items()
    )
    post_calibrator.train(ensemble_train_preds, y_train)

    post_cal_test_preds = post_calibrator.predict(ensemble_test_preds)
    post_cal_brier = brier_score_loss(y_test, post_cal_test_preds)
    post_cal_helps = post_cal_brier < ensemble_brier
    logger.info(f"  Post-calibrated Brier: {post_cal_brier:.4f} (before: {ensemble_brier:.4f})")
    if not post_cal_helps:
        logger.warning(
            f"  Post-calibrator HURTS: {post_cal_brier:.4f} > {ensemble_brier:.4f}. "
            f"Will NOT be saved (raw ensemble is better)."
        )

    # --- Ablation Study ---
    logger.info(f"\n--- Ablation Study ---")
    ablation = ablation_study(y_test, cal_test_preds, xgb_test_preds, lgb_test_preds, weights)
    for name, brier in ablation.items():
        delta = ablation["calibration_only"] - brier
        logger.info(f"  {name}: Brier={brier:.4f} (delta vs cal-only: {delta:+.4f})")

    # --- Comprehensive Metrics ---
    logger.info(f"\n--- Comprehensive Metrics ---")

    # Log-loss
    try:
        ensemble_logloss = log_loss(y_test, np.clip(post_cal_test_preds, 0.01, 0.99))
        logger.info(f"  Ensemble log-loss: {ensemble_logloss:.4f}")
    except ValueError:
        ensemble_logloss = None

    # AUC
    try:
        ensemble_auc = roc_auc_score(y_test, post_cal_test_preds)
        logger.info(f"  Ensemble AUC-ROC: {ensemble_auc:.4f}")
    except ValueError:
        ensemble_auc = 0.0

    # Optimal threshold via precision-recall
    try:
        prec, rec, thresholds = precision_recall_curve(y_test, post_cal_test_preds)
        f1_scores = 2 * prec * rec / (prec + rec + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
        logger.info(f"  Optimal threshold: {best_threshold:.3f} (F1={f1_scores[best_idx]:.3f})")
    except (ValueError, IndexError):
        best_threshold = 0.5

    # --- Profit Simulation (Polymarket-accurate fee model) ---
    logger.info(f"\n--- Profit Simulation (2% on winnings only + 1.5% slippage) ---")
    market_prices_test = X_test_full[:, price_col_full]
    profit_sim = profit_simulation(y_test, post_cal_test_preds, market_prices_test)
    logger.info(f"  Ungated: {profit_sim['ungated_trades']} trades, "
                f"P&L={profit_sim['ungated_pnl']:.4f}, "
                f"win rate={profit_sim['ungated_win_rate']:.1%}")
    logger.info(f"  Gated (>5% edge): {profit_sim['gated_trades']} trades, "
                f"P&L={profit_sim['gated_pnl']:.4f}, "
                f"win rate={profit_sim['gated_win_rate']:.1%}")

    if "threshold_sweep" in profit_sim:
        logger.info("  Threshold sweep:")
        for threshold, result in profit_sim["threshold_sweep"].items():
            logger.info(
                f"    {threshold}: {result['trades']:4d} trades, "
                f"P&L={result['pnl']:+.4f}, "
                f"win={result['win_rate']:.1%}"
            )

    if profit_sim["gated_pnl"] < profit_sim["ungated_pnl"] and profit_sim["gated_trades"] > 0:
        logger.warning("  WARNING: Quality gates not adding value (gated P&L < ungated P&L)")

    # --- Final Results ---
    logger.info(f"\n{'='*55}")
    logger.info(f"FINAL TEST SET RESULTS (n={len(y_test)})")
    logger.info(f"{'='*55}")
    logger.info(f"  Naive baseline:         {naive_brier:.4f}  (predict class prior)")
    logger.info(f"  Market baseline Brier:  {baseline_brier:.4f}")
    logger.info(f"  Calibration (Isotonic): {cal_brier:.4f}  ({_pct(cal_brier, baseline_brier)})")
    if xgb_test_preds is not None:
        logger.info(f"  XGBoost:                {brier_score_loss(y_test, xgb_test_preds):.4f}  "
                     f"({_pct(brier_score_loss(y_test, xgb_test_preds), baseline_brier)})")
    if lgb_test_preds is not None:
        logger.info(f"  LightGBM:               {brier_score_loss(y_test, lgb_test_preds):.4f}  "
                     f"({_pct(brier_score_loss(y_test, lgb_test_preds), baseline_brier)})")
    logger.info(f"  ENSEMBLE (raw):         {ensemble_brier:.4f}  ({_pct(ensemble_brier, baseline_brier)})")
    logger.info(f"  ENSEMBLE (calibrated):  {post_cal_brier:.4f}  ({_pct(post_cal_brier, baseline_brier)})")
    if ensemble_auc > 0:
        logger.info(f"  Ensemble AUC-ROC:       {ensemble_auc:.4f}")
    logger.info(f"{'='*55}")

    # --- Tripwire: ensemble must beat naive baseline ---
    best_ensemble_brier = ensemble_brier if not post_cal_helps else post_cal_brier
    if best_ensemble_brier >= naive_brier:
        logger.warning(
            f"TRIPWIRE: Ensemble Brier ({best_ensemble_brier:.4f}) >= naive baseline ({naive_brier:.4f}). "
            f"Model learned nothing useful."
        )

    # --- Save ---
    logger.info("\nSaving models...")
    cal_model.save()
    if xgb_model:
        xgb_model.save()
    if lgb_model:
        lgb_model.save()

    # Save Post-Calibrator (only if it improves performance)
    import joblib
    save_dir = project_root / "ml" / "saved_models"
    save_dir.mkdir(parents=True, exist_ok=True)
    post_cal_path = save_dir / "post_calibrator.joblib"
    if post_cal_helps:
        post_calibrator.save(post_cal_path)
        logger.info("  Saved: post_calibrator.joblib (improves ensemble)")
    else:
        # Delete stale post-calibrator so production uses raw ensemble
        if post_cal_path.exists():
            post_cal_path.unlink()
            logger.info("  Deleted stale post_calibrator.joblib (raw ensemble is better)")
        else:
            logger.info("  Skipped post_calibrator (raw ensemble is better)")

    # --- Metrics + Model Card ---
    n_with_snapshots = len(price_at_as_of_map)
    n_total_usable = len(y)
    coverage_pct = round(n_with_snapshots / n_total_usable * 100, 1) if n_total_usable else 0.0
    logger.info(
        f"Snapshot coverage: {n_with_snapshots}/{n_total_usable} "
        f"({coverage_pct}%) markets have as_of price history"
    )

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_total_resolved": len(markets),
        "n_usable": len(y),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "snapshot_coverage": {
            "n_with_snapshots": n_with_snapshots,
            "n_total": n_total_usable,
            "coverage_pct": coverage_pct,
            "snapshot_only_mode": snapshot_only,
            "tradeable_range": list(tradeable_range) if tradeable_range else None,
            "as_of_days": _AS_OF_DAYS,
        },
        "class_balance_yes_pct": round(float(y.mean()) * 100, 1),
        "temporal_split_date": cutoff_date,
        "feature_names": active_features,
        "features_dropped": dropped_features,
        "models_included": list(ensemble_models.keys()),
        "models_excluded": [m for m in ["xgboost", "lightgbm"] if m not in ensemble_models],
        "oof_brier": {name: round(r["oof_brier"], 4) for name, r in oof_results.items()},
        "baseline_brier": round(float(baseline_brier), 4),
        "naive_baseline_brier": round(float(naive_brier), 4),
        "calibration_brier": round(float(cal_brier), 4),
        "ensemble_brier": round(float(ensemble_brier), 4),
        "post_calibrated_brier": round(float(post_cal_brier), 4),
        "ensemble_auc": round(float(ensemble_auc), 4),
        "ensemble_logloss": round(float(ensemble_logloss), 4) if ensemble_logloss else None,
        "optimal_threshold": round(best_threshold, 3),
        "ablation": {k: round(v, 4) for k, v in ablation.items()},
        "profit_simulation": profit_sim,
        "brier_by_price_bucket": brier_by_bucket,
        "price_distribution": {
            "pct_near_0": round(pct_near_0, 3),
            "pct_near_1": round(pct_near_1, 3),
        },
        "leakage_warnings": [
            "orderbook_snapshots: filtered to as_of timestamp (fixed as of 2026-02-14)",
            f"volume_features: correlation with resolution = {max(volume_corrs.values(), key=abs) if volume_corrs else 0.0:.3f} (check if >0.3)"
            if volume_corrs else "volume_features: correlation check not performed",
            f"price_distribution: {pct_near_0 + pct_near_1:.1%} near-extremes (inflates metrics, real-world harder)",
            f"survivorship_bias: {len(y)}/{len(markets)} ({100*len(y)/max(1,len(markets)):.1f}%) of resolved markets used (volume>0 filter)",
        ],
    }

    if xgb_model:
        metrics["xgb_feature_importance"] = xgb_model.get_feature_importance()
    if lgb_model:
        metrics["lgb_feature_importance"] = lgb_model.get_feature_importance()

    ensemble = EnsembleModel()
    ensemble.save_weights(weights, metrics)

    # Save model card as JSON
    model_card_path = save_dir / "model_card.json"
    with open(model_card_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"  Saved: model_card.json")

    logger.info("\nAll models saved to ml/saved_models/")
    logger.info("Ensemble ready for API serving.")


def _pct(score: float, baseline: float) -> str:
    """Format improvement percentage vs baseline."""
    if baseline == 0:
        return "N/A"
    improvement = (1 - score / baseline) * 100
    return f"{improvement:+.1f}% vs baseline"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ensemble model on resolved markets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--archive-dir",
        default=None,
        help=(
            "Path to local Parquet archive produced by export_archive_to_local.py. "
            "When provided, training merges historical snapshots from archive with "
            "recent snapshots from DB so the model learns from full history."
        ),
    )
    parser.add_argument(
        "--snapshot-only",
        action="store_true",
        default=False,
        dest="snapshot_only",
        help=(
            "Only train on markets that have a real as_of snapshot price "
            "(eliminates market.price_yes leakage for resolved markets). "
            "Reduces training set size but produces cleaner calibration and "
            "enables all momentum/price features. Recommended when as_of "
            "coverage >= 20%%."
        ),
    )
    parser.add_argument(
        "--tradeable-range",
        default=None,
        dest="tradeable_range",
        metavar="MIN,MAX",
        help=(
            "Comma-separated price range e.g. '0.05,0.95'. When set with "
            "--snapshot-only, only include markets whose as_of price falls within "
            "[MIN, MAX]. Excludes near-decided markets (price near 0 or 1) that "
            "trivially inflate AUC/Brier. Use '0.05,0.95' for the full tradeable "
            "universe or '0.1,0.9' for stricter filtering. Honest Brier will be "
            "HIGHER (worse-looking) but the model will generalise to live uncertain "
            "markets instead of memorising obvious outcomes."
        ),
    )
    parser.add_argument(
        "--as-of-days",
        type=int,
        default=1,
        dest="as_of_days",
        help=(
            "Number of days before resolved_at to use as the as_of reference time "
            "(default: 1 = 24h). Increase to e.g. 7 to train the model on prices "
            "from 7 days before resolution, which are less certain and better "
            "reflect conditions you will face on live uncertain markets."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.snapshot_only:
        _SNAPSHOT_ONLY = True
        logger.info("Snapshot-only mode: training on markets with real as_of prices only (no leakage fallback).")

    if args.tradeable_range:
        try:
            lo_str, hi_str = args.tradeable_range.split(",")
            _TRADEABLE_RANGE = (float(lo_str.strip()), float(hi_str.strip()))
            logger.info(
                f"Tradeable-range filter: [{_TRADEABLE_RANGE[0]:.2f}, {_TRADEABLE_RANGE[1]:.2f}] — "
                "near-decided markets excluded to prevent leakage and force genuine learning."
            )
        except ValueError:
            logger.error(
                f"--tradeable-range must be 'MIN,MAX' e.g. '0.05,0.95'. Got: {args.tradeable_range!r}"
            )
            sys.exit(1)

    if args.as_of_days != 1:
        _AS_OF_DAYS = args.as_of_days
        logger.info(
            f"as-of-days={_AS_OF_DAYS}: using prices from {_AS_OF_DAYS} days before resolution."
        )

    if args.archive_dir:
        _archive_path = Path(args.archive_dir)
        if not _archive_path.is_absolute():
            _archive_path = project_root / _archive_path
        if not _archive_path.exists():
            logger.warning(
                f"--archive-dir '{_archive_path}' does not exist. "
                f"Run export_archive_to_local.py first. Continuing without archive."
            )
        else:
            _ARCHIVE_DIR = _archive_path
            logger.info(f"Archive dir: {_ARCHIVE_DIR}")

    asyncio.run(main())
