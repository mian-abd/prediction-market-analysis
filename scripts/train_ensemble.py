"""Train XGBoost + LightGBM + Calibration ensemble on resolved markets.

Usage:
    python scripts/train_ensemble.py

Prerequisite:
    python scripts/backfill_resolved_markets.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression

from db.database import async_session, init_db
from db.models import Market
from sqlalchemy import select

from ml.features.training_features import (
    build_training_matrix,
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


async def load_resolved_markets() -> list[Market]:
    """Load resolved markets with valid outcomes from DB."""
    async with async_session() as session:
        result = await session.execute(
            select(Market).where(
                Market.is_resolved == True,
                Market.resolution_value != None,
                Market.price_yes != None,
            )
        )
        markets = list(result.scalars().all())
        session.expunge_all()
        return markets


async def main():
    await init_db()

    logger.info("Loading resolved markets...")
    markets = await load_resolved_markets()
    logger.info(f"Loaded {len(markets)} resolved markets")

    # Build feature matrix (filters zero-volume markets)
    logger.info("Extracting features...")
    X, y = build_training_matrix(markets)
    logger.info(f"Feature matrix: {X.shape} ({N_FEATURES} features)")
    logger.info(f"Features: {ENSEMBLE_FEATURE_NAMES}")

    n_yes = int(y.sum())
    n_no = len(y) - n_yes
    logger.info(f"Class balance: {n_yes} YES / {n_no} NO ({100 * y.mean():.1f}% YES)")

    if len(y) < 20:
        logger.error(f"Only {len(y)} samples with volume > 0. Not enough for training.")
        logger.error("Collect more resolved markets or lower the volume filter.")
        return

    # Train/test split (stratified)
    test_size = min(0.2, max(0.1, 20 / len(y)))  # At least 20 test samples
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y,
    )
    logger.info(f"Train: {len(y_train)} ({int(y_train.sum())} YES), "
                f"Test: {len(y_test)} ({int(y_test.sum())} YES)")

    price_col = ENSEMBLE_FEATURE_NAMES.index("price_yes")

    # --- Baseline: raw market price ---
    baseline_brier = brier_score_loss(y_test, X_test[:, price_col])
    baseline_acc = accuracy_score(y_test, (X_test[:, price_col] > 0.5).astype(int))
    logger.info(f"\n{'='*55}")
    logger.info(f"BASELINE (market price as probability)")
    logger.info(f"  Brier: {baseline_brier:.4f}, Accuracy: {baseline_acc:.1%}")

    # --- Logistic Regression baseline ---
    logger.info(f"\n--- Logistic Regression Baseline ---")
    lr = LogisticRegression(
        C=0.1, max_iter=1000, class_weight="balanced", random_state=42,
    )
    lr.fit(X_train, y_train)
    lr_test_preds = lr.predict_proba(X_test)[:, 1]
    lr_brier = brier_score_loss(y_test, lr_test_preds)
    logger.info(f"  Brier: {lr_brier:.4f}")

    # --- Calibration Model ---
    logger.info(f"\n--- Calibration Model (Isotonic Regression) ---")
    cal_model = CalibrationModel()
    cal_model.train(X_train[:, price_col], y_train)
    cal_test_preds = cal_model.predict(X_test[:, price_col])
    cal_brier = brier_score_loss(y_test, cal_test_preds)
    logger.info(f"  Brier: {cal_brier:.4f}")

    # --- XGBoost ---
    logger.info(f"\n--- XGBoost ---")
    xgb_model = XGBoostModel()
    xgb_model.train(X_train, y_train, feature_names=ENSEMBLE_FEATURE_NAMES)
    xgb_test_preds = xgb_model.predict_proba(X_test)
    xgb_brier = brier_score_loss(y_test, xgb_test_preds)
    logger.info(f"  Test Brier: {xgb_brier:.4f}")
    top_feats = list(xgb_model.get_feature_importance().items())[:5]
    logger.info(f"  Top features: {top_feats}")

    # --- LightGBM ---
    logger.info(f"\n--- LightGBM ---")
    lgb_model = LightGBMModel()
    lgb_model.train(X_train, y_train, feature_names=ENSEMBLE_FEATURE_NAMES)
    lgb_test_preds = lgb_model.predict_proba(X_test)
    lgb_brier = brier_score_loss(y_test, lgb_test_preds)
    logger.info(f"  Test Brier: {lgb_brier:.4f}")
    top_feats = list(lgb_model.get_feature_importance().items())[:5]
    logger.info(f"  Top features: {top_feats}")

    # --- Compute Ensemble Weights ---
    logger.info(f"\n--- Computing Ensemble Weights ---")
    # Use OOF Brier scores (from training CV) for weight computation
    brier_scores = {
        "calibration": cal_brier,
        "xgboost": xgb_model.brier_score or xgb_brier,
        "lightgbm": lgb_model.brier_score or lgb_brier,
    }
    logger.info(f"  Model Brier scores: {brier_scores}")
    weights = EnsembleModel.compute_weights(brier_scores)
    logger.info(f"  Weights: {weights}")

    # --- Evaluate Ensemble ---
    ensemble_preds = (
        weights["calibration"] * cal_test_preds
        + weights["xgboost"] * xgb_test_preds
        + weights["lightgbm"] * lgb_test_preds
    )
    ensemble_brier = brier_score_loss(y_test, ensemble_preds)

    # AUC (if both classes present in test set)
    try:
        ensemble_auc = roc_auc_score(y_test, ensemble_preds)
    except ValueError:
        ensemble_auc = 0.0

    # --- Final Results ---
    logger.info(f"\n{'='*55}")
    logger.info(f"FINAL TEST SET RESULTS (n={len(y_test)})")
    logger.info(f"{'='*55}")
    logger.info(f"  Market baseline Brier:  {baseline_brier:.4f}")
    logger.info(f"  Logistic Regression:    {lr_brier:.4f}  ({_pct(lr_brier, baseline_brier)})")
    logger.info(f"  Calibration (Isotonic): {cal_brier:.4f}  ({_pct(cal_brier, baseline_brier)})")
    logger.info(f"  XGBoost:                {xgb_brier:.4f}  ({_pct(xgb_brier, baseline_brier)})")
    logger.info(f"  LightGBM:               {lgb_brier:.4f}  ({_pct(lgb_brier, baseline_brier)})")
    logger.info(f"  ENSEMBLE:               {ensemble_brier:.4f}  ({_pct(ensemble_brier, baseline_brier)})")
    if ensemble_auc > 0:
        logger.info(f"  Ensemble AUC-ROC:       {ensemble_auc:.4f}")
    logger.info(f"{'='*55}")

    # --- Save ---
    logger.info("\nSaving models...")
    cal_model.save()
    xgb_model.save()
    lgb_model.save()

    metrics = {
        "n_total_resolved": len(markets),
        "n_usable": len(y),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "class_balance_yes_pct": round(float(y.mean()) * 100, 1),
        "baseline_brier": round(float(baseline_brier), 4),
        "logistic_brier": round(float(lr_brier), 4),
        "calibration_brier": round(float(cal_brier), 4),
        "xgboost_brier": round(float(xgb_brier), 4),
        "lightgbm_brier": round(float(lgb_brier), 4),
        "ensemble_brier": round(float(ensemble_brier), 4),
        "ensemble_auc": round(float(ensemble_auc), 4),
        "feature_names": ENSEMBLE_FEATURE_NAMES,
        "xgb_feature_importance": xgb_model.get_feature_importance(),
        "lgb_feature_importance": lgb_model.get_feature_importance(),
    }

    ensemble = EnsembleModel()
    ensemble.save_weights(weights, metrics)

    logger.info("All models saved to ml/saved_models/")
    logger.info("Ensemble ready for API serving.")


def _pct(score: float, baseline: float) -> str:
    """Format improvement percentage vs baseline."""
    if baseline == 0:
        return "N/A"
    improvement = (1 - score / baseline) * 100
    return f"{improvement:+.1f}% vs baseline"


if __name__ == "__main__":
    asyncio.run(main())
