"""ML model prediction endpoints."""

import logging
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_session
from db.models import Market
from ml.models.calibration_model import CalibrationModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ml"])

# Singleton models
_calibration_model: CalibrationModel | None = None
_ensemble_model = None


def get_calibration_model() -> CalibrationModel:
    global _calibration_model
    if _calibration_model is None:
        _calibration_model = CalibrationModel()
        _calibration_model.load()
    return _calibration_model


def get_ensemble_model():
    """Lazy-load ensemble model (calibration + XGBoost + LightGBM)."""
    global _ensemble_model
    if _ensemble_model is None:
        from ml.models.ensemble import EnsembleModel
        _ensemble_model = EnsembleModel()
        _ensemble_model.load_all()
    return _ensemble_model


@router.get("/predictions/{market_id}")
async def get_prediction(
    market_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Get ML predictions for a market (calibration + ensemble if available)."""
    market = await session.get(Market, market_id)
    if not market:
        return {"error": "Market not found"}

    if market.price_yes is None:
        return {"error": "No price data available"}

    model = get_calibration_model()
    mispricing = model.get_mispricing(market.price_yes)

    # Try ensemble prediction
    ensemble_data = None
    try:
        ensemble = get_ensemble_model()
        ensemble_data = ensemble.predict_market(market)
    except Exception as e:
        logger.debug(f"Ensemble prediction unavailable: {e}")

    return {
        "market_id": market_id,
        "question": market.question,
        "models": {
            "calibration": {
                "market_price": mispricing["market_price"],
                "calibrated_price": mispricing["calibrated_price"],
                "delta": mispricing["delta"],
                "delta_pct": mispricing["delta_pct"],
                "direction": mispricing["direction"],
                "edge_estimate": mispricing["edge_estimate"],
            },
            "ensemble": ensemble_data,
        },
    }


@router.get("/predictions/{market_id}/ensemble")
async def get_ensemble_prediction(
    market_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Get ensemble (calibration + XGBoost + LightGBM) prediction."""
    market = await session.get(Market, market_id)
    if not market:
        return {"error": "Market not found"}

    if market.price_yes is None:
        return {"error": "No price data available"}

    try:
        ensemble = get_ensemble_model()
        result = ensemble.predict_market(market)
        return {
            "market_id": market_id,
            "question": market.question,
            "ensemble": result,
        }
    except Exception as e:
        # Fallback to calibration-only
        model = get_calibration_model()
        mispricing = model.get_mispricing(market.price_yes)
        return {
            "market_id": market_id,
            "question": market.question,
            "ensemble": None,
            "fallback": "calibration_only",
            "calibration": mispricing,
            "error": str(e),
        }


@router.get("/predictions/top/mispriced")
async def top_mispriced(
    limit: int = 20,
    session: AsyncSession = Depends(get_session),
):
    """Get markets with highest calibration mispricing."""
    from sqlalchemy import select
    markets = (await session.execute(
        select(Market)
        .where(Market.is_active == True, Market.price_yes != None)  # noqa
        .order_by(Market.volume_24h.desc())
        .limit(200)
    )).scalars().all()

    model = get_calibration_model()
    results = []

    for m in markets:
        if m.price_yes is None or m.price_yes <= 0.01 or m.price_yes >= 0.99:
            continue

        mispricing = model.get_mispricing(m.price_yes)
        results.append({
            "market_id": m.id,
            "question": m.question,
            "category": m.category,
            "price_yes": m.price_yes,
            "calibrated_price": mispricing["calibrated_price"],
            "delta_pct": mispricing["delta_pct"],
            "direction": mispricing["direction"],
            "edge_estimate": mispricing["edge_estimate"],
            "volume_24h": m.volume_24h,
        })

    # Sort by absolute delta (biggest mispricings first)
    results.sort(key=lambda x: abs(x["delta_pct"]), reverse=True)
    return {"markets": results[:limit]}


@router.get("/predictions/accuracy")
async def get_model_accuracy():
    """Get model performance metrics from training."""
    try:
        ensemble = get_ensemble_model()
        metrics = ensemble.metrics
        weights = ensemble.weights

        return {
            "trained": bool(metrics),
            "metrics": metrics,
            "weights": weights,
            "models": {
                "calibration": {
                    "name": "Isotonic Regression",
                    "type": "calibration",
                    "brier_score": metrics.get("calibration_brier"),
                    "weight": weights.get("calibration", 0),
                },
                "xgboost": {
                    "name": "XGBoost",
                    "type": "gradient_boosting",
                    "brier_score": metrics.get("xgboost_brier"),
                    "weight": weights.get("xgboost", 0),
                    "feature_importance": metrics.get("xgb_feature_importance", {}),
                },
                "lightgbm": {
                    "name": "LightGBM",
                    "type": "gradient_boosting",
                    "brier_score": metrics.get("lightgbm_brier"),
                    "weight": weights.get("lightgbm", 0),
                    "feature_importance": metrics.get("lgb_feature_importance", {}),
                },
                "ensemble": {
                    "name": "Weighted Ensemble",
                    "type": "ensemble",
                    "brier_score": metrics.get("ensemble_brier"),
                    "auc_roc": metrics.get("ensemble_auc"),
                },
            },
            "baseline_brier": metrics.get("baseline_brier"),
            "training_samples": metrics.get("n_train"),
            "test_samples": metrics.get("n_test"),
            "total_resolved": metrics.get("n_total_resolved"),
            "usable_samples": metrics.get("n_usable"),
        }
    except Exception as e:
        return {
            "trained": False,
            "error": str(e),
            "hint": "Run: python scripts/train_ensemble.py",
        }


@router.get("/calibration/curve")
async def calibration_curve():
    """Get calibration curve data for visualization."""
    model = get_calibration_model()
    return {"curve": model.get_calibration_curve(n_points=20)}
