"""ML model prediction endpoints.

Two modes:
- Coverage (dashboard/research): predict everything with quality metadata
- Precision (trading signals): only return markets that pass quality gates
"""

import logging
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_session
from db.models import Market
from ml.models.calibration_model import CalibrationModel
from ml.strategies.ensemble_edge_detector import (
    detect_edge,
    edge_signal_to_dict,
)

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


# --- Static routes MUST be defined before /predictions/{market_id} ---
# FastAPI matches routes in order; {market_id} would catch "accuracy" etc.

@router.get("/predictions/accuracy/backtest")
async def get_signal_backtest(
    session: AsyncSession = Depends(get_session),
):
    """Backtest: compare our past signals against actual market resolutions.

    Returns hit rate, Brier score, simulated P&L, and breakdowns by direction/tier.
    """
    from ml.evaluation.signal_tracker import compute_signal_accuracy
    return await compute_signal_accuracy(session)


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
            "models_included": metrics.get("models_included", []),
            "models_excluded": metrics.get("models_excluded", []),
            "methodology": {
                "temporal_split": True,
                "walk_forward_cv": True,
                "significance_gated": True,
                "split_date": metrics.get("temporal_split_date"),
            },
            "baseline_brier": metrics.get("baseline_brier"),
            "naive_baseline_brier": metrics.get("naive_baseline_brier"),
            "ensemble_brier": metrics.get("post_calibrated_brier") or metrics.get("ensemble_brier"),
            "ensemble_auc": metrics.get("ensemble_auc"),
            "ablation": metrics.get("ablation"),
            "profit_simulation": metrics.get("profit_simulation"),
            "training_samples": metrics.get("n_train"),
            "test_samples": metrics.get("n_test"),
            "features_used": len(metrics.get("feature_names", [])),
            "features_dropped": len(metrics.get("features_dropped", [])),
        }
    except Exception as e:
        return {
            "trained": False,
            "error": str(e),
            "hint": "Run: python scripts/train_ensemble.py",
        }


@router.get("/predictions/{market_id}")
async def get_prediction(
    market_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Get ML predictions for a market with quality gates and edge signal."""
    market = await session.get(Market, market_id)
    if not market:
        return {"error": "Market not found"}

    if market.price_yes is None:
        return {"error": "No price data available"}

    model = get_calibration_model()
    mispricing = model.get_mispricing(market.price_yes)

    # Ensemble prediction + edge detection
    ensemble_data = None
    edge_data = None
    try:
        ensemble = get_ensemble_model()
        ensemble_data = ensemble.predict_market(market)
        edge = detect_edge(market, ensemble_data)
        edge_data = edge_signal_to_dict(edge)
    except Exception as e:
        logger.debug(f"Ensemble/edge detection unavailable: {e}")

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
        "edge_signal": edge_data,
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
        edge = detect_edge(market, result)
        return {
            "market_id": market_id,
            "question": market.question,
            "ensemble": result,
            "edge_signal": edge_signal_to_dict(edge),
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
    limit: int = Query(default=20, le=100),
    min_edge: float = Query(default=0.0, ge=0.0, description="Minimum edge % (0-100)"),
    session: AsyncSession = Depends(get_session),
):
    """Coverage mode: markets with highest mispricing, with quality metadata."""
    markets = (await session.execute(
        select(Market)
        .where(Market.is_active == True, Market.price_yes != None)  # noqa
        .order_by(Market.volume_24h.desc())
        .limit(200)
    )).scalars().all()

    ensemble = None
    try:
        ensemble = get_ensemble_model()
    except Exception:
        pass

    model = get_calibration_model()
    results = []

    for m in markets:
        if m.price_yes is None or m.price_yes <= 0.01 or m.price_yes >= 0.99:
            continue

        mispricing = model.get_mispricing(m.price_yes)

        # Ensemble + edge if available
        edge_data = None
        ensemble_data = None
        if ensemble:
            try:
                ensemble_data = ensemble.predict_market(m)
                edge = detect_edge(m, ensemble_data)
                edge_data = edge_signal_to_dict(edge)
            except Exception:
                pass

        # Use ensemble delta if available, otherwise calibration
        if ensemble_data:
            delta_pct = ensemble_data["delta_pct"]
            direction = ensemble_data["direction"]
            edge_estimate = ensemble_data["edge_estimate"]
        else:
            delta_pct = mispricing["delta_pct"]
            direction = mispricing["direction"]
            edge_estimate = mispricing["edge_estimate"]

        # Filter by min_edge if specified
        if min_edge > 0 and abs(delta_pct) < min_edge:
            continue

        results.append({
            "market_id": m.id,
            "question": m.question,
            "category": m.normalized_category or m.category,
            "price_yes": m.price_yes,
            "calibrated_price": mispricing["calibrated_price"],
            "delta_pct": delta_pct,
            "direction": direction,
            "edge_estimate": edge_estimate,
            "volume_24h": m.volume_24h,
            "edge_signal": edge_data,
        })

    # Sort by absolute delta (biggest mispricings first)
    results.sort(key=lambda x: abs(x["delta_pct"]), reverse=True)
    return {"markets": results[:limit]}


@router.get("/predictions/top/edges")
async def top_edges(
    limit: int = Query(default=20, le=100),
    min_edge: float = Query(default=3.0, ge=0.0, description="Minimum net edge % after fees"),
    min_confidence: float = Query(default=0.5, ge=0.0, le=1.0),
    quality: str = Query(default="all", pattern="^(high|medium|all)$"),
    session: AsyncSession = Depends(get_session),
):
    """Precision mode: only markets passing ALL quality gates with actionable edges.

    This is the "what should I actually trade?" endpoint.
    """
    markets = (await session.execute(
        select(Market)
        .where(Market.is_active == True, Market.price_yes != None)  # noqa
        .order_by(Market.volume_24h.desc())
        .limit(500)
    )).scalars().all()

    try:
        ensemble = get_ensemble_model()
    except Exception as e:
        return {"error": f"Ensemble not available: {e}", "edges": []}

    results = []

    for m in markets:
        if m.price_yes is None or m.price_yes <= 0.01 or m.price_yes >= 0.99:
            continue

        try:
            ensemble_data = ensemble.predict_market(m)
            edge = detect_edge(m, ensemble_data)
        except Exception:
            continue

        # Quality gate check
        if not edge.quality_gate.passes:
            continue

        # Edge threshold
        if edge.net_ev * 100 < min_edge:
            continue

        # Confidence threshold
        if edge.confidence < min_confidence:
            continue

        # Quality tier filter
        if quality == "high" and edge.quality_tier != "high":
            continue
        if quality == "medium" and edge.quality_tier not in ("high", "medium"):
            continue

        # Must have a tradeable direction
        if edge.direction is None:
            continue

        results.append({
            "market_id": m.id,
            "question": m.question,
            "category": m.normalized_category or m.category,
            **edge_signal_to_dict(edge),
        })

    # Sort by net EV descending
    results.sort(key=lambda x: x["net_ev_pct"], reverse=True)
    return {"edges": results[:limit], "total_scanned": len(markets)}


@router.get("/calibration/curve")
async def calibration_curve():
    """Get calibration curve data for visualization."""
    model = get_calibration_model()
    return {"curve": model.get_calibration_curve(n_points=20)}
