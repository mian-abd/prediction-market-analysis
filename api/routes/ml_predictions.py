"""ML model prediction endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_session
from db.models import Market
from ml.models.calibration_model import CalibrationModel

router = APIRouter(tags=["ml"])

# Singleton calibration model
_calibration_model: CalibrationModel | None = None


def get_calibration_model() -> CalibrationModel:
    global _calibration_model
    if _calibration_model is None:
        _calibration_model = CalibrationModel()
        _calibration_model.load()
    return _calibration_model


@router.get("/predictions/{market_id}")
async def get_prediction(
    market_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Get ML predictions for a market."""
    market = await session.get(Market, market_id)
    if not market:
        return {"error": "Market not found"}

    if market.price_yes is None:
        return {"error": "No price data available"}

    model = get_calibration_model()
    mispricing = model.get_mispricing(market.price_yes)

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
        },
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


@router.get("/calibration/curve")
async def calibration_curve():
    """Get calibration curve data for visualization."""
    model = get_calibration_model()
    return {"curve": model.get_calibration_curve(n_points=20)}
