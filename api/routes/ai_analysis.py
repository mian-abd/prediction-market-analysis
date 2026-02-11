"""Claude AI analysis endpoints. User-triggered only."""

from fastapi import APIRouter, Depends
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_session
from db.models import Market, AIAnalysis, PriceSnapshot
from ai_analysis.claude_client import ClaudeClient
from ai_analysis.prompts.market_analysis import format_analysis_prompt
from ml.models.calibration_model import CalibrationModel

router = APIRouter(tags=["ai"])

_claude_client: ClaudeClient | None = None
_calibration_model: CalibrationModel | None = None


def get_claude() -> ClaudeClient:
    global _claude_client
    if _claude_client is None:
        _claude_client = ClaudeClient()
    return _claude_client


def get_calibration() -> CalibrationModel:
    global _calibration_model
    if _calibration_model is None:
        _calibration_model = CalibrationModel()
        _calibration_model.load()
    return _calibration_model


@router.post("/analyze/{market_id}")
async def analyze_market(
    market_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Trigger Claude analysis for a market. User-triggered, cached."""
    market = await session.get(Market, market_id)
    if not market:
        return {"error": "Market not found"}

    # Get calibration data
    calibration = get_calibration()
    cal_data = calibration.get_mispricing(market.price_yes or 0.5)

    # Get recent price history
    prices = (await session.execute(
        select(PriceSnapshot)
        .where(PriceSnapshot.market_id == market_id)
        .order_by(PriceSnapshot.timestamp.desc())
        .limit(20)
    )).scalars().all()

    price_history = [
        {"price_yes": p.price_yes, "timestamp": p.timestamp.isoformat()}
        for p in reversed(prices)
    ]

    # Format market dict
    market_dict = {
        "question": market.question,
        "description": market.description or "",
        "category": market.category or "other",
        "price_yes": market.price_yes or 0,
        "price_no": market.price_no or 0,
        "volume_24h": market.volume_24h or 0,
        "volume_total": market.volume_total or 0,
        "liquidity": market.liquidity or 0,
        "end_date": market.end_date.isoformat() if market.end_date else "Unknown",
        "time_to_resolution": "Unknown",
    }

    # Format prompts
    system_prompt, user_prompt = format_analysis_prompt(market_dict, cal_data, price_history)

    # Call Claude (or return cache)
    claude = get_claude()
    result = await claude.analyze(
        session=session,
        market_id=market_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    return {
        "market_id": market_id,
        "question": market.question,
        "analysis": result,
    }


@router.get("/analyze/{market_id}/cached")
async def get_cached_analysis(
    market_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Get cached analysis without triggering new API call."""
    result = await session.execute(
        select(AIAnalysis)
        .where(AIAnalysis.market_id == market_id)
        .order_by(AIAnalysis.created_at.desc())
        .limit(1)
    )
    analysis = result.scalar_one_or_none()

    if not analysis:
        return {"cached": False, "message": "No analysis cached for this market"}

    return {
        "cached": True,
        "market_id": market_id,
        "analysis_type": analysis.analysis_type,
        "response_text": analysis.response_text,
        "structured_result": analysis.structured_result,
        "cost": analysis.estimated_cost_usd,
        "created_at": analysis.created_at.isoformat(),
    }


@router.get("/analyze/cost")
async def get_cost_summary(session: AsyncSession = Depends(get_session)):
    """Get Claude API spend summary."""
    total_cost = (await session.execute(
        select(func.sum(AIAnalysis.estimated_cost_usd))
    )).scalar() or 0

    total_calls = (await session.execute(
        select(func.count(AIAnalysis.id))
    )).scalar() or 0

    total_tokens_in = (await session.execute(
        select(func.sum(AIAnalysis.input_tokens))
    )).scalar() or 0

    total_tokens_out = (await session.execute(
        select(func.sum(AIAnalysis.output_tokens))
    )).scalar() or 0

    return {
        "total_cost_usd": total_cost,
        "total_calls": total_calls,
        "total_input_tokens": total_tokens_in,
        "total_output_tokens": total_tokens_out,
        "avg_cost_per_call": total_cost / total_calls if total_calls > 0 else 0,
    }
