"""Market browsing and detail endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_session
from db.models import Market, Platform, PriceSnapshot, CrossPlatformMatch

router = APIRouter(tags=["markets"])


@router.get("/markets")
async def list_markets(
    platform: str | None = None,
    category: str | None = None,
    search: str | None = None,
    is_active: bool = True,
    sort_by: str = "volume_24h",
    limit: int = Query(default=50, le=500),
    offset: int = 0,
    session: AsyncSession = Depends(get_session),
):
    """List markets with filters."""
    query = select(Market)

    if is_active:
        query = query.where(Market.is_active == True)  # noqa

    if platform:
        platform_result = await session.execute(
            select(Platform).where(Platform.name == platform)
        )
        p = platform_result.scalar_one_or_none()
        if p:
            query = query.where(Market.platform_id == p.id)

    if category:
        query = query.where(Market.category == category)

    if search:
        query = query.where(
            or_(
                Market.question.ilike(f"%{search}%"),
                Market.description.ilike(f"%{search}%"),
            )
        )

    # Sorting
    sort_col = getattr(Market, sort_by, Market.volume_24h)
    query = query.order_by(sort_col.desc()).offset(offset).limit(limit)

    result = await session.execute(query)
    markets = result.scalars().all()

    # Get total count
    count_query = select(func.count(Market.id)).where(Market.is_active == True)  # noqa
    total = (await session.execute(count_query)).scalar()

    # Get platform names
    platforms_result = await session.execute(select(Platform))
    platform_map = {p.id: p.name for p in platforms_result.scalars().all()}

    return {
        "markets": [
            {
                "id": m.id,
                "platform": platform_map.get(m.platform_id, "unknown"),
                "external_id": m.external_id,
                "question": m.question,
                "description": m.description,
                "category": m.category,
                "price_yes": m.price_yes,
                "price_no": m.price_no,
                "volume_24h": m.volume_24h,
                "volume_total": m.volume_total,
                "liquidity": m.liquidity,
                "end_date": m.end_date.isoformat() if m.end_date else None,
                "is_neg_risk": m.is_neg_risk,
                "updated_at": m.updated_at.isoformat() if m.updated_at else None,
            }
            for m in markets
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/markets/categories")
async def list_categories(session: AsyncSession = Depends(get_session)):
    """List all market categories with counts."""
    result = await session.execute(
        select(Market.category, func.count(Market.id))
        .where(Market.is_active == True)  # noqa
        .group_by(Market.category)
        .order_by(func.count(Market.id).desc())
    )
    return [{"category": row[0] or "other", "count": row[1]} for row in result.all()]


@router.get("/markets/{market_id}")
async def get_market_detail(
    market_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Get detailed market information including recent prices."""
    market = await session.get(Market, market_id)
    if not market:
        return {"error": "Market not found"}

    platform = await session.get(Platform, market.platform_id)

    # Get recent price history
    prices_result = await session.execute(
        select(PriceSnapshot)
        .where(PriceSnapshot.market_id == market_id)
        .order_by(PriceSnapshot.timestamp.desc())
        .limit(100)
    )
    prices = prices_result.scalars().all()

    # Check for cross-platform match
    match_result = await session.execute(
        select(CrossPlatformMatch).where(
            or_(
                CrossPlatformMatch.market_id_a == market_id,
                CrossPlatformMatch.market_id_b == market_id,
            )
        )
    )
    matches = match_result.scalars().all()

    matched_markets = []
    for m in matches:
        other_id = m.market_id_b if m.market_id_a == market_id else m.market_id_a
        other = await session.get(Market, other_id)
        if other:
            other_platform = await session.get(Platform, other.platform_id)
            matched_markets.append({
                "id": other.id,
                "platform": other_platform.name if other_platform else "unknown",
                "question": other.question,
                "price_yes": other.price_yes,
                "similarity": m.similarity_score,
            })

    return {
        "id": market.id,
        "platform": platform.name if platform else "unknown",
        "external_id": market.external_id,
        "question": market.question,
        "description": market.description,
        "category": market.category,
        "price_yes": market.price_yes,
        "price_no": market.price_no,
        "volume_24h": market.volume_24h,
        "volume_total": market.volume_total,
        "liquidity": market.liquidity,
        "end_date": market.end_date.isoformat() if market.end_date else None,
        "is_neg_risk": market.is_neg_risk,
        "token_id_yes": market.token_id_yes,
        "token_id_no": market.token_id_no,
        "price_history": [
            {
                "timestamp": p.timestamp.isoformat(),
                "price_yes": p.price_yes,
                "price_no": p.price_no,
                "volume": p.volume,
            }
            for p in reversed(prices)
        ],
        "cross_platform_matches": matched_markets,
    }
