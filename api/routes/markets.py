"""Market browsing and detail endpoints."""

from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_session
from db.models import Market, Platform, PriceSnapshot, OrderbookSnapshot, CrossPlatformMatch

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


@router.get("/markets/{market_id}/price-history")
async def get_price_history(
    market_id: int,
    interval: str = Query(default="5m", regex="^(1m|5m|15m|1h|4h|1d)$"),
    limit: int = Query(default=500, le=1000),
    session: AsyncSession = Depends(get_session),
):
    """Get OHLC price history for charting.

    Returns candlestick data grouped by time interval.
    Intervals: 1m, 5m, 15m, 1h, 4h, 1d
    """
    # Verify market exists
    market = await session.get(Market, market_id)
    if not market:
        raise HTTPException(status_code=404, detail="Market not found")

    # Map interval to minutes
    interval_minutes = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
    }

    minutes = interval_minutes.get(interval, 5)

    # Calculate lookback period (limit * interval)
    lookback_minutes = limit * minutes
    cutoff_time = datetime.utcnow() - timedelta(minutes=lookback_minutes)

    # Fetch raw price snapshots
    result = await session.execute(
        select(PriceSnapshot)
        .where(
            PriceSnapshot.market_id == market_id,
            PriceSnapshot.timestamp >= cutoff_time
        )
        .order_by(PriceSnapshot.timestamp.asc())
    )
    snapshots = result.scalars().all()

    if not snapshots:
        return {"data": []}

    # Group snapshots into OHLC candles
    candles = []
    current_candle_start = None
    current_candle_data = []

    for snapshot in snapshots:
        # Determine which candle this snapshot belongs to
        timestamp_minutes = int(snapshot.timestamp.timestamp() / 60)
        candle_start_minutes = (timestamp_minutes // minutes) * minutes
        candle_start = datetime.utcfromtimestamp(candle_start_minutes * 60)

        # New candle?
        if current_candle_start != candle_start:
            # Save previous candle if exists
            if current_candle_data:
                candles.append(create_candle(current_candle_start, current_candle_data))

            # Start new candle
            current_candle_start = candle_start
            current_candle_data = [snapshot]
        else:
            current_candle_data.append(snapshot)

    # Save last candle
    if current_candle_data:
        candles.append(create_candle(current_candle_start, current_candle_data))

    return {"data": candles[-limit:]}  # Return last N candles


def create_candle(start_time: datetime, snapshots: list[PriceSnapshot]) -> dict:
    """Create OHLC candle from price snapshots."""
    if not snapshots:
        return None

    # Use YES price for OHLC (main price)
    prices = [s.price_yes for s in snapshots if s.price_yes is not None]

    if not prices:
        # Fallback to close price (1 - NO price)
        prices = [1.0 - s.price_no for s in snapshots if s.price_no is not None]

    if not prices:
        return None

    # Calculate OHLC
    open_price = prices[0]
    high_price = max(prices)
    low_price = min(prices)
    close_price = prices[-1]

    # Sum volume across all snapshots in this candle
    total_volume = sum(s.volume or 0 for s in snapshots)

    return {
        "timestamp": int(start_time.timestamp()),  # Unix timestamp in seconds
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "volume": total_volume,
    }


@router.get("/markets/{market_id}/orderbook")
async def get_orderbook(
    market_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Get latest orderbook snapshot for a market.

    Returns bids/asks with cumulative depth for visualization.
    """
    # Get the latest orderbook snapshot for this market
    result = await session.execute(
        select(OrderbookSnapshot)
        .where(OrderbookSnapshot.market_id == market_id)
        .order_by(OrderbookSnapshot.timestamp.desc())
        .limit(1)
    )
    snapshot = result.scalar_one_or_none()

    if not snapshot:
        raise HTTPException(status_code=404, detail="No orderbook data available")

    # Parse bids and asks from JSON
    bids_raw = snapshot.bids_json or []
    asks_raw = snapshot.asks_json or []

    # Calculate cumulative depth
    bids_cumulative = []
    cumulative_bid_size = 0
    for bid in sorted(bids_raw, key=lambda x: x.get("price", 0), reverse=True):
        price = bid.get("price", 0)
        size = bid.get("size", 0)
        cumulative_bid_size += size
        bids_cumulative.append({
            "price": price,
            "size": size,
            "cumulative": cumulative_bid_size,
        })

    asks_cumulative = []
    cumulative_ask_size = 0
    for ask in sorted(asks_raw, key=lambda x: x.get("price", 0)):
        price = ask.get("price", 0)
        size = ask.get("size", 0)
        cumulative_ask_size += size
        asks_cumulative.append({
            "price": price,
            "size": size,
            "cumulative": cumulative_ask_size,
        })

    return {
        "market_id": market_id,
        "timestamp": snapshot.timestamp.isoformat(),
        "best_bid": snapshot.best_bid,
        "best_ask": snapshot.best_ask,
        "spread": snapshot.bid_ask_spread,
        "obi": snapshot.obi_level1,  # Order Book Imbalance
        "bids": bids_cumulative[:10],  # Top 10 levels
        "asks": asks_cumulative[:10],  # Top 10 levels
        "bid_depth_total": snapshot.bid_depth_total,
        "ask_depth_total": snapshot.ask_depth_total,
    }
