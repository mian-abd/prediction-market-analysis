"""Market browsing and detail endpoints."""

from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi_cache.decorator import cache
from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_session
from db.models import Market, Platform, PriceSnapshot, OrderbookSnapshot, CrossPlatformMatch
from data_pipeline.category_normalizer import normalize_category

router = APIRouter(tags=["markets"])


@router.get("/markets")
async def list_markets(
    platform: str | None = None,
    category: str | None = None,
    search: str | None = None,
    is_active: bool = True,
    is_resolved: bool | None = None,
    sort_by: str = "volume_24h",
    sort_dir: str = "desc",
    exclude_combos: bool = True,
    price_min: float | None = Query(default=None, ge=0, le=1),
    price_max: float | None = Query(default=None, ge=0, le=1),
    limit: int = Query(default=50, le=500),
    offset: int = 0,
    session: AsyncSession = Depends(get_session),
):
    """List markets with filters. Active markets exclude dead (price 0/1) and expired by default."""
    query = select(Market)

    if is_active:
        query = query.where(Market.is_active == True)  # noqa
        # Exclude effectively-resolved markets (price stuck at 0 or 1)
        query = query.where(Market.price_yes > 0.01, Market.price_yes < 0.99)
        # Exclude expired markets (past end_date)
        query = query.where(
            or_(Market.end_date == None, Market.end_date >= func.now())  # noqa
        )

    if is_resolved is not None:
        query = query.where(Market.is_resolved == is_resolved)

    # Exclude Kalshi combo/parlay markets (contain ",yes " or ",no " patterns)
    if exclude_combos:
        query = query.where(
            ~Market.question.like('%,yes %'),
            ~Market.question.like('%,no %'),
        )

    p_obj = None
    if platform:
        platform_result = await session.execute(
            select(Platform).where(Platform.name == platform)
        )
        p_obj = platform_result.scalar_one_or_none()
        if p_obj:
            query = query.where(Market.platform_id == p_obj.id)

    if category:
        # Filter on normalized_category (pre-computed), fallback to raw category
        query = query.where(
            or_(
                Market.normalized_category == category,
                Market.category == category,
            )
        )

    if search:
        query = query.where(
            or_(
                Market.question.ilike(f"%{search}%"),
                Market.description.ilike(f"%{search}%"),
            )
        )

    if price_min is not None:
        query = query.where(Market.price_yes >= price_min)
    if price_max is not None:
        query = query.where(Market.price_yes <= price_max)

    # Sorting with direction support
    valid_sorts = {
        "volume_24h": Market.volume_24h,
        "volume_total": Market.volume_total,
        "price_yes": Market.price_yes,
        "question": Market.question,
        "end_date": Market.end_date,
        "liquidity": Market.liquidity,
        "updated_at": Market.updated_at,
    }
    sort_col = valid_sorts.get(sort_by, Market.volume_24h)
    if sort_dir == "asc":
        query = query.order_by(sort_col.asc().nullslast()).offset(offset).limit(limit)
    else:
        query = query.order_by(sort_col.desc().nullslast()).offset(offset).limit(limit)

    result = await session.execute(query)
    markets = result.scalars().all()

    # Build count query with same filters
    count_query = select(func.count(Market.id))
    if is_active:
        count_query = count_query.where(Market.is_active == True)  # noqa
        count_query = count_query.where(Market.price_yes > 0.01, Market.price_yes < 0.99)
        count_query = count_query.where(
            or_(Market.end_date == None, Market.end_date >= func.now())  # noqa
        )
    if is_resolved is not None:
        count_query = count_query.where(Market.is_resolved == is_resolved)
    if exclude_combos:
        count_query = count_query.where(
            ~Market.question.like('%,yes %'),
            ~Market.question.like('%,no %'),
        )
    if p_obj:
        count_query = count_query.where(Market.platform_id == p_obj.id)
    if category:
        count_query = count_query.where(
            or_(
                Market.normalized_category == category,
                Market.category == category,
            )
        )
    if search:
        count_query = count_query.where(
            or_(Market.question.ilike(f"%{search}%"), Market.description.ilike(f"%{search}%"))
        )
    if price_min is not None:
        count_query = count_query.where(Market.price_yes >= price_min)
    if price_max is not None:
        count_query = count_query.where(Market.price_yes <= price_max)
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
                "category": m.normalized_category or normalize_category(m.category, m.question),
                "price_yes": m.price_yes,
                "price_no": m.price_no,
                "volume_24h": m.volume_24h,
                "volume_total": m.volume_total,
                "liquidity": m.liquidity,
                "end_date": m.end_date.isoformat() if m.end_date else None,
                "is_neg_risk": m.is_neg_risk,
                "is_resolved": m.is_resolved,
                "is_active": m.is_active,
                "updated_at": m.updated_at.isoformat() if m.updated_at else None,
                "last_fetched_at": m.last_fetched_at.isoformat() if m.last_fetched_at else None,
            }
            for m in markets
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/markets/categories")
@cache(expire=300)  # Cache for 5 minutes
async def list_categories(session: AsyncSession = Depends(get_session)):
    """List all market categories with counts (normalized)."""
    # Use pre-computed normalized_category for fast grouping (10 buckets vs 750+)
    result = await session.execute(
        select(Market.normalized_category, func.count(Market.id))
        .where(Market.is_active == True)  # noqa
        .where(Market.normalized_category != None)  # noqa
        .group_by(Market.normalized_category)
    )
    counts: dict[str, int] = {}
    for cat, count in result.all():
        counts[cat or "other"] = counts.get(cat or "other", 0) + count

    # Fallback: count markets without normalized_category
    fallback_result = await session.execute(
        select(Market.category, func.count(Market.id))
        .where(Market.is_active == True)  # noqa
        .where(Market.normalized_category == None)  # noqa
        .group_by(Market.category)
    )
    for raw_cat, count in fallback_result.all():
        normalized = normalize_category(raw_cat)
        counts[normalized] = counts.get(normalized, 0) + count

    return [
        {"category": cat, "count": cnt}
        for cat, cnt in sorted(counts.items(), key=lambda x: -x[1])
    ]


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

    # Fetch recent news (async, non-blocking)
    news_articles = []
    try:
        from data_pipeline.collectors.gdelt_news import get_market_news as fetch_news
        news_articles = await fetch_news(market.question, market.category)
        news_articles = news_articles[:3]  # Top 3 only
    except Exception as e:
        # Don't fail the whole request if news fetch fails
        print(f"News fetch failed: {e}")

    return {
        "id": market.id,
        "platform": platform.name if platform else "unknown",
        "external_id": market.external_id,
        "question": market.question,
        "description": market.description,
        "category": market.normalized_category or normalize_category(market.category, market.question),
        "price_yes": market.price_yes,
        "price_no": market.price_no,
        "volume_24h": market.volume_24h,
        "volume_total": market.volume_total,
        "liquidity": market.liquidity,
        "end_date": market.end_date.isoformat() if market.end_date else None,
        "is_neg_risk": market.is_neg_risk,
        "token_id_yes": market.token_id_yes,
        "token_id_no": market.token_id_no,
        "last_fetched_at": market.last_fetched_at.isoformat() if market.last_fetched_at else None,
        "updated_at": market.updated_at.isoformat() if market.updated_at else None,
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
        "news": news_articles,  # NEW: Include news context
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
        # Fall back to current market price as a single data point
        if market.price_yes is not None:
            now_ts = int(datetime.utcnow().timestamp())
            return {
                "data": [{
                    "timestamp": now_ts,
                    "open": market.price_yes,
                    "high": market.price_yes,
                    "low": market.price_yes,
                    "close": market.price_yes,
                    "volume": 0,
                }],
                "limited_data": True,
            }
        return {"data": [], "limited_data": True}

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

    return {"data": candles[-limit:], "limited_data": len(candles) < 5}


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


@router.get("/markets/{market_id}/news")
async def get_market_news(
    market_id: int,
    max_articles: int = Query(default=5, le=10),
    session: AsyncSession = Depends(get_session),
):
    """Get relevant news articles for a market via GDELT.

    Returns recent news context to help traders understand price moves.
    """
    from data_pipeline.collectors.gdelt_news import get_market_news as fetch_news

    # Get market to extract question and category
    market = await session.get(Market, market_id)
    if not market:
        raise HTTPException(status_code=404, detail="Market not found")

    # Fetch news from GDELT
    articles = await fetch_news(
        question=market.question,
        category=market.normalized_category or market.category,
    )

    # Return up to max_articles
    return {
        "market_id": market_id,
        "articles": articles[:max_articles],
        "count": len(articles),
    }
