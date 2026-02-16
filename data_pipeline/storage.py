"""Data storage helpers - bulk upsert markets, snapshots, orderbooks into DB."""

import logging
from datetime import datetime, timedelta
from sqlalchemy import select, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import (
    Platform, Market, PriceSnapshot, OrderbookSnapshot,
    ArbitrageOpportunity, NewsEvent, SystemMetric, TraderActivity,
)
from data_pipeline.category_normalizer import normalize_category

logger = logging.getLogger(__name__)


def _strip_timezone(dt):
    """Strip timezone from datetime for PostgreSQL compatibility with timezone-naive columns."""
    if dt and hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


async def ensure_platforms(session: AsyncSession) -> dict[str, int]:
    """Ensure polymarket and kalshi platform rows exist. Returns {name: id}."""
    platforms = {}
    for name, url in [
        ("polymarket", "https://polymarket.com"),
        ("kalshi", "https://kalshi.com"),
    ]:
        result = await session.execute(
            select(Platform).where(Platform.name == name)
        )
        platform = result.scalar_one_or_none()
        if not platform:
            platform = Platform(name=name, base_url=url, is_active=True)
            session.add(platform)
            await session.flush()
        platforms[name] = platform.id
    await session.commit()
    return platforms


async def upsert_markets(
    session: AsyncSession,
    parsed_markets: list[dict],
    platform_id: int,
) -> int:
    """Upsert markets into the database. Returns count of upserted markets.

    Uses no_autoflush blocks and per-market error handling so one bad record
    doesn't crash the entire batch (critical for PostgreSQL strict type checks).
    """
    count = 0
    errors = 0
    for m in parsed_markets:
        try:
            # Use SAVEPOINT so one failure doesn't revert all previous markets
            async with session.begin_nested():
                result = await session.execute(
                    select(Market).where(
                        Market.platform_id == platform_id,
                        Market.external_id == m["external_id"],
                    )
                )
                existing = result.scalar_one_or_none()

                if existing:
                    # Update existing market
                    existing.question = m["question"]
                    existing.description = m.get("description")
                    existing.category = m.get("category")
                    existing.normalized_category = normalize_category(
                        m.get("category"), m.get("question", ""), m.get("description", ""),
                    )
                    existing.price_yes = m.get("price_yes")
                    existing.price_no = m.get("price_no")
                    existing.volume_24h = m.get("volume_24h", 0)
                    existing.volume_total = m.get("volume_total", 0)
                    existing.liquidity = m.get("liquidity", 0)
                    existing.is_active = m.get("is_active", True)
                    existing.is_resolved = m.get("is_resolved", False)
                    existing.end_date = _strip_timezone(m.get("end_date"))
                    existing.is_neg_risk = m.get("is_neg_risk", False)
                    existing.last_fetched_at = datetime.utcnow()
                    existing.updated_at = datetime.utcnow()
                    if m.get("token_id_yes"):
                        existing.token_id_yes = m["token_id_yes"]
                    if m.get("token_id_no"):
                        existing.token_id_no = m["token_id_no"]
                    if m.get("condition_id"):
                        existing.condition_id = m["condition_id"]
                    # Resolution fields
                    if m.get("resolution_value") is not None:
                        existing.resolution_value = m["resolution_value"]
                    if m.get("resolution_outcome"):
                        existing.resolution_outcome = m["resolution_outcome"]
                    if m.get("resolved_at"):
                        existing.resolved_at = _strip_timezone(m["resolved_at"])
                else:
                    # Insert new market
                    market = Market(
                        platform_id=platform_id,
                        external_id=m["external_id"],
                        condition_id=m.get("condition_id"),
                        token_id_yes=m.get("token_id_yes"),
                        token_id_no=m.get("token_id_no"),
                        question=m["question"],
                        description=m.get("description"),
                        category=m.get("category"),
                        normalized_category=normalize_category(
                            m.get("category"), m.get("question", ""), m.get("description", ""),
                        ),
                        slug=m.get("slug"),
                        price_yes=m.get("price_yes"),
                        price_no=m.get("price_no"),
                        volume_24h=m.get("volume_24h", 0),
                        volume_total=m.get("volume_total", 0),
                        liquidity=m.get("liquidity", 0),
                        is_active=m.get("is_active", True),
                        is_resolved=m.get("is_resolved", False),
                        end_date=_strip_timezone(m.get("end_date")),
                        is_neg_risk=m.get("is_neg_risk", False),
                        resolution_value=m.get("resolution_value"),
                        resolution_outcome=m.get("resolution_outcome"),
                        resolved_at=_strip_timezone(m.get("resolved_at")),
                        last_fetched_at=datetime.utcnow(),
                    )
                    session.add(market)
            count += 1
        except Exception as e:
            errors += 1
            if errors <= 3:  # Log first few, avoid spam
                logger.warning(f"Skipping market {m.get('external_id', '?')[:20]}: {e}")

    await session.commit()
    if errors:
        logger.warning(f"Upsert complete: {count} ok, {errors} skipped")
    return count


async def insert_price_snapshots(
    session: AsyncSession,
    market_id: int,
    price_yes: float,
    price_no: float,
    volume: float = 0,
):
    """Insert a price snapshot for a market."""
    now = datetime.utcnow()
    midpoint = (price_yes + (1 - price_no)) / 2 if price_no else price_yes
    spread = abs(price_yes + price_no - 1.0) if price_no else 0.0

    snapshot = PriceSnapshot(
        market_id=market_id,
        timestamp=now,
        price_yes=price_yes,
        price_no=price_no,
        midpoint=midpoint,
        spread=spread,
        volume=volume,
    )
    session.add(snapshot)
    await session.commit()


async def insert_orderbook_snapshot(
    session: AsyncSession,
    market_id: int,
    side: str,
    parsed_ob: dict,
):
    """Insert an orderbook snapshot with pre-computed features."""
    now = datetime.utcnow()
    snapshot = OrderbookSnapshot(
        market_id=market_id,
        side=side,
        timestamp=now,
        best_bid=parsed_ob["best_bid"],
        best_ask=parsed_ob["best_ask"],
        bid_ask_spread=parsed_ob["bid_ask_spread"],
        bid_depth_total=parsed_ob["bid_depth_total"],
        ask_depth_total=parsed_ob["ask_depth_total"],
        obi_level1=parsed_ob["obi_level1"],
        obi_weighted=parsed_ob["obi_weighted"],
        depth_ratio=parsed_ob["depth_ratio"],
        bids_json=parsed_ob["bids"],
        asks_json=parsed_ob["asks"],
    )
    session.add(snapshot)
    await session.commit()


async def cleanup_old_data(session: AsyncSession, days: int = 3) -> dict[str, int]:
    """Delete old time-series data to prevent disk from filling up.

    Keeps the last `days` worth of data for high-volume tables.
    Returns count of deleted rows per table.
    """
    cutoff = datetime.utcnow() - timedelta(days=days)
    deleted = {}

    # Price snapshots — biggest offender (~2000 rows per 20s cycle)
    result = await session.execute(
        delete(PriceSnapshot).where(PriceSnapshot.timestamp < cutoff)
    )
    deleted["price_snapshots"] = result.rowcount

    # Orderbook snapshots (JSON blobs eat space fast)
    result = await session.execute(
        delete(OrderbookSnapshot).where(OrderbookSnapshot.timestamp < cutoff)
    )
    deleted["orderbook_snapshots"] = result.rowcount

    # Expired arbitrage opportunities older than cutoff
    result = await session.execute(
        delete(ArbitrageOpportunity).where(
            ArbitrageOpportunity.detected_at < cutoff,
            ArbitrageOpportunity.expired_at != None,  # noqa: E711
        )
    )
    deleted["arbitrage_opportunities"] = result.rowcount

    # Old news events
    result = await session.execute(
        delete(NewsEvent).where(NewsEvent.fetched_at < cutoff)
    )
    deleted["news_events"] = result.rowcount

    # Old system metrics
    result = await session.execute(
        delete(SystemMetric).where(SystemMetric.timestamp < cutoff)
    )
    deleted["system_metrics"] = result.rowcount

    # Old trader activities
    result = await session.execute(
        delete(TraderActivity).where(TraderActivity.created_at < cutoff)
    )
    deleted["trader_activities"] = result.rowcount

    await session.commit()
    return deleted


async def get_active_markets(
    session: AsyncSession,
    platform_name: str | None = None,
    limit: int = 500,
) -> list[Market]:
    """Get active markets from DB, optionally filtered by platform."""
    query = select(Market).where(Market.is_active == True).limit(limit)  # noqa: E712
    if platform_name:
        platform_result = await session.execute(
            select(Platform).where(Platform.name == platform_name)
        )
        platform = platform_result.scalar_one_or_none()
        if platform:
            query = query.where(Market.platform_id == platform.id)

    query = query.order_by(Market.volume_24h.desc())
    result = await session.execute(query)
    return list(result.scalars().all())
