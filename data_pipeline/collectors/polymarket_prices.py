"""Polymarket price history collector.

Fetches historical price data using the CLOB API's prices-history endpoint
for feature engineering (momentum features, volatility).

GET /prices-history?market={condition_id}&interval={1m|1h|6h|1d}&fidelity={1|full}
"""

import logging
from datetime import datetime, timedelta

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from db.models import Market, PriceSnapshot

logger = logging.getLogger(__name__)

CLOB_BASE = settings.polymarket_clob_url


async def fetch_price_history(
    condition_id: str,
    interval: str = "1h",
    fidelity: int = 1,
) -> list[dict]:
    """Fetch price history for a Polymarket market.

    Args:
        condition_id: Market's condition ID
        interval: "1m", "1h", "6h", or "1d"
        fidelity: 1 for simplified data, or higher for more granular

    Returns list of { t: unix_timestamp, p: price } dicts.
    """
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{CLOB_BASE}/prices-history",
                params={
                    "market": condition_id,
                    "interval": interval,
                    "fidelity": fidelity,
                },
            )
            if resp.status_code != 200:
                logger.debug(f"Price history unavailable for {condition_id[:20]}: {resp.status_code}")
                return []
            data = resp.json()
            history = data.get("history", [])
            return history
    except Exception as e:
        logger.error(f"Price history fetch failed for {condition_id[:20]}: {e}")
        return []


async def backfill_price_snapshots(
    session: AsyncSession,
    market: Market,
    hours_back: int = 48,
) -> int:
    """Backfill price snapshots for a single market using prices-history API.

    Creates PriceSnapshot records from the historical data. Useful for
    building momentum features for markets that were added after our
    real-time collection started.

    Returns number of new snapshots created.
    """
    if not market.condition_id:
        return 0

    history = await fetch_price_history(
        market.condition_id,
        interval="1h",
        fidelity=1,
    )

    if not history:
        return 0

    # Filter to requested time range
    cutoff = datetime.utcnow() - timedelta(hours=hours_back)
    created = 0

    for point in history:
        ts = point.get("t", 0)
        price = point.get("p", 0)

        if ts == 0 or price == 0:
            continue

        snapshot_time = datetime.utcfromtimestamp(ts)
        if snapshot_time < cutoff:
            continue

        # Check if snapshot already exists (avoid duplicates)
        existing = await session.execute(
            select(PriceSnapshot.id)
            .where(
                PriceSnapshot.market_id == market.id,
                PriceSnapshot.timestamp == snapshot_time,
            )
            .limit(1)
        )
        if existing.scalar_one_or_none():
            continue

        snapshot = PriceSnapshot(
            market_id=market.id,
            price_yes=float(price),
            price_no=1.0 - float(price),
            volume=0.0,
            timestamp=snapshot_time,
        )
        session.add(snapshot)
        created += 1

    if created > 0:
        await session.commit()
        logger.info(f"Backfilled {created} price snapshots for market {market.id}")

    return created


async def backfill_all_active_markets(
    session: AsyncSession,
    hours_back: int = 48,
    batch_size: int = 20,
) -> int:
    """Backfill price snapshots for active markets that lack recent data.

    Targets markets with fewer than 5 price snapshots in the last 48h.
    """
    result = await session.execute(
        select(Market)
        .where(
            Market.is_active == True,  # noqa: E712
            Market.condition_id.isnot(None),
        )
        .limit(batch_size * 2)
    )
    markets = result.scalars().all()

    total_created = 0
    processed = 0

    for market in markets:
        if processed >= batch_size:
            break

        # Check existing snapshot count
        cutoff = datetime.utcnow() - timedelta(hours=hours_back)
        count_result = await session.execute(
            select(PriceSnapshot.id)
            .where(
                PriceSnapshot.market_id == market.id,
                PriceSnapshot.timestamp >= cutoff,
            )
        )
        existing_count = len(count_result.scalars().all())

        if existing_count >= 5:
            continue

        created = await backfill_price_snapshots(session, market, hours_back)
        total_created += created
        processed += 1

    if total_created > 0:
        logger.info(f"Price backfill: {total_created} snapshots for {processed} markets")

    return total_created
