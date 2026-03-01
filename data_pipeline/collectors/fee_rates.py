"""Polymarket and Kalshi fee rate collectors.

Polymarket: GET /fee-rate?token_id=TOKEN_ID -> { "fee_rate_bps": N }
Kalshi: GET /series/fee_changes -> scheduled fee changes per series
"""

import logging
from datetime import datetime

import httpx
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from db.models import Market

logger = logging.getLogger(__name__)

CLOB_BASE = settings.polymarket_clob_url


async def fetch_polymarket_fee_rate(token_id: str) -> int:
    """Fetch fee_rate_bps for a Polymarket token. Returns 0 for fee-free markets."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{CLOB_BASE}/fee-rate",
                params={"token_id": token_id},
            )
            if resp.status_code != 200:
                return 0
            data = resp.json()
            return int(data.get("fee_rate_bps", 0))
    except Exception as e:
        logger.debug(f"Fee rate fetch failed for {token_id[:20]}...: {e}")
        return 0


async def update_polymarket_fee_rates(session: AsyncSession, batch_size: int = 50) -> int:
    """Fetch and store fee rates for all Polymarket markets with token IDs.

    Only updates markets that don't already have a fee rate stored
    (to avoid redundant API calls).
    """
    result = await session.execute(
        select(Market)
        .where(
            Market.token_id_yes.isnot(None),
            Market.is_active == True,  # noqa: E712
            Market.taker_fee_bps == None,  # noqa: E711 — only fetch for markets we haven't checked
        )
        .limit(batch_size)
    )
    markets = result.scalars().all()

    if not markets:
        return 0

    updated = 0
    for market in markets:
        fee_bps = await fetch_polymarket_fee_rate(market.token_id_yes)
        market.taker_fee_bps = fee_bps
        market.maker_fee_bps = 0
        if fee_bps > 0:
            # Makers get rebates on fee-enabled markets
            rebate_pct = 20 if fee_bps >= 1000 else 25
            market.maker_fee_bps = -int(fee_bps * rebate_pct / 100)
        updated += 1

    if updated > 0:
        await session.commit()
        fee_free = sum(1 for m in markets if (m.taker_fee_bps or 0) == 0)
        fee_enabled = updated - fee_free
        logger.info(
            f"Fee rates updated: {updated} markets "
            f"({fee_free} fee-free, {fee_enabled} fee-enabled)"
        )

    return updated


async def fetch_kalshi_fee_changes() -> list[dict]:
    """Fetch scheduled fee changes from Kalshi API.

    GET /trade-api/v2/series/fee_changes
    Returns list of { series_ticker, fee_type, fee_multiplier, scheduled_at }
    """
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{settings.kalshi_api_url}/series/fee_changes",
            )
            if resp.status_code != 200:
                logger.warning(f"Kalshi fee_changes returned {resp.status_code}")
                return []
            data = resp.json()
            changes = data.get("fee_changes", [])
            logger.info(f"Kalshi fee changes: {len(changes)} scheduled")
            return changes
    except Exception as e:
        logger.error(f"Kalshi fee_changes fetch failed: {e}")
        return []
