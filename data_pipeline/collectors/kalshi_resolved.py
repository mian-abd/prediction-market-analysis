"""Kalshi resolved markets collector - backfill historical outcomes for ML training."""

import httpx
import logging
from datetime import datetime

from config.settings import settings

logger = logging.getLogger(__name__)

BASE_URL = settings.kalshi_api_url


async def fetch_resolved_markets(
    limit: int = 200,
    cursor: str | None = None,
) -> tuple[list[dict], str | None]:
    """Fetch settled (resolved) markets from Kalshi API.

    Returns:
        (markets, next_cursor) tuple
    """
    params = {
        "status": "settled",  # Only finalized/resolved markets
        "limit": limit,
    }
    if cursor:
        params["cursor"] = cursor

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{BASE_URL}/markets", params=params)
        resp.raise_for_status()
        data = resp.json()

    markets = data.get("markets", [])
    next_cursor = data.get("cursor")

    return markets, next_cursor


async def fetch_all_resolved_markets(max_markets: int = 1000) -> list[dict]:
    """Paginate through all settled Kalshi markets.

    Args:
        max_markets: Maximum number to fetch (default 1000 for training)

    Returns:
        List of resolved market dicts with outcomes
    """
    all_markets = []
    cursor = None

    logger.info(f"Fetching up to {max_markets} settled Kalshi markets...")

    async with httpx.AsyncClient(timeout=30) as client:
        while len(all_markets) < max_markets:
            params = {
                "status": "settled",
                "limit": min(200, max_markets - len(all_markets)),
            }
            if cursor:
                params["cursor"] = cursor

            resp = await client.get(f"{BASE_URL}/markets", params=params)
            resp.raise_for_status()
            data = resp.json()

            batch = data.get("markets", [])
            if not batch:
                logger.info(f"No more settled markets (fetched {len(all_markets)} total)")
                break

            all_markets.extend(batch)
            cursor = data.get("cursor")

            logger.info(f"Fetched {len(all_markets)} settled Kalshi markets so far...")

            if not cursor:  # No more pages
                break

    logger.info(f"Total settled Kalshi markets fetched: {len(all_markets)}")
    return all_markets


def parse_resolved_market(raw: dict) -> dict:
    """Parse resolved Kalshi market.

    Returns dict with outcome for ML training.
    Note: price_yes from parse_kalshi_market uses last_price (pre-settlement
    last traded price in cents). For settled markets with 0 volume, this will
    be 0.0 — these markets are low-signal for ML training.
    """
    from data_pipeline.collectors.kalshi_markets import parse_kalshi_market

    # Use existing parser for base fields
    market = parse_kalshi_market(raw)

    # Add resolution-specific fields
    market["is_resolved"] = True
    resolved_str = raw.get("close_time") or raw.get("expiration_time")
    if resolved_str:
        try:
            market["resolved_at"] = datetime.fromisoformat(resolved_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            market["resolved_at"] = None
    else:
        market["resolved_at"] = None

    # Extract outcome - Kalshi provides clear result field
    result = raw.get("result")  # "yes" or "no"

    if result:
        market["resolution_outcome"] = result.upper()  # "YES" or "NO"
        market["resolution_value"] = 1.0 if result.lower() == "yes" else 0.0
    else:
        # No clear outcome - skip this market
        market["resolution_value"] = None

    return market
