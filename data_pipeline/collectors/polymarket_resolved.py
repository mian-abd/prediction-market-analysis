"""Polymarket resolved markets collector - backfill historical outcomes for ML training."""

import httpx
import logging
from datetime import datetime

from config.settings import settings

logger = logging.getLogger(__name__)

BASE_URL = settings.polymarket_gamma_url


async def fetch_resolved_markets(
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """Fetch closed/resolved markets from Gamma API.

    These are markets that have finalized outcomes - critical for ML training.
    """
    params = {
        "closed": "true",  # Only resolved markets
        "limit": limit,
        "offset": offset,
        "order": "volume24hr",  # Fetch highest volume first (better quality data)
        "ascending": "false",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{BASE_URL}/markets", params=params)
        resp.raise_for_status()
        return resp.json()


async def fetch_all_resolved_markets(max_markets: int = 1000) -> list[dict]:
    """Paginate through resolved markets for ML training data.

    Args:
        max_markets: Maximum number of resolved markets to fetch (default 1000 for training set)

    Returns:
        List of resolved market dicts with outcomes
    """
    all_markets = []
    offset = 0
    batch_size = 100

    logger.info(f"Fetching up to {max_markets} resolved Polymarket markets...")

    async with httpx.AsyncClient(timeout=30) as client:
        while offset < max_markets:
            params = {
                "closed": "true",
                "limit": batch_size,
                "offset": offset,
                "order": "volume24hr",
                "ascending": "false",
            }

            resp = await client.get(f"{BASE_URL}/markets", params=params)
            resp.raise_for_status()
            batch = resp.json()

            if not batch:
                logger.info(f"No more resolved markets found (fetched {len(all_markets)} total)")
                break

            all_markets.extend(batch)
            offset += batch_size

            logger.info(f"Fetched {len(all_markets)} resolved Polymarket markets so far...")

    logger.info(f"Total resolved Polymarket markets fetched: {len(all_markets)}")
    return all_markets


def parse_resolved_market(raw: dict) -> dict:
    """Parse resolved market from Gamma API response.

    Returns dict with outcome information for ML training.
    """
    from data_pipeline.collectors.polymarket_gamma import parse_gamma_market

    # Use existing parser for base fields
    market = parse_gamma_market(raw)

    # Add resolution-specific fields
    market["is_resolved"] = True
    market["resolved_at"] = raw.get("endDate") or raw.get("end_date")

    # Extract outcome from Gamma API
    # Polymarket markets resolve to YES (1.0) or NO (0.0)
    outcome_prices = raw.get("outcomePrices", [])

    # If market is closed, final price = resolution
    if outcome_prices and len(outcome_prices) >= 2:
        # YES outcome is first element (typically)
        yes_outcome = float(outcome_prices[0]) if outcome_prices[0] else None
        market["resolution_value"] = yes_outcome  # 1.0 = YES won, 0.0 = NO won
    else:
        # Fallback: use current price as proxy (less accurate)
        market["resolution_value"] = market.get("price_yes")

    return market
