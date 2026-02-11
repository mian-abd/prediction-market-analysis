"""Polymarket Gamma API client - market metadata, events, categories.
No authentication required. Read-only."""

import json
import httpx
import logging
from datetime import datetime

from config.settings import settings

logger = logging.getLogger(__name__)

BASE_URL = settings.polymarket_gamma_url


async def fetch_active_markets(
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """Fetch active, non-closed markets from Gamma API."""
    params = {
        "active": "true",
        "closed": "false",
        "limit": limit,
        "offset": offset,
        "order": "volume24hr",
        "ascending": "false",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{BASE_URL}/markets", params=params)
        resp.raise_for_status()
        return resp.json()


async def fetch_all_active_markets(max_markets: int = 2000) -> list[dict]:
    """Paginate through all active markets."""
    all_markets = []
    offset = 0
    batch_size = 100

    async with httpx.AsyncClient(timeout=30) as client:
        while offset < max_markets:
            params = {
                "active": "true",
                "closed": "false",
                "limit": batch_size,
                "offset": offset,
                "order": "volume24hr",
                "ascending": "false",
            }
            resp = await client.get(f"{BASE_URL}/markets", params=params)
            resp.raise_for_status()
            batch = resp.json()

            if not batch:
                break

            all_markets.extend(batch)
            offset += batch_size
            logger.info(f"Fetched {len(all_markets)} Polymarket markets so far...")

            if len(batch) < batch_size:
                break

    logger.info(f"Total Polymarket markets fetched: {len(all_markets)}")
    return all_markets


async def fetch_events(
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """Fetch events (groups of related markets)."""
    params = {
        "active": "true",
        "closed": "false",
        "limit": limit,
        "offset": offset,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{BASE_URL}/events", params=params)
        resp.raise_for_status()
        return resp.json()


async def fetch_market_by_id(condition_id: str) -> dict | None:
    """Fetch a single market by condition ID."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{BASE_URL}/markets/{condition_id}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()


def parse_gamma_market(raw: dict) -> dict:
    """Normalize a Gamma API market response into our unified format."""
    # Extract token IDs - API returns JSON-encoded strings like '["id1", "id2"]'
    clob_token_ids = raw.get("clobTokenIds", "")
    if isinstance(clob_token_ids, str):
        try:
            token_ids = json.loads(clob_token_ids)
        except (json.JSONDecodeError, TypeError):
            token_ids = [t.strip() for t in clob_token_ids.split(",") if t.strip()]
    elif isinstance(clob_token_ids, list):
        token_ids = clob_token_ids
    else:
        token_ids = []

    token_id_yes = token_ids[0] if len(token_ids) > 0 else None
    token_id_no = token_ids[1] if len(token_ids) > 1 else None

    # Parse outcomes - also JSON-encoded strings like '["Yes", "No"]'
    outcomes_raw = raw.get("outcomes", "")
    if isinstance(outcomes_raw, str):
        try:
            outcomes = json.loads(outcomes_raw)
        except (json.JSONDecodeError, TypeError):
            outcomes = [o.strip() for o in outcomes_raw.split(",")]
    elif isinstance(outcomes_raw, list):
        outcomes = outcomes_raw
    else:
        outcomes = ["Yes", "No"]

    # Parse prices - JSON-encoded like '["0.0365", "0.9635"]'
    prices_raw = raw.get("outcomePrices", "")
    if isinstance(prices_raw, str):
        try:
            prices = [float(p) for p in json.loads(prices_raw)]
        except (json.JSONDecodeError, TypeError, ValueError):
            prices = []
    elif isinstance(prices_raw, list):
        prices = [float(p) for p in prices_raw]
    else:
        prices = []

    price_yes = prices[0] if len(prices) > 0 else None
    price_no = prices[1] if len(prices) > 1 else None

    # Parse end date
    end_date = None
    end_date_str = raw.get("endDate") or raw.get("end_date_iso")
    if end_date_str:
        try:
            end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            pass

    return {
        "platform": "polymarket",
        "external_id": raw.get("conditionId") or raw.get("id", ""),
        "condition_id": raw.get("conditionId"),
        "token_id_yes": token_id_yes,
        "token_id_no": token_id_no,
        "question": raw.get("question", ""),
        "description": raw.get("description", ""),
        "category": (raw.get("groupItemTitle") or raw.get("category") or "other").lower(),
        "slug": raw.get("slug", ""),
        "price_yes": price_yes,
        "price_no": price_no,
        "volume_24h": float(raw.get("volume24hr", 0) or 0),
        "volume_total": float(raw.get("volume", 0) or 0),
        "liquidity": float(raw.get("liquidity", 0) or 0),
        "is_active": raw.get("active", True),
        "is_resolved": raw.get("closed", False),
        "resolution_outcome": raw.get("resolutionSource"),
        "end_date": end_date,
        "is_neg_risk": raw.get("negRisk", False),
    }
