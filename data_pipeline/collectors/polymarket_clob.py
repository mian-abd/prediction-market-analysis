"""Polymarket CLOB API client - orderbooks, prices, trades.
No authentication required for read-only operations."""

import httpx
import logging
from datetime import datetime

from config.settings import settings

logger = logging.getLogger(__name__)

BASE_URL = settings.polymarket_clob_url


async def fetch_price(token_id: str, side: str = "buy") -> dict | None:
    """Fetch current price for a token. Returns {"price": "0.55"}."""
    params = {"token_id": token_id, "side": side}
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.get(f"{BASE_URL}/price", params=params)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            logger.warning(f"Price fetch failed for {token_id}: {e}")
            return None


async def fetch_prices_batch(params_list: list[dict]) -> list[dict]:
    """Batch price fetch. Each param: {"token_id": "...", "side": "buy"}."""
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(f"{BASE_URL}/prices", json=params_list)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            logger.warning(f"Batch price fetch failed: {e}")
            return []


async def fetch_orderbook(token_id: str) -> dict | None:
    """Fetch full orderbook for a token.
    Returns: {"bids": [{"price": "0.55", "size": "100"}, ...],
              "asks": [{"price": "0.57", "size": "80"}, ...]}
    """
    params = {"token_id": token_id}
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.get(f"{BASE_URL}/book", params=params)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.debug(f"Orderbook 404 for {token_id} (likely delisted)")
            else:
                logger.warning(f"Orderbook fetch failed for {token_id}: {e}")
            return None
        except httpx.HTTPError as e:
            logger.warning(f"Orderbook fetch failed for {token_id}: {e}")
            return None


async def fetch_midpoint(token_id: str) -> float | None:
    """Fetch midpoint price for a token."""
    params = {"token_id": token_id}
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.get(f"{BASE_URL}/midpoint", params=params)
            resp.raise_for_status()
            data = resp.json()
            return float(data.get("mid", 0))
        except httpx.HTTPError as e:
            logger.warning(f"Midpoint fetch failed for {token_id}: {e}")
            return None


async def fetch_price_history(
    token_id: str,
    interval: str = "max",
    fidelity: int = 60,
) -> list[dict]:
    """Fetch price history timeseries.
    interval: '1m', '1w', '1d', '6h', '1h', 'max'
    fidelity: MINUTES between points (60=1hr, 1440=1day)
    Returns: [{"t": 1700000000, "p": "0.55"}, ...]
    """
    params = {
        "market": token_id,  # FIXED: API expects 'market' parameter, not 'token_id'
        "interval": interval,
        "fidelity": fidelity,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.get(f"{BASE_URL}/prices-history", params=params)
            resp.raise_for_status()
            data = resp.json()
            return data.get("history", [])
        except httpx.HTTPError as e:
            logger.warning(f"Price history fetch failed for {token_id}: {e}")
            return []


def parse_orderbook(raw: dict) -> dict:
    """Parse CLOB orderbook into structured format with computed features."""
    bids = []
    asks = []

    for entry in raw.get("bids", []):
        bids.append({
            "price": float(entry.get("price", 0)),
            "size": float(entry.get("size", 0)),
        })
    for entry in raw.get("asks", []):
        asks.append({
            "price": float(entry.get("price", 0)),
            "size": float(entry.get("size", 0)),
        })

    # Sort: bids descending by price, asks ascending
    bids.sort(key=lambda x: x["price"], reverse=True)
    asks.sort(key=lambda x: x["price"])

    # Compute features
    best_bid = bids[0]["price"] if bids else 0.0
    best_ask = asks[0]["price"] if asks else 1.0
    spread = best_ask - best_bid

    bid_depth = sum(b["size"] for b in bids[:5])
    ask_depth = sum(a["size"] for a in asks[:5])

    # Order Book Imbalance (Level 1)
    bid1_qty = bids[0]["size"] if bids else 0.0
    ask1_qty = asks[0]["size"] if asks else 0.0
    denom = bid1_qty + ask1_qty
    obi_level1 = (bid1_qty - ask1_qty) / denom if denom > 0 else 0.0

    # Weighted OBI (top 5 levels, weighted by inverse distance from mid)
    obi_weighted = 0.0
    if bids and asks:
        mid = (best_bid + best_ask) / 2
        total_weight = 0.0
        for i, (b, a) in enumerate(zip(bids[:5], asks[:5])):
            weight = 1.0 / (i + 1)
            obi_weighted += weight * (b["size"] - a["size"])
            total_weight += weight * (b["size"] + a["size"])
        if total_weight > 0:
            obi_weighted /= total_weight

    depth_ratio = bid_depth / ask_depth if ask_depth > 0 else 1.0

    return {
        "bids": bids[:10],
        "asks": asks[:10],
        "best_bid": best_bid,
        "best_ask": best_ask,
        "bid_ask_spread": spread,
        "bid_depth_total": bid_depth,
        "ask_depth_total": ask_depth,
        "obi_level1": obi_level1,
        "obi_weighted": obi_weighted,
        "depth_ratio": depth_ratio,
    }
