"""Kalshi REST API client - markets, orderbooks, trades.
Public endpoints don't require authentication."""

import httpx
import logging
from datetime import datetime

from config.settings import settings
from data_pipeline.category_normalizer import normalize_category

logger = logging.getLogger(__name__)

BASE_URL = settings.kalshi_api_url


async def fetch_markets(
    status: str = "open",
    limit: int = 200,
    cursor: str | None = None,
    series_ticker: str | None = None,
) -> dict:
    """Fetch markets from Kalshi.
    Returns: {"markets": [...], "cursor": "next_page_cursor"}
    """
    params = {"status": status, "limit": limit}
    if cursor:
        params["cursor"] = cursor
    if series_ticker:
        params["series_ticker"] = series_ticker

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{BASE_URL}/markets", params=params)
        resp.raise_for_status()
        return resp.json()


async def fetch_all_active_markets(max_markets: int = 2000) -> list[dict]:
    """Paginate through all open Kalshi markets."""
    all_markets = []
    cursor = None

    async with httpx.AsyncClient(timeout=30) as client:
        while len(all_markets) < max_markets:
            params = {"status": "open", "limit": 200}
            if cursor:
                params["cursor"] = cursor

            resp = await client.get(f"{BASE_URL}/markets", params=params)
            resp.raise_for_status()
            data = resp.json()

            markets = data.get("markets", [])
            if not markets:
                break

            all_markets.extend(markets)
            cursor = data.get("cursor")
            logger.info(f"Fetched {len(all_markets)} Kalshi markets so far...")

            if not cursor:
                break

    logger.info(f"Total Kalshi markets fetched: {len(all_markets)}")
    return all_markets


async def fetch_market_by_ticker(ticker: str) -> dict | None:
    """Fetch a single market by ticker."""
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{BASE_URL}/markets/{ticker}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json().get("market")


async def fetch_orderbook(ticker: str, depth: int = 10) -> dict | None:
    """Fetch orderbook for a Kalshi market.
    Returns: {"orderbook": {"yes": [[price, qty], ...], "no": [[price, qty], ...]}}
    """
    params = {"depth": depth}
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.get(
                f"{BASE_URL}/markets/{ticker}/orderbook",
                params=params,
            )
            resp.raise_for_status()
            return resp.json().get("orderbook")
        except httpx.HTTPError as e:
            logger.warning(f"Kalshi orderbook fetch failed for {ticker}: {e}")
            return None


async def fetch_trades(
    ticker: str | None = None,
    limit: int = 100,
    cursor: str | None = None,
) -> dict:
    """Fetch recent trades."""
    params = {"limit": limit}
    if ticker:
        params["ticker"] = ticker
    if cursor:
        params["cursor"] = cursor

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{BASE_URL}/markets/trades", params=params)
        resp.raise_for_status()
        return resp.json()


async def fetch_event(event_ticker: str) -> dict | None:
    """Fetch an event (group of related markets)."""
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.get(f"{BASE_URL}/events/{event_ticker}")
            resp.raise_for_status()
            return resp.json().get("event")
        except httpx.HTTPError as e:
            logger.warning(f"Kalshi event fetch failed for {event_ticker}: {e}")
            return None


def _cents_to_prob(raw_val) -> float:
    """Convert Kalshi cent-based price (0-100) to probability (0-1).

    Kalshi API returns prices in cents. response_price_units = 'usd_cent'.
    A value of 88 means $0.88 = 88% probability.
    """
    if raw_val is None:
        return 0.0
    val = float(raw_val)
    # Kalshi prices are always in cents (0-100 range)
    # Divide by 100 unless it's already clearly a probability
    if val > 1.0:
        return val / 100.0
    # Values 0 or 1 could be either 0 cents or 0/1 probability
    # Since Kalshi uses cents, 0 means 0% and 1 means 1 cent = 1%
    return val / 100.0 if val == 1.0 else val


def parse_kalshi_market(raw: dict) -> dict:
    """Normalize a Kalshi market into our unified format."""
    # Kalshi prices are in cents (0-100), convert to 0-1
    # Use None-aware fallback (0 is a valid price, not falsy)
    yes_bid = raw.get("yes_bid")
    last_price = raw.get("last_price")
    yes_price_raw = yes_bid if yes_bid is not None and yes_bid > 0 else (last_price if last_price is not None else 0)
    no_price_raw = raw.get("no_bid") if raw.get("no_bid") else 0

    price_yes = _cents_to_prob(yes_price_raw)
    price_no = _cents_to_prob(no_price_raw)

    # If we only have yes price, infer no price
    if price_yes > 0 and price_no == 0:
        price_no = 1.0 - price_yes

    # Parse end date
    end_date = None
    close_time = raw.get("close_time") or raw.get("expiration_time")
    if close_time:
        try:
            end_date = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            pass

    # Volume
    volume_24h = float(raw.get("volume_24h", 0) or 0)
    volume_total = float(raw.get("volume", 0) or 0)

    # Category from subtitle or category field
    category = normalize_category(
        raw.get("category") or raw.get("sub_title"),
        raw.get("title", ""),
    )

    return {
        "platform": "kalshi",
        "external_id": raw.get("ticker", ""),
        "condition_id": None,
        "token_id_yes": None,
        "token_id_no": None,
        "question": raw.get("title", ""),
        "description": raw.get("subtitle") or raw.get("rules_primary", ""),
        "category": category,
        "slug": raw.get("ticker", "").lower(),
        "price_yes": price_yes,
        "price_no": price_no,
        "volume_24h": volume_24h,
        "volume_total": volume_total,
        "liquidity": float(raw.get("open_interest", 0) or 0),
        "is_active": raw.get("status") in ("open", "active"),
        "is_resolved": raw.get("status") == "settled",
        "resolution_outcome": raw.get("result"),
        "end_date": end_date,
        "is_neg_risk": False,
    }


def parse_kalshi_orderbook(raw: dict) -> dict:
    """Parse Kalshi orderbook into structured format."""
    bids = []
    asks = []

    # Kalshi format: {"yes": [[price, qty], ...], "no": [[price, qty], ...]}
    yes_data = raw.get("yes", [])
    no_data = raw.get("no", [])

    # YES bids (people want to buy YES)
    for entry in yes_data:
        if len(entry) >= 2:
            price = entry[0] / 100.0 if entry[0] > 1 else float(entry[0])
            bids.append({"price": price, "size": float(entry[1])})

    # NO bids become YES asks (buying NO at X = selling YES at 1-X)
    for entry in no_data:
        if len(entry) >= 2:
            price = entry[0] / 100.0 if entry[0] > 1 else float(entry[0])
            asks.append({"price": 1.0 - price, "size": float(entry[1])})

    bids.sort(key=lambda x: x["price"], reverse=True)
    asks.sort(key=lambda x: x["price"])

    best_bid = bids[0]["price"] if bids else 0.0
    best_ask = asks[0]["price"] if asks else 1.0
    spread = best_ask - best_bid

    bid_depth = sum(b["size"] for b in bids[:5])
    ask_depth = sum(a["size"] for a in asks[:5])

    bid1_qty = bids[0]["size"] if bids else 0.0
    ask1_qty = asks[0]["size"] if asks else 0.0
    denom = bid1_qty + ask1_qty
    obi_level1 = (bid1_qty - ask1_qty) / denom if denom > 0 else 0.0

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
        "obi_weighted": obi_level1,  # Simplified for Kalshi
        "depth_ratio": depth_ratio,
    }
