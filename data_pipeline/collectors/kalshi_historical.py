"""Kalshi authenticated API client for historical data.

Supports:
- RSA private key signing for authentication
- Historical candlestick data for backtesting & feature engineering
- Historical fills and orders
- Fee schedule API

Kalshi separates live and historical tiers with a moving cutoff date.
Historical endpoints require authentication.
"""

import base64
import logging
import time
from datetime import datetime, timedelta, timezone

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from config.settings import settings

logger = logging.getLogger(__name__)

BASE_URL = settings.kalshi_api_url
ACCESS_KEY = getattr(settings, 'kalshi_access_key_id', '') or ''
PRIVATE_KEY_PEM = getattr(settings, 'kalshi_private_key', '') or ''

RATE_LIMIT_DELAY = 0.5


def _load_private_key():
    """Load RSA private key from PEM string."""
    if not PRIVATE_KEY_PEM:
        return None
    pem_bytes = PRIVATE_KEY_PEM.replace("\\n", "\n").encode()
    try:
        return serialization.load_pem_private_key(pem_bytes, password=None)
    except Exception as e:
        logger.error(f"Failed to load Kalshi private key: {e}")
        return None


def _sign_request(method: str, path: str, timestamp_ms: int) -> str:
    """Sign a Kalshi API request using RSA-SHA256.

    Signature = base64(RSA_SIGN(timestamp_ms + method + path))
    """
    private_key = _load_private_key()
    if not private_key:
        return ""
    message = f"{timestamp_ms}{method}{path}".encode()
    signature = private_key.sign(
        message,
        padding.PKCS1v15(),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode()


def _auth_headers(method: str, path: str) -> dict:
    """Generate authentication headers for Kalshi API."""
    if not ACCESS_KEY or not PRIVATE_KEY_PEM:
        return {}
    timestamp_ms = int(time.time() * 1000)
    signature = _sign_request(method.upper(), path, timestamp_ms)
    return {
        "KALSHI-ACCESS-KEY": ACCESS_KEY,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
        "Content-Type": "application/json",
    }


async def fetch_candlesticks(
    ticker: str,
    start_ts: int | None = None,
    end_ts: int | None = None,
    period_interval: int = 60,
) -> list[dict]:
    """Fetch OHLC candlestick data for a market.

    GET /markets/{ticker}/candlesticks
    Args:
        ticker: Market ticker (e.g., "KXBTC-26FEB28-T100000")
        start_ts: Start timestamp (unix seconds). Default: 7 days ago.
        end_ts: End timestamp (unix seconds). Default: now.
        period_interval: Candle interval in minutes (1, 5, 15, 60, 1440)

    Returns list of { ts, open, high, low, close, volume } dicts.
    """
    if not ACCESS_KEY:
        logger.warning("Kalshi API key not configured, skipping candlestick fetch")
        return []

    now = int(time.time())
    if start_ts is None:
        start_ts = now - 7 * 86400
    if end_ts is None:
        end_ts = now

    path = f"/trade-api/v2/markets/{ticker}/candlesticks"
    params = {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "period_interval": period_interval,
    }

    headers = _auth_headers("GET", path)

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{BASE_URL}/markets/{ticker}/candlesticks",
                params=params,
                headers=headers,
            )
            if resp.status_code == 401:
                logger.error("Kalshi auth failed — check access key and private key")
                return []
            if resp.status_code == 404:
                logger.debug(f"No candlestick data for {ticker}")
                return []
            resp.raise_for_status()
            data = resp.json()
            candles = data.get("candlesticks", [])
            return candles
    except Exception as e:
        logger.error(f"Kalshi candlestick fetch failed for {ticker}: {e}")
        return []


async def fetch_historical_fills(
    ticker: str | None = None,
    limit: int = 100,
    cursor: str | None = None,
    min_ts: int | None = None,
    max_ts: int | None = None,
) -> dict:
    """Fetch historical fill data.

    GET /trade-api/v2/historical/fills
    Returns { fills: [...], cursor: "..." }
    """
    if not ACCESS_KEY:
        return {"fills": [], "cursor": None}

    path = "/trade-api/v2/historical/fills"
    params = {"limit": limit}
    if ticker:
        params["ticker"] = ticker
    if cursor:
        params["cursor"] = cursor
    if min_ts:
        params["min_ts"] = min_ts
    if max_ts:
        params["max_ts"] = max_ts

    headers = _auth_headers("GET", path)

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{BASE_URL.replace('/trade-api/v2', '')}/trade-api/v2/historical/fills",
                params=params,
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.error(f"Kalshi historical fills fetch failed: {e}")
        return {"fills": [], "cursor": None}


async def fetch_historical_markets(
    limit: int = 200,
    cursor: str | None = None,
    status: str = "settled",
) -> dict:
    """Fetch historical market metadata.

    GET /trade-api/v2/historical/markets
    """
    if not ACCESS_KEY:
        return {"markets": [], "cursor": None}

    path = "/trade-api/v2/historical/markets"
    params = {"limit": limit, "status": status}
    if cursor:
        params["cursor"] = cursor

    headers = _auth_headers("GET", path)

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{BASE_URL.replace('/trade-api/v2', '')}/trade-api/v2/historical/markets",
                params=params,
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.error(f"Kalshi historical markets fetch failed: {e}")
        return {"markets": [], "cursor": None}
