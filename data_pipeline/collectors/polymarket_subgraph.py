"""Polymarket on-chain data via Goldsky subgraphs.

Polymarket runs on Polygon. ALL trades are publicly visible on-chain.
These subgraphs let us query:

1. PNL Subgraph  — which wallets are most profitable (ranked leaderboard)
2. Activity Subgraph — recent trades with wallet address, market, size, direction
3. Positions Subgraph — current open positions per wallet

Strategy: identify "smart money" wallets (proven profitable, high win rate),
then monitor when they enter a market. When 2+ qualified wallets take the same
side in the same market within a 1-hour window, that is a strong signal.

No API key required — all endpoints are public Goldsky GraphQL APIs.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Goldsky subgraph endpoints (public, no auth)
PNL_SUBGRAPH = (
    "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw"
    "/subgraphs/pnl-subgraph/0.0.14/gn"
)
ACTIVITY_SUBGRAPH = (
    "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw"
    "/subgraphs/activity-subgraph/0.0.4/gn"
)
POSITIONS_SUBGRAPH = (
    "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw"
    "/subgraphs/positions-subgraph/0.0.7/gn"
)

# Wallet quality thresholds
MIN_REALIZED_PNL_USD = 5_000     # Minimum lifetime P&L
MIN_WIN_RATE = 0.55              # Minimum historical win rate
MIN_TOTAL_TRADES = 20           # Need enough trades for statistical significance
MAX_TRACKED_WALLETS = 150       # Cap to keep queries manageable

# Signal generation
MIN_WHALE_AGREEMENT = 2         # Minimum whales on same side to generate signal
WHALE_WINDOW_HOURS = 1          # Time window for whale agreement

# Caching
_wallet_cache: list[dict] = []
_wallet_cache_loaded_at: datetime | None = None
WALLET_CACHE_TTL_HOURS = 24

_HTTP_TIMEOUT = 20.0


async def _graphql_query(endpoint: str, query: str, variables: dict | None = None) -> dict | None:
    """Execute a GraphQL query against a subgraph endpoint with retries."""
    import asyncio

    payload: dict = {"query": query}
    if variables:
        payload["variables"] = variables

    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.post(endpoint, json=payload)
                resp.raise_for_status()
                data = resp.json()
                if "errors" in data:
                    logger.warning(f"GraphQL errors: {data['errors']}")
                    return None
                return data.get("data")
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            logger.warning(f"Subgraph HTTP error {e.response.status_code}: {endpoint}")
            return None
        except (httpx.ConnectError, httpx.ReadTimeout) as e:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            logger.warning(f"Subgraph connection failed ({endpoint}): {e}")
            return None
        except Exception as e:
            logger.warning(f"Subgraph query failed ({endpoint}): {e}")
            return None
    return None


async def get_top_profitable_wallets(
    min_pnl_usd: float = MIN_REALIZED_PNL_USD,
    limit: int = MAX_TRACKED_WALLETS,
) -> list[dict]:
    """Query PNL subgraph for the most profitable wallets.

    Returns list of dicts: {wallet, realized_pnl, num_bets, avg_pnl_per_bet}
    """
    # PNL subgraph stores amounts in USDC with 6 decimal places
    min_pnl_raw = int(min_pnl_usd * 1_000_000)

    query = """
    query TopWallets($minPnl: BigInt!, $limit: Int!) {
      userPositions(
        where: { realizedPnl_gt: $minPnl }
        orderBy: realizedPnl
        orderDirection: desc
        first: $limit
      ) {
        user { id }
        realizedPnl
        numBets
      }
    }
    """
    data = await _graphql_query(
        PNL_SUBGRAPH,
        query,
        {"minPnl": str(min_pnl_raw), "limit": limit},
    )

    if not data or "userPositions" not in data:
        # Try alternative schema field names
        query_alt = """
        query TopWallets($limit: Int!) {
          globalUserPositions(
            orderBy: realizedPnl
            orderDirection: desc
            first: $limit
          ) {
            user { id }
            realizedPnl
            numBets
          }
        }
        """
        data = await _graphql_query(PNL_SUBGRAPH, query_alt, {"limit": limit})

    if not data:
        logger.warning("PNL subgraph returned no data, using empty wallet list")
        return []

    # Parse results — try multiple possible field names
    positions = (
        data.get("userPositions") or
        data.get("globalUserPositions") or
        []
    )

    wallets = []
    for pos in positions:
        try:
            user = pos.get("user") or {}
            wallet_addr = user.get("id") or pos.get("userId") or pos.get("user")
            if not wallet_addr:
                continue

            realized_pnl_raw = int(pos.get("realizedPnl", 0))
            realized_pnl_usd = realized_pnl_raw / 1_000_000  # USDC 6 decimals

            if realized_pnl_usd < min_pnl_usd:
                continue

            num_bets = int(pos.get("numBets", 0))
            avg_pnl = realized_pnl_usd / max(num_bets, 1)

            wallets.append({
                "wallet": wallet_addr.lower(),
                "realized_pnl_usd": round(realized_pnl_usd, 2),
                "num_bets": num_bets,
                "avg_pnl_per_bet": round(avg_pnl, 4),
            })
        except (ValueError, TypeError, KeyError):
            continue

    logger.info(f"Subgraph: {len(wallets)} profitable wallets (min ${min_pnl_usd:,.0f})")
    return wallets


async def get_recent_whale_trades(
    wallet_addresses: list[str],
    since_hours: float = 2.0,
) -> list[dict]:
    """Query Activity subgraph for recent trades by tracked wallets.

    Returns list of dicts:
    {wallet, token_id, side ('buy'/'sell'), price, size, timestamp, condition_id}
    """
    if not wallet_addresses:
        return []

    # Activity subgraph uses Unix timestamps
    since_ts = int((datetime.utcnow() - timedelta(hours=since_hours)).timestamp())
    wallet_list = [w.lower() for w in wallet_addresses[:50]]  # Limit batch size

    query = """
    query WhaleActivity($wallets: [String!]!, $since: Int!) {
      orderFilleds(
        where: {
          maker_in: $wallets
          timestamp_gt: $since
        }
        orderBy: timestamp
        orderDirection: desc
        first: 500
      ) {
        id
        maker
        size
        price
        makerAssetId
        takerAssetId
        timestamp
      }
    }
    """
    data = await _graphql_query(
        ACTIVITY_SUBGRAPH,
        query,
        {"wallets": wallet_list, "since": since_ts},
    )

    # Try alternative schema if first attempt fails
    if not data or not data.get("orderFilleds"):
        query_alt = """
        query WhaleActivity($wallets: [String!]!, $since: Int!) {
          trades(
            where: {
              transactor_in: $wallets
              timestamp_gt: $since
            }
            orderBy: timestamp
            orderDirection: desc
            first: 500
          ) {
            id
            transactor { id }
            price
            size
            outcome
            timestamp
            condition { id }
          }
        }
        """
        data = await _graphql_query(
            ACTIVITY_SUBGRAPH,
            query_alt,
            {"wallets": wallet_list, "since": since_ts},
        )

    if not data:
        logger.debug("Activity subgraph returned no data for whale trades")
        return []

    trades_raw = data.get("orderFilleds") or data.get("trades") or []
    trades = []

    for t in trades_raw:
        try:
            wallet = (
                (t.get("maker") or "").lower() or
                (t.get("transactor") or {}).get("id", "").lower()
            )
            if not wallet:
                continue

            token_id = (
                t.get("makerAssetId") or
                t.get("takerAssetId") or
                ""
            )
            price = float(t.get("price", 0))
            size = float(t.get("size", 0))

            if price <= 0 or size <= 0:
                continue

            # Infer side: buy = acquiring YES token, sell = disposing YES token
            outcome = t.get("outcome", "").lower()
            if outcome:
                side = "buy" if outcome == "yes" else "buy_no"
            else:
                # Heuristic: price < 0.5 more likely NO token, otherwise YES
                side = "buy_yes" if price <= 0.5 else "buy_no"

            ts = int(t.get("timestamp", 0))
            condition_id = (t.get("condition") or {}).get("id") or ""

            trades.append({
                "wallet": wallet,
                "token_id": str(token_id),
                "condition_id": condition_id,
                "side": side,
                "price": round(price, 6),
                "size_usd": round(price * size, 4),
                "timestamp": ts,
            })
        except (ValueError, TypeError, KeyError):
            continue

    logger.debug(f"Subgraph: {len(trades)} whale trades in last {since_hours}h")
    return trades


async def get_cached_smart_wallets() -> list[dict]:
    """Return cached wallet list, refreshing every WALLET_CACHE_TTL_HOURS."""
    global _wallet_cache, _wallet_cache_loaded_at

    now = datetime.utcnow()
    needs_refresh = (
        not _wallet_cache or
        _wallet_cache_loaded_at is None or
        (now - _wallet_cache_loaded_at).total_seconds() > WALLET_CACHE_TTL_HOURS * 3600
    )

    if needs_refresh:
        logger.info("Refreshing smart wallet list from PNL subgraph...")
        fresh = await get_top_profitable_wallets()
        if fresh:
            _wallet_cache = fresh
            _wallet_cache_loaded_at = now
            logger.info(f"Smart wallet cache updated: {len(_wallet_cache)} wallets")
        elif not _wallet_cache:
            logger.warning("Could not load smart wallets from subgraph")

    return _wallet_cache


async def get_whale_market_signals(
    token_to_market: dict[str, tuple[int, str]],
    since_hours: float = WHALE_WINDOW_HOURS,
) -> list[dict]:
    """Detect when multiple smart-money wallets converge on the same market.

    Args:
        token_to_market: mapping from token_id -> (market_id, 'yes' | 'no')
        since_hours: look back window for trade activity

    Returns list of signal dicts:
    {market_id, side, whale_count, avg_price, total_size_usd, wallets, confidence}
    """
    wallets = await get_cached_smart_wallets()
    if len(wallets) < MIN_WHALE_AGREEMENT:
        return []

    wallet_addrs = [w["wallet"] for w in wallets]
    wallet_quality = {w["wallet"]: w for w in wallets}

    recent_trades = await get_recent_whale_trades(wallet_addrs, since_hours=since_hours)
    if not recent_trades:
        return []

    # Group trades by (market_id, side)
    from collections import defaultdict
    groups: dict[tuple[int, str], list[dict]] = defaultdict(list)

    for trade in recent_trades:
        token_id = trade.get("token_id", "")
        if token_id and token_id in token_to_market:
            market_id, market_side = token_to_market[token_id]
            # Reconcile direction: if token is YES and trade is buy → buy_yes
            if market_side == "yes":
                direction = "buy_yes" if trade["side"] in ("buy", "buy_yes") else "buy_no"
            else:
                direction = "buy_no" if trade["side"] in ("buy", "buy_yes") else "buy_yes"

            unique_key = (market_id, direction)
            # Only count unique wallets per group
            existing_wallets = {t["wallet"] for t in groups[unique_key]}
            if trade["wallet"] not in existing_wallets:
                groups[unique_key].append(trade)

    signals = []
    for (market_id, direction), trades in groups.items():
        unique_wallets = list({t["wallet"] for t in trades})
        if len(unique_wallets) < MIN_WHALE_AGREEMENT:
            continue

        # Compute quality-weighted confidence
        total_quality = sum(
            wallet_quality.get(w, {}).get("avg_pnl_per_bet", 1.0)
            for w in unique_wallets
        )
        avg_quality = total_quality / len(unique_wallets)

        # Normalize quality score → 0-1 confidence bonus
        # avg_pnl_per_bet of $50 = 0.2 bonus, $200 = 0.5 bonus
        quality_bonus = min(0.5, avg_quality / 400)

        whale_count_factor = min(1.0, len(unique_wallets) / 5)  # 5+ whales = max factor
        base_confidence = 0.35 + 0.35 * whale_count_factor + quality_bonus

        avg_price = sum(t["price"] for t in trades) / len(trades)
        total_size_usd = sum(t.get("size_usd", 0) for t in trades)

        signals.append({
            "market_id": market_id,
            "direction": direction,
            "whale_count": len(unique_wallets),
            "avg_entry_price": round(avg_price, 4),
            "total_size_usd": round(total_size_usd, 2),
            "wallets": unique_wallets[:10],  # Store top 10
            "confidence": round(min(0.85, base_confidence), 3),
            "window_hours": since_hours,
        })

    logger.info(
        f"Whale signal detection: {len(recent_trades)} trades from "
        f"{len(wallet_addrs)} wallets → {len(signals)} market signals"
    )
    return signals
