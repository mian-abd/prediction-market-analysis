"""Collect real trader performance data from Polymarket leaderboard API."""

import httpx
import logging
from datetime import datetime
from typing import List, Dict

logger = logging.getLogger(__name__)

POLYMARKET_LEADERBOARD_URL = "https://data-api.polymarket.com/v1/leaderboard"


async def fetch_polymarket_leaderboard(
    time_period: str = "MONTH",  # Options: "DAY", "WEEK", "MONTH", "ALL"
    limit: int = 50,
    order_by: str = "PNL",  # Options: "PNL", "VOL"
    category: str = "OVERALL",  # Options: OVERALL, POLITICS, SPORTS, etc.
    offset: int = 0,
) -> List[Dict]:
    """
    Fetch top traders from Polymarket leaderboard.

    Returns list of trader objects with:
    - proxyWallet: Wallet address
    - userName: Display name
    - pnl: Profit/Loss in USD
    - vol: Total volume traded
    - rank: Leaderboard position
    - profileImage: Avatar URL
    - xUsername: Twitter handle
    - verifiedBadge: Verification status
    """
    params = {
        "timePeriod": time_period,
        "limit": min(limit, 50),  # API max is 50
        "orderBy": order_by,
        "category": category,
        "offset": offset,
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(POLYMARKET_LEADERBOARD_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            logger.info(f"Fetched {len(data)} traders from Polymarket leaderboard")
            return data
    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch Polymarket leaderboard: {e}")
        return []


async def fetch_trader_details(user_address: str) -> Dict | None:
    """
    Fetch detailed stats for a specific trader.

    Polymarket API endpoint: /users/{address}
    """
    url = f"https://gamma-api.polymarket.com/users/{user_address}"

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as e:
        logger.warning(f"Failed to fetch trader details for {user_address}: {e}")
        return None


async def fetch_trader_positions(user_address: str, limit: int = 50) -> List[Dict]:
    """Fetch current positions for a trader."""
    url = f"https://gamma-api.polymarket.com/users/{user_address}/positions"
    params = {"limit": limit}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as e:
        logger.warning(f"Failed to fetch positions for {user_address}: {e}")
        return []


async def fetch_trader_trades(
    user_address: str,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict]:
    """Fetch trade history for a trader."""
    url = f"https://gamma-api.polymarket.com/users/{user_address}/trades"
    params = {"limit": limit, "offset": offset}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as e:
        logger.warning(f"Failed to fetch trades for {user_address}: {e}")
        return []


def calculate_trader_stats(trader_data: Dict, trades: List[Dict]) -> Dict:
    """
    Calculate comprehensive trader statistics.

    Args:
        trader_data: Raw trader data from API
        trades: List of trader's historical trades

    Returns:
        Dict with calculated stats for TraderProfile model
    """
    total_pnl = float(trader_data.get("pnl", 0) or 0)
    volume = float(trader_data.get("volume", 0) or 0)

    # Calculate win rate from trades
    winning_trades = sum(1 for t in trades if float(t.get("pnl", 0)) > 0)
    total_trades = len([t for t in trades if t.get("pnl") is not None])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

    # Calculate ROI
    # Assume starting capital is volume - pnl (rough estimate)
    invested = volume - total_pnl if volume > 0 else 1000
    roi_pct = (total_pnl / invested * 100) if invested > 0 else 0.0

    # Calculate average trade duration from trades
    durations = []
    for trade in trades:
        if trade.get("created_at") and trade.get("closed_at"):
            try:
                created = datetime.fromisoformat(trade["created_at"].replace("Z", "+00:00"))
                closed = datetime.fromisoformat(trade["closed_at"].replace("Z", "+00:00"))
                duration_hrs = (closed - created).total_seconds() / 3600
                durations.append(duration_hrs)
            except (ValueError, TypeError, KeyError):
                continue

    avg_trade_duration_hrs = sum(durations) / len(durations) if durations else 48.0

    # Calculate max drawdown from PnL history
    pnl_values = [float(t.get("pnl", 0)) for t in trades if t.get("pnl") is not None]
    if pnl_values:
        cumulative_pnl = []
        running_total = 0
        for pnl in pnl_values:
            running_total += pnl
            cumulative_pnl.append(running_total)

        max_drawdown = 0
        peak = cumulative_pnl[0]
        for value in cumulative_pnl:
            if value > peak:
                peak = value
            drawdown = peak - value
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        max_drawdown = -max_drawdown  # Negative value
    else:
        max_drawdown = 0.0

    # Calculate risk score (1-10 scale)
    # Based on: volatility, max drawdown, avg position size
    volatility = calculate_pnl_volatility(pnl_values) if len(pnl_values) > 5 else 0
    risk_score = calculate_risk_score(
        max_drawdown=abs(max_drawdown),
        volatility=volatility,
        avg_position_size=volume / total_trades if total_trades > 0 else 0,
    )

    return {
        "total_pnl": total_pnl,
        "roi_pct": roi_pct,
        "win_rate": win_rate,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "avg_trade_duration_hrs": avg_trade_duration_hrs,
        "risk_score": risk_score,
        "max_drawdown": max_drawdown,
    }


def calculate_pnl_volatility(pnl_values: List[float]) -> float:
    """Calculate standard deviation of PnL values."""
    if len(pnl_values) < 2:
        return 0.0

    mean = sum(pnl_values) / len(pnl_values)
    variance = sum((x - mean) ** 2 for x in pnl_values) / len(pnl_values)
    return variance ** 0.5


def calculate_risk_score(
    max_drawdown: float,
    volatility: float,
    avg_position_size: float,
) -> int:
    """
    Calculate risk score (1-10) based on trader behavior.

    1-3: Low risk (conservative, stable)
    4-6: Medium risk (balanced)
    7-10: High risk (aggressive, volatile)
    """
    score = 5  # Start with medium risk

    # Adjust based on max drawdown
    if max_drawdown > 5000:
        score += 3
    elif max_drawdown > 2000:
        score += 2
    elif max_drawdown > 1000:
        score += 1
    elif max_drawdown < 500:
        score -= 1
    elif max_drawdown < 200:
        score -= 2

    # Adjust based on volatility
    if volatility > 1000:
        score += 2
    elif volatility > 500:
        score += 1
    elif volatility < 100:
        score -= 1

    # Adjust based on position size
    if avg_position_size > 5000:
        score += 1
    elif avg_position_size < 500:
        score -= 1

    # Clamp to 1-10 range
    return max(1, min(10, score))


def generate_trader_bio(trader_data: Dict, stats: Dict) -> str:
    """Generate a bio for a trader based on their stats."""
    win_rate = stats["win_rate"]
    risk = stats["risk_score"]
    total_trades = stats["total_trades"]

    # Determine trading style
    if risk <= 3:
        style = "Conservative trader with focus on capital preservation"
    elif risk <= 6:
        style = "Balanced approach with moderate risk tolerance"
    else:
        style = "Aggressive trader seeking high-reward opportunities"

    # Determine skill level
    if win_rate >= 70:
        skill = "Exceptional track record"
    elif win_rate >= 60:
        skill = "Strong performance"
    elif win_rate >= 50:
        skill = "Consistent results"
    else:
        skill = "Active trader"

    return f"{skill}. {style}. {total_trades}+ trades executed on Polymarket."


def anonymize_address(address: str) -> str:
    """Convert wallet address to readable trader name."""
    # Take last 6 chars of address for uniqueness
    suffix = address[-6:].upper()
    return f"Trader_{suffix}"
