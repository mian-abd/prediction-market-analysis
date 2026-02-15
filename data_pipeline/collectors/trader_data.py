"""Collect real trader performance data from Polymarket APIs.

Uses data-api.polymarket.com for:
- /v1/leaderboard — top traders by PnL/volume
- /v1/positions — real position-level P&L per trader
- /v1/trades — individual trade executions
"""

import httpx
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

POLYMARKET_DATA_API = "https://data-api.polymarket.com/v1"


async def fetch_polymarket_leaderboard(
    time_period: str = "MONTH",
    limit: int = 50,
    order_by: str = "PNL",
    category: str = "OVERALL",
    offset: int = 0,
) -> List[Dict]:
    """Fetch top traders from Polymarket leaderboard.

    Returns list of trader objects with real PnL and volume data.
    """
    params = {
        "timePeriod": time_period,
        "limit": min(limit, 50),
        "orderBy": order_by,
        "category": category,
        "offset": offset,
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{POLYMARKET_DATA_API}/leaderboard", params=params)
            resp.raise_for_status()
            data = resp.json()
            logger.info(f"Fetched {len(data)} traders from Polymarket leaderboard")
            return data
    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch Polymarket leaderboard: {e}")
        return []


async def fetch_trader_positions(user_address: str, limit: int = 100) -> List[Dict]:
    """Fetch real positions with P&L for a trader from Polymarket data API.

    Each position includes: size, avgPrice, cashPnl, percentPnl,
    realizedPnl, curPrice, title, outcome, endDate.
    """
    params = {"user": user_address, "limit": limit}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{POLYMARKET_DATA_API}/positions", params=params)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as e:
        logger.warning(f"Failed to fetch positions for {user_address[:10]}...: {e}")
        return []


async def fetch_trader_trades(
    user_address: str,
    limit: int = 100,
) -> List[Dict]:
    """Fetch real trade executions for a trader from Polymarket data API.

    Each trade includes: side (BUY/SELL), size, price, timestamp,
    title, outcome, conditionId.
    """
    params = {"user": user_address, "limit": limit}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{POLYMARKET_DATA_API}/trades", params=params)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as e:
        logger.warning(f"Failed to fetch trades for {user_address[:10]}...: {e}")
        return []


def calculate_trader_stats(trader_data: Dict, positions: List[Dict]) -> Dict:
    """Calculate real trader statistics from Polymarket position data.

    Uses actual position-level P&L (cashPnl field) from the data API,
    NOT estimated/fabricated stats.

    Args:
        trader_data: Leaderboard data (has real pnl, vol)
        positions: Real position data from /v1/positions endpoint
    """
    total_pnl = float(trader_data.get("pnl", 0) or 0)
    volume = float(trader_data.get("vol", 0) or 0)

    # Calculate stats from real position-level P&L
    total_positions = len(positions)
    winning_positions = 0
    losing_positions = 0
    pnl_values = []

    for pos in positions:
        cash_pnl = float(pos.get("cashPnl", 0) or 0)
        pnl_values.append(cash_pnl)
        if cash_pnl > 0:
            winning_positions += 1
        elif cash_pnl < 0:
            losing_positions += 1

    win_rate = (winning_positions / total_positions * 100) if total_positions > 0 else 0.0

    # ROI from leaderboard data
    invested = volume - total_pnl if volume > total_pnl else volume * 0.3
    roi_pct = (total_pnl / max(invested, 1)) * 100

    # Max drawdown from position-level P&L
    max_drawdown = 0.0
    if pnl_values:
        cumulative = []
        running = 0.0
        for pnl in pnl_values:
            running += pnl
            cumulative.append(running)

        peak = cumulative[0]
        for value in cumulative:
            if value > peak:
                peak = value
            dd = peak - value
            if dd > max_drawdown:
                max_drawdown = dd
        max_drawdown = -max_drawdown

    # Risk score from real data
    volatility = _calculate_volatility(pnl_values) if len(pnl_values) > 5 else 0
    risk_score = _calculate_risk_score(
        max_drawdown=abs(max_drawdown),
        volatility=volatility,
        avg_position_size=volume / total_positions if total_positions > 0 else 0,
    )

    return {
        "total_pnl": total_pnl,
        "roi_pct": round(roi_pct, 2),
        "win_rate": round(win_rate, 1),
        "total_trades": total_positions,
        "winning_trades": winning_positions,
        "avg_trade_duration_hrs": 0.0,  # Not available from position data
        "risk_score": risk_score,
        "max_drawdown": round(max_drawdown, 2),
    }


def _calculate_volatility(pnl_values: List[float]) -> float:
    """Standard deviation of P&L values."""
    if len(pnl_values) < 2:
        return 0.0
    mean = sum(pnl_values) / len(pnl_values)
    variance = sum((x - mean) ** 2 for x in pnl_values) / len(pnl_values)
    return variance ** 0.5


def _calculate_risk_score(
    max_drawdown: float,
    volatility: float,
    avg_position_size: float,
) -> int:
    """Risk score (1-10) from real trading behavior."""
    score = 5

    if max_drawdown > 5000:
        score += 3
    elif max_drawdown > 2000:
        score += 2
    elif max_drawdown > 1000:
        score += 1
    elif max_drawdown < 500:
        score -= 1

    if volatility > 1000:
        score += 2
    elif volatility > 500:
        score += 1
    elif volatility < 100:
        score -= 1

    if avg_position_size > 5000:
        score += 1
    elif avg_position_size < 500:
        score -= 1

    return max(1, min(10, score))


def generate_trader_bio(trader_data: Dict, stats: Dict) -> str:
    """Generate bio from real stats."""
    win_rate = stats["win_rate"]
    risk = stats["risk_score"]
    total_trades = stats["total_trades"]

    if risk <= 3:
        style = "Conservative, capital-preservation focused"
    elif risk <= 6:
        style = "Balanced risk approach"
    else:
        style = "Aggressive, high-conviction trader"

    if win_rate >= 70:
        skill = "Exceptional track record"
    elif win_rate >= 60:
        skill = "Strong performer"
    elif win_rate >= 50:
        skill = "Consistent results"
    elif total_trades > 0:
        skill = "Active trader"
    else:
        skill = "Leaderboard trader"

    pnl = stats["total_pnl"]
    pnl_str = f"${pnl:,.0f}" if pnl >= 0 else f"-${abs(pnl):,.0f}"

    return f"{skill}. {style}. {pnl_str} P&L on Polymarket."
