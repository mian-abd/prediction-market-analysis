"""Favorite-Longshot bias exploitation — research-backed standalone strategy.

Evidence: Markets overprice longshots (<20%) and underprice favorites (60-75%).
- Kalshi: 5¢ contracts win 4.18% (not 5%); 95¢ win 95.83%
- Jonathan Becker: 1¢ YES = -41% EV; 1¢ NO = +23% EV (64pp gap)
- Fading <20% longshots → +8.2% ROI; 60-75% favorites underpriced

Strategy rules:
- BUY YES: 0.60 ≤ price ≤ 0.75 (underpriced favorites)
- BUY NO: price ≥ 0.80 (overpriced favorites — fade)
- NEVER BUY YES when price < 0.25 (longshots overpriced)
- Category-specific: Finance ~efficient; Sports/Entertainment/Media have larger gaps
"""

import logging
from dataclasses import dataclass

from ml.features.calibration_features import get_calibration_estimate as _get_calibration_estimate

logger = logging.getLogger(__name__)

# Fee/slippage: per-market fee from DB. Most markets are fee-free.
SLIPPAGE_BUFFER = 0.01

# Category efficiency gaps (Jonathan Becker: maker-taker gap in pp)
# Higher = more edge potential. Finance ~efficient.
CATEGORY_EFFICIENCY_GAP = {
    "finance": 0.17,
    "economics": 0.30,
    "politics": 1.02,
    "sports": 2.23,
    "crypto": 2.69,
    "weather": 2.57,
    "entertainment": 4.79,
    "media": 7.28,
    "culture": 3.0,
    "science": 1.5,
    "technology": 1.2,
    "other": 1.5,
}

MIN_NET_EDGE = 0.03  # 3% minimum
MIN_LIQUIDITY = 2000  # Lower than ensemble to capture liquidity gaps
MIN_VOLUME_TOTAL = 5000
MAX_KELLY = 0.03  # 3% max position


@dataclass
class FavoriteLongshotSignal:
    """Detected favorite-longshot edge."""
    market_id: int
    direction: str  # "buy_yes" or "buy_no"
    calibrated_prob: float
    market_price: float
    raw_edge: float
    fee_cost: float
    net_ev: float
    kelly_fraction: float
    category: str
    efficiency_gap: float
    signal_type: str  # "underpriced_favorite" | "overpriced_favorite" | "fade_longshot"


def _get_category_gap(category: str) -> float:
    """Get efficiency gap for category (pp). Higher = more edge."""
    c = (category or "other").lower()
    return CATEGORY_EFFICIENCY_GAP.get(c, 1.5)


def _compute_fee_cost(direction: str, market_price: float, calibrated_prob: float = 0.5, taker_fee_bps: int = 0) -> float:
    """Expected fee + slippage for the trade, probability-weighted.

    Fee is only charged on winnings, so expected fee = P(win) * fee_rate * winnings.
    """
    fee_rate = taker_fee_bps / 10000
    if direction == "buy_yes":
        fee = calibrated_prob * fee_rate * (1 - market_price)
    else:
        fee = (1 - calibrated_prob) * fee_rate * market_price
    return fee + SLIPPAGE_BUFFER


def _compute_kelly(direction: str, calibrated: float, market_price: float, fee_cost: float) -> float:
    """Fractional Kelly for position sizing."""
    if direction == "buy_yes":
        ev = calibrated * (1 - market_price) - (1 - calibrated) * market_price - fee_cost
        kelly_raw = ev / max(0.01, 1 - market_price)
    else:
        ev = (1 - calibrated) * market_price - calibrated * (1 - market_price) - fee_cost
        kelly_raw = ev / max(0.01, market_price)
    kelly_capped = min(max(0, kelly_raw * 0.25), MAX_KELLY)  # 25% fractional Kelly
    return kelly_capped


def detect_favorite_longshot_edge(market) -> FavoriteLongshotSignal | None:
    """Detect favorite-longshot bias edge for a market.

    Returns signal if edge exists and passes quality gates, else None.
    """
    price = float(market.price_yes or 0.5)
    vol_total = float(market.volume_total or 0)
    liquidity = float(market.liquidity or 0)
    category = getattr(market, "normalized_category", None) or getattr(market, "category", "other") or "other"

    # Resolve platform name from the market object.  Markets may carry either a
    # `platform` string attribute (from API layer) or only a `platform_id` FK.
    platform_name: str | None = None
    if hasattr(market, "platform") and market.platform:
        # Joined ORM object — Platform.name is available directly
        try:
            platform_name = str(market.platform.name).lower()
        except AttributeError:
            pass
    if platform_name is None:
        # Fallback: try a plain string attribute (set by some callers)
        platform_name = getattr(market, "platform_name", None)

    # Quality gates (relaxed vs ensemble to capture liquidity gaps)
    if vol_total < MIN_VOLUME_TOTAL or liquidity < MIN_LIQUIDITY:
        return None
    if price <= 0.02 or price >= 0.98:
        return None

    # Calibrated probability — prefer platform- and category-specific curves when
    # build_calibration_curves.py has been run; otherwise falls back to the static table.
    calibrated = _get_calibration_estimate(price, platform=platform_name, category=category)
    raw_edge = abs(calibrated - price)

    # Rule 1: NEVER buy YES on longshots (price < 0.25)
    if price < 0.25:
        # Fade: BUY NO. Market overprices YES (longshot).
        direction = "buy_no"
        calibrated_no = 1 - calibrated  # True prob of NO
        market_no = 1 - price
        # We think NO is more likely than market implies
        if calibrated_no <= market_no:
            return None  # No edge
        signal_type = "fade_longshot"
    # Rule 2: Underpriced favorites (60-75%) — BUY YES
    elif 0.60 <= price <= 0.75:
        if calibrated <= price:
            return None  # No edge
        direction = "buy_yes"
        signal_type = "underpriced_favorite"
    # Rule 3: Overpriced favorites (80%+) — BUY NO (fade)
    elif price >= 0.80:
        calibrated_no = 1 - calibrated
        market_no = 1 - price
        if calibrated_no <= market_no:
            return None
        direction = "buy_no"
        signal_type = "overpriced_favorite"
    # Rule 4: Mid-range (25-60%) — only BUY NO if strong overpricing
    else:
        # 25-60%: weaker signal, require larger edge
        if calibrated < price:
            # Market overpriced YES → BUY NO
            calibrated_no = 1 - calibrated
            market_no = 1 - price
            if calibrated_no <= market_no or raw_edge < 0.05:
                return None
            direction = "buy_no"
            signal_type = "overpriced_favorite"
        else:
            # Underpriced YES — but mid-range is noisier
            if raw_edge < 0.05:
                return None
            direction = "buy_yes"
            signal_type = "underpriced_favorite"

    taker_fee_bps = getattr(market, 'taker_fee_bps', 0) or 0
    fee_cost = _compute_fee_cost(direction, price, calibrated_prob=calibrated, taker_fee_bps=taker_fee_bps)
    if direction == "buy_yes":
        net_ev = calibrated * (1 - price) - (1 - calibrated) * price - fee_cost
    else:
        net_ev = (1 - calibrated) * price - calibrated * (1 - price) - fee_cost

    if net_ev < MIN_NET_EDGE:
        return None

    # Scale confidence by category efficiency (Finance = low, Media = high)
    gap = _get_category_gap(category)
    category_boost = min(1.2, 0.7 + gap / 5.0)  # 0.7-1.2
    kelly = _compute_kelly(direction, calibrated, price, fee_cost)
    kelly *= category_boost  # Slightly larger in inefficient categories

    return FavoriteLongshotSignal(
        market_id=market.id,
        direction=direction,
        calibrated_prob=calibrated,
        market_price=price,
        raw_edge=raw_edge,
        fee_cost=fee_cost,
        net_ev=net_ev,
        kelly_fraction=min(kelly, MAX_KELLY),
        category=category,
        efficiency_gap=gap,
        signal_type=signal_type,
    )


def signal_to_dict(signal: FavoriteLongshotSignal) -> dict:
    """Serialize for API/DB."""
    return {
        "market_id": signal.market_id,
        "direction": signal.direction,
        "calibrated_prob": round(signal.calibrated_prob, 4),
        "market_price": round(signal.market_price, 4),
        "raw_edge_pct": round(signal.raw_edge * 100, 2),
        "fee_cost_pct": round(signal.fee_cost * 100, 2),
        "net_ev_pct": round(signal.net_ev * 100, 2),
        "kelly_fraction": round(signal.kelly_fraction, 4),
        "category": signal.category,
        "efficiency_gap": signal.efficiency_gap,
        "signal_type": signal.signal_type,
    }
