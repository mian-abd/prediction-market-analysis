"""Order flow features from trade history data.

Computes buy/sell imbalance, trade intensity, large-trade signals, and
VWAP-deviation metrics from historical trade records. These features
capture informed trading patterns that pure price/orderbook data misses.

Requires trade history from the `trades` table (backfilled via
scripts/backfill_trade_history.py).
"""

import numpy as np
from datetime import datetime, timedelta


def compute_orderflow_features(
    trades: list[dict],
    as_of: datetime | None = None,
) -> dict:
    """Extract order flow features from trade history.

    Args:
        trades: List of trade dicts with keys:
            - timestamp: datetime
            - side: "BUY" or "SELL"
            - price: float
            - size: float (USD notional)
        as_of: Reference time. Only trades before this time are used.
               Defaults to latest trade timestamp.

    Returns:
        Dict of feature_name -> float
    """
    if not trades:
        return _empty_features()

    if as_of is not None:
        trades = [t for t in trades if t.get("timestamp") and t["timestamp"] <= as_of]

    if not trades:
        return _empty_features()

    trades.sort(key=lambda t: t["timestamp"])
    ref_time = as_of or trades[-1]["timestamp"]

    # Split into time windows
    t_1h = [t for t in trades if t["timestamp"] >= ref_time - timedelta(hours=1)]
    t_6h = [t for t in trades if t["timestamp"] >= ref_time - timedelta(hours=6)]
    t_24h = [t for t in trades if t["timestamp"] >= ref_time - timedelta(hours=24)]
    t_7d = [t for t in trades if t["timestamp"] >= ref_time - timedelta(days=7)]

    def buy_sell_ratio(trade_list: list[dict]) -> float:
        if not trade_list:
            return 0.0
        buys = sum(t["size"] for t in trade_list if t.get("side", "").upper() == "BUY")
        sells = sum(t["size"] for t in trade_list if t.get("side", "").upper() == "SELL")
        total = buys + sells
        if total == 0:
            return 0.0
        return (buys - sells) / total  # [-1, 1]

    def trade_intensity(trade_list: list[dict], hours: float) -> float:
        if hours <= 0:
            return 0.0
        return len(trade_list) / hours

    def avg_trade_size(trade_list: list[dict]) -> float:
        if not trade_list:
            return 0.0
        return np.mean([t["size"] for t in trade_list])

    def large_trade_count(trade_list: list[dict], threshold: float = 100.0) -> int:
        return sum(1 for t in trade_list if t["size"] >= threshold)

    def vwap(trade_list: list[dict]) -> float:
        if not trade_list:
            return 0.0
        total_value = sum(t["price"] * t["size"] for t in trade_list)
        total_size = sum(t["size"] for t in trade_list)
        if total_size == 0:
            return 0.0
        return total_value / total_size

    def volume_acceleration(trade_list_recent: list, trade_list_older: list) -> float:
        """Compare recent vs older volume to detect acceleration."""
        vol_recent = sum(t["size"] for t in trade_list_recent)
        vol_older = sum(t["size"] for t in trade_list_older)
        if vol_older == 0:
            return 0.0
        return (vol_recent - vol_older) / vol_older

    current_price = trades[-1]["price"] if trades else 0.0
    vwap_24h = vwap(t_24h)
    vwap_dev = (current_price - vwap_24h) / vwap_24h if vwap_24h > 0 else 0.0

    # Volume acceleration: compare last 6h volume to previous 6h
    t_6h_prev = [t for t in trades
                 if ref_time - timedelta(hours=12) <= t["timestamp"] < ref_time - timedelta(hours=6)]

    return {
        "oflow_buy_sell_ratio_1h": buy_sell_ratio(t_1h),
        "oflow_buy_sell_ratio_24h": buy_sell_ratio(t_24h),
        "oflow_trade_intensity_1h": trade_intensity(t_1h, 1.0),
        "oflow_trade_intensity_24h": trade_intensity(t_24h, 24.0),
        "oflow_avg_trade_size_24h": avg_trade_size(t_24h),
        "oflow_large_trade_count_24h": float(large_trade_count(t_24h, 100.0)),
        "oflow_large_trade_count_7d": float(large_trade_count(t_7d, 100.0)),
        "oflow_vwap_deviation_24h": vwap_dev,
        "oflow_volume_acceleration_6h": volume_acceleration(t_6h, t_6h_prev),
        "oflow_total_volume_24h": sum(t["size"] for t in t_24h),
    }


ORDERFLOW_FEATURE_NAMES = [
    "oflow_buy_sell_ratio_1h",
    "oflow_buy_sell_ratio_24h",
    "oflow_trade_intensity_1h",
    "oflow_trade_intensity_24h",
    "oflow_avg_trade_size_24h",
    "oflow_large_trade_count_24h",
    "oflow_large_trade_count_7d",
    "oflow_vwap_deviation_24h",
    "oflow_volume_acceleration_6h",
    "oflow_total_volume_24h",
]


def _empty_features() -> dict:
    return {name: 0.0 for name in ORDERFLOW_FEATURE_NAMES}
