"""Orderbook-derived features for ML models.
Order Book Imbalance (OBI) explains ~65% of midpoint price changes."""

import numpy as np
from typing import Optional


def compute_orderbook_features(
    bids: list[dict],  # [{"price": 0.55, "size": 100}, ...]
    asks: list[dict],  # [{"price": 0.57, "size": 80}, ...]
) -> dict:
    """Extract 8 orderbook features from bid/ask data."""
    if not bids or not asks:
        return _empty_features()

    # Sort
    bids = sorted(bids, key=lambda x: x["price"], reverse=True)
    asks = sorted(asks, key=lambda x: x["price"])

    best_bid = bids[0]["price"]
    best_ask = asks[0]["price"]
    mid = (best_bid + best_ask) / 2

    # Feature 1: Level-1 OBI
    bid1_qty = bids[0]["size"]
    ask1_qty = asks[0]["size"]
    denom = bid1_qty + ask1_qty
    obi_level1 = (bid1_qty - ask1_qty) / denom if denom > 0 else 0.0

    # Feature 2: Weighted OBI (top 5 levels, inverse-distance weighted)
    obi_weighted = 0.0
    weight_sum = 0.0
    for i in range(min(5, len(bids), len(asks))):
        w = 1.0 / (i + 1)
        obi_weighted += w * (bids[i]["size"] - asks[i]["size"])
        weight_sum += w * (bids[i]["size"] + asks[i]["size"])
    obi_weighted = obi_weighted / weight_sum if weight_sum > 0 else 0.0

    # Feature 3: Bid-ask spread (absolute)
    spread_abs = best_ask - best_bid

    # Feature 4: Bid-ask spread (relative to midpoint)
    spread_rel = spread_abs / mid if mid > 0 else 0.0

    # Feature 5: Depth ratio (bid depth / ask depth, top 5 levels)
    bid_depth = sum(b["size"] for b in bids[:5])
    ask_depth = sum(a["size"] for a in asks[:5])
    depth_ratio = bid_depth / ask_depth if ask_depth > 0 else 1.0

    # Feature 6: Total bid depth USD
    bid_depth_usd = sum(b["price"] * b["size"] for b in bids[:5])

    # Feature 7: Total ask depth USD
    ask_depth_usd = sum(a["price"] * a["size"] for a in asks[:5])

    # Feature 8: VWAP deviation
    total_size = sum(b["size"] for b in bids[:5]) + sum(a["size"] for a in asks[:5])
    if total_size > 0:
        vwap = (
            sum(b["price"] * b["size"] for b in bids[:5]) +
            sum(a["price"] * a["size"] for a in asks[:5])
        ) / total_size
        vwap_deviation = (mid - vwap) / mid if mid > 0 else 0.0
    else:
        vwap_deviation = 0.0

    return {
        "obi_level1": obi_level1,
        "obi_weighted_5": obi_weighted,
        "bid_ask_spread_abs": spread_abs,
        "bid_ask_spread_rel": spread_rel,
        "depth_ratio": depth_ratio,
        "bid_depth_usd": bid_depth_usd,
        "ask_depth_usd": ask_depth_usd,
        "vwap_deviation": vwap_deviation,
    }


def _empty_features() -> dict:
    return {
        "obi_level1": 0.0,
        "obi_weighted_5": 0.0,
        "bid_ask_spread_abs": 0.0,
        "bid_ask_spread_rel": 0.0,
        "depth_ratio": 1.0,
        "bid_depth_usd": 0.0,
        "ask_depth_usd": 0.0,
        "vwap_deviation": 0.0,
    }
