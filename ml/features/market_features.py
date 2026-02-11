"""Market-level features: volume, time-to-resolution, category."""

import math
from datetime import datetime


def compute_market_features(
    volume_24h: float,
    volume_total: float,
    liquidity: float,
    open_interest: float,
    end_date: datetime | None,
    category: str,
    price_yes: float,
    num_trades_1h: int = 0,
    avg_trade_size_1h: float = 0.0,
) -> dict:
    """Extract 8 market-level features."""
    now = datetime.utcnow()

    # Feature 1: Time to resolution (hours)
    if end_date:
        delta = end_date.replace(tzinfo=None) - now if end_date.tzinfo else end_date - now
        time_to_resolution_hrs = max(0, delta.total_seconds() / 3600)
    else:
        time_to_resolution_hrs = 8760  # Default 1 year

    # Feature 2: Log volume 24h (avoids scale issues)
    log_volume_24h = math.log1p(volume_24h)

    # Feature 3: Log open interest
    log_open_interest = math.log1p(open_interest)

    # Feature 4: Category encoded (simple hash)
    category_map = {
        "politics": 0, "crypto": 1, "sports": 2, "science": 3,
        "entertainment": 4, "economics": 5, "technology": 6,
        "weather": 7, "culture": 8, "other": 9,
    }
    category_encoded = category_map.get((category or "other").lower(), 9)

    # Feature 5: Price bucket (5% bins, 0-19)
    price_bucket = min(19, int(price_yes / 0.05))

    # Feature 6: Number of trades in last hour
    # Feature 7: Average trade size in last hour

    # Feature 8: Is weekend
    is_weekend = 1 if now.weekday() >= 5 else 0

    return {
        "time_to_resolution_hrs": time_to_resolution_hrs,
        "log_volume_24h": log_volume_24h,
        "log_open_interest": log_open_interest,
        "category_encoded": category_encoded,
        "price_bucket": price_bucket,
        "num_trades_1h": num_trades_1h,
        "avg_trade_size_1h": avg_trade_size_1h,
        "is_weekend": is_weekend,
    }
