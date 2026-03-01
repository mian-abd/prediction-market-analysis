"""Platform-specific constants: fees, API limits, thresholds."""

import math

# ── Polymarket Fee Schedule ──────────────────────────────────────
# Most markets are fee-free. Fee-enabled markets have a curve-based fee
# that must be queried per-token via GET /fee-rate?token_id=TOKEN_ID.
# The response field is `fee_rate_bps`.
POLYMARKET_FEES = {
    "standard": {
        "taker_fee_pct": 0.0,
        "maker_fee_pct": 0.0,
    },
    "crypto_5min": {
        "fee_rate_bps": 2500,
        "exponent": 2,
        "maker_rebate_pct": 20,
    },
    "crypto_15min": {
        "fee_rate_bps": 2500,
        "exponent": 2,
        "maker_rebate_pct": 20,
    },
    "serie_a": {
        "fee_rate_bps": 175,
        "exponent": 1,
        "maker_rebate_pct": 25,
    },
    "ncaab": {
        "fee_rate_bps": 175,
        "exponent": 1,
        "maker_rebate_pct": 25,
    },
}


def compute_polymarket_fee(
    price: float,
    quantity: float,
    fee_rate_bps: int = 0,
    is_maker: bool = False,
) -> float:
    """Compute Polymarket fee for a trade.

    Args:
        price: YES price (0.0 to 1.0)
        quantity: number of shares
        fee_rate_bps: fee rate in basis points (0 = fee-free, from API)
        is_maker: if True, returns negative fee (rebate)

    Returns:
        Fee in absolute terms. Positive for takers, negative for makers (rebate).
    """
    if fee_rate_bps == 0:
        return 0.0

    fee_rate = fee_rate_bps / 10000
    # Determine exponent: crypto markets use quadratic, sports use linear
    # This is approximate; real categorization comes from the API response.
    exponent = 2 if fee_rate_bps >= 1000 else 1
    fee_per_share = fee_rate * (price * (1 - price)) ** exponent
    total_fee = quantity * fee_per_share

    if is_maker:
        rebate_pct = 0.20 if exponent == 2 else 0.25
        return -total_fee * rebate_pct

    return total_fee


def compute_kalshi_fee(
    price: float,
    contracts: int,
    is_maker: bool = False,
) -> float:
    """Compute Kalshi fee using the quadratic formula.

    Taker: fee = roundup(0.07 * contracts * price * (1 - price))
    Maker: fee = roundup(0.0175 * contracts * price * (1 - price))
    """
    multiplier = 0.0175 if is_maker else 0.07
    raw_fee = multiplier * contracts * price * (1 - price)
    return math.ceil(raw_fee * 100) / 100  # Round up to nearest cent


# ── Kalshi Fee Schedule ──────────────────────────────────────────
KALSHI_FEES = {
    "taker_multiplier": 0.07,
    "maker_multiplier": 0.0175,
    "max_taker_pct_at_50": 1.75,
    "max_maker_pct_at_50": 0.44,
}

# ── API Rate Limits ──────────────────────────────────────────────
POLYMARKET_RATE_LIMITS = {
    "gamma_requests_per_minute": 60,
    "clob_requests_per_minute": 100,
}

KALSHI_RATE_LIMITS = {
    "requests_per_second": 10,
}

# ── Market Categories ────────────────────────────────────────────
MARKET_CATEGORIES = [
    "politics", "crypto", "sports", "science", "entertainment",
    "economics", "technology", "weather", "culture", "other",
]

# ── ML Model Constants ───────────────────────────────────────────
PRICE_DIRECTION_THRESHOLD = 0.01    # 1% move = UP/DOWN, else FLAT
CALIBRATION_BUCKET_SIZE = 0.05      # 5% price buckets for calibration
FEATURE_COUNT = 32

# ── Arbitrage Constants ──────────────────────────────────────────
MIN_LIQUIDITY_USD = 10.0            # Skip markets with < $10 liquidity
MAX_MARKET_AGE_DAYS = 365           # Skip markets older than 1 year
STALE_DATA_THRESHOLD_SEC = 60       # Reject arb if data > 60s old
