"""Platform-specific constants: fees, API limits, thresholds."""

# ── Polymarket Fee Schedule ──────────────────────────────────────
POLYMARKET_FEES = {
    "standard": {
        "taker_fee_pct": 0.0,
        "maker_fee_pct": 0.0,
        "winnings_fee_pct": 2.0,   # 2% on net winnings only
    },
    "15min_crypto": {
        "taker_fee_max_pct": 3.15,  # Variable, up to 3.15%
        "maker_fee_pct": 0.0,       # Maker rebate program
        "winnings_fee_pct": 2.0,
    },
}

# ── Kalshi Fee Schedule ──────────────────────────────────────────
KALSHI_FEES = {
    "fee_per_contract": 0.01,       # $0.01 per contract
    "fee_cap_pct": 7.0,             # Capped at 7% of premium
    "avg_effective_pct": 0.7,       # ~0.7% average in practice
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
