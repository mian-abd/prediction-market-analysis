# API Reference

Base URL: `http://localhost:8000/api/v1`

All endpoints return JSON. No authentication required (local deployment).

---

## System

### `GET /health`
Health check.
```json
{"status": "ok"}
```

### `GET /system/stats`
Dashboard statistics for the entire platform.
```json
{
  "total_active_markets": 2013,
  "markets_by_platform": {"polymarket": 1800, "kalshi": 213},
  "price_snapshots": 45000,
  "orderbook_snapshots": 1200,
  "cross_platform_matches": 87,
  "active_arbitrage_opportunities": 3,
  "last_data_fetch": "2026-02-11T14:30:00"
}
```

---

## Markets

### `GET /markets`
List markets with filtering, searching, and sorting.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| platform | string | null | Filter by platform name |
| category | string | null | Filter by category |
| search | string | null | Full-text search on question/description |
| is_active | bool | true | Only active markets |
| sort_by | string | "volume_24h" | Sort column |
| limit | int | 50 | Max 500 |
| offset | int | 0 | Pagination offset |

```json
{
  "markets": [
    {
      "id": 42,
      "platform": "polymarket",
      "external_id": "0x123...",
      "question": "Will BTC exceed $100k by March 2026?",
      "description": "...",
      "category": "crypto",
      "price_yes": 0.72,
      "price_no": 0.28,
      "volume_24h": 150000,
      "volume_total": 2500000,
      "liquidity": 80000,
      "end_date": "2026-03-31T00:00:00",
      "is_neg_risk": false,
      "updated_at": "2026-02-11T14:30:00"
    }
  ],
  "total": 2013,
  "limit": 50,
  "offset": 0
}
```

### `GET /markets/categories`
List all categories with market counts.
```json
[
  {"category": "politics", "count": 450},
  {"category": "crypto", "count": 320}
]
```

### `GET /markets/{market_id}`
Full market detail with price history and cross-platform matches.
```json
{
  "id": 42,
  "platform": "polymarket",
  "question": "...",
  "price_yes": 0.72,
  "price_no": 0.28,
  "volume_24h": 150000,
  "volume_total": 2500000,
  "token_id_yes": "0xabc...",
  "token_id_no": "0xdef...",
  "price_history": [
    {"timestamp": "2026-02-11T12:00:00", "price_yes": 0.70, "price_no": 0.30, "volume": 5000}
  ],
  "cross_platform_matches": [
    {"id": 99, "platform": "kalshi", "question": "...", "price_yes": 0.68, "similarity": 0.82}
  ]
}
```

---

## Arbitrage

### `GET /arbitrage/opportunities`
Current (non-expired) arbitrage opportunities.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| strategy_type | string | null | "single_market" or "cross_platform" |
| min_profit_pct | float | 0.0 | Minimum net profit % |
| limit | int | 50 | Max 200 |

```json
{
  "opportunities": [
    {
      "id": 1,
      "strategy_type": "single_market",
      "detected_at": "2026-02-11T14:25:00",
      "markets": [
        {"id": 42, "question": "...", "price_yes": 0.52, "price_no": 0.46}
      ],
      "gross_spread": 2.08,
      "total_fees": 0.04,
      "net_profit_pct": 2.04,
      "estimated_profit_usd": 2.04,
      "was_executed": false
    }
  ],
  "count": 1
}
```

### `GET /arbitrage/history`
Historical arbitrage opportunities (including expired).

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| strategy_type | string | null | Filter by strategy |
| limit | int | 100 | Max 500 |

---

## ML Predictions

### `GET /predictions/{market_id}`
Calibration model prediction for a specific market.
```json
{
  "market_id": 42,
  "question": "...",
  "models": {
    "calibration": {
      "market_price": 0.72,
      "calibrated_price": 0.67,
      "delta": -0.05,
      "delta_pct": -5.0,
      "direction": "overpriced",
      "edge_estimate": 0.05
    }
  }
}
```

### `GET /predictions/top/mispriced`
Markets with the largest calibration mispricing.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| limit | int | 20 | Number of markets to return |

```json
{
  "markets": [
    {
      "market_id": 42,
      "question": "...",
      "category": "politics",
      "price_yes": 0.85,
      "calibrated_price": 0.76,
      "delta_pct": -9.0,
      "direction": "overpriced",
      "edge_estimate": 0.09,
      "volume_24h": 50000
    }
  ]
}
```

### `GET /calibration/curve`
Calibration curve data for visualization (20 points from 5% to 95%).
```json
{
  "curve": [
    {
      "market_price": 0.05,
      "calibrated_price": 0.06,
      "bias": 0.01,
      "sample_count": 80
    }
  ]
}
```

---

## AI Analysis

### `POST /analyze/{market_id}`
Trigger Claude analysis for a market. Cached by prompt hash.

**Cost**: ~$0.15 per unique analysis, $0 for cached responses.

```json
{
  "market_id": 42,
  "question": "...",
  "analysis": "Extended analysis text from Claude..."
}
```

### `GET /analyze/{market_id}/cached`
Check if analysis exists in cache without triggering a new API call.
```json
{
  "cached": true,
  "market_id": 42,
  "analysis_type": "market_analysis",
  "response_text": "...",
  "cost": 0.15,
  "created_at": "2026-02-11T10:00:00"
}
```

### `GET /analyze/cost`
Claude API spend summary across all analyses.
```json
{
  "total_cost_usd": 2.50,
  "total_calls": 15,
  "total_input_tokens": 45000,
  "total_output_tokens": 30000,
  "avg_cost_per_call": 0.167
}
```

---

## Portfolio (Paper Trading)

### `GET /portfolio/positions`
List paper trading positions.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| status | string | "open" | "open", "closed", or "all" |
| strategy | string | null | Filter by strategy name |
| limit | int | 50 | Max 200 |

```json
{
  "positions": [
    {
      "id": 1,
      "market_id": 42,
      "question": "Will BTC exceed $100k?",
      "platform": "polymarket",
      "side": "yes",
      "entry_price": 0.72,
      "quantity": 100.0,
      "entry_time": "2026-02-11T14:00:00",
      "exit_price": null,
      "exit_time": null,
      "realized_pnl": null,
      "unrealized_pnl": 3.00,
      "current_price": 0.75,
      "strategy": "calibration",
      "is_simulated": true
    }
  ],
  "count": 1
}
```

### `POST /portfolio/positions`
Open a new paper trading position.

**Request body:**
```json
{
  "market_id": 42,
  "side": "yes",
  "entry_price": 0.72,
  "quantity": 100.0,
  "strategy": "calibration"
}
```

Strategy values: `manual`, `single_market_arb`, `cross_platform_arb`, `calibration`

### `POST /portfolio/positions/{position_id}/close`
Close an open position.

**Request body:**
```json
{"exit_price": 0.80}
```

### `GET /portfolio/summary`
Overall portfolio performance.
```json
{
  "open_positions": 3,
  "closed_positions": 12,
  "total_realized_pnl": 15.40,
  "win_rate": 66.7,
  "total_exposure": 300.0,
  "by_strategy": [
    {"strategy": "calibration", "trades": 8, "total_pnl": 12.30},
    {"strategy": "single_market_arb", "trades": 4, "total_pnl": 3.10}
  ]
}
```
