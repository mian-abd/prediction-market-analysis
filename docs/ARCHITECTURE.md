# Architecture

## System Overview

PredictFlow is a quantitative prediction market analysis platform. It collects data from multiple platforms, detects arbitrage opportunities using pure math, applies ML calibration models, and offers on-demand Claude AI analysis.

**Design principle**: Math runs for free. AI is user-triggered only.

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Polymarket   │     │    Kalshi     │     │  GDELT News  │
│  Gamma + CLOB │     │   REST API   │     │   (future)   │
└──────┬───────┘     └──────┬───────┘     └──────────────┘
       │                    │
       ▼                    ▼
┌──────────────────────────────────────────┐
│          DATA PIPELINE (scheduler.py)     │
│                                           │
│  collect_markets()     every 1 hour       │
│  collect_prices()      every 60 seconds   │
│  collect_orderbooks()  every 5 minutes    │
│  run_market_matching() after market refresh│
│  scan_arbitrage()      every 5 minutes    │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│           SQLite DATABASE (12 tables)     │
│                                           │
│  platforms · markets · price_snapshots    │
│  orderbook_snapshots · trades             │
│  cross_platform_matches                   │
│  market_relationships                     │
│  arbitrage_opportunities                  │
│  ml_predictions · ai_analyses             │
│  portfolio_positions · system_metrics     │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│          FastAPI BACKEND (16 endpoints)   │
│                                           │
│  /markets        Browse/search/detail     │
│  /arbitrage      Opportunities + history  │
│  /predictions    ML calibration model     │
│  /calibration    Calibration curve viz    │
│  /analyze        Claude AI (user-trigger) │
│  /portfolio      Paper trading positions  │
│  /system         Health + stats           │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│        React FRONTEND (6 pages)           │
│                                           │
│  Dashboard · Market Browser               │
│  Market Detail · Arbitrage Scanner        │
│  ML Models · Calibration Chart            │
└──────────────────────────────────────────┘
```

## Directory Structure

```
prediction-market-analysis/
├── config/
│   ├── settings.py          # Pydantic BaseSettings, all env vars
│   └── constants.py         # Fee schedules, API URLs, thresholds
├── db/
│   ├── database.py          # SQLAlchemy async engine + session factory
│   └── models.py            # 12 ORM tables
├── data_pipeline/
│   ├── scheduler.py         # Main orchestrator (runs in background)
│   ├── collectors/
│   │   ├── polymarket_gamma.py  # Markets, events, metadata
│   │   ├── polymarket_clob.py   # Orderbooks, prices via CLOB API
│   │   └── kalshi_markets.py    # Markets, orderbooks from Kalshi
│   ├── transformers/
│   │   └── market_matcher.py    # TF-IDF cross-platform matching
│   └── storage.py               # Bulk upsert helpers
├── arbitrage/
│   ├── engine.py            # Orchestrates all strategies
│   ├── strategies/
│   │   ├── single_market.py     # YES + NO < $1 rebalancing
│   │   └── cross_platform.py    # Polymarket vs Kalshi spread
│   └── fee_calculator.py        # Platform-specific fee math
├── ml/
│   ├── features/
│   │   ├── calibration_features.py  # Historical calibration data
│   │   ├── market_features.py       # Volume, time, category
│   │   ├── orderbook_features.py    # OBI, depth, spread
│   │   └── momentum_features.py     # Returns, volatility, RSI
│   ├── models/
│   │   └── calibration_model.py     # Isotonic regression
│   └── saved_models/                # Serialized .joblib files
├── ai_analysis/
│   ├── claude_client.py     # SDK wrapper + cost tracking + cache
│   └── prompts/
│       └── market_analysis.py   # Analysis prompt templates
├── api/
│   ├── main.py              # FastAPI app factory + lifespan
│   └── routes/
│       ├── markets.py       # Browse/search/detail
│       ├── arbitrage.py     # Opportunities + history
│       ├── ml_predictions.py# Calibration model predictions
│       ├── ai_analysis.py   # Claude analysis (user-triggered)
│       ├── portfolio.py     # Paper trading positions
│       └── system.py        # Health + dashboard stats
├── frontend/
│   └── src/
│       ├── api/client.ts    # Axios client with /api/v1 proxy
│       ├── pages/           # 6 page components
│       ├── App.tsx          # Router + sidebar layout
│       └── index.css        # Design system tokens
└── docs/
    ├── ARCHITECTURE.md      # This file
    ├── API.md               # Endpoint reference
    ├── STRATEGIES.md        # Trading strategy documentation
    └── SETUP.md             # Setup and deployment guide
```

## Data Flow

### Price Collection (every 60s)
1. `collect_prices()` reads active markets from DB
2. For each market with a valid price, creates a `PriceSnapshot` row
3. Stores: market_id, timestamp, price_yes, price_no, midpoint, spread, volume

### Arbitrage Detection (every 5 min)
1. `scan_arbitrage()` first expires opportunities older than 30 minutes
2. Runs `scan_single_market_arb()` - checks all markets for YES + NO < $1
3. Runs `scan_cross_platform_arb()` - checks cross-platform matches for price spreads
4. Both strategies apply platform-specific fee calculations
5. Only net-profitable opportunities (after fees) are stored

### Cross-Platform Matching (every 1 hr)
1. `run_market_matching()` loads top 500 markets from each platform
2. Builds TF-IDF vectors from market questions
3. Computes cosine similarity matrix
4. Creates `CrossPlatformMatch` records for pairs above 0.45 similarity
5. These matches feed into cross-platform arbitrage scanning

### ML Predictions (on-demand)
1. `CalibrationModel` uses isotonic regression trained on historical bias data
2. Maps market price -> calibrated (true) probability
3. Returns delta, direction (overpriced/underpriced), and edge estimate
4. Used by `/predictions/top/mispriced` to rank markets

### Claude Analysis (user-triggered)
1. User clicks "Analyze with Claude" on a market detail page
2. Backend formats prompt with market data + calibration + price history
3. Checks SHA-256 hash cache in `ai_analyses` table
4. If not cached: calls Claude API, stores response + cost
5. Returns analysis text. ~$0.15 per call, cached forever.

## Database Schema

| Table | Rows (est.) | Purpose |
|-------|-------------|---------|
| platforms | 2 | Polymarket, Kalshi configs |
| markets | ~2,000 | All active markets |
| price_snapshots | Growing | Time-series prices (60s interval) |
| orderbook_snapshots | Growing | CLOB orderbook state (5min interval) |
| trades | Empty | Future: individual trade tracking |
| cross_platform_matches | ~50-200 | Polymarket-Kalshi event pairs |
| market_relationships | Empty | Future: logical constraints |
| arbitrage_opportunities | Growing | Detected opportunities with P&L |
| ml_predictions | Growing | Model outputs per market |
| ai_analyses | ~15 | Cached Claude responses |
| portfolio_positions | Growing | Paper trading positions |
| system_metrics | Growing | Pipeline health metrics |

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Backend | Python 3.13, FastAPI, SQLAlchemy 2 | Async-native, type-safe |
| Database | SQLite (aiosqlite) | Zero config, file-based |
| ML | scikit-learn, numpy, joblib | Isotonic regression, fast |
| AI | Claude Opus 4.6 (Anthropic SDK) | Extended thinking, web search |
| Frontend | React 19, TypeScript, Vite 7 | Fast builds, type safety |
| Charts | Recharts | React-native charting |
| HTTP | httpx (async), axios (frontend) | Async-first |
| Styling | Tailwind CSS 4 + custom tokens | Utility-first, themeable |

## Key Design Decisions

1. **SQLite over PostgreSQL**: Zero-config for hackathon. Schema designed for easy Postgres migration (no SQLite-specific features used).

2. **Background pipeline in FastAPI lifespan**: No separate scheduler process needed. Pipeline runs as asyncio task within the same process.

3. **Fee-first arbitrage**: Every opportunity passes through `FeeCalculator` before being stored. No false positives from ignoring fees.

4. **Isotonic regression for calibration**: Non-parametric, monotonic, handles the well-documented S-curve bias in prediction markets.

5. **Claude analysis cached by prompt hash**: SHA-256 of the prompt ensures identical questions return cached responses. Zero cost for repeat queries.

6. **Paper trading by default**: All positions are simulated (`is_simulated=True`). Safe for testing strategies before committing real capital.
