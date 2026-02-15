# PredictFlow

> **Quantitative prediction market analysis platform with multi-strategy signal generation, ML ensemble models, and automated paper trading**

Built for the **"Built with Opus 4.6: Claude Code Hackathon"** (Feb 10-16, 2026) — designed for long-term prop trading.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![React 19](https://img.shields.io/badge/react-19-61dafb.svg)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What Makes This Different

Most prediction market tools show prices. PredictFlow generates **fee-adjusted, Kelly-sized trading signals** with honest confidence estimates.

- **3-model ensemble** (Isotonic + XGBoost + LightGBM) with temporal train/test split — no future data leakage
- **Edge credibility capping** — signals >15% edge are auto-flagged as speculative (professional markets don't have 30% edges)
- **Signal accuracy tracking** — every signal is scored against market resolution. Hit rate, Brier score, simulated P&L all queryable via API
- **Risk management** — position limits, exposure caps, daily loss circuit breaker enforced before every trade
- **Fee-aware throughout** — platform fees + 1% slippage deducted from every edge estimate
- **Paper trading automation** — high-confidence signals auto-execute as paper trades with fractional Kelly sizing

---

## Architecture

```
                          ┌─────────────────────────────────────┐
                          │     React 19 Frontend (Vite + TS)   │
                          │  Dashboard · Markets · Signals ·    │
                          │  ML Models · Portfolio · Analytics   │
                          └──────────────┬──────────────────────┘
                                         │ REST API
┌────────────────────────────────────────┴──────────────────────────────────┐
│                        FastAPI Backend (Python 3.13)                      │
│                                                                          │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │ Data Pipeline │  │ ML Ensemble   │  │ Strategy     │  │ Execution  │  │
│  │ (asyncio)    │  │ (3 models)    │  │ Signals      │  │ Engine     │  │
│  │              │  │               │  │              │  │            │  │
│  │ · Prices     │  │ · Calibration │  │ · Ensemble   │  │ · Paper    │  │
│  │ · Orderbooks │  │ · XGBoost     │  │ · Elo Tennis │  │   trades   │  │
│  │ · Matching   │  │ · LightGBM    │  │ · Arbitrage  │  │ · Copy     │  │
│  │ · Snapshots  │  │ · 25 features │  │ · Quality    │  │   trading  │  │
│  └──────┬───────┘  └──────┬────────┘  │   gates      │  │ · Risk     │  │
│         │                  │           └──────┬───────┘  │   limits   │  │
│         │                  │                  │          └─────┬──────┘  │
│  ┌──────┴──────────────────┴──────────────────┴───────────────┴───────┐  │
│  │                   SQLite (16 tables, aiosqlite)                    │  │
│  │  Markets · Prices · Orderbooks · Predictions · Signals · Elo ·    │  │
│  │  Arbitrage · Portfolio · Trades · Copy · AI Cache · Metrics       │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
         │                    │                    │
    ┌────┴────┐         ┌────┴────┐          ┌────┴────┐
    │Polymarket│         │  Kalshi │          │ Claude  │
    │Gamma+CLOB│         │REST API │          │Opus 4.6 │
    └─────────┘         └─────────┘          └─────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.13+
- Node.js 18+
- API key: Anthropic (for Claude deep analysis, optional)

### Backend

```bash
git clone https://github.com/yourusername/prediction-market-analysis.git
cd prediction-market-analysis

# Virtual environment
python -m venv venv
# Linux/Mac: source venv/bin/activate
# Windows: .\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env: add ANTHROPIC_API_KEY (optional)

# Start server (auto-creates DB, starts background pipeline)
uvicorn api.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

### Verify

- Health check: `GET http://localhost:8000/api/v1/health`
- System stats: `GET http://localhost:8000/api/v1/system/stats`
- Frontend: http://localhost:5173

### Optional: Build ML Models

```bash
# Train ensemble (requires resolved markets in DB)
python scripts/train_ensemble.py

# Build Elo ratings for tennis markets
python scripts/build_elo_ratings.py --export-db

# Backfill historical prices from CLOB API
python scripts/backfill_price_history.py
```

---

## ML Ensemble Model

**3-model blend** with temporal train/test split at 2026-02-13:

| Component | Weight | Role |
|-----------|--------|------|
| Isotonic Regression | 20% | Calibration correction |
| XGBoost | 40% | Non-linear patterns |
| LightGBM | 40% | Gradient boosting |

**Performance** (2,840 training markets, 957 YES / 1,883 NO):

| Metric | Value |
|--------|-------|
| Brier Score | 0.0539 (baseline 0.0670, +19.6% improvement) |
| AUC-ROC | 0.9654 |
| Features | 25 extracted, 13 survive pruning |
| Post-calibrator | Disabled (hurts: 0.0579 > 0.0539) |

**Top features** (XGBoost importance): `volume_volatility` (55%), `volume_trend_7d` (26%), `log_open_interest` (5%)

### Signal Generation Pipeline

1. Extract 25 features per market (price, volume, time, orderbook)
2. Ensemble prediction (weighted blend of 3 models)
3. Quality gates: volume > threshold, liquidity check, price range check
4. Fee-aware edge: `raw_edge - platform_fee - slippage`
5. Edge credibility cap: edges >15% flagged as speculative
6. Fractional Kelly sizing (0.25x Kelly, 2% max of bankroll)
7. Signal persistence + resolution scoring for accuracy tracking

---

## Strategies

### ML Ensemble Edge Detection
Find markets where the ensemble model disagrees with market price by more than fees + slippage. Signals include direction (buy YES/NO), net expected value, Kelly fraction, confidence score, and quality tier (high/medium/low/speculative).

### Elo Sports Edge Detection
Glicko-2 ratings for tennis players. When Elo-implied probability diverges from market price by more than fees, generate a signal. Uses canonical Kelly formula shared with ensemble detector.

### Cross-Platform Arbitrage
Detect same-event pricing differences between Polymarket and Kalshi. Fee-aware (platform fees + 1% slippage). Many "opportunities" disappear after honest fee accounting.

### Copy Trading
Follow top Polymarket traders. Auto-replicate their positions with configurable sizing. Activity feed shows leader trades in real-time.

### Claude Deep Analysis
On-demand AI analysis (~$0.15/market). Extended thinking + market context. Cached in SQLite (SHA-256 prompt hash) — never pays twice for the same analysis.

---

## API Reference (25+ endpoints)

### Markets
```
GET  /api/v1/markets                    # List markets (paginated, filterable)
GET  /api/v1/markets/categories         # Category breakdown
GET  /api/v1/markets/{id}               # Market details + price history
```

### ML Predictions
```
GET  /api/v1/predictions/{id}           # Ensemble prediction for market
GET  /api/v1/predictions/top/mispriced  # Top mispriced markets
GET  /api/v1/predictions/top/edges      # Top edge opportunities
GET  /api/v1/predictions/accuracy/backtest  # Signal accuracy proof
GET  /api/v1/calibration/curve          # Calibration curve data
```

### Strategy Signals
```
GET  /api/v1/strategies/signals         # Unified: all signal types
GET  /api/v1/strategies/ensemble-edges  # Active ensemble edge signals
GET  /api/v1/strategies/signal-performance  # Historical signal P&L time-series
```

### Elo Ratings
```
GET  /api/v1/elo/ratings                # All player ratings
GET  /api/v1/elo/player/{name}          # Player rating + history
GET  /api/v1/elo/predict/tennis         # Head-to-head prediction
GET  /api/v1/elo/edges                  # Active Elo edge signals
POST /api/v1/elo/scan                   # Trigger Elo edge scan
```

### Portfolio & Risk
```
GET  /api/v1/portfolio/positions        # Open positions
POST /api/v1/portfolio/positions        # Open new position (risk-checked)
POST /api/v1/portfolio/positions/{id}/close  # Close position
GET  /api/v1/portfolio/summary          # Portfolio summary
GET  /api/v1/portfolio/risk-status      # Risk limit utilization
```

### Arbitrage
```
GET  /api/v1/arbitrage/opportunities    # Current opportunities
GET  /api/v1/arbitrage/history          # Historical opportunities
```

### AI Analysis
```
POST /api/v1/analyze/{id}              # Trigger Claude analysis
GET  /api/v1/analyze/{id}/cached       # Get cached analysis
GET  /api/v1/analyze/cost              # Cost tracking
```

### System
```
GET  /api/v1/health                    # Health check
GET  /api/v1/system/stats              # Dashboard statistics
```

---

## Frontend Pages

| Page | Description |
|------|-------------|
| **Dashboard** | Live stats, risk status card, market overview, quick actions |
| **Markets** | Browse 3,200+ markets with search, filters, sort, grid/list views |
| **Signals** | All active signals with direction arrows, Kelly sizing, quality tiers, model agreement indicators |
| **ML Models** | Training methodology, Brier scores, ablation study, feature importance, signal accuracy charts |
| **Market Detail** | Price chart, ensemble breakdown, quality gate checklist, trading signal card, AI analysis |
| **Analytics** | Correlation matrix, sentiment gauge, orderbook depth, position heatmap |
| **Portfolio** | Open/closed positions, P&L tracking (correct for both YES and NO sides) |

---

## Project Structure

```
prediction-market-analysis/
├── api/                    # FastAPI application
│   ├── routes/             # 25+ API endpoints
│   └── main.py             # App factory + lifespan
├── arbitrage/              # Arbitrage detection + fee calculator (with slippage)
├── config/                 # Settings, constants, risk limits
├── data_pipeline/          # Background pipeline (asyncio)
│   ├── collectors/         # Polymarket, Kalshi data fetchers
│   ├── copy_engine.py      # Copy trading auto-replication
│   └── scheduler.py        # Pipeline orchestrator
├── db/                     # SQLAlchemy models (16 tables), migrations
├── execution/              # Paper trading automation
│   └── paper_executor.py   # Auto-trade from high-confidence signals
├── ml/                     # Machine learning
│   ├── features/           # 25 engineered features + quality metadata
│   ├── models/             # Ensemble (calibration + XGB + LGB)
│   ├── strategies/         # Edge detection (ensemble + Elo)
│   ├── evaluation/         # Signal accuracy tracker + resolution scorer
│   └── saved_models/       # Trained model artifacts + model card
├── risk/                   # Risk management
│   └── risk_manager.py     # Position limits, exposure caps, circuit breaker
├── ai_analysis/            # Claude integration + caching
├── frontend/               # React 19 + TypeScript + Vite 7 + Tailwind 4
│   └── src/
│       ├── pages/          # 8 pages
│       └── components/     # Charts, skeletons, error states
├── scripts/                # Training, backfill, Elo builder
└── tests/                  # Test suite
```

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.13, FastAPI, SQLAlchemy 2 (async), aiosqlite |
| ML | scikit-learn (isotonic), XGBoost, LightGBM, Glicko-2 |
| Frontend | React 19, TypeScript, Vite 7, Tailwind CSS 4, Recharts |
| AI | Claude Opus 4.6 via Anthropic SDK (user-triggered, cached) |
| Data | Polymarket Gamma + CLOB APIs, Kalshi REST API |

## Cost

| Component | Cost |
|-----------|------|
| Data collection | $0 (free APIs) |
| ML training + inference | $0 (local compute) |
| Arbitrage detection | $0 (pure math) |
| Claude analyses | ~$0.15/analysis (cached forever) |
| **Monthly estimate** | **$15-50** |

---

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- **Academic Research**: [AFT 2025 NegRisk paper](https://arxiv.org/abs/2501.05602), [Hediger et al. 2022](https://www.sciencedirect.com/science/article/pii/S0169207021001679), [Polymarket calibration](https://arxiv.org/abs/2409.18044)
- **Platforms**: Polymarket, Kalshi, Anthropic
- **Hackathon**: "Built with Opus 4.6: Claude Code Hackathon" (Feb 10-16, 2026)

---

**Disclaimer:** This platform is for educational and research purposes. Prediction market trading involves financial risk. Not financial advice.
