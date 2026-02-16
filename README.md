# PredictFlow

> **Quantitative prediction market analysis platform — ML ensemble models, automated paper trading, and multi-strategy signal generation**

**[Live Demo](https://prediction-market-analysis-one.vercel.app)** | **[API](https://prediction-market-analysis-production-e6fa.up.railway.app/api/v1/system/stats)**

Built for the **"Built with Claude: Claude Code Hackathon"** (Feb 10-16, 2026)

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![React 19](https://img.shields.io/badge/react-19-61dafb.svg)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## The Problem

Prediction markets are the most accurate forecasting tool we have — but trading them profitably requires quantitative infrastructure that doesn't exist for retail users. Prices are noisy, fees eat edges, and most "mispricing" signals are statistical noise.

## What PredictFlow Does

PredictFlow is a **full-stack quantitative trading platform** that continuously collects data from Polymarket and Kalshi, runs ML models against every active market, generates fee-adjusted trading signals, and auto-executes paper trades with proper risk management.

It doesn't just show prices — it answers: **"Is this market mispriced, by how much after fees, and how much should I bet?"**

---

## Key Results

### ML Ensemble Performance

Trained on **4,341 resolved markets** with strict temporal split (no future data leakage):

| Metric | Ensemble | Market Baseline | Improvement |
|--------|----------|-----------------|-------------|
| **Brier Score** | **0.0662** | 0.0843 | **+21.5%** |
| **AUC-ROC** | **0.9329** | — | — |
| **Tradeable Range (20-80%)** | **0.1349** | 0.1722 | **+21.7%** |

The 20-80% price bucket is where the real money is — markets near 0% or 100% are easy to predict but untradeable. Our model shows a **21.7% Brier improvement in exactly the price range that matters for trading.**

### Honesty in Metrics

We discovered and fixed data leakage mid-hackathon. Our Brier score went from 0.054 (contaminated) to 0.066 (clean). We documented the entire process in our [CHANGELOG](CHANGELOG.md) — the degradation *proves* the audit was correct.

- Edges >15% are auto-flagged as **speculative** and never auto-traded
- Post-calibrator was disabled because it *hurt* performance (0.068 > 0.066)
- Quality gates were tested and found to *not* add value — so we documented that too

---

## How It Works

```
                    ┌────────────────────────────────────────┐
                    │    React 19 Frontend (Vercel)           │
                    │  Dashboard · Markets · Signals ·        │
                    │  ML Models · Portfolio · Copy Trading    │
                    └──────────────────┬─────────────────────┘
                                       │ REST API (32+ endpoints)
┌──────────────────────────────────────┴───────────────────────────────┐
│                     FastAPI Backend (Railway)                         │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────┐ │
│  │ Data Pipeline │  │ ML Ensemble  │  │  Strategy   │  │Execution │ │
│  │  (asyncio)   │  │  (3 models)  │  │  Signals    │  │ Engine   │ │
│  │              │  │              │  │             │  │          │ │
│  │ · Prices     │  │ · Isotonic   │  │ · Ensemble  │  │ · Paper  │ │
│  │ · Orderbooks │  │ · XGBoost    │  │ · Elo/      │  │   trades │ │
│  │ · Matching   │  │ · LightGBM   │  │   Glicko-2  │  │ · Copy   │ │
│  │ · News       │  │ · 19 features│  │ · Arbitrage │  │   trading│ │
│  └──────┬───────┘  └──────┬───────┘  │ · Quality   │  │ · Risk   │ │
│         │                  │          │   gates     │  │   mgmt   │ │
│         │                  │          └──────┬──────┘  └────┬─────┘ │
│  ┌──────┴──────────────────┴────────────────┴───────────────┴─────┐ │
│  │              PostgreSQL (17 tables, asyncpg)                    │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
         │                    │                    │
    ┌────┴─────┐        ┌────┴────┐          ┌────┴────┐
    │Polymarket │        │  Kalshi │          │ Claude  │
    │Gamma+CLOB │        │REST API │          │Opus 4.6 │
    └──────────┘        └─────────┘          └─────────┘
```

### The Pipeline (runs continuously)

1. **Collect** — Fetch prices from Polymarket CLOB + Kalshi every 20s (~2,000 markets)
2. **Predict** — Run 3-model ensemble on all active markets every few minutes
3. **Signal** — Generate fee-aware, Kelly-sized signals with quality tiers
4. **Execute** — Auto-open paper trades for high-confidence signals (>0.6 confidence, >7% net EV)
5. **Close** — Stop-loss at 5%, auto-close on signal expiry, fee-aware P&L
6. **Score** — Track every signal against actual resolution for rolling Brier/hit-rate

---

## Five Trading Strategies

### 1. ML Ensemble Edge Detection
3-model blend finds markets where predicted probability diverges from market price. Every signal includes direction, net EV (after 2% Polymarket fee + 1% slippage), Kelly fraction, and quality tier. **Edge decay with 2-hour half-life** prevents stale signals from executing.

### 2. Elo Sports Pricing (Glicko-2)
Custom Glicko-2 engine for tennis player ratings. When Elo-implied win probability differs from market price by more than fees, generate a signal. Surface-specific ratings (hard, clay, grass).

### 3. Cross-Platform Arbitrage
TF-IDF matching between Polymarket and Kalshi markets, then fee-aware spread detection. Most "opportunities" vanish after honest fee accounting — we show that transparently.

### 4. Copy Trading
Real trader data from Polymarket leaderboard (not synthetic). Follow top traders, auto-replicate positions with configurable sizing, activity feed in real-time.

### 5. Claude Deep Analysis
On-demand AI analysis via Claude Opus 4.6 with extended thinking. Cached by SHA-256 prompt hash — never pays twice for the same analysis. ~$0.15/market.

---

## Frontend (12 pages)

| Page | What It Shows |
|------|---------------|
| **Dashboard** | Live stats, risk status bars, market overview, pipeline status |
| **Markets** | Browse 3,000+ markets with search, filters, categories, grid/list views |
| **Market Detail** | Price chart, ensemble breakdown, quality gates, AI analysis trigger |
| **Signals Hub** | All active signals — direction arrows, Kelly sizing, quality tier pills, model agreement |
| **ML Models** | Training methodology, Brier scores, feature importance, signal accuracy over time |
| **Portfolio** | Open/closed positions, unrealized P&L (correct for both YES and NO sides) |
| **Copy Trading** | Top trader leaderboard, follow/unfollow, position replication |
| **Calibration** | Model calibration curve visualization |
| **Correlation** | Interactive market correlation matrix with zoom/pan |
| **Data Quality** | System health monitoring, pipeline metrics |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 19, TypeScript, Vite 7, Tailwind CSS 4, Recharts, TanStack Query |
| **Backend** | Python 3.13, FastAPI, SQLAlchemy 2 (async), Pydantic |
| **ML** | scikit-learn (isotonic regression), XGBoost, LightGBM, Glicko-2 |
| **AI** | Claude Opus 4.6 via Anthropic SDK (user-triggered, cached) |
| **Database** | PostgreSQL (Railway, asyncpg) / SQLite (local dev) |
| **Data Sources** | Polymarket Gamma + CLOB APIs, Kalshi REST API, GDELT news |
| **Deployment** | Vercel (frontend) + Railway (backend + PostgreSQL) |

---

## API (32+ endpoints)

```
Markets          GET  /markets, /markets/{id}, /markets/categories
ML Predictions   GET  /predictions/{id}, /predictions/top/mispriced, /predictions/accuracy/backtest
Signals          GET  /strategies/signals, /strategies/ensemble-edges, /strategies/signal-performance
Elo              GET  /elo/ratings, /elo/predict/tennis, /elo/edges
Portfolio        GET  /portfolio/positions, /portfolio/summary, /portfolio/risk-status
                 POST /portfolio/positions, /portfolio/positions/{id}/close
Auto-Trading     GET  /auto-trading/status
Arbitrage        GET  /arbitrage/opportunities, /arbitrage/history
AI Analysis      POST /analyze/{id}  |  GET /analyze/{id}/cached
Copy Trading     GET  /copy-trading/traders, /copy-trading/traders/{id}
System           GET  /system/stats, /health
```

All endpoints prefixed with `/api/v1/`.

---

## Ensemble Model Details

**3-model blend** with temporal train/test split at 2026-02-13:

| Component | Weight | Role |
|-----------|--------|------|
| Isotonic Regression | 20% | Probability calibration |
| XGBoost | 40% | Non-linear feature interactions |
| LightGBM | 40% | Gradient boosting (complementary to XGB) |

**19 features extracted** per market (7 survive automatic pruning):

| Feature | XGB Importance | Source |
|---------|---------------|--------|
| `log_open_interest` | 41% | Liquidity signal |
| `volatility_20` | 27% | 20-period price volatility |
| `price_yes` | 16% | Market consensus price |
| `time_to_resolution_hrs` | 8% | Temporal decay |
| `price_distance_from_50` | 5% | Uncertainty measure |
| `return_1h` | 2% | Short-term momentum |
| `zscore_24h` | 1% | Deviation from rolling mean |

### Execution Engine

- **Edge decay**: 2-hour half-life on Kelly sizing (stale signals get smaller bets)
- **Confidence scoring**: Penalizes edges >8% (drops to 0 at 15% — real markets don't have 30% edges)
- **Risk management**: $100 max position, $500 exposure cap, $25 daily loss circuit breaker
- **Fee model**: 2% on net winnings only (Polymarket fee structure), applied asymmetrically in Kelly formula
- **Dual portfolios**: Manual (human + copy trades) vs Auto (ML signals), isolated risk budgets

---

## Quick Start (Local Development)

```bash
# Clone and setup
git clone https://github.com/mian-abd/prediction-market-analysis.git
cd prediction-market-analysis

# Backend
python -m venv venv
.\venv\Scripts\Activate.ps1          # Windows
# source venv/bin/activate            # Linux/Mac
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000

# Frontend (new terminal)
cd frontend
npm install
npm run dev                           # http://localhost:5173
```

### Optional: Train ML Models

```bash
python scripts/train_ensemble.py              # Train ensemble (needs resolved markets)
python scripts/build_elo_ratings.py --export-db  # Build Elo ratings
python scripts/backfill_price_history.py      # Backfill price history from CLOB API
```

---

## Project Structure

```
prediction-market-analysis/
├── api/                    # FastAPI app + 32 route handlers
├── arbitrage/              # Cross-platform arb detection + fee calculator
├── ai_analysis/            # Claude integration + SHA-256 caching
├── config/                 # Settings, risk limits
├── data_pipeline/          # Background pipeline (asyncio scheduler)
│   ├── collectors/         # Polymarket Gamma/CLOB, Kalshi, GDELT, trader data
│   ├── copy_engine.py      # Copy trading position replication
│   └── scheduler.py        # Orchestrator (prices → signals → trades → scoring)
├── db/                     # SQLAlchemy 2 models (17 tables)
├── execution/              # Paper executor + auto-closer (stop-loss, fee-aware P&L)
├── ml/
│   ├── features/           # 19 engineered features + automatic pruning
│   ├── models/             # Ensemble (Isotonic + XGB + LGB) + Glicko-2 Elo
│   ├── strategies/         # Edge detection + quality gates + Kelly sizing
│   └── evaluation/         # Signal accuracy tracker + resolution scorer
├── risk/                   # Position limits, exposure caps, circuit breaker
├── frontend/               # React 19 + TypeScript + Vite 7 + Tailwind 4
│   └── src/pages/          # 12 pages
├── scripts/                # Training, backfill, Elo builder
└── docs/                   # Architecture, API reference, strategies deep-dive
```

---

## Cost to Run

| Component | Cost |
|-----------|------|
| Data collection (Polymarket, Kalshi, GDELT) | $0 |
| ML training + inference | $0 (CPU) |
| Claude deep analysis | ~$0.15/market (cached) |
| Railway backend + PostgreSQL | ~$5/month |
| Vercel frontend | $0 (free tier) |

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/ARCHITECTURE.md) | System design, data flow, design decisions |
| [API Reference](docs/API.md) | Full endpoint documentation with examples |
| [Strategies](docs/STRATEGIES.md) | Deep-dive on all 5 trading strategies |
| [Changelog](CHANGELOG.md) | Version history including data leakage discovery + fix |

---

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- **Academic**: [AFT 2025 NegRisk paper](https://arxiv.org/abs/2501.05602), [Hediger et al. 2022](https://www.sciencedirect.com/science/article/pii/S0169207021001679), [Polymarket calibration](https://arxiv.org/abs/2409.18044)
- **Platforms**: Polymarket, Kalshi, Anthropic
- **Hackathon**: "Built with Claude: Claude Code Hackathon" (Feb 10-16, 2026)

---

*This platform is for educational and research purposes. Prediction market trading involves financial risk. Not financial advice.*
