# PredictFlow — Technical Documentation

> **Purpose**: Architecture, APIs, data pipeline, ML, execution, and how to extend.  
> **Audience**: Engineers working on the codebase.  
> **Last updated**: 2026-02-28  
> **Update cadence**: Refresh after every few hours of work (see [Documentation Index](DOCUMENTATION_INDEX.md)).  
> **Audit/fixes**: See [AUDIT_AND_IMPLEMENTATION.md](AUDIT_AND_IMPLEMENTATION.md) for train/serve parity, feature fixes, and L4/L6/L8 implementation details.

---

## Table of Contents

1. [System overview](#1-system-overview)
2. [Directory structure](#2-directory-structure)
3. [Configuration and environment](#3-configuration-and-environment)
4. [Database](#4-database)
5. [Data pipeline](#5-data-pipeline)
6. [API](#6-api)
7. [ML stack](#7-ml-stack)
8. [Execution and risk](#8-execution-and-risk)
9. [Frontend](#9-frontend)
10. [Deployment](#10-deployment)
11. [How to extend](#11-how-to-extend)

---

## 1. System overview

- **Backend**: Python 3.13, FastAPI, SQLAlchemy 2 (async), Pydantic. Runs on Railway (or local).
- **Database**: PostgreSQL in production (Railway), SQLite for local dev (`data/markets.db`). Async via asyncpg/aiosqlite.
- **Pipeline**: Single process; pipeline runs as an asyncio loop started in FastAPI lifespan. No separate worker.
- **Frontend**: React 19, TypeScript, Vite 7, Tailwind CSS 4. Deployed on Vercel.
- **Data sources**: Polymarket Gamma + CLOB, Kalshi REST API, GDELT (news), Polymarket leaderboard (copy trading).

Flow: **Collect** (markets, prices, orderbooks) → **Predict** (ensemble) → **Signal** (multiple strategies) → **Execute** (paper) → **Score** (resolution/scoring).

---

## 2. Directory structure

```
prediction-market-analysis/
├── api/                    # FastAPI app
│   ├── main.py              # App factory, lifespan, CORS, pipeline start/stop
│   └── routes/              # Route handlers (see Section 6)
├── arbitrage/               # Cross-platform and single-market arb
│   ├── engine.py
│   ├── strategies/          # single_market, cross_platform
│   └── fee_calculator.py
├── ai_analysis/             # Claude integration, SHA-256 cache
├── config/
│   ├── settings.py          # Pydantic BaseSettings (env)
│   └── constants.py         # Fee schedules, URLs, thresholds
├── data_pipeline/
│   ├── scheduler.py         # Main loop: collect, match, arb, signals, score
│   ├── collectors/          # Polymarket, Kalshi, GDELT, traders, fee_rates, etc.
│   ├── streams.py           # WebSocket price cache, Polymarket stream
│   ├── copy_engine.py       # Copy-trading position replication
│   └── storage.py           # Bulk upsert helpers
├── db/
│   ├── database.py          # Async engine, session factory, init_db
│   └── models.py            # SQLAlchemy ORM (17+ tables)
├── execution/
│   ├── paper_executor.py    # Auto-open/close paper positions
│   └── auto_closer.py       # Stop-loss, edge invalidation, stale close
├── ml/
│   ├── features/            # training_features.py (18 features, load_serving_context, pruning)
│   ├── models/              # ensemble, xgboost, lightgbm, calibration, Glicko-2
│   ├── strategies/          # Edge detectors, market_making (AS), intra_market_arb, etc. (Section 7)
│   └── evaluation/          # signal_tracker, resolution_scorer, confidence_adjuster, validation_gates, execution_simulator (seeded), tradability_backtest, model_monitor
├── risk/
│   └── risk_manager.py      # Position/exposure/daily limits, circuit breaker
├── frontend/                # React 19 + Vite 7 + Tailwind 4
│   └── src/pages/           # 12 pages
├── scripts/                  # train_ensemble.py, build_elo_ratings.py, validate_deployment, backfill_all_prices,
│                             # backtest_market_making.py, run_paper_mm.py, collect_orderbooks_daemon, data_coverage_report, etc.
└── docs/                    # All .md docs
```

Additional modules present in repo: `execution/clob_manager.py`, `ml/strategies/endgame_maker.py`, `ml/strategies/ml_informed_mm.py`, `ml/evaluation/deployment_gate.py`, `data_pipeline/collectors/nba_results.py`, `scripts/backtest_endgame.py`, `scripts/build_elo_ratings_nba.py`, and other backfill/backtest/collection scripts. See repo tree for full list.

---

## 3. Configuration and environment

- **Entry point**: `config.settings.Settings` (Pydantic BaseSettings). Loads from env and `.env`.
- **Key vars**: `DATABASE_URL`, `ANTHROPIC_API_KEY`, `POLYMARKET_*`, `KALSHI_*`, pipeline intervals, risk limits.
- **Defaults**: See `config/settings.py` (e.g. `max_position_usd=100`, `max_total_exposure_usd=1000`, `max_loss_per_day_usd=50`).
- **Example**: `.env.example` lists all optional env vars; copy to `.env` and fill.

---

## 4. Database

- **Engine**: Async SQLAlchemy; PostgreSQL (production) or SQLite (local). URL normalized in settings for asyncpg.
- **Tables** (from `db/models.py`):

| Table | Purpose |
|-------|---------|
| platforms | Polymarket, Kalshi config |
| markets | Active/resolved markets, prices, volume, resolution, fee fields |
| price_snapshots | Time-series prices per market |
| orderbook_snapshots | Orderbook state per market |
| trades | (Future) individual trade log |
| cross_platform_matches | Polymarket–Kalshi TF-IDF matches |
| news_events | GDELT / news |
| market_relationships | (Future) logical constraints |
| arbitrage_opportunities | Detected arb opportunities |
| ml_predictions | Model outputs per market |
| ai_analyses | Cached Claude analyses (hash-keyed) |
| portfolio_positions | Paper/live positions |
| system_metrics | Pipeline health |
| auto_trading_configs | Per-strategy config (ensemble, elo, etc.) |
| trader_profiles | Copy-trading leaderboard |
| followed_traders | User → trader follow |
| copy_trades | Replicated positions |
| trader_activities | Activity feed |
| ensemble_edge_signals | Ensemble strategy signals |
| elo_ratings | Glicko-2 ratings |
| strategy_signals | Unified strategy signal log |
| elo_edge_signals | Elo strategy signals |
| favorite_longshot_edge_signals | Favorite-longshot signals |

- **Migrations**: No Alembic in tree; schema changes are applied via `init_db()` (create tables). For production, consider adding migrations.

---

## 5. Data pipeline

- **Owner**: `data_pipeline/scheduler.py`. Started in `api/main.py` lifespan; runs `run_pipeline_loop()`.
- **Main tasks** (conceptually):
  - **Market refresh**: Polymarket Gamma, Kalshi markets (and optional historical/price collectors).
  - **Prices**: Polymarket CLOB (and optional `polymarket_prices`), Kalshi; stored as `PriceSnapshot`.
  - **Orderbooks**: CLOB/Kalshi; stored as `OrderbookSnapshot`.
  - **Cross-platform matching**: TF-IDF on questions; populate `cross_platform_matches`.
  - **Arbitrage scan**: Single-market (YES+NO &lt; $1) and cross-platform; fee-aware; write `arbitrage_opportunities`.
  - **Trader refresh**: Leaderboard + positions for copy trading; `trader_profiles`, activities.
  - **Ensemble / Elo / other strategies**: Generate edge signals; paper executor consumes them.
  - **Resolution scoring**: Compare signals to resolved outcomes; update accuracy/Brier.
  - **Model monitor**: Record resolved predictions, run drift/Brier/edge checks, emit retrain signals — `ml/evaluation/model_monitor`.
  - **Confidence adjuster**: Adaptive confidence (Phase 2.5) — `ml/evaluation/confidence_adjuster`.
  - **Cleanup**: Old snapshots/metrics.
- **Collectors** (under `data_pipeline/collectors/`):  
  `polymarket_gamma`, `polymarket_clob`, `polymarket_prices`, `kalshi_markets`, `kalshi_historical`, `fee_rates`, `gdelt_news`, `trader_data`, `polymarket_resolved`, `kalshi_resolved`, `ufc_results`, `sports_results`, `polymarket_subgraph`.
- **Streaming**: `data_pipeline/streams.py` — `PriceCache`, `PolymarketStream` for real-time prices (used if enabled in pipeline).

---

## 6. API

- **Base path**: `/api/v1/`. All endpoints return JSON.
- **Route modules** and main handlers:

| Module | Key endpoints |
|--------|----------------|
| system | `GET /health`, `GET /system/stats`, websocket status, confidence adjuster stats |
| markets | `GET /markets`, `GET /markets/categories`, `GET /markets/{id}`, price history, orderbook, news |
| ml_predictions | `GET /predictions/{id}`, ensemble prediction, top mispriced, top edges, calibration curve, accuracy backtest |
| strategy_signals | `GET /strategies/signals`, signal performance, new strategy signals, performance by strategy, ensemble edges |
| portfolio | List/open/close positions, summary, equity curve, win rate by strategy, risk status, reset |
| arbitrage | `GET /arbitrage/opportunities`, history |
| elo_sports | Ratings, player rating, predict tennis, Elo edges, trigger edge scan |
| ai_analysis | `POST /analyze/{id}`, cached analysis, cost summary |
| copy_trading | Traders, follow/unfollow, following, copy performance, activity, positions, equity, drawdown |
| auto_trading | Get/update configs, toggle strategy, status |
| admin | Reset auto portfolio, init auto trading, backfill traders |
| analytics | Market correlations |

- **Auth**: Optional `X-User-Id` header for portfolio/copy-trading user context; no required auth for read-only.
- **Full reference**: `docs/API.md`.

---

## 7. ML stack

### Models

- **Ensemble**: Isotonic + XGBoost + LightGBM; weights (e.g. ~49% XGB, ~49% LGB post-retrain). Trained with temporal 60/20/20 split; `as_of_days` and tradeable-range filter applied consistently.
- **Features**: 18 in `ENSEMBLE_FEATURE_NAMES` (`ml/features/training_features.py`). **Removed**: `log_open_interest` (leaky), `cross_platform_spread` (always 0). **Pruning**: “Near-constant” = dominant value &gt; 97% (not unique-ratio &lt; 5%, which had killed binary features). Momentum and orderbook features require snapshots/orderbook at as_of.
- **Serving parity**: Callers must load **serving context** before prediction: `load_serving_context(session, market_id)` returns recent price snapshots and latest orderbook; pass into `ensemble.predict_market(market, price_snapshots=..., orderbook_snapshot=...)`. Used in scheduler, `ml_predictions` routes, and `validate_deployment.py`.
- **Training**: `scripts/train_ensemble.py` — `--tradeable-range 0.10,0.90`, `--as-of-days 1`, optional `--snapshot-only`. Saves to `ml/saved_models/` and writes `monitor_state.json` baseline for model monitor.

### Strategies (signal generators)

- **ensemble_edge_detector**: Uses ensemble + fee model + Kelly; quality tiers, edge decay.
- **elo_edge_detector**: Glicko-2 tennis; Elo-implied vs market price.
- **favorite_longshot_detector**: Favorite-longshot bias edge.
- **smart_money**: Heuristic (not full on-chain).
- **intra_market_arbitrage**: YES+NO &lt; $1; **uses orderbook best ask** when available (not mid-prices) for realistic arb cost; fallback to mid when no orderbook.
- **market_making**: Avellaneda-Stoikov (2008) adapted for prediction markets — `ml/strategies/market_making.py`; reservation price, optimal spread, inventory skew, maker rebates, kill switch, backtest.
- **market_clustering**, **signal_consensus**, **orderflow_analyzer**: Noisy; disabled or deprioritized.
- **Others**: `llm_forecaster`, `longshot_bias`, `news_catalyst`, `resolution_convergence`.

### Evaluation

- **signal_tracker**: Hit rate, Brier, simulated P&L for resolved signals.
- **resolution_scorer**: Score ensemble/Elo vs actual resolutions; feeds model monitor.
- **confidence_adjuster**: Adaptive confidence (Phase 2.5).
- **validation_gates**, **execution_simulator** (seeded RNG for determinism), **tradability_backtest**: Pre-live validation (L5).
- **model_monitor**: Drift (KS), rolling Brier, edge decay, model age, retrain signals; state in `ml/saved_models/monitor_state.json`; integrated in train pipeline (write baseline) and scheduler (record predictions, run checks).

---

## 8. Execution and risk

- **Paper executor** (`execution/paper_executor.py`): Opens/closes paper positions from strategy signals; uses per-strategy `AutoTradingConfig` (min_confidence, min_net_ev, max_kelly, stop_loss_pct, limits). Fee-aware P&L; correct YES/NO cost and exposure (see CHANGELOG 0.4.x).
- **Auto-closer** (`execution/auto_closer.py`): Stop-loss, edge invalidation (e.g. 10% deviation), stale unprofitable timeout (e.g. 24h). Can be tuned per roadmap.
- **Risk manager** (`risk/risk_manager.py`): Max position size, max total exposure, daily loss circuit breaker, daily trade cap. Enforced on `POST /portfolio/positions` and by executor.

---

## 9. Frontend

- **Stack**: React 19, TypeScript, Vite 7, Tailwind CSS 4, TanStack Query, Recharts.
- **Pages**: Dashboard, Markets, Market Detail, Signals Hub, ML Models, Portfolio, Copy Trading, Calibration, Correlation, Data Quality, etc. (see README).
- **API client**: Axios with `/api/v1` base (proxy in dev to backend).
- **Docs**: UX_IMPROVEMENTS.md, README “Frontend” section.

---

## 10. Deployment

- **Backend**: Railway (or `uvicorn api.main:app`). Needs `DATABASE_URL` (PostgreSQL on Railway), env vars for Polymarket/Kalshi/Anthropic if used.
- **Frontend**: Vercel; build from `frontend/`; env points to backend API URL.
- **Guides**: `docs/DEPLOYMENT.md`, `docs/RAILWAY_SETUP.md`, `docs/PRODUCTION_READINESS.md`.

---

## 11. How to extend

- **New strategy**: Add a detector in `ml/strategies/` (or re-use strategy_signals table); register in scheduler and (if auto-traded) in `AutoTradingConfig` and paper executor.
- **New API**: Add route in `api/routes/` and register in `main.py` under the v1 router.
- **New collector**: Add module in `data_pipeline/collectors/`, call from `scheduler.py` at desired interval.
- **New DB table**: Add model in `db/models.py`; ensure `init_db()` creates it (or add migration).
- **New feature for ML**: Add in `ml/features/training_features.py`; retrain with `scripts/train_ensemble.py`; ensure serving uses same feature logic. Any new code that calls `ensemble.predict_market()` must call `load_serving_context(session, market.id)` and pass `price_snapshots` and `orderbook_snapshot` for train/serve parity.

---

*Keep this file in sync with the codebase after each work session. For business context and “when we’re done,” see [BUSINESS_DOCUMENTATION.md](BUSINESS_DOCUMENTATION.md).*
