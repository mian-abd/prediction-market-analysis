# PredictFlow — Business Documentation

> **Purpose**: Business and product view: what we built, why, current state, roadmap, and when we're done.  
> **Audience**: Product owner, stakeholders, future you.  
> **Last updated**: 2026-02-28  
> **Update cadence**: Refresh after every few hours of work (see [Documentation Index](DOCUMENTATION_INDEX.md)).

### Implementation status (2026-02-28)

| Layer | Status | Notes |
|-------|--------|--------|
| L1 Stop the bleeding | **COMPLETE** | Disabled noise strategies, execution bugs fixed |
| L2 Honest measurement | **COMPLETE** | Train/serve parity, fee model, pruning gate, as_of_days (see [AUDIT_AND_IMPLEMENTATION.md](AUDIT_AND_IMPLEMENTATION.md)) |
| L3 Data infrastructure | **COMPLETE** | Fee rates, Kalshi historical, price history collectors |
| L4 Market making engine | **COMPLETE** | Avellaneda-Stoikov core in `ml/strategies/market_making.py` |
| L5 Validation framework | **COMPLETE** | 5-gate validation, execution simulator (seeded RNG), tradability backtest |
| L6 Retrain ensemble | **COMPLETE** | 19.1% Brier improvement, 4/5 gates, $72 profit sim |
| L7 Optimism tax | **DEFERRED** | Requires live market data collection period |
| L8 Long-term infra | **COMPLETE** | Model monitor: drift detection, retrain triggers (`ml/evaluation/model_monitor.py`) |

---

## Table of Contents

1. [Product vision and goals](#1-product-vision-and-goals)
2. [What PredictFlow does today](#2-what-predictflow-does-today)
3. [Current business metrics](#3-current-business-metrics)
4. [Why the system loses money (root causes)](#4-why-the-system-loses-money-root-causes)
5. [Profitability roadmap (layers)](#5-profitability-roadmap-layers)
6. [Success criteria and “done” definition](#6-success-criteria-and-done-definition)
7. [References to other docs](#7-references-to-other-docs)

---

## 1. Product vision and goals

### Vision

**PredictFlow** is a quantitative prediction-market analysis and (paper/live) trading platform. The goal is to answer: *“Is this market mispriced, by how much after fees, and how much should I bet?”* — and eventually to trade profitably after costs.

### Goals

| Goal | Status | Notes |
|------|--------|------|
| Demo at hackathon | ✅ Done | Built with Claude hackathon (Feb 10–16, 2026) |
| Honest metrics (no data leakage) | ✅ Addressed | Audit done; clean features; Brier degraded as expected |
| Paper trading with risk controls | ✅ Done | Paper executor, risk manager, auto-closer |
| Profitable after fees (real or paper) | ❌ Not yet | System is net negative; roadmap below |
| Production-ready for real money | ⏳ 60–90 days | Per EXECUTIVE_SUMMARY and PROFITABILITY_ROADMAP |

### Target users

- Researchers / quants exploring prediction markets  
- Users who want fee-aware signals and backtested accuracy  
- (Future) Users who want automated or copy trading with clear risk limits  

---

## 2. What PredictFlow does today

### High level

- **Collects** data from Polymarket (Gamma + CLOB) and Kalshi (REST).
- **Predicts** with an ML ensemble (Isotonic + XGBoost + LightGBM) on active markets.
- **Signals** multiple strategies: ensemble edges, Elo (sports), arbitrage, copy trading, etc.
- **Executes** paper trades with Kelly sizing, risk limits, and fee-aware P&L.
- **Scores** signals against resolutions (Brier, hit rate, simulated P&L).

### Delivered capabilities

| Capability | Description |
|------------|-------------|
| **ML ensemble** | 3-model blend, 18 features (log_open_interest/cross_platform_spread removed), train/serve parity via `load_serving_context`, temporal split, 19.1% Brier improvement over market baseline (post-audit retrain) |
| **Strategies** | Ensemble edge, Elo/Glicko-2 (tennis), cross-platform arb, **intra-market arb (orderbook ask/bid)**, **market making (Avellaneda-Stoikov)**, copy trading, favorite-longshot, smart money, market clustering, consensus, orderflow (several disabled or phased out) |
| **Execution** | Paper executor, auto-closer (stop-loss, edge invalidation, stale timeout), risk manager (position/exposure/daily limits); **execution simulator** (seeded RNG) for validation |
| **Model monitoring** | Drift detection (KS, rolling Brier, edge decay), retrain triggers, baseline from training; integrated in scheduler and train pipeline |
| **Frontend** | Dashboard, Markets, Market Detail, Signals Hub, ML Models, Portfolio, Copy Trading, Calibration, Correlation, Data Quality |
| **API** | 32+ REST endpoints under `/api/v1/` for markets, predictions, signals, portfolio, arbitrage, Elo, copy trading, system |
| **Data pipeline** | Scheduler for prices, orderbooks, markets, arbitrage scan, cross-platform matching, trader refresh, resolution scoring, confidence adjuster, **model monitor** |

### What we explicitly do *not* promise yet

- Real-money profitability.  
- “Production-ready” for live capital without the 60–90 day hardening in the roadmap.

---

## 3. Current business metrics

### Platform P&L (pre-fix audit baseline)

| Metric | Value |
|--------|--------|
| Total P&L | **-$358.30** (historical; pre–honest retrain) |
| Win rate | **13.5%** (488 trades) |
| Sharpe ratio | **-10.66** |
| Max drawdown | **-949.66%** |
| Calmar ratio | **-0.38** |

*Post-audit: fee model fixed (0% for fee-free markets; Polymarket 2% profit fee removed per docs). Retrained ensemble uses honest features and serving path; see “ML ensemble (post-audit)” below.*

### Per-strategy breakdown (from roadmap audit)

| Strategy | Trades | Win rate | P&L | Root cause (summary) |
|----------|--------|----------|-----|----------------------|
| market_clustering | 255 | 9.0% | -$184.63 | Spurious correlations, ~10 data points |
| consensus | 27 | 7.4% | -$100.89 | Amplifies bad signals (e.g. 1.5x Kelly) |
| ensemble | 160 | 23.8% | -$38.54 | Train/serve feature mismatch (fixed in L2) |
| orderflow | 38 | 2.6% | -$35.17 | Equity OBI not applicable here |
| smart_money | 8 | 25.0% | -$0.60 | Heuristic; not real on-chain data |

### ML ensemble (post-audit retrain, 2026-02-28)

- **Brier**: Ensemble 0.0878 vs market baseline 0.1086 → **19.1% improvement**.  
- **Training**: 7,671 markets, tradeable range 10–90%, as_of_days=1; no settlement-price leakage.  
- **Profit simulation**: $72.55 over 1,535 trades, 75% win rate (fee-free + 1.5% slippage).  
- **Validation gates**: 4/5 passed; calibration quality gate fails (19.8% max bin deviation &gt; 10%).  
- **Top features**: price_bucket (33%), price_yes (21%), calibration_bias (16%), volatility_20 (8%), category_encoded (8%).  
- **Fee model**: Polymarket profit fee confirmed removed; per-market `taker_fee_bps` used for fee-enabled markets only.

---

## 4. Why the system loses money (root causes)

Summary from [PROFITABILITY_ROADMAP.md](PROFITABILITY_ROADMAP.md). **Many of these have been addressed in the 2026-02-28 audit**; see [AUDIT_AND_IMPLEMENTATION.md](AUDIT_AND_IMPLEMENTATION.md).

1. **Directional taker disadvantage**  
   Research (e.g. Becker 2026, 72.1M Kalshi trades): takers lose ~1.12% per trade on average; makers gain. We mostly take.

2. **Noisy strategies**  
   market_clustering, consensus, orderflow produce most trades with very low win rates; they are being disabled or deprioritized.

3. **Training vs serving mismatch**  
   Features like `time_to_resolution_hrs`, `is_weekend`, `return_1h`, `volatility_20`, `zscore_24h` differ (or are zero) at serve time vs training.

4. **Wrong fee model**  
   Hardcoded 2% Polymarket fee; many markets are fee-free. We reject real edges and may mis-size in fee-enabled markets.

5. **Execution bugs**  
   Stale entry price, wrong slippage for NO side, no exit slippage → overstated P&L.

6. **Auto-closer too aggressive**  
   E.g. 10% edge invalidation, 24h stale close, stop-losses that fire on noise.

7. **Evaluation bias**  
   Shuffled K-fold breaks temporal order; Brier/calibration can be optimistic; no tradability/friction check.

Details, tables, and code references are in **docs/PROFITABILITY_ROADMAP.md**.

---

## 5. Profitability roadmap (layers)

Execution order (from roadmap): **L1 → L2 → L3 → L5 (simulator + gates) → L4 (market making) → L6 → L7 → L8**.

| Layer | Name | Purpose | Status |
|-------|------|---------|--------|
| **1** | Stop the bleeding | Disable broken strategies; fix critical bugs (fees, P&L, execution). | **COMPLETE** |
| **2** | Honest measurement | Unified features train/serve; temporal CV; real fee model; pruning gate fix; as_of_days. | **COMPLETE** |
| **3** | Data infrastructure | Tick-level / depth data; Kalshi candlesticks; fee-rate API. | **COMPLETE** |
| **5** | Strategy validation | Execution simulator + validation gates (build before L4). | **COMPLETE** |
| **4** | Market making engine | Two-sided quoting, spread capture (after simulator validates fills). | **COMPLETE** (core AS engine) |
| **6** | Retrain ensemble | Retrain on honest features and data. | **COMPLETE** (19.1% improvement) |
| **7** | Optimism tax | Exploit maker-side / favorite-longshot structural edge. | **DEFERRED** (needs live data period) |
| **8** | Long-term infra | Backtesting, continuous learning, monitoring. | **COMPLETE** (model monitor) |

Full rationale, research refs, and implementation notes: **docs/PROFITABILITY_ROADMAP.md**. Audit and fix details: **docs/AUDIT_AND_IMPLEMENTATION.md**.

---

## 6. Success criteria and “done” definition

### When is the project “done” (current definition)

Use this as a checklist; update as goals change.

**Phase 1 — Demo / research (current)**  
- [x] End-to-end pipeline: collect → predict → signal → paper execute → score.  
- [x] Honest ML metrics and leakage fixes documented.  
- [x] Frontend and API usable for exploration and paper trading.  
- [ ] **Not done**: Profitable paper P&L over a sustained period.

**Phase 2 — Path to profitability**  
- [x] L1–L3 and L5 implemented (stop bleeding, honest measurement, data infra, validation).  
- [x] L4 (market making core) and L6 (retrain) and L8 (monitoring) implemented.  
- [ ] L7 (Optimism Tax) deferred until live data collection period.  
- [ ] Paper trading shows positive expectancy after fees over 30+ days (profit sim $72 on 1,535 trades is encouraging; live paper run still needed).  
- [ ] Risk limits and circuit breakers validated.

**Phase 3 — Production (optional)**  
- [ ] Real-money pilot (e.g. $1K) with same risk rules.  
- [ ] Monitoring, alerts, and ops runbook.  
- [ ] Documented “done” for v1 production in CHANGELOG or roadmap.

### How to use this doc when you come back

- Re-read **Section 3** and **Section 4** and refresh numbers from the app/DB or latest audit.  
- Tick/untick **Section 6** as you complete roadmap items.  
- After every few hours of work, ask the assistant to **update both BUSINESS_DOCUMENTATION.md and TECHNICAL_DOCUMENTATION.md** so the “done” state and technical state stay in sync.

---

## 7. References to other docs

| Document | Use |
|----------|-----|
| [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) | Master index and update workflow |
| [AUDIT_AND_IMPLEMENTATION.md](AUDIT_AND_IMPLEMENTATION.md) | Deep audit (17 issues, 4 CRITICAL), what was verified, fixes, L4/L6/L8 implementation log |
| [PROFITABILITY_ROADMAP.md](PROFITABILITY_ROADMAP.md) | Full profitability analysis, research, layers, appendices |
| [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md) | Production readiness audit, 60–90 day plan |
| [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md) | Architecture, APIs, pipeline, ML, execution (for engineers) |
| [README.md](../README.md) | Quick start, stack, high-level overview |
| [CHANGELOG.md](../CHANGELOG.md) | Version history and notable fixes |

---

*This file is the single place for business/product view of “what we did and when we’re done.” Keep it updated after each work session.*
