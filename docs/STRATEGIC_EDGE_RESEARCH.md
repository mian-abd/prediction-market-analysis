# Strategic Edge Research: Prediction Markets Deep Dive

**Purpose**: Long-term plan to build sustainable edge in prediction markets (Polymarket, Kalshi). Based on extensive research of academic papers, documentation, and your codebase.

---

## Part 1: How Prediction Markets Actually Work

### Polymarket Mechanics (CLOB)

- **Central Limit Order Book**: Prices emerge from supply/demand, not set by Polymarket
- **Price = Probability**: $0.75 = 75% implied probability
- **Displayed price**: Midpoint of bid-ask; if spread > $0.10, shows last traded price
- **Execution reality**: You pay the **ask** when buying, receive the **bid** when selling — never the midpoint
- **Order types**: GTC, GTD, FOK, FAK — all technically limit orders
- **Settlement**: Offchain matching, onchain settlement; you maintain custody

### Critical Insight: Bid-Ask Matters

Your strategies use **mid-price** (`price_yes`). Real execution uses **ask** (buy) or **bid** (sell). A 2–4% spread is common in thin markets. **You cannot arbitrage or edge-trade without modeling bid-ask and slippage.**

### NegRisk Markets (Polymarket)

- Multi-outcome events where 1 No share converts to 1 Yes in every other outcome
- **Sum > 1.0**: Capital efficiency (lower collateral), NOT arbitrage
- **Sum < 1.0**: True arbitrage — buy all outcomes for < $1, receive $1
- AFT 2025 paper: $29M extracted from NegRisk markets by bots

---

## Part 2: Research-Backed Systematic Edges

### 2.1 Favorite-Longshot Bias (Most Reliable)

**What**: Markets systematically overprice longshots (<20%) and underprice favorites (60–75%).

**Evidence**:
- NFL: <15% teams covered 38% of time; >70% favorites covered 61%
- Fading extreme underdogs (<20%) → +8.2% ROI
- Kalshi: 5¢ contracts win 4.18% (not 5%); 95¢ win 95.83%
- Jonathan Becker: 1¢ YES = -41% EV; 1¢ NO = +23% EV (64pp gap!)

**Your implementation**: Calibration model partially captures this. But `calibration_features.py` uses generic research data — **retrain on YOUR resolved markets** for platform-specific calibration.

**Action**: 
- Add **category-specific calibration** (Sports vs Finance vs Politics)
- Exploit **YES/NO asymmetry**: Prefer BUY NO on overpriced favorites; avoid BUY YES on longshots
- Focus on 60–75% favorites (underpriced) and fade <20% longshots

### 2.2 Recency Bias (News Overreaction)

**What**: Markets overweight last 24h of information. Panic sells create mean-reversion opportunities.

**Evidence**:
- Biden debate June 2024: 38% → 19% in 2h, recovered to 28% in 72h
- Fading the panic → 47% returns
- Information percolation: Short-term momentum, longer-term mean reversion

**Your implementation**: `return_1h`, `volatility_20`, `zscore_24h` exist but are **pruned as near-constant** (6.3% snapshot coverage). This is a **data problem**, not a strategy problem.

**Action**:
- Backfill price history (ROADMAP_90_DAYS.md Task 1.3)
- Add **recency bias detector**: Flag markets with >15% move in 24h
- Strategy: Fade extreme moves when fundamentals (polls, Elo) disagree

### 2.3 Optimism Tax / Maker-Taker Dynamics (Jonathan Becker)

**What**: Takers overpay for YES contracts; makers profit by selling into biased flow. Not forecasting — **structural arbitrage**.

**Evidence**:
- Taker: -1.12% avg excess return; Maker: +1.12%
- 1¢ contracts: Takers win 0.43%; Makers win 1.57%
- **Category gaps**: Finance 0.17pp (efficient) vs Entertainment 4.79pp vs Media 7.28pp
- YES buyers lose 1.02%; NO buyers gain 0.83%

**Your implementation**: None. You are a taker (market orders / crossing spread).

**Action**:
- **Become a maker**: Place limit orders instead of crossing spread
- **Prefer NO side** when edge exists (structural advantage)
- **Category selection**: Avoid Finance (efficient); focus Sports, Entertainment, Media
- Add `category_efficiency_gap` feature (Finance=0.17, Sports=2.23, etc.)

### 2.4 Liquidity Gaps

**What**: Thin markets show wider mispricings but higher execution risk.

**Evidence**:
- Montana Senate: 8% bid-ask, 12-point mispricings vs polls → 3× edges of liquid PA
- Requires limit orders, patience

**Your implementation**: Quality gates (MIN_LIQUIDITY=5K) filter these out.

**Action**:
- **Separate strategy**: "Liquidity gap" — lower liquidity threshold, use limit orders only, smaller size
- Target 2–5K liquidity markets with 8%+ model-market divergence

### 2.5 Arbitrage (What You Have)

**Single-market**: YES+NO < $1. Implemented. Works. Execution is the gap.

**Cross-platform**: Polymarket vs Kalshi. Implemented. Research: opportunities exist 2–15 seconds; need sub-100ms execution for real profit. One bot: $313 → $414K.

**Combinatorial**: Related markets (Trump wins + Republican Senate). **62% failure rate** in practice; only 0.24% of total arb profits. Speed dominates, not analytical complexity. **Skip for now.**

---

## Part 3: Your Codebase — Honest Assessment

### What's Working

| Component | Status |
|-----------|--------|
| Data pipeline | Markets, prices, orderbooks, cross-platform matching |
| Single-market arb | Logic correct, fee-aware |
| Cross-platform arb | TF-IDF matching, fee calc |
| Calibration model | Isotonic regression, historical bias |
| Elo (Glicko-2) | Tennis/UFC, surface-specific |
| Risk manager | Position limits, circuit breaker |
| Arbitrage executor | **Paper only** — real execution TODO |

### What's Broken or Degraded

| Issue | Impact |
|-------|--------|
| **Snapshot coverage 6.3%** | Momentum features pruned; 93.7% use `market.price_yes` fallback (leakage risk) |
| **Only 2 features active** | log_open_interest, time_to_resolution_hrs. Calibration, price_bucket, etc. pruned |
| **Orderbook 0%** | All orderbook features zero variance |
| **Mid-price only** | No bid/ask in strategies; execution reality ignored |
| **20–80% Brier 0.16** | Tradeable range performs poorly; 0–20% Brier 0.008 (suspicious), 80–100% Brier 0.91 (terrible) |
| **Class imbalance 16.7% YES** | Model biased toward NO predictions |
| **Real arb execution** | Placeholder; WebSocket signals only log |

### Model Card Snapshot (2026-02-23)

- **Active**: calibration only (XGBoost/LightGBM excluded)
- **Features**: 2 active, 18 dropped
- **Ensemble Brier**: 0.0732; Calibration: 0.0532
- **Profit sim**: 92.5% win rate (suspicious — likely near-extremes inflation)

---

## Part 4: New Strategies Beyond Elo and Arbitrage

### Strategy A: Favorite-Longshot Exploitation (Standalone)

**Logic**: 
- BUY YES only when 0.60 ≤ price ≤ 0.75 (underpriced favorites)
- BUY NO when price ≥ 0.85 (overpriced favorites — fade)
- NEVER BUY YES when price < 0.25 (longshots overpriced)
- Use calibration model for magnitude; category-specific curves

**Data**: Your resolved markets. Build calibration curve per category.

**Implementation**: New `ml/strategies/favorite_longshot_detector.py`

### Strategy B: Recency Bias / Mean Reversion

**Logic**:
- Detect: |price_now - price_24h_ago| > 0.15
- If move is panic (e.g., news-driven) and polls/ fundamentals disagree → fade
- Entry: Limit order at mean-reversion target (e.g., 50% retracement)

**Data**: Price snapshots (need backfill). Momentum features.

**Implementation**: Add to `ensemble_edge_detector` or new `recency_bias_detector.py`

### Strategy C: Maker / Limit Order Strategy

**Logic**:
- Instead of crossing spread, place limit orders at your edge price
- E.g., model says fair = 0.72, market ask = 0.75 → place bid at 0.71
- Capture spread + edge when filled
- Requires order placement integration (Polymarket CLOB API)

**Implementation**: `execution/limit_order_executor.py` — place, monitor, cancel

### Strategy D: Category-Specific Models

**Logic**:
- Finance: ~efficient, skip or tiny size
- Sports: 2.23pp maker-taker gap — use Elo + calibration
- Entertainment/Media: 4.79–7.32pp — largest edge, focus here
- Politics: 1.02pp — moderate, use polls + calibration

**Implementation**: `ml/models/category_ensemble.py` — train per category, weight by efficiency gap

### Strategy E: Orderbook Imbalance (When Data Exists)

**Logic**:
- OBI (order book imbalance) explains ~65% of next-period price moves
- 32 features in `orderbook_features.py` — all zero variance today
- Need orderbook snapshots at as_of for training

**Implementation**: After orderbook backfill (ROADMAP Month 1 Week 3), retrain with OB features

### Strategy F: Cross-Platform Lead-Lag

**Logic**:
- Research: Polymarket leads Kalshi (higher liquidity)
- When Polymarket moves first, Kalshi may follow
- Trade Kalshi in direction of Polymarket move (with delay)

**Implementation**: Add `polymarket_price_change_1h` as feature for Kalshi markets; or standalone signal

### Strategy G: Copy Trading / Smart Follow

**Logic**:
- You have `copy_engine.py` and trader profiles
- Filter: Only copy traders with **maker** behavior (limit orders) and positive ROI
- Avoid copying takers (they lose 1.12% on average)

**Implementation**: Add maker/taker classification to trader_data if available; filter leaderboard

---

## Part 5: Prioritized Action Plan

### Phase 1: Data Foundation (Weeks 1–4)

**Goal**: Snapshot coverage 6.3% → 50%+ so momentum/orderbook features activate.

1. **Run `analyze_training_universe.py`** (create if missing) — identify which markets need backfill
2. **Targeted backfill** — `backfill_price_history.py --market-ids data/missing_snapshot_markets.json` for top 2000 Polymarket markets
3. **Retrain** — Expect 10+ features active, Brier improvement
4. **Add bid/ask to strategy** — Fetch orderbook at signal time; use ask for buy, bid for sell in EV calc

### Phase 2: Model Improvements (Weeks 5–8)

1. **Category-specific calibration** — Train separate curves per category (Finance, Sports, Politics, etc.)
2. **Filter near-extremes in training** — Exclude <5% and >95% from training (or train separate "hard markets" model)
3. **Fix 20–80% Brier** — This is where you trade; focus validation on this bucket
4. **Platform-specific calibration** — Retrain calibration on YOUR Polymarket vs Kalshi resolved data (not synthetic)

### Phase 3: New Strategies (Weeks 9–12)

1. **Favorite-Longshot detector** — Standalone strategy with category-specific rules
2. **Recency bias detector** — Use momentum features once backfill done
3. **Category ensemble** — Per-category models
4. **Maker strategy** — Limit order placement when ready

### Phase 4: Execution Reality (Weeks 13–16)

1. **Real arbitrage execution** — Wire `arbitrage/executor.py` to Polymarket/Kalshi APIs
2. **Bid/ask in all EV calculations** — No more mid-price
3. **Slippage model** — Per-market based on liquidity
4. **WebSocket arb execution** — Real-time single-market arb when YES+NO < 1

### Phase 5: Advanced (Ongoing)

1. **Orderbook strategy** — When OB snapshots available
2. **Cross-platform lead-lag** — Polymarket → Kalshi signals
3. **Combinatorial arb** — Low priority; 62% failure rate in research
4. **GDELT / news** — If you add alternative data

---

## Part 6: Key Research References

| Source | Key Finding |
|--------|-------------|
| Polymarket Docs | CLOB, bid-ask, order types |
| AFT 2025 "Unravelling the Probabilistic Forest" | $40M arb on Polymarket; combinatorial 0.24% of profits |
| Jonathan Becker "Microstructure of Wealth Transfer" | Maker-taker, optimism tax, category gaps |
| QuantPedia "Systematic Edges" | Inter/intra arbitrage, longshot bias |
| Jepstar "Systematic Edge" | Favorite-longshot, recency bias, liquidity gaps |
| Snowberg & Wolfers (NBER) | Favorite-longshot bias: misperceptions, not risk-love |
| Navnoor Bawa (Medium) | Combinatorial arb 62% failure; speed > complexity |

---

## Part 7: Summary — One Strategy to Rule Them All?

If you must pick **one** to perfect first:

**Favorite-Longshot + Calibration + Category Selection**

- Most reliable (decades of evidence)
- Works with your current data (no backfill required for calibration)
- Add category-specific curves from your resolved markets
- Focus on: BUY NO on 80%+ overpriced favorites; BUY YES on 60–75% underpriced favorites
- Avoid: BUY YES on <25% longshots; Finance category (efficient)

Then layer in:
1. **Data** (backfill → momentum, recency bias)
2. **Execution** (bid/ask, limit orders, real arb)
3. **Maker positioning** (limit orders, category selection)

---

*Document created from research + codebase analysis. Update as you implement.*
