# PredictFlow: 90-Day Production Roadmap

**Status**: 3-Feature Model (Hackathon-Ready) → 16-Feature Model (Production-Ready)
**Goal**: Transform 6.3% snapshot coverage → 50%+ coverage to activate momentum/orderbook features
**Timeline**: February 14 – May 14, 2026

---

## Current State (Day 0)

### What Works ✅
- **3-feature ensemble**: price_yes, log_open_interest, time_to_resolution_hrs
- **Performance**: Brier 0.0788 (15.9% better than baseline)
- **Training set**: 3,638 markets (up from 2,840)
- **All critical leakage fixes active**: Orderbook as_of filtering, volume contamination removed, Kelly formula fixed
- **Validation passed**: Live predictions work, API functional (43 endpoints)

### What's Broken ⚠️
- **Snapshot coverage**: Only 6.3% (230/3,638 markets)
- **Momentum features inactive**: return_1h, volatility_20, zscore_24h pruned as "near-constant"
- **Orderbook features inactive**: Zero variance (0% coverage)
- **Fallback rate**: 93.7% of training uses `market.price_yes` (not as_of snapshots)
- **Impact**: 13+ features designed but only 3 active → performance ceiling at 0.0788

### The Gap
**Root cause**: Price history backfill (1,205 markets) doesn't overlap with training set (3,638 markets).
**Solution**: Align backfill with training universe, prioritize resolved Polymarket markets with `token_id_yes`.

---

## Month 1: Data Foundation (Days 1-30)

**Goal**: Snapshot coverage 6.3% → 50%+ → Activate 10+ features → Target Brier 0.06-0.07

### Week 1: Align Backfill with Training Set

#### Task 1.1: Define Training Universe (2 days)
**File**: `scripts/analyze_training_universe.py` (NEW)

**Action**: Query the exact markets used by `train_ensemble.py` and analyze coverage.

```python
# Pseudo-code
async def analyze_training_universe():
    """Which markets does training use? Which have snapshots?"""
    # Same filters as train_ensemble.py
    resolved_markets = query(
        Market.resolution_value.isnot(None),
        Market.volume_total > 0,  # Usable filter
    ).all()

    # Check snapshot coverage
    for market in resolved_markets:
        as_of = market.resolved_at - timedelta(hours=24)
        snapshots = query_price_snapshots(market.id, before=as_of)

        if snapshots:
            has_coverage[market.id] = len(snapshots)
        else:
            missing_coverage.append(market.id)

    # Output
    print(f"Training universe: {len(resolved_markets)} markets")
    print(f"  With snapshots: {len(has_coverage)} ({pct}%)")
    print(f"  Missing: {len(missing_coverage)} ({pct}%)")
    print(f"  By platform:")
    print(f"    Polymarket: {pm_with} / {pm_total} ({pm_pct}%)")
    print(f"    Kalshi: {kalshi_with} / {kalshi_total} ({kalshi_pct}%)")

    # Save missing IDs to JSON for targeted backfill
    save_json("data/missing_snapshot_markets.json", missing_coverage)
```

**Output**:
- `data/training_universe_analysis.json` (coverage stats by platform, category, date)
- `data/missing_snapshot_markets.json` (3,408 market IDs to backfill)

**Acceptance**: Run script, verify it matches training logs (230 with snapshots, 3,408 missing).

---

#### Task 1.2: Prioritize Backfill Targets (1 day)
**File**: `scripts/prioritize_backfill.py` (NEW)

**Action**: Rank markets by "training value" (volume, recency, platform).

```python
# Pseudo-code
def prioritize_backfill(missing_markets):
    """Rank markets by training value."""
    # Load from missing_snapshot_markets.json
    for market in missing_markets:
        score = 0

        # Platform: Polymarket backfill works (has CLOB API)
        if market.platform == "polymarket" and market.token_id_yes:
            score += 100  # High priority
        elif market.platform == "kalshi":
            score += 10   # Low priority (no historical API yet)

        # Volume: Higher volume = more important
        score += min(market.volume_total / 1000, 50)

        # Recency: Recent = more relevant
        days_old = (now - market.resolved_at).days
        score += max(0, 50 - days_old)  # Up to +50 for recent

        market.priority_score = score

    # Sort descending
    return sorted(missing_markets, key=lambda m: m.priority_score, reverse=True)
```

**Output**: `data/backfill_priority_list.json` (3,408 markets ranked by priority)

**Acceptance**:
- Top 1,000 should be mostly Polymarket with `token_id_yes`
- Bottom 1,000 should be mostly Kalshi (no backfill possible yet)

---

#### Task 1.3: Targeted Backfill (4 days)
**File**: `scripts/backfill_price_history.py` (MODIFY)

**Action**: Add `--market-ids` parameter to backfill specific markets from priority list.

```python
# Add to backfill_price_history.py
parser.add_argument(
    "--market-ids",
    type=str,
    help="Path to JSON file with market IDs to backfill (overrides --limit)",
)

# In main():
if args.market_ids:
    with open(args.market_ids) as f:
        target_ids = json.load(f)
    markets = query(Market).filter(Market.id.in_(target_ids)).all()
else:
    # Existing logic (top N by volume)
    markets = query(Market).order_by(desc(volume_total)).limit(args.limit).all()
```

**Run**:
```bash
# Backfill top 2,000 priority markets (Polymarket only)
python scripts/backfill_price_history.py \
    --market-ids data/backfill_priority_list.json \
    --limit 2000 \
    --days 60

# Expected: ~1,800 Polymarket markets × 60 days × 24 hours = 2.6M snapshots
# Time: ~1,800 markets × 1 sec/market = 30 minutes
```

**Acceptance**:
- Snapshot count increases from 600K → 3.2M
- Re-run `analyze_training_universe.py` → coverage 6.3% → 50%+

---

### Week 2: Retrain and Validate

#### Task 2.1: Retrain with Enhanced Coverage (1 day)
**Action**: Run `train_ensemble.py` with new snapshot data.

```bash
python scripts/train_ensemble.py
```

**Expected outcomes**:
- Training log shows: "Price snapshots: 1,800+ markets have snapshot data (50%+)"
- Features active: 3 → 10+ (momentum features no longer "near-constant")
- Brier: 0.0788 → target 0.06-0.07 (improved with momentum signals)
- XGBoost importance shifts: log_open_interest 51% → 30%, momentum features 20%+

**Acceptance**:
- Model card shows `feature_names` includes return_1h, volatility_20, zscore_24h
- `features_dropped` no longer lists momentum features as "near-constant"
- Ensemble Brier ≤ 0.070 (if not, investigate)

---

#### Task 2.2: Add Coverage Tracking to Validation (1 day)
**File**: `scripts/validate_deployment.py` (MODIFY)

**Action**: Add check for snapshot coverage percentage.

```python
# Add to validate_deployment.py
def check_snapshot_coverage():
    """Ensure snapshot coverage meets minimum threshold."""
    model_card = load_model_card()

    # Parse from leakage_warnings or add explicit key
    # Example: Extract from "3638/23600 (15.4%) of resolved markets used"
    coverage_pct = model_card.get("snapshot_coverage_pct", 0.0)

    min_coverage = 30.0  # Minimum 30% for production
    warn_coverage = 50.0  # Target 50%

    if coverage_pct < min_coverage:
        print(f"❌ FAIL: Snapshot coverage {coverage_pct:.1f}% < {min_coverage}% minimum")
        return False
    elif coverage_pct < warn_coverage:
        print(f"⚠️  WARN: Snapshot coverage {coverage_pct:.1f}% < {warn_coverage}% target")
        return True  # Pass but warn
    else:
        print(f"✅ PASS: Snapshot coverage {coverage_pct:.1f}% ≥ {warn_coverage}% target")
        return True
```

**Acceptance**: Run validation, verify coverage check present.

---

#### Task 2.3: Update Model Card with Coverage Metric (1 day)
**File**: `scripts/train_ensemble.py` (MODIFY)

**Action**: Add explicit `snapshot_coverage_pct` key to model card.

```python
# In train_ensemble.py, after building training matrix:
n_with_snapshots = len([m for m in markets if m.id in price_at_as_of_map])
n_total = len(markets)
coverage_pct = (n_with_snapshots / n_total * 100) if n_total > 0 else 0.0

# Add to model_card dict:
model_card["snapshot_coverage"] = {
    "n_with_snapshots": n_with_snapshots,
    "n_total": n_total,
    "coverage_pct": coverage_pct,
    "target_pct": 50.0,
}
```

**Acceptance**: Open `model_card.json`, verify `snapshot_coverage` key exists with coverage_pct.

---

### Week 3: Orderbook Backfill (Stretch Goal)

**Goal**: If Polymarket CLOB API supports historical orderbooks, backfill top 500 markets.

#### Task 3.1: Research Polymarket CLOB Historical API (2 days)
**Action**:
1. Check CLOB API docs for historical orderbook endpoint
2. Test if `GET /book?token_id=X&timestamp=Y` works
3. Document findings in `docs/POLYMARKET_API.md`

**Scenarios**:
- ✅ **API supports historical**: Proceed to Task 3.2
- ❌ **API only returns current snapshot**: Skip orderbook backfill, document as "Month 3 stretch goal"

---

#### Task 3.2: Backfill Orderbook Snapshots (5 days, if API works)
**File**: `scripts/backfill_orderbook_snapshots.py` (NEW, similar to backfill_price_history.py)

**Action**: For top 500 resolved markets, fetch orderbook at `resolved_at - 24h`.

```python
async def backfill_orderbook(market, session, as_of_timestamp):
    """Fetch historical orderbook snapshot at specific time."""
    if not market.token_id_yes:
        return None

    # Try CLOB historical API (hypothetical)
    orderbook_data = await fetch_orderbook_at_time(
        token_id=market.token_id_yes,
        timestamp=as_of_timestamp,
    )

    if orderbook_data:
        snapshot = OrderbookSnapshot(
            market_id=market.id,
            timestamp=datetime.fromtimestamp(as_of_timestamp),
            bids=orderbook_data["bids"][:10],  # Top 10 levels
            asks=orderbook_data["asks"][:10],
            # ... compute OBI, spread, depth, etc.
        )
        session.add(snapshot)
        await session.commit()
        return snapshot

    return None
```

**Expected**:
- 200-500 markets with orderbook snapshots at as_of
- Orderbook features (obi_level1, bid_ask_spread_abs, depth_ratio) no longer zero variance
- Next retrain: Orderbook features contribute 5-10% importance

**Acceptance**:
- OrderbookSnapshot table grows from 3,341 → 8,000+
- Re-run training → orderbook features active (if variance > 0)

---

### Week 4: Documentation and Monitoring

#### Task 4.1: Add Coverage Dashboard to Frontend (3 days)
**File**: `frontend/src/pages/DataQuality.tsx` (MODIFY or NEW)

**Action**: Display snapshot coverage metrics from model card.

```tsx
// Fetch model card via API
const modelCard = await fetch("/api/v1/ml/model_card").then(r => r.json());

// Display coverage
<div className="card">
  <h2>Snapshot Coverage</h2>
  <p>Markets with historical snapshots: {modelCard.snapshot_coverage.n_with_snapshots} / {modelCard.snapshot_coverage.n_total}</p>
  <ProgressBar value={modelCard.snapshot_coverage.coverage_pct} target={modelCard.snapshot_coverage.target_pct} />
  {modelCard.snapshot_coverage.coverage_pct < 50 && (
    <Alert variant="warning">
      Coverage below 50% target. Momentum features may be inactive.
    </Alert>
  )}
</div>
```

**Acceptance**: Navigate to Data Quality page, see coverage bar chart.

---

#### Task 4.2: Update README.md Roadmap Section (1 day)
**File**: `README.md` (MODIFY)

**Action**: Add "Data Coverage Roadmap" section linking to this document.

```markdown
## Data Coverage Roadmap

**Current**: 3-feature model (50% snapshot coverage)
**Target**: 16-feature model (50%+ coverage)

See [90-Day Roadmap](docs/ROADMAP_90_DAYS.md) for detailed plan.

### Quick Status
- ✅ Month 0 (Hackathon): 3 features, Brier 0.0788, 6.3% coverage
- ⏳ Month 1 (Foundation): Target 50%+ coverage → 10+ features → Brier 0.06-0.07
- ⏳ Month 2 (Refinement): Near-extremes filtering, clean volume_24h, orderbook active
- ⏳ Month 3 (Production): Multi-horizon models, paper trading, PostgreSQL migration
```

---

#### Task 4.3: Automate Coverage Tracking (2 days)
**File**: `scripts/track_coverage_history.py` (NEW)

**Action**: Log coverage metrics after each training run to CSV.

```python
# After each train_ensemble.py run:
def log_coverage(model_card):
    """Append coverage metrics to history CSV."""
    import csv
    from datetime import datetime

    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "n_total_resolved": model_card["n_total_resolved"],
        "n_usable": model_card["n_usable"],
        "n_with_snapshots": model_card["snapshot_coverage"]["n_with_snapshots"],
        "coverage_pct": model_card["snapshot_coverage"]["coverage_pct"],
        "ensemble_brier": model_card["ensemble_brier"],
        "features_active": len(model_card["feature_names"]),
    }

    with open("data/coverage_history.csv", "a") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if f.tell() == 0:  # Empty file
            writer.writeheader()
        writer.writerow(row)
```

**Acceptance**:
- Run training 3× → CSV has 3 rows
- Plot coverage_pct vs ensemble_brier → see correlation (higher coverage → lower Brier)

---

### Month 1 Success Criteria ✅

- [ ] **Snapshot coverage ≥ 50%** (1,800+ markets / 3,638 training markets)
- [ ] **Momentum features active** (return_1h, volatility_20, zscore_24h no longer pruned)
- [ ] **Ensemble Brier ≤ 0.070** (improved from 0.0788)
- [ ] **10+ features in model** (up from 3)
- [ ] **Coverage tracked** (in model card, validation, frontend dashboard)
- [ ] **Roadmap documented** (README, CHANGELOG, this file)

**Estimated effort**: 15-20 days (full-time) or 30-40 days (part-time)

---

## Month 2: Model Refinement (Days 31-60)

**Goal**: Address remaining biases → Target Brier 0.055-0.060 → Win rate 65-75%

### Week 5: Near-Extremes Filtering

#### Task 5.1: Filter Training Set to Mid-Priced Markets (2 days)
**File**: `scripts/train_ensemble.py` (MODIFY)

**Action**: Add parameter `--filter-extremes` to exclude <5% and >95% prices.

```python
# Add filter after loading resolved markets
if args.filter_extremes:
    before_count = len(resolved_markets)
    resolved_markets = [
        m for m in resolved_markets
        if 0.05 <= m.price_yes <= 0.95  # Mid-priced only
    ]
    after_count = len(resolved_markets)
    logger.info(f"Filtered extremes: {before_count} → {after_count} markets")
```

**Expected**:
- Training set: 3,638 → ~2,600 markets (28% removed)
- Brier increases slightly on test set (0.070 → 0.075) but win rate drops significantly (86% → 65-70%)
- This is HONEST: model no longer trained on trivial predictions

**Acceptance**: Model card shows `filter_extremes: true` and updated price distribution (<5% near-extremes).

---

#### Task 5.2: Train Separate "Hard Markets" Model (3 days)
**File**: `scripts/train_ensemble_hard.py` (NEW or --mode=hard)

**Action**: Train on 0.20 ≤ price ≤ 0.80 (most uncertain markets).

**Expected**:
- Brier 0.08-0.12 (realistic for uncertain markets)
- Win rate 54-58% (realistic)
- Sharpe ratio 0.3-0.6 (modest but real edge)

**Acceptance**: Model card shows `hard_markets_only: true`, Brier ≤ 0.12, win rate 54-60%.

---

### Week 6: Clean Volume Features

#### Task 6.1: Implement Clean volume_24h (3 days)
**File**: `ml/features/training_features.py` (MODIFY)

**Action**: Compute volume_24h from price snapshot deltas (not market.volume_24h).

```python
def compute_clean_volume_24h(market, as_of, session):
    """Compute 24h volume from snapshot deltas."""
    snapshots = query_price_snapshots(
        market.id,
        start=as_of - timedelta(hours=24),
        end=as_of,
    )

    if len(snapshots) < 2:
        return 0.0  # Not enough data

    # Sum absolute price changes × open_interest
    volume = 0.0
    for i in range(1, len(snapshots)):
        price_change = abs(snapshots[i].price_yes - snapshots[i-1].price_yes)
        avg_oi = (snapshots[i].open_interest + snapshots[i-1].open_interest) / 2
        volume += price_change * avg_oi

    return volume
```

**Acceptance**:
- Re-train with clean volume_24h
- Verify XGBoost importance < 20% (not 51% like contaminated version)
- Model card documents "volume_24h: computed from snapshot deltas (clean)"

---

### Week 7: Multi-Horizon Models

#### Task 7.1: Train 3 Horizon-Specific Models (5 days)
**Files**:
- `scripts/train_ensemble_24h.py` (predict 1 day before resolution)
- `scripts/train_ensemble_7d.py` (predict 7 days before)
- `scripts/train_ensemble_30d.py` (predict 30 days before)

**Action**: Set `as_of = resolved_at - timedelta(days=horizon)` for each model.

**Expected performance**:
- 24h model: Brier 0.06-0.08 (lots of information)
- 7d model: Brier 0.10-0.14 (moderate information)
- 30d model: Brier 0.16-0.22 (mostly prior)

**Acceptance**:
- 3 models saved to `ml/saved_models/ensemble_{horizon}.joblib`
- API endpoint `/predictions/{id}?horizon=7d` returns appropriate model

---

### Week 8: Orderbook Integration (if data available)

#### Task 8.1: Activate Orderbook Features (2 days)
**Prerequisite**: Task 3.2 completed (historical orderbook data backfilled)

**Action**: Re-train with orderbook features no longer zero-variance.

**Expected**:
- Orderbook features: obi_level1 (3-5% importance), bid_ask_spread_rel (2-4%)
- Ensemble Brier: 0.070 → 0.055-0.060 (orderbook depth adds signal)

**Acceptance**: Model card shows orderbook features in `feature_names` (not `features_dropped`).

---

### Month 2 Success Criteria ✅

- [ ] **Near-extremes filtered** (<5% of training at price extremes)
- [ ] **Clean volume_24h implemented** (from snapshot deltas)
- [ ] **Multi-horizon models trained** (3 models: 24h, 7d, 30d)
- [ ] **Orderbook features active** (if historical data available)
- [ ] **Ensemble Brier ≤ 0.060** (target: 0.055-0.060)
- [ ] **Win rate 65-75%** (realistic, not inflated)

**Estimated effort**: 15-20 days (full-time)

---

## Month 3: Production Hardening (Days 61-90)

**Goal**: Validate in paper trading → Deploy with confidence → Scale infrastructure

### Week 9: Risk Controls

#### Task 9.1: Implement PositionManager (3 days)
**File**: `risk/position_manager.py` (NEW)

**Action**: Enforce per-market, total, category, and correlation limits.

```python
class PositionManager:
    def can_trade(self, market, size_usd, category):
        # Check per-market limit
        if size_usd > self.max_position_usd:
            return False, "Exceeds per-market limit"

        # Check total exposure
        if self.total_exposure + size_usd > self.max_total_exposure:
            return False, "Exceeds total exposure"

        # Check category concentration
        if self.category_exposure[category] + size_usd > self.max_category_pct * self.bankroll:
            return False, f"Category {category} > {self.max_category_pct}%"

        # Check daily loss circuit breaker
        if self.daily_pnl < -self.max_loss_per_day:
            return False, "Circuit breaker: daily loss limit"

        return True, "OK"
```

**Acceptance**:
- Try position > $100 → rejected
- Try 11th position → rejected (total exposure)
- Simulate -$55 P&L → circuit breaker active

---

#### Task 9.2: Add Correlation Risk Check (2 days)
**File**: `risk/correlation.py` (NEW)

**Action**: Prevent >20% exposure to correlated markets.

```python
def compute_correlation(market1, market2):
    """Estimate correlation via keyword overlap + category."""
    # Keyword similarity (TF-IDF cosine)
    similarity = cosine_similarity(market1.question, market2.question)

    # Category overlap
    if market1.category == market2.category:
        similarity += 0.3

    return min(similarity, 1.0)

def check_correlation_risk(new_market, open_positions, max_correlated_pct=0.20):
    """Sum exposure to correlated markets."""
    correlated_exposure = 0.0
    for pos in open_positions:
        corr = compute_correlation(new_market, pos.market)
        if corr > 0.7:  # High correlation
            correlated_exposure += pos.size_usd

    if correlated_exposure > bankroll * max_correlated_pct:
        return False, "Correlated exposure > 20%"

    return True, "OK"
```

**Acceptance**: Open 5× Trump positions → 6th rejected (correlated).

---

### Week 10: Paper Trading

#### Task 10.1: Launch Paper Trading Engine (2 days)
**File**: `portfolio/paper_trader.py` (MODIFY existing or NEW)

**Action**: On each ensemble edge signal, log to `PaperTrade` table (don't execute).

```python
# In data_pipeline/scheduler.py, after edge detection:
if signal.net_ev > 0.03 and quality_gates_pass(signal.market):
    paper_trade = PaperTrade(
        market_id=signal.market_id,
        side=signal.direction,
        price=signal.market_price,
        size_usd=signal.kelly_pct * BANKROLL,
        ensemble_prob=signal.ensemble_prob,
        expected_ev=signal.net_ev,
        status="open",
        opened_at=datetime.utcnow(),
    )
    session.add(paper_trade)
```

**Acceptance**: After 24h, PaperTrade table has 5-20 entries.

---

#### Task 10.2: Resolve Paper Trades (1 day)
**File**: `portfolio/paper_trader.py` (ADD resolve logic)

**Action**: Check resolved markets, compute P&L for paper trades.

```python
# Every 5 minutes:
def resolve_paper_trades(session):
    open_trades = query(PaperTrade).filter(status="open").all()

    for trade in open_trades:
        if trade.market.resolved_at:
            # Compute P&L
            if trade.side == "buy_yes":
                pnl = trade.size_usd * (trade.market.resolution_value - trade.price)
            else:  # buy_no
                pnl = trade.size_usd * ((1 - trade.market.resolution_value) - trade.price)

            # Apply fees
            pnl -= trade.size_usd * 0.02  # Polymarket 2% winning fee

            trade.pnl = pnl
            trade.status = "closed"
            trade.closed_at = datetime.utcnow()
            session.commit()
```

**Acceptance**: After market resolves, paper trade shows P&L in database.

---

#### Task 10.3: Paper Trading Dashboard (3 days)
**File**: `frontend/src/pages/PaperTrading.tsx` (NEW)

**Action**: Display paper trading performance metrics.

```tsx
// Fetch paper trades via API
const summary = await fetch("/api/v1/portfolio/paper_summary").then(r => r.json());

<div className="card">
  <h2>Paper Trading Results (30 Days)</h2>
  <MetricGrid>
    <Metric label="Total Trades" value={summary.total_trades} />
    <Metric label="Win Rate" value={`${summary.win_rate * 100}%`} />
    <Metric label="Avg P&L" value={`$${summary.avg_pnl.toFixed(2)}`} />
    <Metric label="Sharpe Ratio" value={summary.sharpe.toFixed(2)} />
    <Metric label="Max Drawdown" value={`$${summary.max_drawdown.toFixed(2)}`} />
  </MetricGrid>

  {summary.win_rate < 0.54 && (
    <Alert variant="warning">
      Win rate below 54% target. Review edge threshold or feature set.
    </Alert>
  )}
</div>
```

**Acceptance**: Navigate to Paper Trading page, see 30-day rolling metrics.

---

### Week 11: Infrastructure

#### Task 11.1: PostgreSQL Migration (3 days)
**Action**:
1. Install PostgreSQL, create DB
2. Update `database_url` to `postgresql+asyncpg://...`
3. Run Alembic migrations
4. Migrate SQLite data to PostgreSQL
5. Run full integration test

**Acceptance**:
- All tests pass on PostgreSQL
- Benchmark: 10K concurrent writes <5s (was timing out on SQLite)

---

#### Task 11.2: Rate Limiting + Circuit Breakers (2 days)
**File**: `data_pipeline/collectors/polymarket_clob.py` (MODIFY)

**Action**: Add token bucket rate limiter + exponential backoff.

```python
from aiohttp_retry import ExponentialRetry

rate_limiter = TokenBucket(rate=10, per=60)  # 10 req/min

async def fetch_orderbook(token_id):
    await rate_limiter.acquire()  # Block if over limit

    try:
        async with retry_session(retry_options=ExponentialRetry(attempts=5)) as session:
            response = await session.get(f"/book?token_id={token_id}")
            return response.json()
    except ClientError as e:
        if e.status == 429:  # Rate limited
            logger.warning("Rate limited, backing off...")
            await asyncio.sleep(60)  # Wait 1 min
            raise  # Retry will handle exponential backoff
```

**Acceptance**: Simulate 100 req/min → limiter blocks excess, no 429 errors.

---

#### Task 11.3: Monitoring Stack (2 days)
**Action**:
1. Install Prometheus + Grafana
2. Add `prometheus_client` to FastAPI
3. Export metrics: request rate, latency p95, error rate, Brier rolling 30d
4. Create Grafana dashboard (6 panels)
5. Set up alerts: error rate >5% for 5 min, Brier >0.10 for 7 days

**Acceptance**: Open Grafana, all panels show live data, alerts configured.

---

### Week 12: Validation and Launch

#### Task 12.1: 30-Day Paper Trading Validation (30 days continuous)
**Start**: Day 60
**End**: Day 90

**Criteria**:
- [ ] Total trades: 100-500 (3-15/day)
- [ ] Win rate: 54-65% (realistic)
- [ ] Brier: 0.055-0.070 (consistent with backtest)
- [ ] Sharpe: 0.5-1.2 (modest but positive)
- [ ] Max drawdown: <10% of bankroll
- [ ] No circuit breaker triggers (daily loss <$50)

**Action**: Monitor daily, investigate if metrics diverge from backtest.

**Acceptance**: After 30 days, paper trading Brier within 10% of backtest Brier.

---

#### Task 12.2: Operations Runbook (2 days)
**File**: `docs/OPERATIONS.md` (NEW)

**Sections**:
1. Deployment (API, migrations, rollback)
2. Monitoring (Grafana tour, key metrics)
3. Incident response (API down, DB full, circuit breaker)
4. Database ops (backup, restore, archival)
5. Common tasks (retrain, backfill, add platform)

**Acceptance**: Teammate can follow runbook to recover from simulated incident in <15 min.

---

#### Task 12.3: Production Pilot (if paper trading passes)
**Bankroll**: $1,000
**Duration**: 30 days
**Risk limits**: Max $100/position, $1K total, $50 daily loss

**Action**: Deploy to production with real API keys (paper=false).

**Monitoring**: Daily check P&L, Brier, risk limits.

**Circuit breaker**: Pause if any of:
- Daily loss > $50
- Rolling Brier > 0.15 (model degraded)
- Error rate > 10%
- Correlation > 0.8 with unexpected market

**Acceptance**: After 30 days, positive P&L and Sharpe > 0.5.

---

### Month 3 Success Criteria ✅

- [ ] **PositionManager enforced** (all risk limits active)
- [ ] **Paper trading validated** (30 days, 54-65% win rate, Brier consistent)
- [ ] **PostgreSQL deployed** (SQLite replaced)
- [ ] **Monitoring live** (Prometheus + Grafana)
- [ ] **Operations runbook complete**
- [ ] **Production pilot launched** ($1K bankroll, real money)

**Estimated effort**: 20-25 days (full-time)

---

## Success Metrics (90-Day Summary)

### Day 0 (Hackathon)
- ✅ 3-feature model, Brier 0.0788, 6.3% snapshot coverage
- ✅ All critical leakage fixes in place
- ✅ Validation passed, API functional

### Day 30 (Month 1 Complete)
- [ ] Snapshot coverage 50%+ (1,800+ markets)
- [ ] 10+ features active (momentum restored)
- [ ] Brier ≤ 0.070 (improved from 0.0788)
- [ ] Coverage tracked (model card, validation, frontend)

### Day 60 (Month 2 Complete)
- [ ] Near-extremes filtered (<5% at extremes)
- [ ] Multi-horizon models (3 models: 24h, 7d, 30d)
- [ ] Orderbook features active (if data available)
- [ ] Brier ≤ 0.060 (target: 0.055-0.060)

### Day 90 (Month 3 Complete)
- [ ] Paper trading: 30 days, win rate 54-65%, Sharpe 0.5-1.2
- [ ] PostgreSQL deployed, rate limiting active
- [ ] Monitoring dashboard live
- [ ] Production pilot: $1K bankroll, real money validated

---

## Long-Term Principles (Sustaining Quality)

### 1. Honest Methodology as Default
- Always use temporal splits (no shuffling)
- Always enforce `as_of` filtering (no fallback to market.price_yes in production)
- Always document leakage risks for new features
- Always track coverage metrics alongside Brier

### 2. Retrain Cadence
- After every +2,000 resolved markets: retrain + document
- After every +20% snapshot coverage: retrain + validate improvement
- Monthly retrain (at minimum) to prevent staleness
- Log coverage_pct, Brier, features_active to `data/coverage_history.csv`

### 3. Coverage as North Star Metric
**Goal**: Treat snapshot coverage like uptime (99% target long-term).

**Tracking**:
- Model card: `snapshot_coverage.coverage_pct`
- Validation gate: `coverage_pct >= 30%` (warn), `>= 50%` (pass)
- Frontend dashboard: Real-time coverage bar chart
- CSV history: Plot coverage vs Brier over time

**Actions when coverage drops**:
1. Investigate: Which markets missing snapshots?
2. Prioritize: Run targeted backfill
3. Retrain: Restore feature activation
4. Validate: Ensure Brier doesn't degrade

### 4. Validation Gates Before Deploy
**Never deploy if**:
- Snapshot coverage < 30%
- Brier regression > 15% (0.070 → 0.080)
- Volume features dominate (> 30% importance)
- Win rate > 75% (unrealistic, check for leakage)
- Feature count mismatch (expected != actual)

**Always run before deploy**:
```bash
python scripts/validate_deployment.py
# Must pass all checks
```

### 5. Roadmap Hygiene
- Update this file quarterly (not just once)
- When completing a task, mark [x] and note actual effort vs estimate
- When discovering new issues, add to "Next 90 Days" section
- Keep README.md "Quick Status" in sync with this file

---

## Appendix: Quick Reference

### Key Files
- **Training**: `scripts/train_ensemble.py`
- **Backfill**: `scripts/backfill_price_history.py`, `scripts/backfill_resolved_markets.py`
- **Validation**: `scripts/validate_deployment.py`
- **Model Card**: `ml/saved_models/model_card.json`
- **Coverage History**: `data/coverage_history.csv` (created in Month 1)
- **Roadmap**: `docs/ROADMAP_90_DAYS.md` (this file)

### Key Metrics
- **Snapshot coverage**: 6.3% (Day 0) → 50%+ (Day 30) → 80%+ (Day 90)
- **Features active**: 3 (Day 0) → 10+ (Day 30) → 16 (Day 60)
- **Brier**: 0.0788 (Day 0) → 0.070 (Day 30) → 0.060 (Day 60)
- **Win rate**: 86.5% (Day 0) → 65-75% (Day 60) → 54-65% (Day 90, paper trading)

### Commands
```bash
# Analyze training universe (Month 1, Week 1)
python scripts/analyze_training_universe.py

# Targeted backfill (Month 1, Week 1)
python scripts/backfill_price_history.py --market-ids data/backfill_priority_list.json --limit 2000 --days 60

# Retrain (any time after data changes)
python scripts/train_ensemble.py

# Validate before deploy (always)
python scripts/validate_deployment.py

# Track coverage history (Month 1, Week 4)
python scripts/track_coverage_history.py

# Paper trading summary (Month 3, Week 10)
curl http://localhost:8000/api/v1/portfolio/paper_summary
```

---

**Status**: Living document. Update quarterly or when major milestones reached.
**Owner**: Maintain in `docs/ROADMAP_90_DAYS.md`
**Last Updated**: 2026-02-14 (Day 0 - Hackathon baseline)
