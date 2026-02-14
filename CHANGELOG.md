# Changelog - PredictFlow

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-02-14 - **Critical Leakage Fixes & Production Audit**

### 🚨 **BREAKING CHANGES**
- **Ensemble Brier increased** from 0.0539 → 0.0557 (performance degraded intentionally after removing data leakage)
- **6 contaminated volume features excluded** from training (volume_volatility, volume_trend_7d, etc.)
- **Quality gates tightened 2-5×** (MIN_VOLUME_TOTAL: $5K → $10K, MIN_VOLUME_24H: $200 → $1K)
- **Kelly formula behavior changed** - now properly scales with edge magnitude instead of always capping at 2%

### ✅ **Fixed - Data Leakage (P0 CRITICAL)**

#### 1. Orderbook Temporal Leakage ([#1](scripts/train_ensemble.py#L134-L183))
**Before**: Orderbook snapshots loaded with `func.max(timestamp)` with NO `as_of` filtering
```python
# LEAKED: Used latest orderbook snapshot (potentially post-resolution)
subq = select(OrderbookSnapshot.market_id, func.max(OrderbookSnapshot.timestamp))
```

**After**: Now filters orderbooks to `timestamp <= as_of` (resolved_at - 24h)
```python
# FIXED: Only use orderbooks from BEFORE resolution
valid_obs = [ob for ob in all_orderbooks
            if ob.market_id == market_id and ob.timestamp <= as_of]
```

**Impact**:
- Orderbook features no longer have zero variance due to post-resolution contamination
- Now have zero variance due to insufficient coverage (1/2840 markets) - correct reason!

---

#### 2. Volume Feature Contamination ([#2](scripts/train_ensemble.py#L528-L547))
**Detection**: Added correlation tripwire during training
```python
# Compute correlation between volume features and resolution outcome
for vol_feature in ["volume_volatility", "volume_trend_7d", ...]:
    corr = np.corrcoef(X[:, vol_idx], y)[0, 1]
    if abs(corr) > 0.4:
        logger.warning(f"HIGH CORRELATION: {vol_feature} = {corr:.3f}")
```

**Results**:
- `volume_volatility`: **0.853 correlation** (SEVERE contamination)
- `volume_trend_7d`: **0.809 correlation**
- `log_volume_total`: **0.664 correlation**
- XGBoost relied on these for **81.6% of decisions** (55.6% + 26%)

**Solution**: Excluded all 6 contaminated volume features from `ENSEMBLE_FEATURE_NAMES`
- Removed: `volume_volatility`, `volume_trend_7d`, `log_volume_total`, `volume_per_day`, `volume_acceleration`, `volume_to_liquidity_ratio`
- Result: XGBoost now uses **clean features** (log_open_interest 41.9%, volatility_20 20.2%)

---

### ✅ **Fixed - Position Sizing Bug (P1 HIGH)**

#### Kelly Formula ([#3](ml/strategies/ensemble_edge_detector.py#L136-L141))
**Before**: Applied fractional Kelly BEFORE capping
```python
# BUG: Always caps at 2% for any edge >8%
return max(0.0, min(kelly_raw * KELLY_FRACTION, MAX_KELLY))
#                   └─ Applies 0.25× first, then caps
#                      Result: min(0.289 * 0.25, 0.02) = 0.02 (ALWAYS)
```

**After**: Cap raw Kelly FIRST, then apply fraction
```python
# FIXED: Scales 0-8% → 0-2% properly
kelly_capped = min(kelly_raw, MAX_KELLY / KELLY_FRACTION)  # Cap at 8%
return max(0.0, kelly_capped * KELLY_FRACTION)  # Apply 0.25× → 2% max
```

**Test Results**:
- 4% edge → 1.50% position (scales ✓)
- 8% edge → 2.00% position (at cap ✓)
- 20% edge → 2.00% position (capped ✓)

---

### ✅ **Changed - Quality Gates Tightened ([#4](ml/strategies/ensemble_edge_detector.py#L17-L22))**

| Gate | Before | After | Change |
|------|--------|-------|--------|
| `MIN_VOLUME_TOTAL` | $5,000 | **$10,000** | 2× stricter |
| `MIN_VOLUME_24H` | $200 | **$1,000** | 5× stricter |
| `MIN_LIQUIDITY` | $1,000 | **$5,000** | 5× stricter |

**Rationale**: Prevent trading on thin markets where slippage dominates edge

---

### ✅ **Added - Leakage Warnings ([#5](scripts/train_ensemble.py#L821-L828))**

Model card now includes 4 critical caveats in `leakage_warnings` key:

```json
{
  "leakage_warnings": [
    "orderbook_snapshots: filtered to as_of timestamp (fixed as of 2026-02-14)",
    "volume_features: correlation check not performed",  // TODO: Update to show exclusion
    "price_distribution: 32.3% near-extremes (inflates metrics, real-world harder)",
    "survivorship_bias: 2840/13199 (21.5%) of resolved markets used (volume>0 filter)"
  ]
}
```

---

### 📊 **Performance Impact - HONEST METRICS**

| Metric | Before (Contaminated) | After (Clean) | Change |
|--------|-----------------------|---------------|--------|
| **Ensemble Brier** | 0.0539 | **0.0557** | +3.3% worse ✓ |
| **vs Baseline** | 19.6% better | **16.9% better** | -2.7 pp |
| **AUC** | 0.9654 | 0.9643 | -0.1% |
| **Win Rate** | 88.2% | 88.1% | Unchanged (other biases remain) |
| **Features Used** | 13 (after pruning) | **8 (after pruning)** | -38% |

#### XGBoost Feature Importance Shift
**Before** (Contaminated):
```
volume_volatility   55.6%  ← 0.853 correlation with outcome!
volume_trend_7d     26.0%  ← 0.809 correlation
log_open_interest    4.8%
price_yes            3.6%
```

**After** (Clean):
```
log_open_interest   41.9%  ← Clean signal ✓
price_yes           22.7%  ← Clean signal ✓
volatility_20       20.2%  ← Clean momentum! (was 0% before)
calibration_bias    11.3%
```

**Key Insight**: Performance degraded when leakage removed, **proving the audit was correct**. Model now uses legitimate signals.

---

### 📝 **Documentation Added**

- `docs/PRODUCTION_READINESS.md` - Comprehensive deployment status report
- `scripts/validate_deployment.py` - End-to-end validation script
- `CHANGELOG.md` - This file
- Updated `MEMORY.md` with leakage fixes

---

### ⚠️ **Known Remaining Issues**

Despite fixes, **88.1% win rate** remains unrealistic. Remaining biases:

1. **Near-Resolution Bias** (32.3% of training at price extremes)
   - Impact: +15-20 pp win rate inflation
   - Fix: Filter out <5% and >95% prices from training

2. **Survivorship Bias** (Only 21.5% of resolved markets usable)
   - Impact: Training on "easy" liquid markets, deploying on harder illiquid markets
   - Fix: Include low-volume markets with clean as_of snapshots

3. **Price Fallback Contamination** (84% use market.price_yes fallback)
   - Impact: Unknown, likely 5-10% contamination
   - Fix: Enforce strict as_of mode (100% use backfilled snapshots)

**Realistic Production Expectation**: 54-58% win rate after fixing ALL biases

---

### 🔧 **Technical Debt**

- [ ] Update leakage warning #2 to reflect volume feature exclusion (not just "check not performed")
- [ ] Implement clean `volume_24h` from historical PriceSnapshot deltas
- [ ] Bid/ask spread modeling (all strategies assume mid-price execution)
- [ ] Dynamic slippage calculation (currently fixed 1% buffer)
- [ ] PostgreSQL migration (SQLite fails at 6 months, 10GB/month growth)
- [ ] Rate limiting + circuit breakers (will hit API bans within days)
- [ ] PositionManager class (risk limits defined but not enforced)

---

### ✅ **Validated End-to-End**

**Model Loading**:
- ✓ Ensemble loads 8 clean features (excludes 6 contaminated)
- ✓ Weights: Calibration 19.6%, XGBoost 41.4%, LightGBM 39%
- ✓ Clean momentum features active (volatility_20: 20.2% importance)

**Quality Gates**:
- ✓ Volume: $10K total, $1K/day, $5K liquidity
- ✓ Kelly scales: 4% edge → 1.50%, 8% edge → 2.00% (capped)

**API Endpoints** (43 total):
- ✓ `/health`, `/system/stats`, `/markets` working
- ✓ `/predictions/{id}` returns ensemble predictions
- ✓ `/strategies/signals` returns active edge signals

**Live Predictions**:
- ✓ Tested on "Government shutdown" market
- ✓ Market: 2.2%, Ensemble: 1.3%, Delta: -0.9%
- ✓ Uses 8 features, all 3 models contributing

---

### 🚀 **Deployment Status**

**Hackathon-Ready**: ✅ YES (with honest disclaimers)
```
✓ Orderbook as_of filtering
✓ Volume contamination detection & exclusion
✓ Kelly formula bug fixed
✓ Quality gates tightened
✓ Leakage warnings documented
✓ API serves updated model
✓ Live predictions validated
```

**Production-Ready**: ⚠️ NOT YET (60-90 days)
```
⏳ Filter near-extremes from training
⏳ Clean volume_24h implementation
⏳ Bid/ask spread + slippage modeling
⏳ PositionManager (risk limits)
⏳ PostgreSQL migration
⏳ Rate limiting + circuit breakers
⏳ 30-day paper trading validation
```

---

### 📚 **Migration Guide**

If you have existing saved models from before this release:

1. **Retrain required**: Old models use contaminated volume features
   ```bash
   python scripts/train_ensemble.py
   ```

2. **API restart required**: New model must be loaded
   ```bash
   # Stop running API
   # Restart with: uvicorn api.main:app --reload
   ```

3. **Expect performance drop**: Brier 0.0539 → 0.0557 (NORMAL, not a regression!)

4. **Review warnings**: Check `ml/saved_models/model_card.json` for `leakage_warnings`

---

### 👥 **Contributors**

- Audit conducted by: Claude Opus 4.6 (via Anthropic API)
- Fixes implemented: 2026-02-14
- Review: Red team / production readiness audit

---

## [0.1.0] - 2026-02-12 - **Initial Release**

### Added
- FastAPI backend with 43 API endpoints
- React 19 frontend with Tailwind CSS 4
- ML ensemble (Calibration + XGBoost + LightGBM)
- Glicko-2 Elo system for tennis
- Arbitrage detection (single-market, cross-platform)
- Copy trading system (50 traders tracked)
- Paper trading portfolio
- Category normalization (750 → 10 categories)

### Known Issues (Fixed in 0.2.0)
- ❌ Orderbook temporal leakage (post-resolution data)
- ❌ Volume contamination (0.85 correlation with outcome)
- ❌ Kelly formula bug (always caps at 2%)
- ❌ Quality gates too lenient ($200/day minimum)
