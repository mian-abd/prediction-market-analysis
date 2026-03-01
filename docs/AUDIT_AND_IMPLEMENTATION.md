# PredictFlow — Audit and Implementation Log

> **Purpose**: Record of the deep self-audit (2026-02-28), critical bugs fixed, and implementation status for Layers 4, 6, and 8.  
> **Audience**: Engineers and future you reviewing “what was verified and what was fixed.”  
> **Last updated**: 2026-02-28

---

## Table of Contents

1. [What was verified (no bug)](#1-what-was-verified-no-bug)
2. [Fee model verification](#2-fee-model-verification)
3. [Critical bugs fixed (17 issues, 4 CRITICAL)](#3-critical-bugs-fixed-17-issues-4-critical)
4. [Other fixes](#4-other-fixes)
5. [New implementations (L4, L6, L8)](#5-new-implementations-l4-l6-l8)
6. [What remains](#6-what-remains)

---

## 1. What was verified (no bug)

| Item | Location / context | Result |
|------|-------------------|--------|
| **walk_forward_oof** | `train_ensemble.py` — OOF predictions | Uses `fold_X_val_pruned` correctly; no train/val swap. Earlier report was from an older version. |
| **as_of refactor** | `build_training_matrix` callers | Only caller is `train_ensemble.py`, which passes `as_of_days`; time_to_resolution_hrs now uses `end_date - as_of` (time remaining), matching serving. |
| **Per-market taker_fee_bps** | Fee calculations | Correct; fee-free markets use 0; fee-enabled use API-derived bps. |
| **2% profit fee removal** | Polymarket docs + help | Confirmed: 2% profit fee at resolution has been removed. Current Polymarket is largely fee-free; only certain market types (crypto 5m/15m, NCAAB, Serie A) have curve-based taker fees. |

---

## 2. Fee model verification

- **Two fee types on Polymarket** (conceptually): (1) **Trading taker/maker fees** — curve-based on certain market types; tracked by `taker_fee_bps` from the fee-rate API. (2) **Resolution/profit fee** — historically 2% on net winnings at resolution.
- **Finding**: Official Polymarket docs (docs.polymarket.com/trading/fees) describe only taker/maker trading fees. Third-party and help (polymarket.help) confirmed the **2% profit fee has been removed**.
- **Implication**: Using 0% for fee-free markets and only `taker_fee_bps` for fee-enabled markets is **correct**. The previous “2% on all markets” was wrong and was rightly removed.

---

## 3. Critical bugs fixed (17 issues, 4 CRITICAL)

### CRITICAL 1: Train/serve feature skew (11 features always 0.0 at serve)

- **Problem**: Momentum and orderbook features had real data during training (from snapshots/orderbook) but at serving time `predict_market()` did not load snapshots or orderbook from DB, so those features were always 0.0. Model learned patterns it could never use at inference.
- **Fix**: Added `load_serving_context(session, market_id)` in `ml/features/training_features.py` to load recent price snapshots and latest orderbook for the market. All callers of `predict_market()` now load this context and pass `price_snapshots` and `orderbook_snapshot` into `ensemble.predict_market()`. Callers updated: `data_pipeline/scheduler.py`, `api/routes/ml_predictions.py` (multiple endpoints), `scripts/validate_deployment.py`.

### CRITICAL 2: Leaky feature — log_open_interest

- **Problem**: `log_open_interest` was 51% of XGBoost importance. For resolved markets, `market.liquidity` is the post-resolution value and collapses toward 0, creating target leakage. No historical liquidity snapshots exist for a clean as_of value.
- **Fix**: Removed `log_open_interest` from `ENSEMBLE_FEATURE_NAMES`. Documented in comments. Volume/liquidity features are all excluded for leakage reasons.

### CRITICAL 3: Pruning gate killed all price-derived features

- **Problem**: “Near-constant” gate used `n_unique / len(col) < 0.05`. Binary features (e.g. `is_weekend`: 2 unique values in 12k samples ≈ 0.016%) were dropped. So were low-cardinality useful features (price_bucket, category_encoded, etc.).
- **Fix**: Replaced with a “dominant value” check: if a single value accounts for >97% of the column, treat as near-constant. Binary and low-cardinality features are no longer incorrectly pruned.

### CRITICAL 4: Tradeable-range filter only on snapshot markets

- **Problem**: Without `--snapshot-only`, many markets used fallback “as_of” price. For resolved markets that fallback was often the settlement price (0 or 1) — direct target leakage. The tradeable-range filter (skip price &lt; 0.10 or &gt; 0.90) was only applied when a snapshot-derived price existed, so ~73% of training data could still use 0/1 as `price_yes`.
- **Fix**: Tradeable-range filter now applies to **all** markets: if `tradeable_range` is set and we have a price (snapshot or fallback), we skip markets outside [lo, hi]. For fallback prices that are 0 or 1, they are now excluded when using e.g. `--tradeable-range 0.10,0.90`.

### CRITICAL 5: as_of_days inconsistency

- **Problem**: Snapshot lookup in `train_ensemble.py` used `resolved_at - _AS_OF_DAYS` (e.g. 7 days), but `build_training_matrix` always computed features at `resolved_at - 1 day`. Mismatch between the price’s as_of and the time features’ as_of.
- **Fix**: `build_training_matrix()` now accepts `as_of_days` and uses it to compute the feature as_of. `train_ensemble.py` passes `_AS_OF_DAYS` through so snapshot lookup and feature computation use the same window.

### Dead / misleading code

- **cross_platform_spread**: Always 0.0 (no cross-platform matching in serving). Removed from `ENSEMBLE_FEATURE_NAMES`.
- **Fee logging**: Training and profit simulation logs now say “fee-free” and “fee-free + 1.5% slippage” instead of “2% on winnings.”
- **price_yes falsy check**: Replaced `if market.price_yes` with `if market.price_yes is not None` so 0.0 is a valid price (in `training_features.py` and `ensemble.py`).
- **Wilcoxon test**: Switched to one-sided (`alternative='greater'`) for calibration vs model comparison.
- **Platt scaling**: When the calibration window (Window B) is small (~60 samples), isotonic regression overfits. Post-calibrator now uses Platt (logistic) scaling for small N and isotonic for larger N.
- **Dead temporal_split**: Unused function removed from `train_ensemble.py`.
- **Cutoff date logging**: Fixed indexing so we don’t use matrix indices into the unfiltered markets list; approximate cutoff is derived from proportions.

---

## 4. Other fixes

| Fix | Location | Description |
|-----|----------|-------------|
| **Intra-market arb** | `ml/strategies/intra_market_arbitrage.py` | Arb detection now uses **orderbook best ask** for YES and NO when available, not mid-prices. Real arb = buy both sides at ask for &lt; $1 total. Falls back to mid when no orderbook. |
| **Execution simulator** | `ml/evaluation/execution_simulator.py` | Seeded RNG (`np.random.RandomState(seed)`) for deterministic backtests; all `np.random.*` replaced with `self.rng.*`. |

---

## 5. New implementations (L4, L6, L8)

### Layer 4: Market Making Engine

- **File**: `ml/strategies/market_making.py`
- **Description**: Avellaneda-Stoikov (2008) style market making adapted for prediction markets:
  - Binary terminal value (0 or 1 at resolution), resolution deadline, information events, maker rebates.
  - Reservation price and optimal spread from inventory, volatility, time to resolution, risk aversion (`gamma`), order intensity (`kappa`).
  - Config: min/max spread bps, max inventory, quote size, min time-to-resolution, kill switch, maker rebate fraction, min book depth.
  - Outputs: two-sided quotes (bid/ask), quote actions (POST/CANCEL/WIDEN/HOLD), backtesting support.
- **References**: Polymarket MM docs (market-makers/trading, maker-rebates); Avellaneda & Stoikov (2008).

### Layer 6: Retrain ensemble on honest foundations

- **Training run** (after all fixes): `--tradeable-range 0.10,0.90 --as-of-days 1` (no `--snapshot-only`; tradeable-range now applies to all markets).
- **Results**:
  - 7,671 usable markets; 0% near 0, 0% near 1 (tradeable-range filter working).
  - Class balance: 37.3% YES / 62.7% NO (test: 14.3% YES).
  - **Brier**: Ensemble 0.0878 vs market baseline 0.1086 → **19.1% improvement**.
  - XGBoost and LightGBM both included (~49% weight each).
  - **Top features**: price_bucket (33%), price_yes (21%), calibration_bias (16%), volatility_20 (8%), category_encoded (8%).
  - **Profit simulation**: $72.55 over 1,535 trades, 75% win rate (fee-free + 1.5% slippage).
  - **Validation gates**: 4/5 passed. Calibration quality gate fails (max bin deviation 19.8% &gt; 10% threshold).
  - Post-calibrator **hurts** performance; correctly not saved.

### Layer 8: Model monitoring and retrain triggers

- **File**: `ml/evaluation/model_monitor.py`
- **Description**: Production monitoring for the deployed ensemble:
  - Feature distribution drift (KS test vs training baseline).
  - Rolling Brier degradation vs baseline.
  - Edge decay (rolling profitability).
  - Concept drift; model age alert; retrain cooldown.
- **State**: `ml/saved_models/monitor_state.json` (training baseline and config).
- **Integration**: Training pipeline writes baseline after a successful train. Scheduler’s resolution-scoring loop records resolved predictions and runs the monitor; can emit retrain signals.

### Layer 7: Optimism Tax

- **Status**: **Deferred**. Requires a period of live (or high-fidelity) market data collection to measure maker/taker asymmetry on our specific markets. Not implemented in this session.

---

## 6. What remains

- **Calibration quality gate**: Max bin deviation 19.8% (threshold 10%). Needs more data or improved calibration method.
- **Orderbook features**: Still zero variance in training (no orderbook snapshots in DB for historical markets). Need to collect orderbook snapshots to unlock the 8 orderbook features.
- **Layer 7 (Optimism Tax)**: Implement after data collection period.
- **Market making execution path**: Core AS logic is in place; integration with live two-sided quoting and order management is a separate step.

---

*Use this doc when reviewing “what was audited, what was fixed, and what was implemented” in the 2026-02-28 session. For roadmap status see [PROFITABILITY_ROADMAP.md](PROFITABILITY_ROADMAP.md); for business metrics see [BUSINESS_DOCUMENTATION.md](BUSINESS_DOCUMENTATION.md).*
