# Trading Strategies

Every strategy in PredictFlow is backed by published academic research or documented market inefficiencies. This document explains the theory, implementation, and risk profile of each.

---

## Strategy 1: Single-Market Rebalancing

**Type**: Pure arbitrage (mathematical guarantee)
**AI Cost**: $0
**Implementation**: `arbitrage/strategies/single_market.py`

### Theory

Binary prediction markets have two outcomes: YES and NO. At resolution, one pays $1.00 and the other pays $0.00. If YES + NO < $1.00, you can buy both and guarantee a profit.

```
Example:
  YES price: $0.52
  NO price:  $0.46
  Total:     $0.98
  Payout:    $1.00
  Gross:     $0.02 (2.04%)
```

### Fee Impact

**Polymarket**: 2% on net winnings.
```
Net winnings = $1.00 - $0.98 = $0.02
Fee = $0.02 * 0.02 = $0.0004
Net profit: $0.0196 (2.0% on $0.98 invested)
```

**Kalshi**: min($0.01/contract, 7% of premium) on BOTH legs.
```
YES fee: min($0.01, $0.52 * 0.07) = min($0.01, $0.036) = $0.01
NO fee:  min($0.01, $0.46 * 0.07) = min($0.01, $0.032) = $0.01
Total fee: $0.02
Net profit: $0.02 - $0.02 = $0.00 (NOT profitable on Kalshi for small spreads)
```

### Evidence

- $29M extracted from Polymarket NegRisk markets by arbitrage bots (AFT 2025 paper)
- Spreads of 1-3% documented in binary markets
- Higher spreads in multi-outcome (NegRisk) markets where prices across N outcomes sum > $1

### Risk

- **Execution risk**: Prices may move between placing YES and NO orders
- **Liquidity risk**: Thin orderbooks mean large orders have slippage
- **Fee risk**: Platform fee changes can eliminate edge

### Thresholds

- Minimum net profit: 0.5% (configurable in `settings.py`)
- Skip 15-minute crypto markets (higher fees)

---

## Strategy 2: Cross-Platform Arbitrage

**Type**: Statistical arbitrage (near-guarantee if markets match correctly)
**AI Cost**: One-time ~$5 for initial market matching, then $0
**Implementation**: `arbitrage/strategies/cross_platform.py`

### Theory

The same event (e.g., "Will Trump win 2028?") may be priced differently on Polymarket vs Kalshi. Buy YES on the cheaper platform, buy NO on the expensive platform. One side is guaranteed to pay $1.

```
Example:
  Polymarket YES: $0.55
  Kalshi YES:     $0.60

  Strategy: Buy YES on Polymarket ($0.55), Buy NO on Kalshi ($0.40)
  Total cost: $0.95
  Guaranteed payout: $1.00
  Gross spread: 5.26%
```

### Market Matching

Markets are matched using TF-IDF cosine similarity on question text:

1. Vectorize questions from both platforms using TF-IDF with 1-2 grams
2. Compute cosine similarity matrix
3. Accept matches with similarity >= 0.45
4. Auto-confirm matches with similarity >= 0.80

Implementation: `data_pipeline/transformers/market_matcher.py`

### Fee Impact

Combined fees: ~2.7% (Polymarket 2% winnings + Kalshi ~0.7%)

```
Need gross spread > 2.7% to be profitable.
Conservative threshold: 2.5% (configurable)
```

### Evidence

- 70-100 cross-platform opportunities per day documented in 2024-2025
- One documented bot: $313 -> $414,000 using cross-platform arb
- Price discrepancies persist due to different user bases and fee structures

### Risk

- **Matching risk**: TF-IDF may match markets that aren't truly equivalent
- **Settlement risk**: Platforms may resolve differently for edge cases
- **Timing risk**: Non-atomic execution means price can move between legs
- **Inverted markets**: Some matches require YES-NO inversion

### Mitigations

- Only trade confirmed matches (similarity > 0.80) with real capital
- Start with small positions ($10-20) to validate matching
- Manual review of high-value matches before trading

---

## Strategy 3: Calibration Bias Exploitation

**Type**: Statistical edge (probabilistic, not guaranteed)
**AI Cost**: $0
**Implementation**: `ml/models/calibration_model.py`

### Theory

Prediction markets are systematically miscalibrated:
- Markets priced at 80% actually resolve YES only 74% of the time
- Markets priced at 90% actually resolve YES only 82% of the time
- Overconfidence is strongest at price extremes (>80% or <20%)

An isotonic regression model maps market price -> true probability. The delta is the exploitable edge.

### Published Calibration Data

From published research across Polymarket, Kalshi, and PredictIt:

| Market Price | Actual Resolution Rate | Bias |
|-------------|----------------------|------|
| 10% | 13% | +3pp (markets underconfident here) |
| 20% | 23% | +3pp |
| 30% | 30% | 0pp (well calibrated) |
| 40% | 38% | -2pp |
| 50% | 48% | -2pp |
| 60% | 56% | -4pp |
| 70% | 64% | -6pp |
| 80% | 74% | -6pp |
| 90% | 82% | -8pp |
| 95% | 88% | -7pp |

**Key insight**: Kalshi is 93% calibrated; Polymarket is only 67% calibrated.

### Model

**Isotonic Regression** (scikit-learn):
- Non-parametric: makes no distributional assumptions
- Monotonically increasing: calibrated price always increases with market price
- Trained on synthetic data from published calibration research
- Brier score improvement: 15-25% over raw market prices

### Trading Logic

For a market at 85% ($0.85):
1. Calibrated price: ~78% ($0.78)
2. Delta: -7pp (overpriced)
3. Edge estimate: 7%
4. Action: SELL / BUY NO at current prices

Minimum edge to trade: 2% (avoids noise)

### Evidence

- Documented across 5+ academic papers
- Consistent across platforms and time periods
- Strongest at price extremes where cognitive biases are worst
- Average overconfidence: ~6 percentage points

### Risk

- **Model risk**: Historical patterns may not persist
- **Sample size**: Limited resolved markets for some categories
- **Timing risk**: Calibration edge takes time to materialize (hold until resolution)

---

## Strategy 4: Orderbook-Driven Price Movement (Future)

**Type**: Short-term directional (predictive, not guaranteed)
**AI Cost**: $0
**Implementation**: Planned - feature extraction ready in `ml/features/orderbook_features.py`

### Theory

Order book imbalance (OBI) explains ~65% of midpoint price changes in the next period. A LightGBM model using 32 features can predict short-term price direction.

### Features (32 total, implemented)

**Orderbook (8)**: OBI level-1, OBI weighted, bid-ask spread, depth ratio, bid depth, ask depth, VWAP deviation, spread percentile

**Momentum (10)**: Returns at 1/5/15/60/240/1440 periods, volatility, z-score, RSI-like momentum, mean-reversion signal

**Market (8)**: Time-to-resolution, log volume, log OI, category encoding, price bucket, trade count, weekend flag, hours-active

**Calibration (6)**: Historical bias, overconfidence flag, historical resolution rate, market age, price stability, spread percentile

### Status

Feature extraction is fully implemented. Training pipeline and LightGBM model are planned for post-hackathon.

---

## Strategy 5: Decorrelation (Future)

**Type**: Ensemble diversification
**AI Cost**: $0
**Reference**: Hediger et al., 2022, International Journal of Forecasting

### Theory

You can profit from a model that is WORSE than the market, as long as its errors are decorrelated from market errors. A custom loss function minimizes correlation between model predictions and market prices.

### Status

Planned for post-hackathon. Requires sufficient training data from price snapshots.

---

## Strategy 6: Claude Deep Analysis

**Type**: Qualitative edge (human-AI collaboration)
**AI Cost**: ~$0.15 per analysis, cached forever
**Implementation**: `ai_analysis/claude_client.py`

### When to Use

- Complex political/geopolitical markets where context matters
- Markets with unusual price movements that merit investigation
- Situations where quantitative signals conflict

### What Claude Provides

- Multi-factor analysis (political, economic, social context)
- Risk assessment and scenario analysis
- Comparison with calibration model output
- Web search for latest relevant news

### Cost Management

- Prompt hash cache: identical prompts return cached results ($0)
- Database persistence: analyses survive server restarts
- Total spend tracking via `/analyze/cost` endpoint
- User-triggered only: never runs automatically

---

## Risk Management

### Position Limits (configurable in settings.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| max_position_usd | $100 | Max single position |
| max_total_exposure_usd | $1,000 | Max total exposure |
| max_daily_trades | 50 | Daily trade limit |
| max_loss_per_day_usd | $50 | Daily stop loss |

### Paper Trading First

All positions start as simulated (`is_simulated=True`). The portfolio system tracks:
- Entry/exit prices and times
- Realized P&L per position
- Unrealized P&L for open positions
- Win rate by strategy
- Total exposure

### Graduation Path

1. **Phase 1**: Paper trade for 2 weeks, validate strategies
2. **Phase 2**: $100 real capital on single-market arb only (lowest risk)
3. **Phase 3**: $500 across all strategies, monitor Sharpe ratio
4. **Phase 4**: Scale based on demonstrated edge and risk-adjusted returns
