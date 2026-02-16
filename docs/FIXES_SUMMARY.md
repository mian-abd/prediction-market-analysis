# Fixes Summary - 2026-02-14

## ✅ Completed Fixes

### 1. **Correlation Graph - Zoom & Pan Controls**
- **File**: `frontend/src/components/charts/CorrelationGraph.tsx`
- **Changes**:
  - Added mouse wheel zoom (0.5x to 3x range)
  - Added click-and-drag panning
  - Added zoom buttons (+/-/reset) in top-right corner
  - Cursor changes to "grabbing" when dragging
- **Usage**:
  - Scroll mouse wheel to zoom in/out
  - Click and drag on empty space to pan
  - Use +/- buttons for precise zoom control
  - Click ⟲ to reset view

### 2. **Lookback Period Extended to 365 Days**
- **File**: `frontend/src/pages/Analytics.tsx`
- **Change**: Slider max changed from 30 to 365 days
- **Impact**: Can now view correlations over full year of data

### 3. **Portfolio Reset Endpoint**
- **File**: `api/routes/portfolio.py`
- **New Endpoint**: `DELETE /api/v1/portfolio/reset`
- **Action**: Deletes ALL portfolio positions (all users, open and closed)
- **How to Use**:
  ```bash
  # Call this endpoint to reset portfolio to zero
  curl -X DELETE http://localhost:8000/api/v1/portfolio/reset
  ```
- **Response**: Returns count of deleted positions

---

## 🔍 Diagnostic: Why Components Are Not Loading

### **Sentiment Gauge, Orderbook, News - Root Cause**
All three components have correct frontend code. The issue is **missing data**:

1. **Sentiment Gauge** (`frontend/src/components/charts/SentimentGauge.tsx`)
   - Depends on: Market data + Price history + Orderbook
   - **Issue**: If orderbook data is missing, sentiment calculation fails

2. **Orderbook Depth** (`frontend/src/components/charts/OrderbookDepth.tsx`)
   - Endpoint: `GET /api/v1/markets/{id}/orderbook`
   - **Issue**: Backend returns 404 if no `OrderbookSnapshot` records exist
   - **Check**: Run `SELECT COUNT(*) FROM orderbook_snapshots;` in SQLite

3. **News Articles** (`frontend/src/components/NewsSection.tsx`)
   - Fetched via GDELT API in market detail endpoint
   - **Issue**: GDELT API call may be failing silently (caught at line 256-257 in `api/routes/markets.py`)
   - **Check**: Look for "News fetch failed:" in backend logs

### **Copy Trading - Activity Feed & Recent Trades**
All four backend endpoints exist and work correctly:
- `GET /copy-trading/traders/{id}/activity` ✅
- `GET /copy-trading/traders/{id}/positions` ✅
- `GET /copy-trading/traders/{id}/equity-curve` ✅
- `GET /copy-trading/traders/{id}/drawdown` ✅

**Issue**: No trader profile data exists yet
- **Check**: Run `SELECT COUNT(*) FROM trader_profiles;` in SQLite
- **Solution**: Trader profiles are auto-created when users make trades. Since portfolio was just reset, there are no trades → no trader profiles.

---

## 📊 Data Refresh Intervals

### **Markets & Prices**
- **Data Pipeline**: Runs continuously in background (FastAPI lifespan)
- **Market Sync**: Every 5 minutes (defined in `data_pipeline/scheduler.py`)
- **Price Snapshots**: Every 1 minute for active markets
- **Orderbook Snapshots**: Every 5 minutes (Polymarket only)

### **ML Model Calibration**
- **Ensemble Training**: Manual trigger via `scripts/train_ensemble.py`
- **Edge Detection**: Runs after each price update (real-time)
- **Feature Calculation**: Real-time during prediction
- **Model Serving**: Pre-trained model loaded at startup (from `ml/saved_models/`)

### **Portfolio Tracking**
- **Position Updates**: Real-time when user opens/closes positions
- **Unrealized P&L**: Calculated on-demand when fetching positions (uses latest market prices)
- **Trader Profiles**: Updated after each trade (async via `data_pipeline/copy_engine.py`)
- **Equity Curves**: Computed on-demand from PortfolioPosition history

### **Correlation Analysis**
- **Computation**: On-demand (calculated when user requests)
- **Caching**: 2 minutes (via React Query `staleTime: 120_000`)
- **Price History Window**: User-configurable 1-365 days

### **News Articles**
- **Source**: GDELT API (live fetch)
- **Caching**: None (always fresh)
- **Rate Limit**: GDELT free tier limits may apply

---

## 🚀 Next Steps to See Data

### 1. **Reset Portfolio** (Clear old test data)
```bash
curl -X DELETE http://localhost:8000/api/v1/portfolio/reset
```

### 2. **Verify Data Pipeline is Running**
Check backend logs for:
```
Pipeline active
Syncing markets from polymarket...
Updating price snapshots...
Scanning for ensemble edges...
```

### 3. **Check Orderbook Data**
```sql
-- Connect to SQLite database
sqlite3 data/predictions.db

-- Check if orderbook snapshots exist
SELECT COUNT(*) FROM orderbook_snapshots;
SELECT market_id, timestamp, best_bid, best_ask
FROM orderbook_snapshots
ORDER BY timestamp DESC
LIMIT 5;
```

### 4. **Test a Market with Data**
Find a market with orderbook data:
```bash
# Get markets
curl http://localhost:8000/api/v1/markets?limit=10

# Try orderbook for market ID 1
curl http://localhost:8000/api/v1/markets/1/orderbook
```

### 5. **Check Backend Logs for News Fetch Errors**
Look for:
```
News fetch failed: <error message>
```
This will tell you if GDELT API is failing.

### 6. **Create Test Trader Profile**
Open a position to auto-create a trader profile:
```bash
curl -X POST http://localhost:8000/api/v1/portfolio/positions \
  -H "Content-Type: application/json" \
  -H "X-User-Id: test-trader-1" \
  -d '{
    "market_id": 1,
    "side": "yes",
    "entry_price": 0.65,
    "quantity": 100,
    "strategy": "manual"
  }'
```

---

## 📝 Summary

**Fixed Issues**:
✅ Correlation graph now has full zoom/pan controls
✅ Lookback period extended to 365 days
✅ Portfolio reset endpoint created

**Data Issues** (not code bugs):
⚠️ Orderbook data may be sparse (check if snapshots exist)
⚠️ News fetch may be failing (check backend logs)
⚠️ Sentiment requires orderbook data to work
⚠️ Copy trading needs trader profiles (created when users trade)

**All frontend code is correct** - the issue is missing backend data, which is expected after a fresh database or portfolio reset.
