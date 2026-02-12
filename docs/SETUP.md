# Setup & Deployment Guide

## Prerequisites

- Python 3.11+ (3.13 recommended)
- Node.js 18+ (for frontend)
- Git

## Quick Start

### 1. Clone and Configure

```bash
git clone <repo-url>
cd prediction-market-analysis

# Create environment file
copy .env.example .env
# Edit .env with your API keys
```

Required `.env` values:
```env
ANTHROPIC_API_KEY=sk-ant-...   # For Claude analysis (optional for core functionality)
```

Optional `.env` values:
```env
DATABASE_URL=sqlite+aiosqlite:///./data/markets.db
PRICE_POLL_INTERVAL_SEC=60
ORDERBOOK_POLL_INTERVAL_SEC=300
MARKET_REFRESH_INTERVAL_SEC=3600
MIN_SINGLE_MARKET_PROFIT_PCT=0.5
MIN_CROSS_PLATFORM_SPREAD_PCT=2.5
MAX_POSITION_USD=100.0
MAX_TOTAL_EXPOSURE_USD=1000.0
```

### 2. Backend Setup

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Initialize Database

Database tables are created automatically on first startup. No migration step needed.

```bash
# Optional: Run data pipeline once to seed initial data
python -m data_pipeline.scheduler
```

### 4. Start Backend

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The backend will:
1. Create all database tables
2. Start background data pipeline
3. Begin collecting markets from Polymarket + Kalshi
4. Start price snapshots every 60 seconds
5. Run arbitrage scanning every 5 minutes
6. Run cross-platform matching every hour

**Verify**: `http://localhost:8000/api/v1/health` should return `{"status": "ok"}`

### 5. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

**Open**: `http://localhost:5173`

The Vite dev server proxies `/api/*` requests to `http://localhost:8000`.

### 6. Production Build

```bash
cd frontend
npm run build
# Static files in frontend/dist/
```

---

## Commands Reference

| Command | Description |
|---------|-------------|
| `uvicorn api.main:app --reload` | Start backend with hot-reload |
| `python -m data_pipeline.scheduler` | Run pipeline once (seeding) |
| `cd frontend && npm run dev` | Start frontend dev server |
| `cd frontend && npm run build` | Production frontend build |

### Windows vs Linux/Mac

| Operation | Windows (PowerShell) | Linux/Mac |
|-----------|---------------------|-----------|
| Activate venv | `.\venv\Scripts\Activate.ps1` | `source venv/bin/activate` |
| Set env var | `$env:VAR="value"` | `export VAR="value"` |
| Run backend | `python -m uvicorn api.main:app` | Same |
| Run frontend | `npm run dev` | Same |

---

## Architecture Notes

### Background Pipeline

The data pipeline runs as an asyncio background task within the FastAPI process. It starts automatically on server startup via the `lifespan` context manager.

**Pipeline timing:**
- Price snapshots: every 60 seconds
- Arbitrage scan: every 5 minutes
- Orderbook collection: every 5 minutes
- Market refresh: every 1 hour
- Cross-platform matching: after each market refresh

### Database

SQLite file at `data/markets.db`. Grows at approximately:
- ~50KB per hour (price snapshots)
- ~10KB per hour (orderbook snapshots)
- ~1MB per market refresh (market data)

For production, migrate to PostgreSQL by changing `DATABASE_URL`:
```env
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/predictflow
```

### API Keys

**Required for full functionality:**
- `ANTHROPIC_API_KEY`: Claude analysis (~$0.15 per unique analysis)

**Not required (free APIs):**
- Polymarket Gamma API: public, no auth
- Polymarket CLOB API: read-only, no auth
- Kalshi API: public market data, no auth

### Cost Breakdown

| Component | Cost | Frequency |
|-----------|------|-----------|
| Data collection | $0 | Continuous |
| Arbitrage scanning | $0 | Every 5 min |
| ML predictions | $0 | On-demand |
| Orderbook collection | $0 | Every 5 min |
| Cross-platform matching | $0 | Every 1 hr |
| Claude analysis | ~$0.15 | User-triggered |

**Estimated monthly cost**: $5-15 for moderate Claude usage.

---

## Troubleshooting

### "Failed to load dashboard" in frontend
- Check that backend is running at `http://localhost:8000`
- Verify: `curl http://localhost:8000/api/v1/health`

### No markets showing
- Wait 2-3 minutes for initial data collection
- Check backend logs for collection errors
- Verify internet connectivity (Polymarket/Kalshi APIs)

### No arbitrage opportunities
- Cross-platform matching must run first (wait for initial market refresh)
- Single-market arb requires YES + NO < $1.00 (spreads may be small)
- Check `min_single_market_profit_pct` threshold in settings

### Orderbook collection fails
- Polymarket CLOB API may rate-limit aggressive requests
- Collection includes 200ms delay between requests
- Only top 50 markets by volume are collected

### Claude analysis fails
- Check `ANTHROPIC_API_KEY` in `.env`
- Verify API key has sufficient credits
- Check `/analyze/cost` endpoint for spend tracking

### Database grows too large
- Price snapshots accumulate over time
- To trim: `DELETE FROM price_snapshots WHERE timestamp < datetime('now', '-7 days')`
- Consider PostgreSQL for production workloads

---

## Paper Trading Setup

### Opening Positions

```bash
# Open a position via API
curl -X POST http://localhost:8000/api/v1/portfolio/positions \
  -H "Content-Type: application/json" \
  -d '{
    "market_id": 42,
    "side": "yes",
    "entry_price": 0.72,
    "quantity": 100,
    "strategy": "calibration"
  }'
```

### Checking Performance

```bash
# Portfolio summary
curl http://localhost:8000/api/v1/portfolio/summary

# Open positions with unrealized P&L
curl http://localhost:8000/api/v1/portfolio/positions?status=open

# Closed positions with realized P&L
curl http://localhost:8000/api/v1/portfolio/positions?status=closed
```

### Closing Positions

```bash
curl -X POST http://localhost:8000/api/v1/portfolio/positions/1/close \
  -H "Content-Type: application/json" \
  -d '{"exit_price": 0.80}'
```

### Strategy Validation Checklist

Before moving to real capital:

- [ ] Paper traded for 2+ weeks
- [ ] Positive total P&L across 50+ trades
- [ ] Win rate > 55% for calibration strategy
- [ ] Sharpe ratio > 1.0
- [ ] No single loss > 5% of portfolio
- [ ] Verified cross-platform matches are correct
- [ ] Tested with small real positions ($10-20)
