# Data Collection & Model Learning (Automation)

## What the live app does automatically

When the backend is running (e.g. on Railway), the **scheduler** already:

| Task | Frequency | Purpose |
|------|-----------|--------|
| Collect markets | ~1 h | Refresh active markets from Polymarket/Kalshi |
| Collect prices | ~20 s | Live prices for signals and PnL |
| Collect orderbooks | ~2 min | Depth for OBI overlay and slippage |
| Scan ensemble edges | ~2 min | ML-based edge detection |
| Scan Elo edges | ~10 min | Tennis (ATP/WTA) + UFC |
| Auto paper-trade | ~2 min | Open positions from signals |
| Auto-close | Every cycle | Stop-loss, trailing stop, time-decay exits |
| **Refresh confidence adjuster** | ~30 min | **Learn from closed trades** (realized PnL by segment) |
| Score resolved signals | ~30 min | Backfill outcome for training data |

So **live data collection** and **learning from streams** (confidence adjuster using your closed paper trades) are already automated. No extra setup needed.

---

## Periodic retrain (use latest data and retrain models)

To refresh **resolved market data**, **Elo ratings**, and **ensemble models** (so the app uses the best possible models), run the full pipeline periodically (e.g. weekly).

### One command (recommended)

```bash
# Full pipeline: backfill → tennis Elo → UFC Elo → train ensemble
python scripts/run_full_retrain.py

# Also export Elo to DB (do this when deploying to Railway so API serves latest Elo)
python scripts/run_full_retrain.py --export-db

# Only refresh data + ensemble (skip Elo rebuild)
python scripts/run_full_retrain.py --skip-elo
```

### After retrain: deploy to Railway

1. Commit the updated model artifacts and push:
   ```bash
   git add ml/saved_models/
   git commit -m "Retrain: updated ensemble and Elo models"
   git push origin main
   ```
2. Railway will redeploy; the app will load the new `ml/saved_models/` from the repo.

---

## Scheduling the retrain (optional)

- **Windows (Task Scheduler):** Create a weekly task that runs:
  `d:\prediction-market-analysis\venv\Scripts\python.exe d:\prediction-market-analysis\scripts\run_full_retrain.py --export-db`
- **Linux / cron:** Add a weekly cron job, e.g.:
  `0 3 * * 0 cd /path/to/prediction-market-analysis && python scripts/run_full_retrain.py --export-db`
- **Railway:** No built-in cron; run the script locally (or from a separate cron host) and push the updated `ml/saved_models/` so the next deploy uses the new models.

---

## Summary

- **Live:** The app collects data and learns from closed trades automatically.
- **Periodic:** Run `scripts/run_full_retrain.py` (and optionally `--export-db`), then commit `ml/saved_models/` and push so Railway uses the latest models.
