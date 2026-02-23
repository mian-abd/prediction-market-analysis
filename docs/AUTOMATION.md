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

---

## Hot/cold data split — keep Postgres lean, never lose history

The Railway Postgres volume is limited.  To prevent disk-full crashes the app
uses a hot/cold split:

- **Hot (Postgres):** keeps only the last `RETENTION_DAYS` (default 7) of
  `price_snapshots`, `orderbook_snapshots`, `arbitrage_opportunities`,
  `news_events`, `system_metrics`, `trader_activities`.
- **Cold (local disk):** old rows are exported to
  `data/archive/<table>/YYYY-MM-DD.parquet` and read by the training script.

Cleanup is **disabled by default** (`CLEANUP_ENABLED=false`).  You must run
the export script first so no data is ever deleted before it is safely archived.

### Step 1 — Export old data to local archive (run weekly, before any cleanup)

```bash
# Export rows older than 7 days to data/archive/
python scripts/export_archive_to_local.py

# Custom options
python scripts/export_archive_to_local.py --archive-dir data/archive --older-than-days 7
python scripts/export_archive_to_local.py --older-than-days 7 --include-arb   # also arb opportunities
python scripts/export_archive_to_local.py --older-than-days 7 --force          # re-export existing days
```

After export, check `data/archive/manifest_<run_id>.json` to confirm row counts
and file list look correct before enabling cleanup.

### Step 2 — Enable cleanup (only after verifying the archive)

Set these environment variables (locally in `.env`, or in Railway Variables):

```
CLEANUP_ENABLED=true
RETENTION_DAYS=7
CLEANUP_BATCH_SIZE=1000
```

The scheduler will then run **batched** deletes (1 000 rows per commit) every
~1 hour for rows older than `RETENTION_DAYS`, keeping WAL growth bounded and
preventing the disk-full panic that occurred previously.

**Important:** Always run `export_archive_to_local.py` before each cleanup
cycle (i.e. weekly) so new data is archived before being deleted.

### Step 3 — Train with full history (DB + archive)

Pass `--archive-dir` to the training script so it merges archive snapshots with
DB snapshots.  Markets that have no recent snapshot in DB will be supplemented
from the Parquet archive, giving the model access to the complete history.

```bash
python scripts/train_ensemble.py --archive-dir data/archive

# Or as part of the full retrain pipeline:
python scripts/run_full_retrain.py --export-db  # after exporting and committing models
```

### Ongoing workflow (no data loss)

```
1. (weekly) python scripts/export_archive_to_local.py --older-than-days 7
2. Check manifest — confirm row counts
3. (first time only) Set CLEANUP_ENABLED=true in .env / Railway
4. (weekly) python scripts/train_ensemble.py --archive-dir data/archive
5. git add ml/saved_models/ && git commit -m "Retrain" && git push
```

Data is never deleted from Postgres without first being archived locally.  The
archive grows on your local machine and is only read by the training script —
it is not committed to git and not deployed to Railway.
