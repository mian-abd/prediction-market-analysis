"""Data coverage report — shows exactly what data we have and what's missing.

Quick diagnostic script. Read-only, uses direct sqlite3 for speed.

Usage:
  python scripts/data_coverage_report.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import sqlite3
import json
from datetime import datetime, timedelta, timezone

DB_PATH = project_root / "data" / "markets.db"


def run_report():
    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.execute("PRAGMA query_only = 1")
    c = conn.cursor()

    cutoff_24h = (datetime.now(timezone.utc) - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S")

    # ── Markets ──────────────────────────────────────────────────────────────
    total          = c.execute("SELECT COUNT(*) FROM markets").fetchone()[0]
    resolved       = c.execute("SELECT COUNT(*) FROM markets WHERE resolution_value IS NOT NULL").fetchone()[0]
    active         = c.execute("SELECT COUNT(*) FROM markets WHERE is_active = 1").fetchone()[0]
    with_token     = c.execute("SELECT COUNT(*) FROM markets WHERE token_id_yes IS NOT NULL").fetchone()[0]
    with_condition = c.execute("SELECT COUNT(*) FROM markets WHERE condition_id IS NOT NULL").fetchone()[0]

    # ── Price Snapshots ───────────────────────────────────────────────────────
    total_snaps        = c.execute("SELECT COUNT(*) FROM price_snapshots").fetchone()[0]
    markets_with_snaps = c.execute("SELECT COUNT(DISTINCT market_id) FROM price_snapshots").fetchone()[0]

    resolved_with_snaps = c.execute("""
        SELECT COUNT(DISTINCT ps.market_id) FROM price_snapshots ps
        JOIN markets m ON m.id = ps.market_id
        WHERE m.resolution_value IS NOT NULL
    """).fetchone()[0]

    # Use a single pass to compute all bucket sizes efficiently
    snap_counts = c.execute("""
        SELECT market_id, COUNT(*) as cnt
        FROM price_snapshots GROUP BY market_id
    """).fetchall()
    snap_count_vals = [row[1] for row in snap_counts]
    avg_snaps = sum(snap_count_vals) / len(snap_count_vals) if snap_count_vals else 0
    gt_10  = sum(1 for v in snap_count_vals if v >= 10)
    gt_48  = sum(1 for v in snap_count_vals if v >= 48)
    gt_100 = sum(1 for v in snap_count_vals if v >= 100)
    gt_500 = sum(1 for v in snap_count_vals if v >= 500)

    snap_dates = c.execute("SELECT MIN(timestamp), MAX(timestamp) FROM price_snapshots").fetchone()
    oldest_snap, newest_snap = snap_dates

    snaps_24h   = c.execute("SELECT COUNT(*) FROM price_snapshots WHERE timestamp >= ?", (cutoff_24h,)).fetchone()[0]
    markets_24h = c.execute("SELECT COUNT(DISTINCT market_id) FROM price_snapshots WHERE timestamp >= ?", (cutoff_24h,)).fetchone()[0]

    # ── Orderbook Snapshots ───────────────────────────────────────────────────
    total_obs        = c.execute("SELECT COUNT(*) FROM orderbook_snapshots").fetchone()[0]
    markets_with_obs = c.execute("SELECT COUNT(DISTINCT market_id) FROM orderbook_snapshots").fetchone()[0]
    obs_24h          = c.execute("SELECT COUNT(*) FROM orderbook_snapshots WHERE timestamp >= ?", (cutoff_24h,)).fetchone()[0]
    ob_dates         = c.execute("SELECT MIN(timestamp), MAX(timestamp) FROM orderbook_snapshots").fetchone()
    oldest_ob, newest_ob = ob_dates

    # ── Trades ────────────────────────────────────────────────────────────────
    total_trades        = c.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    markets_with_trades = c.execute("SELECT COUNT(DISTINCT market_id) FROM trades").fetchone()[0]
    trades_24h          = c.execute("SELECT COUNT(*) FROM trades WHERE timestamp >= ?", (cutoff_24h,)).fetchone()[0]
    trade_dates         = c.execute("SELECT MIN(timestamp), MAX(timestamp) FROM trades").fetchone()
    oldest_trade, newest_trade = trade_dates

    # ── Gaps ──────────────────────────────────────────────────────────────────
    resolved_with_token_no_snaps = c.execute("""
        SELECT COUNT(*) FROM markets m
        WHERE m.resolution_value IS NOT NULL
          AND m.token_id_yes IS NOT NULL
          AND m.id NOT IN (SELECT DISTINCT market_id FROM price_snapshots)
    """).fetchone()[0]

    # ── Elo ratings ───────────────────────────────────────────────────────────
    try:
        elo_counts = dict(c.execute(
            "SELECT sport, COUNT(*) FROM elo_ratings GROUP BY sport"
        ).fetchall())
    except Exception:
        elo_counts = {}

    conn.close()

    # ── Print ──────────────────────────────────────────────────────────────────
    W = 62
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    print()
    print("=" * W)
    print("  DATA COVERAGE REPORT")
    print(f"  Generated: {now}")
    print("=" * W)

    print(f"\n{'MARKETS':-<{W}}")
    print(f"  Total:                  {total:>10,}")
    print(f"  Resolved:               {resolved:>10,}")
    print(f"  Active:                 {active:>10,}")
    print(f"  With token_id_yes:      {with_token:>10,}")
    print(f"  With condition_id:      {with_condition:>10,}")

    print(f"\n{'PRICE SNAPSHOTS':-<{W}}")
    print(f"  Total snapshots:        {total_snaps:>12,}")
    pct = markets_with_snaps / max(total, 1) * 100
    print(f"  Markets covered:        {markets_with_snaps:>8,} / {total:,}  ({pct:.1f}%)")
    r_pct = resolved_with_snaps / max(resolved, 1) * 100
    print(f"  Resolved covered:       {resolved_with_snaps:>8,} / {resolved:,}  ({r_pct:.1f}%)")
    print(f"  Avg snaps/market:       {avg_snaps:>10.1f}")
    print(f"  Date range:             {oldest_snap or 'N/A'}  ->  {newest_snap or 'N/A'}")
    print(f"  Last 24h snapshots:     {snaps_24h:>10,}")
    print(f"  Last 24h markets:       {markets_24h:>10,}")
    print(f"  Markets >= 10 snaps:    {gt_10:>10,}")
    print(f"  Markets >= 48 snaps:    {gt_48:>10,}  (2 days hourly)")
    print(f"  Markets >= 100 snaps:   {gt_100:>10,}  (4+ days hourly)")
    print(f"  Markets >= 500 snaps:   {gt_500:>10,}  (21+ days hourly)")

    print(f"\n{'ORDERBOOK SNAPSHOTS':-<{W}}")
    print(f"  Total snapshots:        {total_obs:>12,}")
    print(f"  Markets covered:        {markets_with_obs:>10,}")
    print(f"  Last 24h:               {obs_24h:>10,}")
    print(f"  Date range:             {oldest_ob or 'N/A'}  ->  {newest_ob or 'N/A'}")

    print(f"\n{'TRADE HISTORY':-<{W}}")
    print(f"  Total trades:           {total_trades:>12,}")
    print(f"  Markets covered:        {markets_with_trades:>10,}")
    print(f"  Last 24h trades:        {trades_24h:>10,}")
    print(f"  Date range:             {oldest_trade or 'N/A'}  ->  {newest_trade or 'N/A'}")

    if elo_counts:
        print(f"\n{'ELO RATINGS (DB)':-<{W}}")
        for sport, cnt in sorted(elo_counts.items()):
            print(f"  {sport:<20s}  {cnt:>8,} ratings")

    print(f"\n{'GAPS (ACTION NEEDED)':-<{W}}")
    resolved_no_snaps = resolved - resolved_with_snaps
    print(f"  Resolved missing snapshots:    {resolved_no_snaps:>8,}")
    print(f"    With token (backfillable):   {resolved_with_token_no_snaps:>8,}")
    print(f"  Active missing orderbooks:     {max(active - markets_with_obs, 0):>8,}")

    if resolved_with_token_no_snaps > 0:
        print(f"\n  [hint] Run: python scripts/prioritize_backfill.py")
    if total_trades < 50000:
        print(f"  [hint] trades low — collectors are still filling up")

    print()
    print("=" * W)

    # ── Active checkpoints ────────────────────────────────────────────────────
    ckpt_dir = project_root / "data" / "checkpoints"
    if ckpt_dir.exists():
        ckpts = list(ckpt_dir.glob("*.json"))
        if ckpts:
            print("\nACTIVE CHECKPOINTS:")
            for f in sorted(ckpts):
                try:
                    with open(f) as fh:
                        data = json.load(fh)
                    print(f"  {f.name}:")
                    for k in ("started_at", "last_updated", "processed",
                              "total_snapshots", "total_trades", "success",
                              "total_cycles", "success_rate"):
                        if k in data:
                            print(f"    {k}: {data[k]}")
                except Exception:
                    pass
            print()


if __name__ == "__main__":
    run_report()
