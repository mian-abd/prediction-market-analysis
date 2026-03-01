"""Quick DB state check - fast queries only."""
import sqlite3, time
conn = sqlite3.connect("data/markets.db")
c = conn.cursor()

fast_queries = [
    ("Active markets (is_active=1)", "SELECT COUNT(*) FROM markets WHERE is_active=1"),
    ("Live (end_date future)", "SELECT COUNT(*) FROM markets WHERE is_active=1 AND end_date > datetime('now')"),
    ("Inactive markets", "SELECT COUNT(*) FROM markets WHERE is_active=0"),
    ("Total markets", "SELECT COUNT(*) FROM markets"),
    ("Active w/ token_id_yes", "SELECT COUNT(*) FROM markets WHERE is_active=1 AND token_id_yes IS NOT NULL"),
    ("Active w/ NULL end_date", "SELECT COUNT(*) FROM markets WHERE is_active=1 AND end_date IS NULL"),
    ("Price snapshots (approx)", "SELECT MAX(rowid) FROM price_snapshots"),
    ("Orderbook snapshots (approx)", "SELECT MAX(rowid) FROM orderbook_snapshots"),
]

for label, sql in fast_queries:
    t0 = time.time()
    c.execute(sql)
    val = c.fetchone()[0]
    dt = time.time() - t0
    print(f"{label}: {val:,} ({dt:.1f}s)")

print("\n--- Trade backfill stats (from checkpoint) ---")
import json
try:
    with open("data/checkpoints/trade_backfill_stats.json") as f:
        s = json.load(f)
    print(f"  Total trades stored: {s['total_trades']:,}")
    print(f"  Markets processed: {s['processed']:,}")
    print(f"  Success: {s['success']:,}, No trades: {s['no_trades']:,}")
except Exception as e:
    print(f"  Error: {e}")

print("\n--- Sample live markets ---")
c.execute("SELECT id, question, end_date FROM markets WHERE is_active=1 AND end_date > datetime('now') AND token_id_yes IS NOT NULL ORDER BY volume_total DESC NULLS LAST LIMIT 5")
for row in c.fetchall():
    print(f"  ID={row[0]}, end={row[1]}, q={row[2][:70] if row[2] else 'N/A'}")

conn.close()
