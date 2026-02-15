"""Monitor feature coverage for model training.

Tracks price snapshot and orderbook coverage across active markets.
Run daily to monitor Phase 2 data collection progress.
"""

import sqlite3
from datetime import datetime, timedelta

def main():
    conn = sqlite3.connect('data/markets.db')
    cursor = conn.cursor()

    # Active market count
    cursor.execute('SELECT COUNT(*) FROM markets WHERE is_active = 1')
    active_markets = cursor.fetchone()[0]

    # Markets with recent snapshots (24h)
    yesterday = (datetime.utcnow() - timedelta(days=1)).isoformat()
    cursor.execute('SELECT COUNT(DISTINCT market_id) FROM price_snapshots WHERE timestamp > ?', (yesterday,))
    markets_with_snapshots = cursor.fetchone()[0]

    # Markets with orderbook data
    cursor.execute('SELECT COUNT(DISTINCT market_id) FROM orderbook_snapshots')
    markets_with_orderbooks = cursor.fetchone()[0]

    # Avg snapshots per market
    cursor.execute('''
        SELECT AVG(cnt) FROM (
            SELECT COUNT(*) as cnt FROM price_snapshots ps
            JOIN markets m ON ps.market_id = m.id
            WHERE m.is_active = 1 AND ps.timestamp > ?
            GROUP BY market_id
        )
    ''', (yesterday,))
    avg_snapshots_result = cursor.fetchone()[0]
    avg_snapshots = avg_snapshots_result if avg_snapshots_result else 0

    print(f"Active Markets: {active_markets}")
    print(f"Snapshot Coverage (24h): {markets_with_snapshots} ({markets_with_snapshots/active_markets*100:.1f}%)")
    print(f"Orderbook Coverage: {markets_with_orderbooks} ({markets_with_orderbooks/active_markets*100:.1f}%)")
    print(f"Avg Snapshots/Market (24h): {avg_snapshots:.0f}")

    if avg_snapshots >= 60:
        print(f"✓ Momentum features ready (avg {avg_snapshots:.0f} >= 60 snapshots)")
    else:
        time_needed_min = (60 - avg_snapshots) / 3  # 3 snapshots per minute (20s interval)
        print(f"⏱ Need {60-avg_snapshots:.0f} more snapshots (~{time_needed_min:.0f} min for momentum features)")

    conn.close()

if __name__ == "__main__":
    main()
