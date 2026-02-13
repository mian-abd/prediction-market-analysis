"""Check resolved markets available for ML training."""
import sqlite3
from datetime import datetime, timedelta

conn = sqlite3.connect('data/markets.db')
cursor = conn.cursor()

# Check for resolved markets
cursor.execute('''
    SELECT COUNT(*) as total_resolved
    FROM markets
    WHERE is_resolved = 1
''')
total_resolved = cursor.fetchone()[0]
print(f"Total resolved markets: {total_resolved}")

# Check resolved in last 90 days
ninety_days_ago = (datetime.utcnow() - timedelta(days=90)).isoformat()
cursor.execute('''
    SELECT COUNT(*) as recent_resolved
    FROM markets
    WHERE is_resolved = 1
    AND updated_at >= ?
''', (ninety_days_ago,))
recent_resolved = cursor.fetchone()[0]
print(f"Resolved in last 90 days: {recent_resolved}")

# Check if we have price snapshots for resolved markets
cursor.execute('''
    SELECT COUNT(DISTINCT m.id)
    FROM markets m
    JOIN price_snapshots ps ON m.id = ps.market_id
    WHERE m.is_resolved = 1
''')
resolved_with_prices = cursor.fetchone()[0]
print(f"Resolved markets with price history: {resolved_with_prices}")

# Sample some resolved markets
cursor.execute('''
    SELECT id, question, price_yes, is_resolved, updated_at
    FROM markets
    WHERE is_resolved = 1
    LIMIT 10
''')
print("\nSample resolved markets:")
for row in cursor.fetchall():
    print(f"  ID {row[0]}: {row[1][:60]}... (YES={row[2]}, resolved_at={row[4]})")

conn.close()
