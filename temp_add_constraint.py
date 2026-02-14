import sqlite3

conn = sqlite3.connect('data/markets.db')
cursor = conn.cursor()

# Add unique constraint
cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS uq_price_market_time ON price_snapshots(market_id, timestamp)')
conn.commit()
print('Unique constraint added')

# Verify it exists
cursor.execute("SELECT sql FROM sqlite_master WHERE type='index' AND name='uq_price_market_time'")
result = cursor.fetchone()
print(f'Constraint: {result[0] if result else "Not found"}')

conn.close()
