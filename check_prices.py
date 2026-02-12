import sqlite3

conn = sqlite3.connect('data/markets.db')
cursor = conn.cursor()

# Check price snapshots
cursor.execute('SELECT COUNT(*) FROM price_snapshots')
price_count = cursor.fetchone()[0]
print(f'Total price snapshots: {price_count}')

# Check markets
cursor.execute('SELECT COUNT(*) FROM markets')
market_count = cursor.fetchone()[0]
print(f'Total markets: {market_count}')

if price_count > 0:
    # Show top markets by snapshot count
    cursor.execute('''
        SELECT market_id, COUNT(*) as cnt
        FROM price_snapshots
        GROUP BY market_id
        ORDER BY cnt DESC
        LIMIT 5
    ''')
    print('\nTop 5 markets by snapshot count:')
    for row in cursor.fetchall():
        print(f'  Market ID {row[0]}: {row[1]} snapshots')

    # Show most recent snapshots
    cursor.execute('''
        SELECT market_id, timestamp, price_yes
        FROM price_snapshots
        ORDER BY timestamp DESC
        LIMIT 5
    ''')
    print('\nMost recent price snapshots:')
    for row in cursor.fetchall():
        print(f'  Market {row[0]}: {row[1]}, price_yes={row[2]:.4f}')
else:
    print('\n⚠️  NO PRICE SNAPSHOTS IN DATABASE!')
    print('This is why the price history chart is empty.')

conn.close()
