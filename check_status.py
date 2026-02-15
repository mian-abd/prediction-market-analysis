import sqlite3

conn = sqlite3.connect('data/markets.db')
c = conn.cursor()

# Check auto-trading status
print('=== AUTO-TRADING STATUS ===')
c.execute('SELECT strategy, is_enabled, min_confidence, min_net_ev, min_quality_tier, stop_loss_pct FROM auto_trading_configs')
for row in c.fetchall():
    print(f'{row[0]}: enabled={row[1]}, min_conf={row[2]}, min_ev={row[3]}, tier={row[4]}, stop_loss={row[5]}')

print()
print('=== PORTFOLIO SUMMARY ===')
# Manual portfolio
c.execute('''SELECT COUNT(*), SUM(realized_pnl) FROM portfolio_positions
             WHERE portfolio_type IN ('manual', 'copy') AND exit_time IS NOT NULL''')
man_closed, man_realized = c.fetchone()
c.execute('''SELECT COUNT(*), SUM((SELECT price_yes FROM markets WHERE id = market_id) - entry_price) * quantity
             FROM portfolio_positions
             WHERE portfolio_type IN ('manual', 'copy') AND exit_time IS NULL AND side = 'yes' ''')
man_open_yes = c.fetchone()
c.execute('''SELECT COUNT(*), SUM((entry_price - (SELECT price_yes FROM markets WHERE id = market_id)) * quantity)
             FROM portfolio_positions
             WHERE portfolio_type IN ('manual', 'copy') AND exit_time IS NULL AND side = 'no' ''')
man_open_no = c.fetchone()
man_unrealized = (man_open_yes[1] or 0) + (man_open_no[1] or 0)
print(f'Manual: {man_closed} closed, realized=${man_realized or 0:.2f}, {man_open_yes[0] + man_open_no[0]} open, unrealized=${man_unrealized:.2f}')

# Auto portfolio
c.execute('''SELECT COUNT(*), SUM(realized_pnl) FROM portfolio_positions
             WHERE portfolio_type = 'auto' AND exit_time IS NOT NULL''')
auto_closed, auto_realized = c.fetchone()
c.execute('''SELECT COUNT(*), SUM((SELECT price_yes FROM markets WHERE id = market_id) - entry_price) * quantity
             FROM portfolio_positions
             WHERE portfolio_type = 'auto' AND exit_time IS NULL AND side = 'yes' ''')
auto_open_yes = c.fetchone()
c.execute('''SELECT COUNT(*), SUM((entry_price - (SELECT price_yes FROM markets WHERE id = market_id)) * quantity)
             FROM portfolio_positions
             WHERE portfolio_type = 'auto' AND exit_time IS NULL AND side = 'no' ''')
auto_open_no = c.fetchone()
auto_unrealized = (auto_open_yes[1] or 0) + (auto_open_no[1] or 0)
print(f'Auto: {auto_closed} closed, realized=${auto_realized or 0:.2f}, {auto_open_yes[0] + auto_open_no[0]} open, unrealized=${auto_unrealized:.2f}')

print()
print('=== ACTIVE SIGNALS ===')
c.execute('SELECT COUNT(*) FROM ensemble_edge_signals WHERE expired_at IS NULL')
print(f'Ensemble signals: {c.fetchone()[0]}')
c.execute('SELECT COUNT(*) FROM elo_edge_signals WHERE expired_at IS NULL')
print(f'Elo signals: {c.fetchone()[0]}')

print()
print('=== RECENT AUTO POSITIONS ===')
c.execute('''SELECT m.question, p.side, p.entry_price, p.realized_pnl, p.entry_time, p.exit_time
             FROM portfolio_positions p JOIN markets m ON p.market_id = m.id
             WHERE p.portfolio_type = 'auto' AND p.exit_time IS NOT NULL
             ORDER BY p.exit_time DESC LIMIT 5''')
for row in c.fetchall():
    print(f'{row[5][:16]} | {row[0][:40]} | {row[1]} @ {row[2]:.3f} | P&L: ${row[3]:.2f}')

conn.close()
