-- Reset Auto Portfolio for Demo
-- Closes all open auto positions at current market price and archives old signals

-- Close all open auto positions at current market price (only where price is available)
UPDATE portfolio_positions
SET
    exit_time = datetime('now'),
    exit_price = (SELECT price_yes FROM markets WHERE markets.id = portfolio_positions.market_id),
    realized_pnl = CASE
        WHEN side = 'yes' THEN
            ((SELECT price_yes FROM markets WHERE markets.id = portfolio_positions.market_id) - entry_price) * quantity
        ELSE
            (entry_price - (SELECT price_yes FROM markets WHERE markets.id = portfolio_positions.market_id)) * quantity
    END
WHERE portfolio_type = 'auto'
  AND exit_time IS NULL
  AND (SELECT price_yes FROM markets WHERE markets.id = portfolio_positions.market_id) IS NOT NULL;

-- Archive old ensemble signals
UPDATE ensemble_edge_signals
SET expired_at = datetime('now')
WHERE expired_at IS NULL;

-- Archive old elo signals
UPDATE elo_edge_signals
SET expired_at = datetime('now')
WHERE expired_at IS NULL;

-- Verification queries
SELECT 'Auto positions closed:' as label, COUNT(*) as count
FROM portfolio_positions
WHERE portfolio_type = 'auto' AND exit_time IS NOT NULL AND exit_time > datetime('now', '-1 minute');

SELECT 'Open auto positions remaining:' as label, COUNT(*) as count
FROM portfolio_positions
WHERE portfolio_type = 'auto' AND exit_time IS NULL;

SELECT 'Active ensemble signals:' as label, COUNT(*) as count
FROM ensemble_edge_signals
WHERE expired_at IS NULL;

SELECT 'Active elo signals:' as label, COUNT(*) as count
FROM elo_edge_signals
WHERE expired_at IS NULL;
