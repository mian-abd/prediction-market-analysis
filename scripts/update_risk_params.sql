-- Update Risk Parameters for Demo
-- Loosens stop-loss, lowers confidence threshold, accepts medium-quality signals

-- Update ensemble config for wider stops and longer hold
UPDATE auto_trading_configs
SET
    stop_loss_pct = 0.25,              -- 15% → 25%
    min_confidence = 0.5,              -- 0.7 → 0.5
    min_quality_tier = 'medium',       -- Accept medium+ quality signals (was 'high')
    close_on_signal_expiry = 1
WHERE strategy = 'ensemble';

-- Verify changes
SELECT 'Updated ensemble config:' as label;
SELECT strategy, stop_loss_pct, min_confidence, min_quality_tier, min_net_ev, is_enabled
FROM auto_trading_configs
WHERE strategy = 'ensemble';
