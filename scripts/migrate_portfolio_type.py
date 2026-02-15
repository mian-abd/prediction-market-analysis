"""Idempotent migration: add portfolio_type column + seed AutoTradingConfig.

Safe to run multiple times. Uses PRAGMA table_info to check before ALTER TABLE.
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from db.database import engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def migrate():
    async with engine.begin() as conn:
        # 1. Check if portfolio_type column exists
        rows = await conn.execute(text("PRAGMA table_info(portfolio_positions)"))
        columns = {row[1] for row in rows.fetchall()}

        if "portfolio_type" not in columns:
            logger.info("Adding portfolio_type column to portfolio_positions...")
            await conn.execute(text(
                "ALTER TABLE portfolio_positions "
                "ADD COLUMN portfolio_type VARCHAR(10) NOT NULL DEFAULT 'manual'"
            ))
            logger.info("Column added.")
        else:
            logger.info("portfolio_type column already exists, skipping ALTER TABLE.")

        # 2. Tag existing auto positions
        result = await conn.execute(text(
            "UPDATE portfolio_positions SET portfolio_type = 'auto' "
            "WHERE strategy IN ('auto_ensemble', 'auto_elo') AND portfolio_type = 'manual'"
        ))
        logger.info(f"Tagged {result.rowcount} existing auto positions.")

        # 3. Create auto_trading_configs table if not exists
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS auto_trading_configs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy VARCHAR(50) NOT NULL UNIQUE,
                is_enabled BOOLEAN DEFAULT 0,
                min_quality_tier VARCHAR(10) DEFAULT 'high',
                min_confidence FLOAT DEFAULT 0.7,
                min_net_ev FLOAT DEFAULT 0.05,
                bankroll FLOAT DEFAULT 1000.0,
                max_kelly_fraction FLOAT DEFAULT 0.02,
                max_position_usd FLOAT DEFAULT 100.0,
                max_total_exposure_usd FLOAT DEFAULT 500.0,
                max_loss_per_day_usd FLOAT DEFAULT 25.0,
                max_daily_trades INTEGER DEFAULT 20,
                stop_loss_pct FLOAT DEFAULT 0.15,
                close_on_signal_expiry BOOLEAN DEFAULT 1,
                updated_at DATETIME
            )
        """))

        # 4. Seed configs (INSERT OR IGNORE = idempotent)
        await conn.execute(text(
            "INSERT OR IGNORE INTO auto_trading_configs "
            "(strategy, is_enabled, min_quality_tier, min_confidence, min_net_ev, "
            " bankroll, max_kelly_fraction, max_position_usd, "
            " max_total_exposure_usd, max_loss_per_day_usd, max_daily_trades, "
            " stop_loss_pct, close_on_signal_expiry) "
            "VALUES ('ensemble', 1, 'high', 0.7, 0.05, "
            " 1000.0, 0.02, 100.0, 500.0, 25.0, 20, 0.15, 1)"
        ))
        await conn.execute(text(
            "INSERT OR IGNORE INTO auto_trading_configs "
            "(strategy, is_enabled, min_quality_tier, min_confidence, min_net_ev, "
            " bankroll, max_kelly_fraction, max_position_usd, "
            " max_total_exposure_usd, max_loss_per_day_usd, max_daily_trades, "
            " stop_loss_pct, close_on_signal_expiry) "
            "VALUES ('elo', 0, 'high', 0.5, 0.03, "
            " 500.0, 0.02, 100.0, 500.0, 25.0, 20, 0.15, 1)"
        ))
        logger.info("AutoTradingConfig seeded (ensemble=enabled, elo=disabled).")

    # 5. Add index if missing (separate connection since CREATE INDEX isn't transactional in all cases)
    async with engine.begin() as conn:
        try:
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_position_portfolio_type "
                "ON portfolio_positions (portfolio_type)"
            ))
        except Exception:
            pass  # Index already exists

    logger.info("Migration complete.")


if __name__ == "__main__":
    asyncio.run(migrate())
