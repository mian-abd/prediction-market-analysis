"""Enable ELO auto-trading with conservative initial settings.

Run this script to activate ELO strategy trading with small bankroll for testing.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from db.database import async_session
from db.models import AutoTradingConfig
from datetime import datetime


async def enable_elo_trading():
    """Enable ELO auto-trading with conservative initial parameters."""

    async with async_session() as session:
        # Get ELO config
        result = await session.execute(
            select(AutoTradingConfig).where(AutoTradingConfig.strategy == "elo")
        )
        elo_config = result.scalar_one_or_none()

        if not elo_config:
            print("ERROR: ELO config not found. Run admin seed first.")
            return False

        print(f"Current ELO config:")
        print(f"  is_enabled: {elo_config.is_enabled}")
        print(f"  bankroll: ${elo_config.bankroll}")
        print(f"  min_confidence: {elo_config.min_confidence}")
        print(f"  min_net_ev: {elo_config.min_net_ev}")
        print(f"  max_kelly_fraction: {elo_config.max_kelly_fraction}")
        print()

        # Update to conservative settings
        elo_config.is_enabled = True
        elo_config.bankroll = 100.0  # Start small
        elo_config.min_confidence = 0.6  # High confidence only
        elo_config.min_net_ev = 0.04  # 4% minimum edge
        elo_config.max_kelly_fraction = 0.015  # 1.5% max position
        elo_config.stop_loss_pct = 0.15  # 15% stop-loss
        elo_config.max_position_usd = 50.0  # Small max position
        elo_config.max_total_exposure_usd = 100.0  # Limited exposure
        elo_config.max_daily_trades = 10  # Limit trades per day
        elo_config.updated_at = datetime.utcnow()

        await session.commit()
        await session.refresh(elo_config)

        print("✅ ELO auto-trading ENABLED with conservative settings:")
        print(f"  is_enabled: {elo_config.is_enabled}")
        print(f"  bankroll: ${elo_config.bankroll}")
        print(f"  min_confidence: {elo_config.min_confidence}")
        print(f"  min_net_ev: {elo_config.min_net_ev}")
        print(f"  max_kelly_fraction: {elo_config.max_kelly_fraction}")
        print(f"  max_position_usd: ${elo_config.max_position_usd}")
        print(f"  max_total_exposure_usd: ${elo_config.max_total_exposure_usd}")
        print(f"  max_daily_trades: {elo_config.max_daily_trades}")
        print()
        print("📊 Monitor ELO performance at /api/v1/auto-trading/status")
        print("⚠️  Monitor for 1 week before scaling up")

        return True


if __name__ == "__main__":
    success = asyncio.run(enable_elo_trading())
    sys.exit(0 if success else 1)
