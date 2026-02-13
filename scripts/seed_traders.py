"""Seed database with sample traders for copy trading testing."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select
from db.database import async_session
from db.models import TraderProfile, PortfolioPosition
from datetime import datetime, timedelta
import random


async def seed_traders():
    """Create sample trader profiles with realistic stats."""

    async with async_session() as session:
        # Check if traders already exist
        result = await session.execute(select(TraderProfile))
        existing = result.scalars().all()
        if existing:
            print(f"Found {len(existing)} existing traders. Skipping seed.")
            return

        # Sample trader data
        traders = [
            {
                "user_id": "trader_alice",
                "display_name": "Alice Chen",
                "bio": "Quantitative analyst focusing on political markets. 5+ years experience in prediction markets.",
                "total_pnl": 12450.75,
                "roi_pct": 24.9,
                "win_rate": 68.5,
                "total_trades": 127,
                "winning_trades": 87,
                "avg_trade_duration_hrs": 48.3,
                "risk_score": 4,
                "max_drawdown": -850.25,
                "follower_count": 23,
                "is_public": True,
                "accepts_copiers": True,
            },
            {
                "user_id": "trader_bob",
                "display_name": "Bob Martinez",
                "bio": "Sports betting specialist with strong track record in entertainment markets.",
                "total_pnl": 8920.50,
                "roi_pct": 17.8,
                "win_rate": 62.3,
                "total_trades": 203,
                "winning_trades": 126,
                "avg_trade_duration_hrs": 72.1,
                "risk_score": 6,
                "max_drawdown": -1200.00,
                "follower_count": 15,
                "is_public": True,
                "accepts_copiers": True,
            },
            {
                "user_id": "trader_carol",
                "display_name": "Carol Wang",
                "bio": "AI/tech market expert. Data-driven approach with emphasis on calibration.",
                "total_pnl": 15800.25,
                "roi_pct": 31.6,
                "win_rate": 71.2,
                "total_trades": 89,
                "winning_trades": 63,
                "avg_trade_duration_hrs": 36.5,
                "risk_score": 3,
                "max_drawdown": -420.50,
                "follower_count": 42,
                "is_public": True,
                "accepts_copiers": True,
            },
            {
                "user_id": "trader_david",
                "display_name": "David Johnson",
                "bio": "Crypto and finance markets. High risk, high reward strategy.",
                "total_pnl": 6230.00,
                "roi_pct": 12.5,
                "win_rate": 58.1,
                "total_trades": 156,
                "winning_trades": 91,
                "avg_trade_duration_hrs": 96.7,
                "risk_score": 8,
                "max_drawdown": -2100.75,
                "follower_count": 8,
                "is_public": True,
                "accepts_copiers": True,
            },
            {
                "user_id": "trader_emma",
                "display_name": "Emma Thompson",
                "bio": "Conservative long-term strategy. Focus on well-researched positions with strong fundamentals.",
                "total_pnl": 4560.80,
                "roi_pct": 9.1,
                "win_rate": 74.5,
                "total_trades": 51,
                "winning_trades": 38,
                "avg_trade_duration_hrs": 168.2,
                "risk_score": 2,
                "max_drawdown": -180.25,
                "follower_count": 31,
                "is_public": True,
                "accepts_copiers": True,
            },
            {
                "user_id": "user_1",  # Default demo user
                "display_name": "Demo User",
                "bio": "Testing copy trading platform features.",
                "total_pnl": 2100.50,
                "roi_pct": 4.2,
                "win_rate": 55.0,
                "total_trades": 20,
                "winning_trades": 11,
                "avg_trade_duration_hrs": 24.0,
                "risk_score": 5,
                "max_drawdown": -300.00,
                "follower_count": 0,
                "is_public": False,
                "accepts_copiers": False,
            },
        ]

        # Create trader profiles
        for trader_data in traders:
            trader = TraderProfile(**trader_data)
            session.add(trader)

        await session.commit()
        print(f"[OK] Created {len(traders)} trader profiles")

        # Verify
        result = await session.execute(select(TraderProfile))
        all_traders = result.scalars().all()
        print(f"[OK] Database now has {len(all_traders)} traders")

        # Show top 3 by P&L
        result = await session.execute(
            select(TraderProfile)
            .where(TraderProfile.is_public == True)
            .order_by(TraderProfile.total_pnl.desc())
            .limit(3)
        )
        top_traders = result.scalars().all()

        print("\nTop 3 Traders by P&L:")
        for i, trader in enumerate(top_traders, 1):
            print(f"  {i}. {trader.display_name}: ${trader.total_pnl:.2f} ({trader.win_rate:.1f}% win rate)")


if __name__ == "__main__":
    asyncio.run(seed_traders())
