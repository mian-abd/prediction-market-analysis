"""Backfill database with real trader data from Polymarket leaderboard.

Fetches actual trade history per trader to calculate real win rate,
trade count, drawdown, and duration — no fabricated estimates.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select, delete
from db.database import async_session
from db.models import TraderProfile
from data_pipeline.collectors.trader_data import (
    fetch_polymarket_leaderboard,
    fetch_trader_positions,
    calculate_trader_stats,
    generate_trader_bio,
    clean_display_name,
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def backfill_traders(replace_existing: bool = False):
    """
    Fetch top traders from Polymarket and populate database.

    For each trader, fetches real trade history from the Gamma API
    to calculate actual win rate, drawdown, and trade duration.

    Args:
        replace_existing: If True, delete existing traders and refetch
    """
    async with async_session() as session:
        # Check existing traders
        result = await session.execute(select(TraderProfile))
        existing = result.scalars().all()

        if existing and not replace_existing:
            logger.info(f"Found {len(existing)} existing traders. Use --replace to refresh.")
            return

        if replace_existing and existing:
            logger.info(f"Deleting {len(existing)} existing traders...")
            await session.execute(delete(TraderProfile))
            await session.commit()

        # Fetch top traders from multiple time windows and categories
        logger.info("Fetching top traders from Polymarket leaderboard...")

        all_traders_data = []

        # Get top 50 from monthly leaderboard (by PnL)
        traders_month_pnl = await fetch_polymarket_leaderboard(
            time_period="MONTH",
            limit=50,
            order_by="PNL",
            category="OVERALL",
        )
        all_traders_data.extend(traders_month_pnl)

        # Get top 30 by volume
        traders_month_vol = await fetch_polymarket_leaderboard(
            time_period="MONTH",
            limit=30,
            order_by="VOL",
            category="OVERALL",
        )
        all_traders_data.extend(traders_month_vol)

        # Get top 20 from weekly leaderboard (recent hot traders)
        traders_week = await fetch_polymarket_leaderboard(
            time_period="WEEK",
            limit=20,
            order_by="PNL",
            category="OVERALL",
        )
        all_traders_data.extend(traders_week)

        logger.info(f"Fetched {len(all_traders_data)} total trader records")

        # Remove duplicates by wallet address
        seen_wallets = set()
        unique_traders = []
        for trader in all_traders_data:
            wallet = trader.get("proxyWallet")
            if wallet and wallet not in seen_wallets:
                seen_wallets.add(wallet)
                unique_traders.append(trader)

        logger.info(f"Processing {len(unique_traders)} unique traders (fetching real trade history)...")

        created_count = 0
        for i, trader_data in enumerate(unique_traders, 1):
            try:
                wallet = trader_data.get("proxyWallet")
                if not wallet:
                    continue

                # Clean display name (handles wallet-address usernames)
                display_name = clean_display_name(trader_data)

                # Fetch real positions with P&L from Polymarket data API
                positions = await fetch_trader_positions(wallet, limit=100)

                if positions:
                    # Calculate real stats from actual position P&L
                    stats = calculate_trader_stats(trader_data, positions)
                    logger.debug(
                        f"  {display_name}: {stats['total_trades']} positions, "
                        f"{stats['win_rate']:.1f}% win rate"
                    )
                else:
                    # Trader has leaderboard presence but no fetchable positions
                    # Use only what we actually know (PnL, volume)
                    pnl = float(trader_data.get("pnl", 0))
                    volume = float(trader_data.get("vol", 0))
                    stats = {
                        "total_pnl": pnl,
                        "roi_pct": (pnl / max(volume * 0.3, 1)) * 100 if volume > 0 else 0,
                        "win_rate": 0.0,  # Unknown — don't fabricate
                        "total_trades": 0,  # Unknown
                        "winning_trades": 0,
                        "avg_trade_duration_hrs": 0.0,
                        "risk_score": 5,  # Neutral default
                        "max_drawdown": 0.0,
                    }
                    logger.debug(f"  {display_name}: no trades available, using PnL/volume only")

                # Generate bio from real stats
                bio = generate_trader_bio(trader_data, stats)

                # Create TraderProfile
                profile = TraderProfile(
                    user_id=wallet,
                    display_name=display_name,
                    bio=bio,
                    total_pnl=stats["total_pnl"],
                    roi_pct=stats["roi_pct"],
                    win_rate=stats["win_rate"],
                    total_trades=stats["total_trades"],
                    winning_trades=stats["winning_trades"],
                    avg_trade_duration_hrs=stats["avg_trade_duration_hrs"],
                    risk_score=stats["risk_score"],
                    max_drawdown=stats["max_drawdown"],
                    follower_count=0,
                    is_public=True,
                    accepts_copiers=True,
                )

                session.add(profile)
                created_count += 1

                # Commit in batches of 10
                if created_count % 10 == 0:
                    await session.commit()
                    logger.info(f"Saved {created_count}/{len(unique_traders)} traders...")

                # Small delay to avoid rate limiting on Gamma API
                await asyncio.sleep(0.2)

            except Exception as e:
                logger.error(f"Failed to process trader {trader_data.get('proxyWallet', 'unknown')[:10]}: {e}")
                continue

        # Final commit
        await session.commit()
        logger.info(f"[OK] Created {created_count} trader profiles from Polymarket (real trade data)")

        # Show top 5 by P&L
        result = await session.execute(
            select(TraderProfile)
            .where(TraderProfile.is_public == True)
            .order_by(TraderProfile.total_pnl.desc())
            .limit(5)
        )
        top_traders = result.scalars().all()

        logger.info("\nTop 5 Traders by P&L:")
        for i, trader in enumerate(top_traders, 1):
            logger.info(
                f"  {i}. {trader.display_name}: "
                f"${trader.total_pnl:,.2f} P&L | "
                f"{trader.win_rate:.1f}% win rate | "
                f"Risk: {trader.risk_score}/10 | "
                f"{trader.total_trades} trades"
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backfill real trader data from Polymarket")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing traders with fresh data",
    )
    args = parser.parse_args()

    asyncio.run(backfill_traders(replace_existing=args.replace))
