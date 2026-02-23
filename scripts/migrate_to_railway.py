"""Migrate portfolio positions, copy trades, trader profiles, and Elo ratings
from local SQLite to a Railway Postgres instance.

This resolves market_id and position_id foreign keys by matching markets via
external_id, so Railway's auto-increment IDs don't have to match SQLite's.

Usage:
    python scripts/migrate_to_railway.py --dest-url "postgresql+asyncpg://postgres:PASSWORD@HOST:PORT/railway"

Get the destination URL from:
    Railway → your Postgres service → Connect → "Postgres Connection URL"
    Then replace  postgresql://  with  postgresql+asyncpg://
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime, timezone
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert

from db.database import async_session as sqlite_session, init_db as init_sqlite
from db.models import (
    Market, Platform, TraderProfile, FollowedTrader, CopyTrade,
    PortfolioPosition, EloRating,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def migrate(dest_url: str, dry_run: bool) -> None:
    # ── Source: local SQLite ──────────────────────────────────────────
    logger.info("Connecting to local SQLite source...")
    await init_sqlite()

    # ── Destination: Railway Postgres ─────────────────────────────────
    logger.info("Connecting to Railway Postgres destination...")
    dest_engine = create_async_engine(dest_url, echo=False)
    DestSession = async_sessionmaker(dest_engine, expire_on_commit=False)

    async with sqlite_session() as src, DestSession() as dst:

        # ── 1. Build market remapping (SQLite id → Railway id) ────────
        logger.info("Building market ID remapping (external_id matching)...")
        src_markets_result = await src.execute(
            select(Market.id, Market.external_id, Market.platform_id)
        )
        src_markets = {row.id: row.external_id for row in src_markets_result.all()}

        dst_markets_result = await dst.execute(
            select(Market.id, Market.external_id)
        )
        dst_ext_to_id = {row.external_id: row.id for row in dst_markets_result.all()}

        # Build: SQLite market_id → Railway market_id
        market_id_map: dict[int, int] = {}
        unmatched_markets = 0
        for src_id, ext_id in src_markets.items():
            if ext_id and ext_id in dst_ext_to_id:
                market_id_map[src_id] = dst_ext_to_id[ext_id]
            else:
                unmatched_markets += 1

        logger.info(
            f"Market remapping: {len(market_id_map)} matched, "
            f"{unmatched_markets} unmatched (those positions will be skipped)"
        )

        # ── 2. Build platform remapping ───────────────────────────────
        src_platforms_result = await src.execute(select(Platform.id, Platform.name))
        src_platform_map = {row.id: row.name for row in src_platforms_result.all()}

        dst_platforms_result = await dst.execute(select(Platform.id, Platform.name))
        dst_platform_map = {row.name: row.id for row in dst_platforms_result.all()}

        platform_id_map: dict[int, int] = {}
        for src_pid, name in src_platform_map.items():
            if name in dst_platform_map:
                platform_id_map[src_pid] = dst_platform_map[name]

        # ── 3. Migrate trader_profiles (bulk) ────────────────────────
        logger.info("Migrating trader_profiles...")
        src_traders = (await src.execute(select(TraderProfile))).scalars().all()
        # Pre-fetch existing user_ids in destination to avoid per-row queries
        dst_existing_traders = set(
            row[0] for row in (
                await dst.execute(text("SELECT user_id FROM trader_profiles"))
            ).all()
        )
        tp_count = 0
        for tp in src_traders:
            if tp.user_id in dst_existing_traders:
                continue  # Skip duplicates
            if not dry_run:
                dst.add(TraderProfile(
                    user_id=tp.user_id,
                    display_name=tp.display_name,
                    bio=tp.bio,
                    total_pnl=tp.total_pnl,
                    roi_pct=tp.roi_pct,
                    win_rate=tp.win_rate,
                    total_trades=tp.total_trades,
                    winning_trades=tp.winning_trades,
                    avg_trade_duration_hrs=tp.avg_trade_duration_hrs,
                    risk_score=tp.risk_score,
                    max_drawdown=tp.max_drawdown,
                    follower_count=tp.follower_count,
                    is_public=tp.is_public,
                    accepts_copiers=tp.accepts_copiers,
                    created_at=tp.created_at,
                    updated_at=tp.updated_at,
                ))
            tp_count += 1
        if not dry_run:
            await dst.commit()
        logger.info(f"  trader_profiles: {tp_count} migrated")

        # ── 4. Migrate portfolio_positions ────────────────────────────
        logger.info("Migrating portfolio_positions...")
        src_positions = (await src.execute(select(PortfolioPosition))).scalars().all()

        position_id_map: dict[int, int] = {}  # old id → new id
        pos_ok = 0
        pos_skip = 0

        for pos in src_positions:
            new_market_id = market_id_map.get(pos.market_id)
            new_platform_id = platform_id_map.get(pos.platform_id)

            if not new_market_id or not new_platform_id:
                pos_skip += 1
                continue

            if not dry_run:
                new_pos = PortfolioPosition(
                    user_id=pos.user_id,
                    market_id=new_market_id,
                    platform_id=new_platform_id,
                    side=pos.side,
                    entry_price=pos.entry_price,
                    quantity=pos.quantity,
                    entry_time=pos.entry_time,
                    exit_price=pos.exit_price,
                    exit_time=pos.exit_time,
                    realized_pnl=pos.realized_pnl,
                    strategy=pos.strategy,
                    portfolio_type=pos.portfolio_type,
                    is_simulated=pos.is_simulated,
                )
                dst.add(new_pos)
                await dst.flush()  # Get the new auto-increment ID
                position_id_map[pos.id] = new_pos.id
            pos_ok += 1

        if not dry_run:
            await dst.commit()
        logger.info(f"  portfolio_positions: {pos_ok} migrated, {pos_skip} skipped (market not in Railway yet)")

        # ── 5. Migrate followed_traders ───────────────────────────────
        logger.info("Migrating followed_traders...")
        src_followed = (await src.execute(select(FollowedTrader))).scalars().all()
        ft_count = 0
        for ft in src_followed:
            # Check trader still exists in destination
            tp_exists = await dst.execute(
                select(TraderProfile).where(TraderProfile.user_id == ft.trader_id)
            )
            if not tp_exists.scalar_one_or_none():
                continue
            if not dry_run:
                dst.add(FollowedTrader(
                    follower_id=ft.follower_id,
                    trader_id=ft.trader_id,
                    allocation_amount=ft.allocation_amount,
                    copy_percentage=ft.copy_percentage,
                    max_position_size=ft.max_position_size,
                    auto_copy=ft.auto_copy,
                    copy_settings=ft.copy_settings,
                    followed_at=ft.followed_at,
                    unfollowed_at=ft.unfollowed_at,
                    is_active=ft.is_active,
                ))
            ft_count += 1
        if not dry_run:
            await dst.commit()
        logger.info(f"  followed_traders: {ft_count} migrated")

        # ── 6. Migrate copy_trades ────────────────────────────────────
        logger.info("Migrating copy_trades...")
        src_copy = (await src.execute(select(CopyTrade))).scalars().all()
        ct_ok = 0
        ct_skip = 0
        for ct in src_copy:
            new_follower_pos = position_id_map.get(ct.follower_position_id)
            new_trader_pos = position_id_map.get(ct.trader_position_id)
            if not new_follower_pos or not new_trader_pos:
                ct_skip += 1
                continue
            if not dry_run:
                dst.add(CopyTrade(
                    follower_position_id=new_follower_pos,
                    trader_position_id=new_trader_pos,
                    follower_id=ct.follower_id,
                    trader_id=ct.trader_id,
                    copy_ratio=ct.copy_ratio,
                    copied_at=ct.copied_at,
                ))
            ct_ok += 1
        if not dry_run:
            await dst.commit()
        logger.info(f"  copy_trades: {ct_ok} migrated, {ct_skip} skipped")

        # ── 7. Migrate elo_ratings (bulk upsert) ─────────────────────
        # Elo ratings are rebuilt by scripts/build_elo_ratings.py from .joblib
        # files, so skipping if destination already has data.
        logger.info("Migrating elo_ratings (bulk upsert)...")
        dst_elo_count = (await dst.execute(text("SELECT COUNT(*) FROM elo_ratings"))).scalar()
        if dst_elo_count and dst_elo_count > 0:
            logger.info(f"  elo_ratings: destination already has {dst_elo_count} rows, skipping")
        else:
            src_elo = (await src.execute(select(EloRating))).scalars().all()
            elo_count = 0
            BATCH = 500
            batch = []
            for er in src_elo:
                batch.append({
                    "sport": er.sport,
                    "player_name": er.player_name,
                    "surface": er.surface,
                    "mu": er.mu,
                    "phi": er.phi,
                    "sigma": er.sigma,
                    "match_count": er.match_count,
                    "last_match_date": er.last_match_date,
                    "updated_at": er.updated_at,
                })
                elo_count += 1
                if len(batch) >= BATCH:
                    if not dry_run:
                        stmt = pg_insert(EloRating).values(batch)
                        stmt = stmt.on_conflict_do_nothing(
                            index_elements=["sport", "player_name", "surface"]
                        )
                        await dst.execute(stmt)
                        await dst.commit()
                    batch = []
                    logger.info(f"  elo_ratings: {elo_count} processed...")
            if batch and not dry_run:
                stmt = pg_insert(EloRating).values(batch)
                stmt = stmt.on_conflict_do_nothing(
                    index_elements=["sport", "player_name", "surface"]
                )
                await dst.execute(stmt)
                await dst.commit()
            logger.info(f"  elo_ratings: {elo_count} migrated")

    await dest_engine.dispose()

    # ── Summary ────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 55)
    logger.info(f"MIGRATION {'DRY RUN ' if dry_run else ''}COMPLETE")
    logger.info("=" * 55)
    if dry_run:
        logger.info("DRY RUN — nothing was written. Remove --dry-run to apply.")
    else:
        logger.info("All data migrated to Railway Postgres.")
        logger.info("Check your frontend — positions and copy trades should be restored.")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate portfolio/copy-trade data from local SQLite to Railway Postgres.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dest-url",
        required=True,
        help=(
            "Railway Postgres PUBLIC connection URL. "
            "Railway dashboard → Postgres → Connect → copy URL, "
            "then change 'postgresql://' to 'postgresql+asyncpg://'."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Simulate migration without writing anything.",
    )
    args = parser.parse_args()

    dest_url = args.dest_url
    if dest_url.startswith("postgresql://") and "asyncpg" not in dest_url:
        dest_url = dest_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        logger.info(f"Auto-converted URL to asyncpg driver.")

    asyncio.run(migrate(dest_url=dest_url, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
