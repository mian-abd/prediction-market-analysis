"""Migration: Add normalized_category column and populate it.

Adds the column via ALTER TABLE (safe for SQLite), then runs the normalizer
on all existing markets. Preserves the raw `category` column intact.

Usage:
    python scripts/migrate_normalized_categories.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import logging
from sqlalchemy import text, select, update
from db.database import async_session, init_db, engine
from db.models import Market
from data_pipeline.category_normalizer import normalize_category

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


async def add_column_if_missing():
    """Add normalized_category column via ALTER TABLE if it doesn't exist."""
    async with engine.begin() as conn:
        # Check if column already exists
        result = await conn.execute(text("PRAGMA table_info(markets)"))
        columns = [row[1] for row in result.fetchall()]

        if "normalized_category" not in columns:
            logger.info("Adding normalized_category column to markets table...")
            await conn.execute(
                text("ALTER TABLE markets ADD COLUMN normalized_category VARCHAR(50)")
            )
            logger.info("Column added successfully.")
        else:
            logger.info("normalized_category column already exists.")


async def normalize_all_markets():
    """Re-normalize all market categories using the updated normalizer."""
    async with async_session() as session:
        result = await session.execute(select(Market))
        markets = result.scalars().all()

        logger.info(f"Normalizing categories for {len(markets)} markets...")

        # Track category distribution
        distribution: dict[str, int] = {}
        changed = 0
        batch_size = 500

        for i, market in enumerate(markets):
            normalized = normalize_category(
                raw_category=market.category,
                question=market.question or "",
                description=market.description or "",
            )

            distribution[normalized] = distribution.get(normalized, 0) + 1

            if market.normalized_category != normalized:
                market.normalized_category = normalized
                changed += 1

            # Commit in batches
            if (i + 1) % batch_size == 0:
                await session.commit()
                logger.info(f"  Processed {i + 1}/{len(markets)} ({changed} changed)...")

        await session.commit()

        logger.info(f"\nNormalization complete: {changed} markets updated")
        logger.info(f"\nCategory distribution:")
        for cat, count in sorted(distribution.items(), key=lambda x: -x[1]):
            logger.info(f"  {cat:20s} {count:>6,d}")

        # Show some examples of re-categorized markets
        logger.info(f"\nSample re-categorizations:")
        result = await session.execute(
            select(Market)
            .where(Market.category != Market.normalized_category)
            .where(Market.normalized_category == "sports")
            .limit(10)
        )
        for m in result.scalars().all():
            logger.info(f"  [{m.category}] -> [{m.normalized_category}] | {m.question[:80]}")


async def main():
    await init_db()
    await add_column_if_missing()
    await normalize_all_markets()
    logger.info("\nMigration complete!")


if __name__ == "__main__":
    asyncio.run(main())
