"""Data pipeline scheduler - orchestrates periodic data collection."""

import asyncio
import logging
from datetime import datetime

from db.database import async_session
from data_pipeline.storage import ensure_platforms, upsert_markets, insert_price_snapshots
from data_pipeline.collectors import polymarket_gamma, kalshi_markets
from data_pipeline.collectors.polymarket_gamma import parse_gamma_market
from data_pipeline.collectors.kalshi_markets import parse_kalshi_market

logger = logging.getLogger(__name__)


async def collect_markets():
    """Fetch and store all active markets from both platforms."""
    logger.info("Starting market collection...")
    async with async_session() as session:
        platforms = await ensure_platforms(session)

        # Polymarket
        try:
            raw_poly = await polymarket_gamma.fetch_all_active_markets(max_markets=2000)
            parsed_poly = [parse_gamma_market(m) for m in raw_poly]
            count = await upsert_markets(session, parsed_poly, platforms["polymarket"])
            logger.info(f"Upserted {count} Polymarket markets")
        except Exception as e:
            logger.error(f"Polymarket collection failed: {e}")

        # Kalshi
        try:
            raw_kalshi = await kalshi_markets.fetch_all_active_markets(max_markets=2000)
            parsed_kalshi = [parse_kalshi_market(m) for m in raw_kalshi]
            count = await upsert_markets(session, parsed_kalshi, platforms["kalshi"])
            logger.info(f"Upserted {count} Kalshi markets")
        except Exception as e:
            logger.error(f"Kalshi collection failed: {e}")

    logger.info("Market collection complete")


async def collect_prices():
    """Snapshot current prices for all active markets."""
    logger.info("Starting price collection...")
    async with async_session() as session:
        from data_pipeline.storage import get_active_markets

        markets = await get_active_markets(session, limit=500)
        count = 0
        for market in markets:
            if market.price_yes is not None:
                try:
                    await insert_price_snapshots(
                        session,
                        market_id=market.id,
                        price_yes=market.price_yes,
                        price_no=market.price_no or (1.0 - market.price_yes),
                        volume=market.volume_24h or 0,
                    )
                    count += 1
                except Exception as e:
                    logger.warning(f"Price snapshot failed for market {market.id}: {e}")

    logger.info(f"Collected {count} price snapshots")


async def run_pipeline_once():
    """Run all collection tasks once."""
    await collect_markets()
    await collect_prices()
    logger.info("Pipeline run complete")


async def run_pipeline_loop():
    """Run the pipeline in a continuous loop with configurable intervals."""
    from config.settings import settings

    logger.info("Starting pipeline loop...")

    # Initial full collection
    await collect_markets()
    await collect_prices()

    market_refresh_counter = 0
    price_interval = settings.price_poll_interval_sec
    market_interval = settings.market_refresh_interval_sec
    cycles_per_market_refresh = market_interval // price_interval

    while True:
        await asyncio.sleep(price_interval)

        # Always collect prices
        try:
            await collect_prices()
        except Exception as e:
            logger.error(f"Price collection error: {e}")

        # Periodically refresh full market list
        market_refresh_counter += 1
        if market_refresh_counter >= cycles_per_market_refresh:
            market_refresh_counter = 0
            try:
                await collect_markets()
            except Exception as e:
                logger.error(f"Market refresh error: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    async def main():
        from db.database import init_db
        await init_db()
        await run_pipeline_once()

    asyncio.run(main())
