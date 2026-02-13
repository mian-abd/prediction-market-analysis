"""Data pipeline scheduler - orchestrates periodic data collection,
arbitrage scanning, cross-platform matching, and orderbook snapshots."""

import asyncio
import logging
from datetime import datetime, timedelta

from sqlalchemy import update, func

from db.database import async_session
from db.models import Market
from data_pipeline.storage import (
    ensure_platforms, upsert_markets, insert_price_snapshots,
    get_active_markets, insert_orderbook_snapshot,
)
from data_pipeline.collectors import polymarket_gamma, kalshi_markets
from data_pipeline.collectors.polymarket_gamma import parse_gamma_market
from data_pipeline.collectors.kalshi_markets import parse_kalshi_market
from data_pipeline.collectors import polymarket_clob

logger = logging.getLogger(__name__)


# ── Market Lifecycle ─────────────────────────────────────────────────

async def deactivate_expired_markets():
    """Mark markets as inactive if past their end_date or price is stuck at 0/1."""
    async with async_session() as session:
        # Deactivate past-end-date markets
        result = await session.execute(
            update(Market)
            .where(
                Market.is_active == True,  # noqa
                Market.end_date != None,  # noqa
                Market.end_date < func.now(),
            )
            .values(is_active=False)
        )
        expired_count = result.rowcount

        # Deactivate markets with price stuck at 0 or 1 (effectively resolved)
        result2 = await session.execute(
            update(Market)
            .where(
                Market.is_active == True,  # noqa
                Market.price_yes != None,  # noqa
                (Market.price_yes <= 0.01) | (Market.price_yes >= 0.99),
            )
            .values(is_active=False)
        )
        dead_count = result2.rowcount

        await session.commit()

        if expired_count or dead_count:
            logger.info(f"Deactivated markets: {expired_count} expired, {dead_count} dead price")


# ── Market & Price Collection ────────────────────────────────────────

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
    async with async_session() as session:
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


# ── Orderbook Collection ─────────────────────────────────────────────

async def collect_orderbooks():
    """Collect CLOB orderbook snapshots for top Polymarket markets by volume."""
    logger.info("Starting orderbook collection...")
    async with async_session() as session:
        markets = await get_active_markets(session, platform_name="polymarket", limit=50)
        count = 0

        for market in markets:
            token_id = market.token_id_yes
            if not token_id:
                continue
            try:
                raw_ob = await polymarket_clob.fetch_orderbook(token_id)
                if raw_ob:
                    parsed = polymarket_clob.parse_orderbook(raw_ob)
                    await insert_orderbook_snapshot(session, market.id, "yes", parsed)
                    count += 1
            except Exception as e:
                logger.warning(f"Orderbook fetch failed for market {market.id}: {e}")

            # Rate limit: avoid hammering the API
            await asyncio.sleep(0.2)

    logger.info(f"Collected {count} orderbook snapshots")


# ── Cross-Platform Matching ──────────────────────────────────────────

async def run_market_matching():
    """Find cross-platform matches between Polymarket and Kalshi via TF-IDF."""
    logger.info("Starting cross-platform market matching...")
    async with async_session() as session:
        try:
            from data_pipeline.transformers.market_matcher import find_cross_platform_matches
            matches = await find_cross_platform_matches(session)
            logger.info(f"Found {len(matches)} cross-platform matches")
        except Exception as e:
            logger.error(f"Market matching failed: {e}")


# ── Arbitrage Scanning ───────────────────────────────────────────────

async def scan_arbitrage():
    """Expire stale opportunities, then run all arbitrage strategies."""
    logger.info("Starting arbitrage scan...")
    async with async_session() as session:
        try:
            # Expire opportunities older than 30 minutes
            from sqlalchemy import update
            from db.models import ArbitrageOpportunity

            cutoff = datetime.utcnow() - timedelta(minutes=30)
            await session.execute(
                update(ArbitrageOpportunity)
                .where(
                    ArbitrageOpportunity.expired_at == None,  # noqa: E711
                    ArbitrageOpportunity.detected_at < cutoff,
                )
                .values(expired_at=datetime.utcnow())
            )
            await session.commit()

            # Run full scan (single-market + cross-platform)
            from arbitrage.engine import run_full_scan
            opportunities = await run_full_scan(session)
            logger.info(f"Arbitrage scan complete: {len(opportunities)} active opportunities")
        except Exception as e:
            logger.error(f"Arbitrage scan failed: {e}")


# ── Orchestration ────────────────────────────────────────────────────

async def run_pipeline_once():
    """Run all collection tasks once (useful for testing/seeding)."""
    await collect_markets()
    await collect_prices()
    await run_market_matching()
    await scan_arbitrage()
    logger.info("Pipeline single-run complete")


async def run_pipeline_loop():
    """Continuous pipeline loop with configurable intervals.

    Timing (default 60s price interval):
      - Prices:     every cycle     (60s)
      - Arbitrage:  every 5 cycles  (5 min)
      - Orderbooks: every 5 cycles  (5 min)
      - Markets:    every 60 cycles (1 hr) + re-match after refresh
    """
    from config.settings import settings

    logger.info("Starting pipeline loop...")

    # ── Initial full collection ──
    await collect_markets()
    await collect_prices()

    # Initial matching + arb scan after data is loaded
    try:
        await run_market_matching()
    except Exception as e:
        logger.error(f"Initial matching failed: {e}")

    try:
        await scan_arbitrage()
    except Exception as e:
        logger.error(f"Initial arb scan failed: {e}")

    # ── Loop configuration ──
    price_interval = settings.price_poll_interval_sec
    market_interval = settings.market_refresh_interval_sec
    cycles_per_market_refresh = max(1, market_interval // price_interval)
    cycles_per_arb_scan = 5       # Every 5 price cycles
    cycles_per_orderbook = max(1, settings.orderbook_poll_interval_sec // price_interval)

    cycle = 0

    while True:
        await asyncio.sleep(price_interval)
        cycle += 1

        # ── Always collect prices ──
        try:
            await collect_prices()
        except Exception as e:
            logger.error(f"Price collection error: {e}")

        # ── Arbitrage scan every ~5 min ──
        if cycle % cycles_per_arb_scan == 0:
            try:
                await scan_arbitrage()
            except Exception as e:
                logger.error(f"Arbitrage scan error: {e}")

        # ── Orderbook collection every ~5 min ──
        if cycle % cycles_per_orderbook == 0:
            try:
                await collect_orderbooks()
            except Exception as e:
                logger.error(f"Orderbook collection error: {e}")

        # ── Full market refresh + deactivate expired + re-match every ~1 hr ──
        if cycle % cycles_per_market_refresh == 0:
            try:
                await collect_markets()
            except Exception as e:
                logger.error(f"Market refresh error: {e}")
            try:
                await deactivate_expired_markets()
            except Exception as e:
                logger.error(f"Expired market deactivation error: {e}")
            try:
                await run_market_matching()
            except Exception as e:
                logger.error(f"Market matching error: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    async def main():
        from db.database import init_db
        await init_db()
        await run_pipeline_once()

    asyncio.run(main())
