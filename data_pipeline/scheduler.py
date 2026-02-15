"""Data pipeline scheduler - orchestrates periodic data collection,
arbitrage scanning, cross-platform matching, and orderbook snapshots."""

import asyncio
import logging
from datetime import datetime, timedelta

from sqlalchemy import update, select, func

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

from db.models import TraderProfile, EloEdgeSignal, EnsembleEdgeSignal

logger = logging.getLogger(__name__)


# ── Trader Data Refresh ──────────────────────────────────────────────

async def refresh_trader_stats():
    """Refresh trader profiles with latest data from Polymarket leaderboard.

    Runs periodically to keep trader PnL, win rate, and volume stats current.
    Updates existing traders and adds new ones from the leaderboard.
    """
    logger.info("Refreshing trader stats from Polymarket...")
    try:
        from data_pipeline.collectors.trader_data import fetch_polymarket_leaderboard
        from scripts.backfill_real_traders import estimate_trader_stats, generate_bio

        # Fetch latest leaderboard data
        leaderboard = await fetch_polymarket_leaderboard(
            time_period="MONTH", limit=50, order_by="PNL"
        )

        if not leaderboard:
            logger.warning("Empty leaderboard response, skipping refresh")
            return

        async with async_session() as session:
            updated = 0
            created = 0

            for trader_data in leaderboard:
                wallet = trader_data.get("proxyWallet")
                if not wallet:
                    continue

                # Check if trader exists
                result = await session.execute(
                    select(TraderProfile).where(TraderProfile.user_id == wallet)
                )
                existing = result.scalar_one_or_none()

                stats = estimate_trader_stats(trader_data)

                if existing:
                    # Update existing trader stats
                    existing.total_pnl = stats["total_pnl"]
                    existing.roi_pct = stats["roi_pct"]
                    existing.win_rate = stats["win_rate"]
                    existing.total_trades = stats["total_trades"]
                    existing.winning_trades = stats["winning_trades"]
                    existing.risk_score = stats["risk_score"]
                    existing.max_drawdown = stats["max_drawdown"]
                    updated += 1
                else:
                    # Add new trader
                    username = trader_data.get("userName")
                    display_name = username if username else f"Trader_{wallet[-6:].upper()}"
                    bio = generate_bio(trader_data, stats)

                    new_trader = TraderProfile(
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
                    session.add(new_trader)
                    created += 1

            await session.commit()
            logger.info(f"Trader refresh: {updated} updated, {created} new")

    except Exception as e:
        logger.error(f"Trader stats refresh failed: {e}")


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
    """Snapshot current FRESH prices for all active markets from CLOB API."""
    async with async_session() as session:
        markets = await get_active_markets(session, platform_name="polymarket", limit=2000)

        # Build batch request for fresh prices
        params_list = []
        market_lookup = {}
        for market in markets:
            if market.token_id_yes:
                params_list.append({"token_id": market.token_id_yes, "side": "buy"})
                market_lookup[market.token_id_yes] = market

        if not params_list:
            logger.warning("No markets with token_id_yes for price fetch")
            return

        # Fetch fresh prices in batches (100 per request)
        count = 0
        batch_size = 100

        for i in range(0, len(params_list), batch_size):
            batch = params_list[i:i+batch_size]
            try:
                # CLOB returns dict: { "token_id": { "BUY": "0.55", "SELL": "0.54" }, ... }
                prices_dict = await polymarket_clob.fetch_prices_batch(batch)

                for token_id, sides in prices_dict.items():
                    # Use BUY price for YES side
                    buy_price_str = sides.get("BUY")
                    if not buy_price_str:
                        continue

                    market = market_lookup.get(token_id)
                    if not market:
                        continue

                    try:
                        price_yes = float(buy_price_str)
                        price_no = 1.0 - price_yes

                        # Update market table with fresh price
                        market.price_yes = price_yes
                        market.price_no = price_no

                        await insert_price_snapshots(
                            session, market_id=market.id,
                            price_yes=price_yes, price_no=price_no,
                            volume=market.volume_24h or 0,
                        )
                        count += 1
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Price parse failed for {token_id}: {e}")

            except Exception as e:
                logger.warning(f"Batch price fetch failed (batch {i//batch_size}): {e}")

        if count > 0:
            await session.commit()

    logger.info(f"Collected {count} FRESH price snapshots from CLOB API")


# ── Orderbook Collection ─────────────────────────────────────────────

async def collect_orderbooks():
    """Collect CLOB orderbook snapshots for top Polymarket markets by volume."""
    logger.info("Starting orderbook collection...")
    async with async_session() as session:
        markets = await get_active_markets(session, platform_name="polymarket", limit=200)
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


# ── Elo Edge Scanning ────────────────────────────────────────────────

async def scan_elo_edges():
    """Scan active sports markets for Elo-based edge signals."""
    logger.info("Starting Elo edge scan...")
    try:
        from ml.strategies.elo_edge_detector import scan_for_edges, get_active_edges
        from ml.models.elo_sports import Glicko2Engine

        engine = Glicko2Engine.load("ml/saved_models/elo_atp_ratings.joblib")
        if not engine or not engine.ratings:
            logger.info("No Elo ratings loaded yet, skipping scan")
            return

        async with async_session() as session:
            new_signals = await scan_for_edges(session, engine)
            active = await get_active_edges(session)
            logger.info(f"Elo scan: {len(new_signals)} new signals, {len(active)} active edges")
    except FileNotFoundError:
        logger.info("Elo ratings file not found, skipping scan (run build_elo_ratings.py first)")
    except Exception as e:
        logger.error(f"Elo edge scan failed: {e}")


# ── Ensemble Edge Scanning ──────────────────────────────────────────

async def scan_ensemble_edges():
    """Scan active markets for ML ensemble-based edge signals."""
    logger.info("Starting ensemble edge scan...")
    try:
        from ml.models.ensemble import EnsembleModel
        from ml.strategies.ensemble_edge_detector import (
            check_quality_gates, detect_edge, edge_signal_to_dict,
        )

        # Load ensemble (cached singleton pattern)
        ensemble = EnsembleModel()
        if not ensemble.load_all():
            logger.info("Ensemble not fully loaded, skipping scan")
            return

        async with async_session() as session:
            # Expire old signals (>4 hrs for longer position hold times)
            cutoff = datetime.utcnow() - timedelta(hours=4)
            from sqlalchemy import update as sa_update
            await session.execute(
                sa_update(EnsembleEdgeSignal)
                .where(
                    EnsembleEdgeSignal.expired_at == None,  # noqa: E711
                    EnsembleEdgeSignal.detected_at < cutoff,
                )
                .values(expired_at=datetime.utcnow())
            )
            await session.commit()

            # Query top 2000 liquid active markets (scan all active)
            markets = (await session.execute(
                select(Market)
                .where(Market.is_active == True, Market.price_yes != None)  # noqa
                .order_by(Market.volume_24h.desc())
                .limit(2000)
            )).scalars().all()

            edges_found = 0
            for market in markets:
                try:
                    # Quick quality gate filter (fast, no ML)
                    gate = check_quality_gates(market)
                    if not gate.passes:
                        continue

                    # Full ensemble prediction + edge detection
                    result = ensemble.predict_market(market)
                    edge = detect_edge(market, result)

                    # Only persist tradeable edges
                    if edge.direction and edge.net_ev > 0.03 and edge.confidence >= 0.4:
                        # Don't create signal if there's already an open position on this market
                        # Prevents signal expiry → re-creation → duplicate position loop
                        from db.models import PortfolioPosition
                        existing_pos = await session.execute(
                            select(PortfolioPosition.id)
                            .where(
                                PortfolioPosition.market_id == market.id,
                                PortfolioPosition.exit_time == None,  # noqa: E711
                                PortfolioPosition.portfolio_type == "auto",
                            )
                            .limit(1)
                        )
                        if existing_pos.scalar_one_or_none():
                            continue  # Skip: position already exists, don't create new signal

                        signal = EnsembleEdgeSignal(
                            market_id=market.id,
                            detected_at=datetime.utcnow(),
                            direction=edge.direction,
                            ensemble_prob=edge.ensemble_prob,
                            market_price=edge.market_price,
                            raw_edge=edge.raw_edge,
                            fee_cost=edge.fee_cost,
                            net_ev=edge.net_ev,
                            kelly_fraction=edge.kelly_fraction,
                            confidence=edge.confidence,
                            quality_tier=edge.quality_tier,
                            model_predictions=edge.model_predictions,
                        )
                        session.add(signal)
                        edges_found += 1

                except Exception as e:
                    logger.debug(f"Edge detection failed for market {market.id}: {e}")

            await session.commit()
            logger.info(f"Ensemble scan: {edges_found} new edges from {len(markets)} markets")

    except Exception as e:
        logger.error(f"Ensemble edge scan failed: {e}")


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

    # ── Initial ML scans (so signals appear immediately after deploy) ──
    try:
        await scan_ensemble_edges()
    except Exception as e:
        logger.error(f"Initial ensemble scan failed: {e}")

    try:
        await scan_elo_edges()
    except Exception as e:
        logger.error(f"Initial elo scan failed: {e}")

    # ── Initial auto-trade (execute any signals found above) ──
    try:
        from execution.paper_executor import execute_paper_trades
        async with async_session() as exec_session:
            created = await execute_paper_trades(exec_session)
            if created:
                logger.info(f"Initial auto paper trades: {len(created)} positions opened")
    except Exception as e:
        logger.error(f"Initial paper executor error: {e}")

    # ── Backfill trader profiles if empty (for Copy Trading page) ──
    try:
        from sqlalchemy import select as sa_select
        from db.models import TraderProfile
        async with async_session() as session:
            count = (await session.execute(
                sa_select(TraderProfile)
            )).scalars().first()
            if not count:
                logger.info("No trader profiles found, backfilling from Polymarket leaderboard...")
                from data_pipeline.collectors.trader_data import fetch_polymarket_leaderboard
                from scripts.backfill_real_traders import estimate_trader_stats, generate_bio

                all_traders = []
                for period, order, limit in [("MONTH", "PNL", 50), ("MONTH", "VOL", 30), ("WEEK", "PNL", 20)]:
                    try:
                        traders = await fetch_polymarket_leaderboard(time_period=period, limit=limit, order_by=order, category="OVERALL")
                        all_traders.extend(traders)
                    except Exception as e:
                        logger.warning(f"Leaderboard fetch {period}/{order} failed: {e}")

                seen, created_count = set(), 0
                for td in all_traders:
                    wallet = td.get("proxyWallet")
                    if not wallet or wallet in seen:
                        continue
                    seen.add(wallet)
                    stats = estimate_trader_stats(td)
                    session.add(TraderProfile(
                        user_id=wallet,
                        display_name=td.get("userName") or f"Trader_{wallet[-6:].upper()}",
                        bio=generate_bio(td, stats),
                        total_pnl=stats["total_pnl"], roi_pct=stats["roi_pct"],
                        win_rate=stats["win_rate"], total_trades=stats["total_trades"],
                        winning_trades=stats["winning_trades"],
                        avg_trade_duration_hrs=stats["avg_trade_duration_hrs"],
                        risk_score=stats["risk_score"], max_drawdown=stats["max_drawdown"],
                        follower_count=0, is_public=True, accepts_copiers=True,
                    ))
                    created_count += 1
                await session.commit()
                logger.info(f"Backfilled {created_count} trader profiles")
            else:
                logger.info("Trader profiles already exist, skipping backfill")
    except Exception as e:
        logger.error(f"Trader backfill failed: {e}")

    # ── Loop configuration ──
    price_interval = settings.price_poll_interval_sec
    market_interval = settings.market_refresh_interval_sec
    cycles_per_market_refresh = max(1, market_interval // price_interval)
    cycles_per_arb_scan = 5       # Every 5 price cycles
    cycles_per_orderbook = max(1, settings.orderbook_poll_interval_sec // price_interval)
    cycles_per_trader_refresh = max(1, 1800 // price_interval)  # Every ~30 min
    cycles_per_elo_scan = max(1, settings.elo_scan_interval_sec // price_interval)  # Every ~10 min
    cycles_per_ensemble_scan = 5   # Every 5 cycles (~2-3 min) - faster for more trades
    cycles_per_auto_trade = 5      # Every 5 cycles - matches ensemble scan
    cycles_per_resolution_score = max(1, 1800 // price_interval)  # Every ~30 min

    cycle = 0

    while True:
        await asyncio.sleep(price_interval)
        cycle += 1

        # ── Always collect prices ──
        try:
            await collect_prices()
        except Exception as e:
            logger.error(f"Price collection error: {e}")

        # ── Auto-close check EVERY cycle for fast stop-loss response ──
        try:
            from execution.auto_closer import auto_close_positions
            async with async_session() as close_session:
                closed = await auto_close_positions(close_session)
                if closed:
                    logger.info(f"Auto-closed {len(closed)} positions")
        except Exception as e:
            logger.error(f"Auto-closer error: {e}")

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

        # ── Elo edge scan every ~10 min ──
        if cycle % cycles_per_elo_scan == 0:
            try:
                await scan_elo_edges()
            except Exception as e:
                logger.error(f"Elo edge scan error: {e}")

        # ── Ensemble edge scan every ~15 min ──
        if cycle % cycles_per_ensemble_scan == 0:
            try:
                await scan_ensemble_edges()
            except Exception as e:
                logger.error(f"Ensemble edge scan error: {e}")

        # ── Auto-trade + auto-close every ~15 min (independent cycle) ──
        if cycle % cycles_per_auto_trade == 0:
            try:
                from execution.paper_executor import execute_paper_trades
                async with async_session() as exec_session:
                    created = await execute_paper_trades(exec_session)
                    if created:
                        logger.info(f"Auto paper trades: {len(created)} positions opened")
            except Exception as e:
                logger.error(f"Paper executor error: {e}")

        # ── Trader stats refresh every ~30 min ──
        if cycle % cycles_per_trader_refresh == 0:
            try:
                await refresh_trader_stats()
            except Exception as e:
                logger.error(f"Trader stats refresh error: {e}")

        # ── Score resolved signals every ~30 min ──
        if cycle % cycles_per_resolution_score == 0:
            try:
                from ml.evaluation.resolution_scorer import score_resolved_signals
                async with async_session() as score_session:
                    result = await score_resolved_signals(score_session)
                    if result["scored_ensemble"] or result["scored_elo"]:
                        logger.info(
                            f"Resolution scoring: {result['scored_ensemble']} ensemble, "
                            f"{result['scored_elo']} Elo signals scored"
                        )
            except Exception as e:
                logger.error(f"Resolution scoring failed: {e}")

            # Log rolling forward performance (7-day window)
            try:
                from ml.evaluation.signal_tracker import compute_signal_accuracy
                async with async_session() as perf_session:
                    perf = await compute_signal_accuracy(perf_session, days_back=7)
                    n = perf["n_signals_evaluated"]
                    if n > 0:
                        logger.info(
                            f"FORWARD PERFORMANCE (7d): {n} signals | "
                            f"hit_rate={perf['hit_rate']:.1%} | "
                            f"Brier={perf['brier_score']} | "
                            f"P&L=${perf['simulated_pnl']:.2f} | "
                            f"improvement={perf.get('brier_improvement_pct', 'N/A')}%"
                        )
            except Exception as e:
                logger.error(f"Forward performance log failed: {e}")

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
