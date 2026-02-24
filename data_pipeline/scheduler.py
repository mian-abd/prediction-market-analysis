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
    get_active_markets, insert_orderbook_snapshot, cleanup_old_data,
)
from data_pipeline.collectors import polymarket_gamma, kalshi_markets
from data_pipeline.collectors.polymarket_gamma import parse_gamma_market
from data_pipeline.collectors.kalshi_markets import parse_kalshi_market
from data_pipeline.collectors import polymarket_clob

from db.models import TraderProfile, EloEdgeSignal, EnsembleEdgeSignal

# Real-time WebSocket streaming
from data_pipeline.streams import PriceCache, PolymarketStream

# Adaptive confidence adjustment (Phase 2.5)
from ml.evaluation.confidence_adjuster import (
    init_confidence_adjuster,
    refresh_confidence_adjuster,
    get_adjuster_stats,
)

logger = logging.getLogger(__name__)

# Global WebSocket stream instances (initialized in run_pipeline_loop)
_price_cache: PriceCache | None = None
_polymarket_stream: PolymarketStream | None = None


# ── Trader Data Refresh ──────────────────────────────────────────────

async def refresh_trader_stats():
    """Refresh trader profiles with real data from Polymarket.

    Fetches leaderboard for PnL/volume, then real trade history (5 traders
    per cycle to avoid rate limiting) for accurate win rate and drawdown.
    """
    import asyncio as _asyncio
    logger.info("Refreshing trader stats from Polymarket...")
    try:
        from data_pipeline.collectors.trader_data import (
            fetch_polymarket_leaderboard, fetch_trader_positions,
            calculate_trader_stats, generate_trader_bio, clean_display_name,
        )

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
            positions_fetched = 0

            for trader_data in leaderboard:
                wallet = trader_data.get("proxyWallet")
                if not wallet:
                    continue

                # Check if trader exists
                result = await session.execute(
                    select(TraderProfile).where(TraderProfile.user_id == wallet)
                )
                existing = result.scalar_one_or_none()

                # Fetch real positions for up to 5 traders per cycle (rate limit)
                positions = []
                if positions_fetched < 5:
                    positions = await fetch_trader_positions(wallet, limit=100)
                    if positions:
                        positions_fetched += 1
                    await _asyncio.sleep(0.2)  # Rate limit

                if positions:
                    stats = calculate_trader_stats(trader_data, positions)
                else:
                    # Only update PnL (the one thing we KNOW from leaderboard)
                    pnl = float(trader_data.get("pnl", 0))
                    volume = float(trader_data.get("vol", 0))
                    if existing:
                        # Keep existing real stats, just update PnL
                        existing.total_pnl = pnl
                        existing.roi_pct = (pnl / max(volume * 0.3, 1)) * 100 if volume > 0 else 0
                        updated += 1
                        continue
                    else:
                        stats = {
                            "total_pnl": pnl,
                            "roi_pct": (pnl / max(volume * 0.3, 1)) * 100 if volume > 0 else 0,
                            "win_rate": 0.0,
                            "total_trades": 0,
                            "winning_trades": 0,
                            "avg_trade_duration_hrs": 0.0,
                            "risk_score": 5,
                            "max_drawdown": 0.0,
                        }

                if existing:
                    existing.total_pnl = stats["total_pnl"]
                    existing.roi_pct = stats["roi_pct"]
                    existing.win_rate = stats["win_rate"]
                    existing.total_trades = stats["total_trades"]
                    existing.winning_trades = stats["winning_trades"]
                    existing.risk_score = stats["risk_score"]
                    existing.max_drawdown = stats["max_drawdown"]
                    updated += 1
                else:
                    display_name = clean_display_name(trader_data)
                    bio = generate_trader_bio(trader_data, stats)

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
            logger.info(f"Trader refresh: {updated} updated, {created} new, {positions_fetched} with real trades")

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
    """Fetch and store all active markets from both platforms.

    Uses separate sessions per platform so one failure doesn't cascade
    (PostgreSQL rolls back the entire session on error).
    """
    logger.info("Starting market collection...")

    # Ensure platforms exist first (shared lookup)
    async with async_session() as session:
        platforms = await ensure_platforms(session)

    # Polymarket — isolated session
    try:
        raw_poly = await polymarket_gamma.fetch_all_active_markets(max_markets=2000)
        parsed_poly = [parse_gamma_market(m) for m in raw_poly]
        async with async_session() as session:
            count = await upsert_markets(session, parsed_poly, platforms["polymarket"])
            logger.info(f"Upserted {count} Polymarket markets")
    except Exception as e:
        logger.error(f"Polymarket collection failed: {e}")

    # Kalshi — isolated session
    try:
        raw_kalshi = await kalshi_markets.fetch_all_active_markets(max_markets=2000)
        parsed_kalshi = [parse_kalshi_market(m) for m in raw_kalshi]
        async with async_session() as session:
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
        markets = await get_active_markets(session, platform_name="polymarket", limit=500)  # Increased from 200 to 500 for better feature coverage
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
    """Scan active sports markets for Elo-based edge signals (Tennis + UFC)."""
    logger.info("Starting Elo edge scan (Tennis + UFC)...")
    try:
        from ml.strategies.elo_edge_detector import scan_for_edges, get_active_edges
        from ml.models.elo_sports import Glicko2Engine
        from pathlib import Path

        total_new = 0

        # ATP Tennis
        try:
            atp_engine = Glicko2Engine.load("ml/saved_models/elo_atp_ratings.joblib")
            if atp_engine and atp_engine.ratings:
                async with async_session() as session:
                    atp_signals = await scan_for_edges(session, atp_engine)
                    total_new += len(atp_signals)
                    logger.info(f"ATP scan: {len(atp_signals)} new signals")
        except FileNotFoundError:
            logger.info("ATP ratings not found, skipping")

        # WTA Tennis
        try:
            wta_engine = Glicko2Engine.load("ml/saved_models/elo_wta_ratings.joblib")
            if wta_engine and wta_engine.ratings:
                async with async_session() as session:
                    wta_signals = await scan_for_edges(session, wta_engine)
                    total_new += len(wta_signals)
                    logger.info(f"WTA scan: {len(wta_signals)} new signals")
        except FileNotFoundError:
            logger.info("WTA ratings not found, skipping")

        # UFC/MMA
        ufc_path = "ml/saved_models/elo_ufc_ratings.joblib"
        if Path(ufc_path).exists():
            try:
                from ml.strategies.elo_edge_detector import scan_ufc_edges
                ufc_engine = Glicko2Engine.load(ufc_path)
                if ufc_engine and ufc_engine.ratings:
                    async with async_session() as session:
                        ufc_signals = await scan_ufc_edges(session, ufc_engine)
                        total_new += len(ufc_signals)
                        logger.info(f"UFC scan: {len(ufc_signals)} new signals")
            except Exception as e:
                logger.error(f"UFC scan failed: {e}")
        else:
            logger.info("UFC ratings not found (run: python scripts/build_elo_ratings_ufc.py)")

        async with async_session() as session:
            active = await get_active_edges(session)
            logger.info(f"Elo total: {total_new} new signals, {len(active)} active edges")

    except Exception as e:
        logger.error(f"Elo edge scan failed: {e}")


async def scan_ufc_edges():
    """Scan active sports markets for UFC Elo-based edge signals."""
    logger.info("Starting Elo edge scan (UFC)...")
    try:
        from ml.strategies.elo_edge_detector import scan_ufc_edges as _scan_ufc, get_active_edges
        from ml.models.elo_sports import Glicko2Engine

        engine = Glicko2Engine.load("ml/saved_models/elo_ufc_ratings.joblib")
        if not engine or not engine.ratings:
            logger.info("No UFC Elo ratings loaded yet, skipping scan")
            return

        async with async_session() as session:
            new_signals = await _scan_ufc(session, engine)
            active = await get_active_edges(session)
            ufc_active = sum(1 for a in active if a.get("sport") == "ufc")
            logger.info(f"Elo (UFC) scan: {len(new_signals)} new signals, {ufc_active} UFC active edges")
    except FileNotFoundError:
        logger.info("UFC Elo ratings not found, skipping (run build_elo_ratings_ufc.py first)")
    except Exception as e:
        logger.error(f"UFC Elo edge scan failed: {e}")


# ── Ensemble Edge Scanning ──────────────────────────────────────────

async def scan_ensemble_edges():
    """Scan active markets for ML ensemble-based edge signals."""
    logger.info("Starting ensemble edge scan...")
    try:
        from ml.models.ensemble import EnsembleModel
        from ml.strategies.ensemble_edge_detector import (
            check_quality_gates, detect_edge,
        )
        from db.models import PortfolioPosition
        from sqlalchemy import update as sa_update

        # Load ensemble (cached singleton pattern)
        ensemble = EnsembleModel()
        if not ensemble.load_all():
            logger.info("Ensemble not fully loaded, skipping scan")
            return

        async with async_session() as session:
            # Get market IDs with open auto positions — protect their signals from expiry.
            # Without this, signals expire at 4h and auto_closer force-closes positions
            # because the scanner skips open-position markets (no new signal → expiry → close loop).
            open_pos_result = await session.execute(
                select(PortfolioPosition.market_id)
                .where(
                    PortfolioPosition.exit_time == None,  # noqa: E711
                    PortfolioPosition.portfolio_type == "auto",
                )
            )
            open_position_market_ids = {row[0] for row in open_pos_result.all()}

            # Expire old signals (>4 hrs), but NOT for markets with open positions.
            # Positions stay open until stop-loss / edge-invalidation / market resolution.
            cutoff = datetime.utcnow() - timedelta(hours=4)
            expiry_query = sa_update(EnsembleEdgeSignal).where(
                EnsembleEdgeSignal.expired_at == None,  # noqa: E711
                EnsembleEdgeSignal.detected_at < cutoff,
            )
            if open_position_market_ids:
                expiry_query = expiry_query.where(
                    EnsembleEdgeSignal.market_id.not_in(open_position_market_ids)
                )
            await session.execute(expiry_query.values(expired_at=datetime.utcnow()))
            await session.commit()

            # Query top 2000 liquid active markets
            markets = (await session.execute(
                select(Market)
                .where(Market.is_active == True, Market.price_yes != None)  # noqa
                .order_by(Market.volume_24h.desc())
                .limit(2000)
            )).scalars().all()

            # Per-stage diagnostic counters
            n_quality_gate = 0
            n_has_edge = 0
            n_skipped_open_pos = 0
            edges_found = 0

            for market in markets:
                try:
                    gate = check_quality_gates(market)
                    if not gate.passes:
                        continue
                    n_quality_gate += 1

                    result = ensemble.predict_market(market)
                    edge = detect_edge(market, result)

                    if edge.direction and edge.net_ev >= 0.04 and edge.confidence >= 0.50:
                        n_has_edge += 1

                        if market.id in open_position_market_ids:
                            n_skipped_open_pos += 1
                            continue  # Signal kept alive by expiry protection above

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
            logger.info(
                f"Ensemble scan: {edges_found} new signals | "
                f"{len(markets)} scanned → {n_quality_gate} passed gates → "
                f"{n_has_edge} with edge → {n_skipped_open_pos} skipped (open pos)"
            )

    except Exception as e:
        logger.error(f"Ensemble edge scan failed: {e}")


# ── Real-Time Streaming ──────────────────────────────────────────────

async def init_realtime_streams():
    """Initialize WebSocket streams for ultra-low-latency arbitrage detection.

    Connects to Redis price cache and Polymarket WebSocket, then subscribes
    to top markets by volume. Runs stream in background task.

    Performance: <100ms end-to-end (vs 5-10min with polling)
    """
    global _price_cache, _polymarket_stream

    try:
        logger.info("Initializing real-time WebSocket streams...")

        # 1. Initialize Redis price cache
        _price_cache = PriceCache()
        await _price_cache.connect()

        if not _price_cache.redis:
            logger.warning("⚠️  Redis not available - WebSocket streaming disabled")
            logger.warning("   Install Redis: https://redis.io/download")
            return

        logger.info("✅ Redis price cache connected")

        # 2. Initialize Polymarket WebSocket stream
        _polymarket_stream = PolymarketStream(_price_cache)
        await _polymarket_stream.connect()

        if not _polymarket_stream.running:
            logger.warning("⚠️  Polymarket WebSocket connection failed - streaming disabled")
            return

        logger.info("✅ Polymarket WebSocket connected")

        # 3. Get top markets by volume to subscribe
        async with async_session() as session:
            result = await session.execute(
                select(Market.token_id_yes)
                .where(
                    Market.token_id_yes.isnot(None),
                    Market.resolution_value.is_(None),  # Only active markets
                )
                .order_by(Market.volume_total.desc())
                .limit(100)  # Top 100 markets by volume
            )
            market_ids = [str(row[0]) for row in result.all()]

        if not market_ids:
            logger.warning("No markets found for WebSocket subscription")
            return

        # 4. Subscribe to markets
        await _polymarket_stream.subscribe_markets(market_ids)
        logger.info(f"📡 Subscribed to {len(market_ids)} markets")

        # 5. Start streaming in background (runs forever)
        asyncio.create_task(_polymarket_stream.stream())
        logger.info("🌐 WebSocket streaming started in background")

    except Exception as e:
        logger.error(f"❌ Failed to initialize WebSocket streams: {e}")
        logger.error("   Continuing without real-time streaming...")


async def check_arbitrage_signals():
    """Check for real-time arbitrage signals from WebSocket streams.

    Called every cycle to process signals detected by the price cache.
    Signals are automatically generated when cross-platform spread >3%.
    """
    if not _price_cache:
        return

    try:
        signals = await _price_cache.get_arbitrage_signals()

        if signals:
            logger.info(f"🚨 {len(signals)} real-time arbitrage signals detected!")
            for signal in signals:
                logger.info(
                    f"   Market {signal['market_id']}: "
                    f"{signal['spread_pct']:.1f}% spread "
                    f"({signal['platform_1']} vs {signal['platform_2']})"
                )
                # TODO: Execute arbitrage trade (Phase 3)
                # For now, just log the opportunity

    except Exception as e:
        logger.error(f"Arbitrage signal check error: {e}")


async def cleanup_realtime_streams():
    """Gracefully close WebSocket streams and Redis connections."""
    global _price_cache, _polymarket_stream

    try:
        if _polymarket_stream:
            logger.info("Closing Polymarket WebSocket...")
            await _polymarket_stream.close()
            _polymarket_stream = None

        if _price_cache:
            logger.info("Closing Redis connection...")
            await _price_cache.close()
            _price_cache = None

        logger.info("✅ Real-time streams cleanup complete")

    except Exception as e:
        logger.error(f"Stream cleanup error: {e}")


# ── New Strategy Scans ───────────────────────────────────────────────

async def scan_new_strategies():
    """Run all new strategy scans and persist signals to unified strategy_signals table."""
    from config.settings import settings
    from db.models import StrategySignal
    from datetime import timedelta

    logger.info("Starting new strategy scans...")

    async with async_session() as session:
        # Expire old strategy signals (>6 hours)
        cutoff = datetime.utcnow() - timedelta(hours=6)
        from sqlalchemy import update as sa_update
        await session.execute(
            sa_update(StrategySignal).where(
                StrategySignal.expired_at == None,  # noqa: E711
                StrategySignal.detected_at < cutoff,
            ).values(expired_at=datetime.utcnow())
        )
        await session.commit()

    total_signals = 0

    # 1. Favorite-Longshot Bias (always run, no API cost)
    if settings.longshot_bias_enabled:
        try:
            from ml.strategies.longshot_bias import scan_longshot_bias
            async with async_session() as session:
                edges = await scan_longshot_bias(session)
                for edge in edges:
                    sig = StrategySignal(
                        market_id=edge["market_id"],
                        strategy="longshot_bias",
                        direction=edge["direction"],
                        implied_prob=edge.get("true_prob"),
                        market_price=edge["market_price"],
                        raw_edge=edge.get("raw_edge"),
                        net_ev=edge["net_ev"],
                        fee_cost=edge.get("fee_cost", 0),
                        kelly_fraction=edge.get("kelly_fraction"),
                        confidence=edge.get("confidence"),
                        quality_tier="high" if edge["net_ev"] >= 0.05 else "medium",
                        signal_metadata={"bias_type": edge.get("bias_type"), "time_mult": edge.get("time_horizon_multiplier")},
                    )
                    session.add(sig)
                await session.commit()
                total_signals += len(edges)
        except Exception as e:
            logger.error(f"Longshot bias scan failed: {e}")

    # 2. Resolution Convergence (no API cost)
    if settings.resolution_convergence_enabled:
        try:
            from ml.strategies.resolution_convergence import scan_resolution_convergence
            async with async_session() as session:
                edges = await scan_resolution_convergence(session)
                for edge in edges:
                    sig = StrategySignal(
                        market_id=edge["market_id"],
                        strategy="resolution_convergence",
                        direction=edge["direction"],
                        implied_prob=edge.get("true_prob"),
                        market_price=edge["market_price"],
                        raw_edge=edge.get("raw_edge"),
                        net_ev=edge["net_ev"],
                        fee_cost=edge.get("fee_cost", 0),
                        kelly_fraction=edge.get("kelly_fraction"),
                        confidence=edge.get("confidence"),
                        quality_tier=edge.get("quality_tier", "medium"),
                        signal_metadata={"hours_to_resolution": edge.get("hours_to_resolution"), "time_factor": edge.get("time_factor")},
                    )
                    session.add(sig)
                await session.commit()
                total_signals += len(edges)
        except Exception as e:
            logger.error(f"Resolution convergence scan failed: {e}")

    # 3. Order Flow Analysis (no API cost, uses existing orderbook data)
    if settings.orderflow_enabled:
        try:
            from ml.strategies.orderflow_analyzer import scan_orderflow_signals
            async with async_session() as session:
                edges = await scan_orderflow_signals(session)
                for edge in edges:
                    sig = StrategySignal(
                        market_id=edge["market_id"],
                        strategy="orderflow",
                        direction=edge["direction"],
                        implied_prob=edge.get("implied_prob"),
                        market_price=edge["market_price"],
                        raw_edge=edge.get("raw_edge"),
                        net_ev=edge["net_ev"],
                        fee_cost=edge.get("fee_cost", 0),
                        kelly_fraction=edge.get("kelly_fraction"),
                        confidence=edge.get("confidence"),
                        quality_tier="medium",
                        signal_metadata=edge.get("flow_analysis"),
                    )
                    session.add(sig)
                await session.commit()
                total_signals += len(edges)
        except Exception as e:
            logger.error(f"Orderflow scan failed: {e}")

    # 4. Smart Money (no API cost, uses existing trader data)
    if settings.smart_money_enabled:
        try:
            from ml.strategies.smart_money import analyze_smart_money_positioning
            async with async_session() as session:
                edges = await analyze_smart_money_positioning(session)
                for edge in edges:
                    sig = StrategySignal(
                        market_id=edge["market_id"],
                        strategy="smart_money",
                        direction=edge["direction"],
                        implied_prob=edge.get("implied_prob"),
                        market_price=edge["market_price"],
                        raw_edge=edge.get("raw_edge"),
                        net_ev=edge["net_ev"],
                        fee_cost=edge.get("fee_cost", 0),
                        kelly_fraction=edge.get("kelly_fraction"),
                        confidence=edge.get("confidence"),
                        quality_tier="medium",
                        signal_metadata={"volume_surge": edge.get("volume_surge"), "smart_wallets": edge.get("smart_wallet_count")},
                    )
                    session.add(sig)
                await session.commit()
                total_signals += len(edges)
        except Exception as e:
            logger.error(f"Smart money scan failed: {e}")

    logger.info(f"New strategy scans complete: {total_signals} total signals")
    return total_signals


async def scan_news_strategy():
    """Run news catalyst scan (separate due to API rate limits)."""
    from config.settings import settings
    from db.models import StrategySignal

    if not settings.news_catalyst_enabled:
        return

    try:
        from ml.strategies.news_catalyst import scan_news_catalysts
        async with async_session() as session:
            edges = await scan_news_catalysts(session)
            for edge in edges:
                sig = StrategySignal(
                    market_id=edge["market_id"],
                    strategy="news_catalyst",
                    direction=edge["direction"],
                    implied_prob=edge.get("implied_prob"),
                    market_price=edge["market_price"],
                    raw_edge=edge.get("raw_edge"),
                    net_ev=edge["net_ev"],
                    fee_cost=edge.get("fee_cost", 0),
                    kelly_fraction=edge.get("kelly_fraction"),
                    confidence=edge.get("confidence"),
                    quality_tier="medium",
                    signal_metadata=edge.get("sentiment"),
                )
                session.add(sig)
            await session.commit()
            logger.info(f"News catalyst scan: {len(edges)} signals")
    except Exception as e:
        logger.error(f"News catalyst scan failed: {e}")


async def scan_llm_strategy():
    """Run LLM superforecaster scan (separate due to API cost)."""
    from config.settings import settings
    from db.models import StrategySignal

    if not settings.llm_forecast_enabled or not settings.anthropic_api_key:
        return

    try:
        from ml.strategies.llm_forecaster import scan_llm_forecasts
        async with async_session() as session:
            edges = await scan_llm_forecasts(session, max_markets=settings.llm_forecast_max_markets)
            for edge in edges:
                sig = StrategySignal(
                    market_id=edge["market_id"],
                    strategy="llm_forecast",
                    direction=edge["direction"],
                    implied_prob=edge.get("llm_probability"),
                    market_price=edge["market_price"],
                    raw_edge=edge.get("raw_edge"),
                    net_ev=edge["net_ev"],
                    fee_cost=edge.get("fee_cost", 0),
                    kelly_fraction=edge.get("kelly_fraction"),
                    confidence=edge.get("confidence"),
                    quality_tier="high" if edge["net_ev"] >= 0.05 else "medium",
                    signal_metadata={
                        "base_rate": edge.get("base_rate"),
                        "key_factors": edge.get("key_factors"),
                        "reasoning": edge.get("reasoning"),
                    },
                )
                session.add(sig)
            await session.commit()
            logger.info(f"LLM forecast scan: {len(edges)} signals")
    except Exception as e:
        logger.error(f"LLM forecast scan failed: {e}")


async def scan_clustering_strategy():
    """Run market correlation clustering scan."""
    from config.settings import settings
    from db.models import StrategySignal

    if not settings.market_clustering_enabled:
        return

    try:
        from ml.strategies.market_clustering import scan_market_clusters
        async with async_session() as session:
            edges = await scan_market_clusters(session)
            for edge in edges:
                sig = StrategySignal(
                    market_id=edge["market_id"],
                    strategy="market_clustering",
                    direction=edge["direction"],
                    implied_prob=edge.get("implied_prob"),
                    market_price=edge["market_price"],
                    raw_edge=edge.get("raw_edge"),
                    net_ev=edge["net_ev"],
                    fee_cost=edge.get("fee_cost", 0),
                    kelly_fraction=edge.get("kelly_fraction"),
                    confidence=edge.get("confidence"),
                    quality_tier="medium",
                    signal_metadata={
                        "correlation": edge.get("correlation"),
                        "related_market_id": edge.get("related_market_id"),
                        "related_question": edge.get("related_market_question"),
                    },
                )
                session.add(sig)
            await session.commit()
            logger.info(f"Market clustering scan: {len(edges)} signals")
    except Exception as e:
        logger.error(f"Market clustering scan failed: {e}")


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

    Timing (default 20s price interval, set via price_poll_interval_sec):
      - Prices:          every cycle      (~20s)
      - Auto-close:      every cycle      (~20s) — fast stop-loss response
      - Arbitrage:       every 5 cycles   (~2 min)
      - Orderbooks:      every 5 cycles   (~2 min)
      - Ensemble scan:   every 5 cycles   (~2 min)
      - Auto-trade:      every 5 cycles   (~2 min)
      - Elo scan:        every 30 cycles  (~10 min)
      - Trader refresh:  every 90 cycles  (~30 min)
      - Markets:         every 180 cycles (~1 hr) + re-match after refresh
    """
    from config.settings import settings

    logger.info("Starting pipeline loop...")

    # ── Backfill trader profiles FIRST (so Copy Trading page works immediately) ──
    try:
        from sqlalchemy import select as sa_select
        from db.models import TraderProfile
        async with async_session() as session:
            count = (await session.execute(
                sa_select(TraderProfile)
            )).scalars().first()
            if not count:
                logger.info("No trader profiles found, backfilling from Polymarket leaderboard...")
                from data_pipeline.collectors.trader_data import (
                    fetch_polymarket_leaderboard, fetch_trader_positions,
                    calculate_trader_stats, generate_trader_bio, clean_display_name,
                )
                import asyncio as _asyncio

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
                    pnl = float(td.get("pnl", 0))
                    volume = float(td.get("vol", 0))
                    # Fetch real positions (with rate limit)
                    positions = await fetch_trader_positions(wallet, limit=100)
                    await _asyncio.sleep(0.2)
                    if positions:
                        stats = calculate_trader_stats(td, positions)
                    else:
                        stats = {
                            "total_pnl": pnl, "roi_pct": (pnl / max(volume * 0.3, 1)) * 100 if volume > 0 else 0,
                            "win_rate": 0.0, "total_trades": 0, "winning_trades": 0,
                            "avg_trade_duration_hrs": 0.0, "risk_score": 5, "max_drawdown": 0.0,
                        }
                    session.add(TraderProfile(
                        user_id=wallet,
                        display_name=clean_display_name(td),
                        bio=generate_trader_bio(td, stats),
                        total_pnl=stats["total_pnl"], roi_pct=stats["roi_pct"],
                        win_rate=stats["win_rate"], total_trades=stats["total_trades"],
                        winning_trades=stats["winning_trades"],
                        avg_trade_duration_hrs=stats["avg_trade_duration_hrs"],
                        risk_score=stats["risk_score"], max_drawdown=stats["max_drawdown"],
                        follower_count=0, is_public=True, accepts_copiers=True,
                    ))
                    created_count += 1
                await session.commit()
                logger.info(f"Backfilled {created_count} trader profiles (real trade data)")
            else:
                logger.info("Trader profiles already exist, skipping backfill")
    except Exception as e:
        logger.error(f"Trader backfill failed: {e}")

    # ── Initialize real-time WebSocket streams ──
    try:
        await init_realtime_streams()
    except Exception as e:
        logger.error(f"WebSocket stream initialization failed: {e}")
        logger.error("Continuing without real-time streaming...")

    # ── Initialize adaptive confidence adjuster (Phase 2.5) ──
    try:
        async with async_session() as adj_session:
            await init_confidence_adjuster(adj_session)
            stats = get_adjuster_stats()
            logger.info(f"Confidence adjuster loaded: {stats.get('total_segments', 0)} segments")
    except Exception as e:
        logger.error(f"Confidence adjuster initialization failed: {e}")
        logger.error("Continuing without adaptive confidence adjustment...")

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

    # ── Initial new strategy scans ──
    try:
        await scan_new_strategies()
    except Exception as e:
        logger.error(f"Initial new strategy scan failed: {e}")

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
    cycles_per_new_strategies = max(1, 300 // price_interval)  # Every ~5 min
    cycles_per_news_scan = max(1, settings.news_catalyst_interval_sec // price_interval)  # Every ~15 min
    cycles_per_llm_scan = max(1, settings.llm_forecast_interval_sec // price_interval)  # Every ~30 min
    cycles_per_clustering = max(1, settings.market_clustering_interval_sec // price_interval)  # Every ~1 hr

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

        # ── Real-time arbitrage signal check EVERY cycle (<1ms, just queue read) ──
        try:
            await check_arbitrage_signals()
        except Exception as e:
            logger.error(f"Real-time arb signal check error: {e}")

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

        # ── Ensemble edge scan every ~2 min ──
        if cycle % cycles_per_ensemble_scan == 0:
            try:
                await scan_ensemble_edges()
            except Exception as e:
                logger.error(f"Ensemble edge scan error: {e}")

        # ── Auto-trade every ~2 min (same cadence as ensemble scan) ──
        if cycle % cycles_per_auto_trade == 0:
            try:
                from execution.paper_executor import execute_paper_trades
                async with async_session() as exec_session:
                    created = await execute_paper_trades(exec_session)
                    if created:
                        logger.info(f"Auto paper trades: {len(created)} positions opened")
            except Exception as e:
                logger.error(f"Paper executor error: {e}")

            # Copy trading sync (same cadence as auto-trading)
            try:
                from data_pipeline.copy_engine import sync_copy_positions
                async with async_session() as copy_session:
                    result = await sync_copy_positions(copy_session)
                    if result["opened"] or result["closed"]:
                        logger.info(
                            f"Copy sync: {result['opened']} opened, {result['closed']} closed"
                        )
            except Exception as e:
                logger.error(f"Copy sync error: {e}")

        # ── New strategy scans every ~5 min ──
        if cycle % cycles_per_new_strategies == 0:
            try:
                await scan_new_strategies()
            except Exception as e:
                logger.error(f"New strategy scan error: {e}")

        # ── News catalyst scan every ~15 min ──
        if cycle % cycles_per_news_scan == 0:
            try:
                await scan_news_strategy()
            except Exception as e:
                logger.error(f"News catalyst scan error: {e}")

        # ── LLM forecast scan every ~30 min ──
        if cycle % cycles_per_llm_scan == 0:
            try:
                await scan_llm_strategy()
            except Exception as e:
                logger.error(f"LLM forecast scan error: {e}")

        # ── Market clustering every ~1 hour ──
        if cycle % cycles_per_clustering == 0:
            try:
                await scan_clustering_strategy()
            except Exception as e:
                logger.error(f"Market clustering scan error: {e}")

        # ── Trader stats refresh every ~30 min ──
        if cycle % cycles_per_trader_refresh == 0:
            try:
                await refresh_trader_stats()
            except Exception as e:
                logger.error(f"Trader stats refresh error: {e}")

            # ── Refresh confidence adjuster (Phase 2.5) ──
            try:
                async with async_session() as adj_session:
                    await refresh_confidence_adjuster(adj_session)
                    stats = get_adjuster_stats()
                    logger.info(
                        f"Confidence adjuster refreshed: {stats.get('total_segments', 0)} segments, "
                        f"{stats.get('total_trades_analyzed', 0)} trades analyzed"
                    )
            except Exception as e:
                logger.error(f"Confidence adjuster refresh error: {e}")

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

            # Log P&L breakdown by price zone and direction
            try:
                from sqlalchemy import func as sql_func
                async with async_session() as zone_session:
                    from db.models import PortfolioPosition
                    from datetime import timedelta
                    cutoff = datetime.utcnow() - timedelta(days=7)
                    closed_result = await zone_session.execute(
                        select(PortfolioPosition).where(
                            PortfolioPosition.portfolio_type == "auto",
                            PortfolioPosition.exit_time != None,  # noqa
                            PortfolioPosition.exit_time >= cutoff,
                        )
                    )
                    closed_positions = closed_result.scalars().all()
                    if closed_positions:
                        # Bucket by entry price zone
                        zone_pnl = {"0-20%": 0.0, "20-40%": 0.0, "40-60%": 0.0, "60-80%": 0.0, "80-100%": 0.0}
                        zone_count = {"0-20%": 0, "20-40%": 0, "40-60%": 0, "60-80%": 0, "80-100%": 0}
                        dir_pnl = {"buy_yes": 0.0, "buy_no": 0.0}
                        dir_count = {"buy_yes": 0, "buy_no": 0}
                        dir_wins = {"buy_yes": 0, "buy_no": 0}
                        for p in closed_positions:
                            price = p.entry_price or 0.5
                            pnl = p.realized_pnl or 0.0
                            direction = "buy_yes" if p.side == "yes" else "buy_no"
                            if price < 0.20: zone = "0-20%"
                            elif price < 0.40: zone = "20-40%"
                            elif price < 0.60: zone = "40-60%"
                            elif price < 0.80: zone = "60-80%"
                            else: zone = "80-100%"
                            zone_pnl[zone] += pnl
                            zone_count[zone] += 1
                            dir_pnl[direction] += pnl
                            dir_count[direction] += 1
                            if pnl > 0:
                                dir_wins[direction] += 1
                        zone_str = " | ".join(f"{z}: ${v:.2f}({zone_count[z]})" for z, v in zone_pnl.items() if zone_count[z] > 0)
                        yes_wr = f"{dir_wins['buy_yes']/max(dir_count['buy_yes'],1):.0%}" if dir_count["buy_yes"] > 0 else "N/A"
                        no_wr = f"{dir_wins['buy_no']/max(dir_count['buy_no'],1):.0%}" if dir_count["buy_no"] > 0 else "N/A"
                        logger.info(
                            f"P&L BY ZONE (7d): {zone_str} | "
                            f"buy_yes: ${dir_pnl['buy_yes']:.2f}({dir_count['buy_yes']} trades, {yes_wr} win) | "
                            f"buy_no: ${dir_pnl['buy_no']:.2f}({dir_count['buy_no']} trades, {no_wr} win)"
                        )
            except Exception as e:
                logger.error(f"P&L zone tracking failed: {e}")

        # ── Full market refresh + deactivate expired + re-match + cleanup every ~1 hr ──
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

            # Data retention: prune old time-series (only when explicitly enabled).
            # Default is cleanup_enabled=False — run scripts/export_archive_to_local.py
            # first, then set CLEANUP_ENABLED=true in env to enable batched deletes.
            if settings.cleanup_enabled:
                try:
                    async with async_session() as cleanup_session:
                        deleted = await cleanup_old_data(
                            cleanup_session,
                            days=settings.retention_days,
                            batch_size=settings.cleanup_batch_size,
                        )
                        total = sum(deleted.values())
                        if total > 0:
                            logger.info(f"Data cleanup: {total} old rows pruned {deleted}")
                except Exception as e:
                    logger.error(f"Data cleanup error: {e}")
            else:
                logger.debug(
                    "Data cleanup skipped (cleanup_enabled=False). "
                    "Run export_archive_to_local.py first, then set CLEANUP_ENABLED=true."
                )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    async def main():
        from db.database import init_db
        await init_db()
        await run_pipeline_once()

    asyncio.run(main())
