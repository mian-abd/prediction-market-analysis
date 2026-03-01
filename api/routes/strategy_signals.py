"""Unified strategy signals endpoint — all trading signals in one response."""

from datetime import datetime
from fastapi import APIRouter, Query
from sqlalchemy import select, func

from db.database import async_session
from db.models import (
    EnsembleEdgeSignal, EloEdgeSignal, FavoriteLongshotEdgeSignal,
    ArbitrageOpportunity, Market,
    StrategySignal,
)

router = APIRouter(tags=["strategies"])


@router.get("/strategies/signals")
async def get_all_strategy_signals(
    limit: int = Query(default=50, le=200),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    quality_tier: str | None = Query(default=None, description="Filter by quality tier: high, medium, low"),
):
    """Unified endpoint for all active strategy signals.

    Returns ensemble edges, Elo edges, and arbitrage opportunities
    in a single response for the dashboard.
    """
    async with async_session() as session:
        # --- Ensemble Edge Signals ---
        ensemble_query = (
            select(EnsembleEdgeSignal, Market.question, Market.slug)
            .join(Market, EnsembleEdgeSignal.market_id == Market.id)
            .where(EnsembleEdgeSignal.expired_at == None)  # noqa: E711
            .where(EnsembleEdgeSignal.confidence >= min_confidence)
            .order_by(EnsembleEdgeSignal.net_ev.desc())
            .limit(limit)
        )
        if quality_tier:
            ensemble_query = ensemble_query.where(
                EnsembleEdgeSignal.quality_tier == quality_tier
            )
        result = await session.execute(ensemble_query)
        ensemble_rows = result.all()

        ensemble_edges = []
        for signal, question, slug in ensemble_rows:
            ensemble_edges.append({
                "id": signal.id,
                "market_id": signal.market_id,
                "market_question": question,
                "market_slug": slug,
                "detected_at": signal.detected_at.isoformat() if signal.detected_at else None,
                "direction": signal.direction,
                "ensemble_prob": round(signal.ensemble_prob, 4),
                "market_price": round(signal.market_price, 4),
                "raw_edge_pct": round(signal.raw_edge * 100, 2),
                "net_ev_pct": round(signal.net_ev * 100, 2),
                "kelly_fraction": round(signal.kelly_fraction, 4) if signal.kelly_fraction else 0,
                "confidence": signal.confidence,
                "quality_tier": signal.quality_tier,
                "model_predictions": signal.model_predictions,
            })

        # --- Elo Edge Signals ---
        elo_result = await session.execute(
            select(EloEdgeSignal, Market.question, Market.slug)
            .join(Market, EloEdgeSignal.market_id == Market.id)
            .where(EloEdgeSignal.expired_at == None)  # noqa: E711
            .order_by(EloEdgeSignal.net_edge.desc())
            .limit(limit)
        )
        elo_rows = elo_result.all()

        elo_edges = []
        for signal, question, slug in elo_rows:
            elo_edges.append({
                "id": signal.id,
                "market_id": signal.market_id,
                "market_question": question,
                "market_slug": slug,
                "sport": signal.sport,
                "player_a": signal.player_a,
                "player_b": signal.player_b,
                "surface": signal.surface,
                "elo_prob_a": round(signal.elo_prob_a, 4),
                "market_price_yes": round(signal.market_price_yes, 4),
                "raw_edge_pct": round(signal.raw_edge * 100, 2),
                "net_edge_pct": round(signal.net_edge * 100, 2),
                "kelly_fraction": round(signal.kelly_fraction, 4) if signal.kelly_fraction else 0,
                "elo_confidence": round(signal.elo_confidence, 3),
            })

        # --- Favorite-Longshot Edge Signals ---
        fl_result = await session.execute(
            select(FavoriteLongshotEdgeSignal, Market.question, Market.slug)
            .join(Market, FavoriteLongshotEdgeSignal.market_id == Market.id)
            .where(FavoriteLongshotEdgeSignal.expired_at == None)  # noqa: E711
            .order_by(FavoriteLongshotEdgeSignal.net_ev.desc())
            .limit(limit)
        )
        fl_rows = fl_result.all()

        favorite_longshot_edges = []
        for signal, question, slug in fl_rows:
            favorite_longshot_edges.append({
                "id": signal.id,
                "market_id": signal.market_id,
                "market_question": question,
                "market_slug": slug,
                "detected_at": signal.detected_at.isoformat() if signal.detected_at else None,
                "direction": signal.direction,
                "calibrated_prob": round(signal.calibrated_prob, 4),
                "market_price": round(signal.market_price, 4),
                "raw_edge_pct": round(signal.raw_edge * 100, 2),
                "net_ev_pct": round(signal.net_ev * 100, 2),
                "kelly_fraction": round(signal.kelly_fraction, 4) if signal.kelly_fraction else 0,
                "category": signal.category,
                "signal_type": signal.signal_type,
            })

        # --- Arbitrage Opportunities ---
        arb_result = await session.execute(
            select(ArbitrageOpportunity)
            .where(ArbitrageOpportunity.expired_at == None)  # noqa: E711
            .order_by(ArbitrageOpportunity.net_profit_pct.desc())
            .limit(limit)
        )
        arb_rows = arb_result.scalars().all()

        arbitrage_opps = []
        for opp in arb_rows:
            arbitrage_opps.append({
                "id": opp.id,
                "strategy_type": opp.strategy_type,
                "detected_at": opp.detected_at.isoformat() if opp.detected_at else None,
                "market_ids": opp.market_ids,
                "net_profit_pct": round(opp.net_profit_pct, 2),
                "estimated_profit_usd": round(opp.estimated_profit_usd, 2) if opp.estimated_profit_usd else 0,
            })

        # --- New Strategy Signals ---
        new_strat_query = (
            select(StrategySignal, Market.question, Market.slug)
            .join(Market, StrategySignal.market_id == Market.id)
            .where(StrategySignal.expired_at == None)  # noqa: E711
            .order_by(StrategySignal.net_ev.desc())
            .limit(limit)
        )
        if min_confidence > 0:
            new_strat_query = new_strat_query.where(
                StrategySignal.confidence >= min_confidence
            )
        new_strat_result = await session.execute(new_strat_query)
        new_strat_rows = new_strat_result.all()

        new_strategy_signals = []
        for signal, question, slug in new_strat_rows:
            new_strategy_signals.append({
                "id": signal.id,
                "market_id": signal.market_id,
                "market_question": question,
                "market_slug": slug,
                "strategy": signal.strategy,
                "detected_at": signal.detected_at.isoformat() if signal.detected_at else None,
                "direction": signal.direction,
                "implied_prob": round(signal.implied_prob, 4) if signal.implied_prob else None,
                "market_price": round(signal.market_price, 4),
                "raw_edge_pct": round(signal.raw_edge * 100, 2) if signal.raw_edge else 0,
                "net_ev_pct": round(signal.net_ev * 100, 2),
                "kelly_fraction": round(signal.kelly_fraction, 4) if signal.kelly_fraction else 0,
                "confidence": signal.confidence,
                "quality_tier": signal.quality_tier,
                "metadata": signal.signal_metadata,
            })

        # Group new strategies by type
        strategy_breakdown = {}
        for s in new_strategy_signals:
            st = s["strategy"]
            strategy_breakdown.setdefault(st, []).append(s)

        # --- Summary stats ---
        total_signals = (
            len(ensemble_edges) + len(elo_edges) + len(favorite_longshot_edges) +
            len(arbitrage_opps) + len(new_strategy_signals)
        )
        high_confidence = (
            sum(1 for e in ensemble_edges if e["quality_tier"] == "high") +
            sum(1 for e in favorite_longshot_edges if e.get("quality_tier") == "high") +
            sum(1 for s in new_strategy_signals if s.get("quality_tier") == "high")
        )
        all_kelly = (
            [e["kelly_fraction"] for e in ensemble_edges if e["kelly_fraction"] > 0] +
            [e["kelly_fraction"] for e in favorite_longshot_edges if e["kelly_fraction"] > 0] +
            [s["kelly_fraction"] for s in new_strategy_signals if s["kelly_fraction"] > 0]
        )
        avg_kelly = round(sum(all_kelly) / len(all_kelly), 4) if all_kelly else 0.0

        return {
            "ensemble_edges": ensemble_edges,
            "elo_edges": elo_edges,
            "favorite_longshot_edges": favorite_longshot_edges,
            "arbitrage_opportunities": arbitrage_opps,
            "new_strategy_signals": new_strategy_signals,
            "strategy_breakdown": {
                st: len(sigs) for st, sigs in strategy_breakdown.items()
            },
            "summary": {
                "total_signals": total_signals,
                "ensemble_count": len(ensemble_edges),
                "elo_count": len(elo_edges),
                "favorite_longshot_count": len(favorite_longshot_edges),
                "arbitrage_count": len(arbitrage_opps),
                "new_strategy_count": len(new_strategy_signals),
                "high_confidence_count": high_confidence,
                "avg_kelly_fraction": avg_kelly,
                "strategies_active": list(strategy_breakdown.keys()),
            },
        }


@router.get("/strategies/signal-performance")
async def get_signal_performance():
    """Get historical signal performance time-series.

    Returns daily aggregates of signals generated, correct predictions,
    and cumulative P&L for the signal accuracy chart.
    """
    from collections import defaultdict

    async with async_session() as session:
        # Score any newly resolved signals first
        try:
            from ml.evaluation.resolution_scorer import score_resolved_signals
            await score_resolved_signals(session)
        except Exception as e:
            pass  # Non-critical, continue with existing scores

        # Get all scored ensemble signals
        result = await session.execute(
            select(EnsembleEdgeSignal)
            .where(EnsembleEdgeSignal.was_correct != None)  # noqa
            .order_by(EnsembleEdgeSignal.detected_at)
        )
        signals = result.scalars().all()

        if not signals:
            return {"data": [], "summary": {"total_scored": 0}}

        # Aggregate by date
        daily = defaultdict(lambda: {
            "signals_generated": 0,
            "signals_correct": 0,
            "pnl": 0.0,
        })

        cumulative_pnl = 0.0
        for s in signals:
            date_key = s.detected_at.strftime("%Y-%m-%d") if s.detected_at else "unknown"
            daily[date_key]["signals_generated"] += 1
            if s.was_correct:
                daily[date_key]["signals_correct"] += 1
            daily[date_key]["pnl"] += (s.actual_pnl or 0.0)

        # Build time series
        data = []
        cumulative_pnl = 0.0
        total_correct = 0
        total_generated = 0
        for date_key in sorted(daily.keys()):
            d = daily[date_key]
            cumulative_pnl += d["pnl"]
            total_correct += d["signals_correct"]
            total_generated += d["signals_generated"]
            data.append({
                "date": date_key,
                "signals_generated": d["signals_generated"],
                "signals_correct": d["signals_correct"],
                "daily_pnl": round(d["pnl"], 2),
                "cumulative_pnl": round(cumulative_pnl, 2),
                "cumulative_hit_rate": round(
                    total_correct / total_generated, 4
                ) if total_generated > 0 else 0,
            })

        return {
            "data": data,
            "summary": {
                "total_scored": len(signals),
                "total_correct": total_correct,
                "hit_rate": round(total_correct / len(signals), 4) if signals else 0,
                "cumulative_pnl": round(cumulative_pnl, 2),
            },
        }


@router.get("/strategies/new-signals")
async def get_new_strategy_signals(
    strategy: str | None = Query(default=None, description="Filter by strategy: llm_forecast, longshot_bias, news_catalyst, resolution_convergence, orderflow, smart_money, market_clustering"),
    limit: int = Query(default=50, le=200),
    min_ev: float = Query(default=2.0, description="Minimum net EV in percent"),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
):
    """Get active signals from new strategies.

    Strategies include:
    - llm_forecast: LLM Superforecaster (CHAMP framework)
    - longshot_bias: Favorite-Longshot Bias exploitation
    - news_catalyst: News-driven sentiment trading
    - resolution_convergence: Time-decay near resolution
    - orderflow: Order book dynamics analysis
    - smart_money: Whale/smart money following
    - market_clustering: Correlation-based statistical arbitrage
    """
    async with async_session() as session:
        query = (
            select(StrategySignal, Market.question, Market.slug, Market.normalized_category)
            .join(Market, StrategySignal.market_id == Market.id)
            .where(
                StrategySignal.expired_at == None,  # noqa: E711
                StrategySignal.net_ev >= min_ev / 100.0,
            )
        )
        if strategy:
            query = query.where(StrategySignal.strategy == strategy)
        if min_confidence > 0:
            query = query.where(StrategySignal.confidence >= min_confidence)

        query = query.order_by(StrategySignal.net_ev.desc()).limit(limit)
        result = await session.execute(query)
        rows = result.all()

        signals = []
        for signal, question, slug, category in rows:
            signals.append({
                "id": signal.id,
                "market_id": signal.market_id,
                "market_question": question,
                "market_slug": slug,
                "market_category": category,
                "strategy": signal.strategy,
                "detected_at": signal.detected_at.isoformat() if signal.detected_at else None,
                "direction": signal.direction,
                "implied_prob": round(signal.implied_prob, 4) if signal.implied_prob else None,
                "market_price": round(signal.market_price, 4),
                "raw_edge_pct": round(signal.raw_edge * 100, 2) if signal.raw_edge else 0,
                "net_ev_pct": round(signal.net_ev * 100, 2),
                "fee_cost_pct": round(signal.fee_cost * 100, 2) if signal.fee_cost else 0,
                "kelly_fraction": round(signal.kelly_fraction, 4) if signal.kelly_fraction else 0,
                "confidence": signal.confidence,
                "quality_tier": signal.quality_tier,
                "metadata": signal.signal_metadata,
            })

        # Breakdown by strategy
        by_strategy = {}
        for s in signals:
            by_strategy.setdefault(s["strategy"], []).append(s)

        return {
            "signals": signals,
            "count": len(signals),
            "by_strategy": {st: len(sigs) for st, sigs in by_strategy.items()},
        }


@router.get("/strategies/performance-by-strategy")
async def get_performance_by_strategy():
    """P&L, hit rate, and trade count broken down by auto-trading strategy.

    Covers ensemble, elo, AND all new strategies in a single response.
    Used by the frontend to show which strategies are actually making money.
    """
    from collections import defaultdict
    from db.models import PortfolioPosition

    async with async_session() as session:
        result = await session.execute(
            select(PortfolioPosition)
            .where(
                PortfolioPosition.portfolio_type == "auto",
                PortfolioPosition.exit_time.isnot(None),
            )
            .order_by(PortfolioPosition.exit_time.desc())
        )
        positions = result.scalars().all()

        if not positions:
            return {"strategies": {}, "total_trades": 0, "total_pnl": 0.0}

        by_strategy: dict[str, dict] = defaultdict(lambda: {
            "trades": 0, "wins": 0, "pnl": 0.0,
            "best_trade": 0.0, "worst_trade": 0.0,
            "total_exposure": 0.0,
        })

        for p in positions:
            strat = p.strategy or "unknown"
            pnl = p.realized_pnl or 0.0
            cost = (p.entry_price if p.side == "yes" else (1.0 - p.entry_price)) * p.quantity

            by_strategy[strat]["trades"] += 1
            by_strategy[strat]["pnl"] += pnl
            by_strategy[strat]["total_exposure"] += cost
            if pnl > 0:
                by_strategy[strat]["wins"] += 1
            by_strategy[strat]["best_trade"] = max(by_strategy[strat]["best_trade"], pnl)
            by_strategy[strat]["worst_trade"] = min(by_strategy[strat]["worst_trade"], pnl)

        strategies = {}
        for strat, data in by_strategy.items():
            trades = data["trades"]
            strategies[strat] = {
                "trades": trades,
                "wins": data["wins"],
                "losses": trades - data["wins"],
                "hit_rate": round(data["wins"] / trades, 4) if trades > 0 else 0,
                "total_pnl": round(data["pnl"], 2),
                "avg_pnl_per_trade": round(data["pnl"] / trades, 2) if trades > 0 else 0,
                "roi_pct": round((data["pnl"] / data["total_exposure"]) * 100, 2) if data["total_exposure"] > 0 else 0,
                "best_trade": round(data["best_trade"], 2),
                "worst_trade": round(data["worst_trade"], 2),
            }

        total_pnl = sum(s["total_pnl"] for s in strategies.values())
        total_trades = sum(s["trades"] for s in strategies.values())

        return {
            "strategies": strategies,
            "total_trades": total_trades,
            "total_pnl": round(total_pnl, 2),
            "total_wins": sum(s["wins"] for s in strategies.values()),
            "overall_hit_rate": round(
                sum(s["wins"] for s in strategies.values()) / total_trades, 4
            ) if total_trades > 0 else 0,
        }


@router.get("/strategies/ensemble-edges")
async def get_ensemble_edges(
    limit: int = Query(default=50, le=200),
    min_ev: float = Query(default=3.0, description="Minimum net EV in percent"),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
):
    """Get active ensemble edge signals sorted by expected value."""
    async with async_session() as session:
        result = await session.execute(
            select(EnsembleEdgeSignal, Market.question, Market.slug)
            .join(Market, EnsembleEdgeSignal.market_id == Market.id)
            .where(
                EnsembleEdgeSignal.expired_at == None,  # noqa: E711
                EnsembleEdgeSignal.net_ev >= min_ev / 100.0,
                EnsembleEdgeSignal.confidence >= min_confidence,
            )
            .order_by(EnsembleEdgeSignal.net_ev.desc())
            .limit(limit)
        )
        rows = result.all()

        edges = []
        for signal, question, slug in rows:
            edges.append({
                "id": signal.id,
                "market_id": signal.market_id,
                "market_question": question,
                "market_slug": slug,
                "direction": signal.direction,
                "ensemble_prob": round(signal.ensemble_prob, 4),
                "market_price": round(signal.market_price, 4),
                "raw_edge_pct": round(signal.raw_edge * 100, 2),
                "fee_cost_pct": round(signal.fee_cost * 100, 2),
                "net_ev_pct": round(signal.net_ev * 100, 2),
                "kelly_fraction": round(signal.kelly_fraction, 4) if signal.kelly_fraction else 0,
                "confidence": signal.confidence,
                "quality_tier": signal.quality_tier,
                "detected_at": signal.detected_at.isoformat() if signal.detected_at else None,
            })

        return {"edges": edges, "count": len(edges)}
