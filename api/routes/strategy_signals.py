"""Unified strategy signals endpoint — all trading signals in one response."""

from datetime import datetime
from fastapi import APIRouter, Query
from sqlalchemy import select, func

from db.database import async_session
from db.models import (
    EnsembleEdgeSignal, EloEdgeSignal, ArbitrageOpportunity, Market,
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

        # --- Summary stats ---
        total_signals = len(ensemble_edges) + len(elo_edges) + len(arbitrage_opps)
        high_confidence = sum(1 for e in ensemble_edges if e["quality_tier"] == "high")
        avg_kelly = 0.0
        kelly_values = [e["kelly_fraction"] for e in ensemble_edges if e["kelly_fraction"] > 0]
        if kelly_values:
            avg_kelly = round(sum(kelly_values) / len(kelly_values), 4)

        return {
            "ensemble_edges": ensemble_edges,
            "elo_edges": elo_edges,
            "arbitrage_opportunities": arbitrage_opps,
            "summary": {
                "total_signals": total_signals,
                "ensemble_count": len(ensemble_edges),
                "elo_count": len(elo_edges),
                "arbitrage_count": len(arbitrage_opps),
                "high_confidence_count": high_confidence,
                "avg_kelly_fraction": avg_kelly,
            },
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
