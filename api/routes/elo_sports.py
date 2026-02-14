"""Elo/Glicko-2 sports rating API endpoints.

Provides:
- Player ratings lookup
- Head-to-head win probability predictions
- Active edge signals (mispriced markets)
- Rating leaderboards by surface
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_session
from ml.models.elo_sports import Glicko2Engine
from ml.strategies.elo_edge_detector import get_active_edges, scan_for_edges

router = APIRouter(tags=["elo-sports"])

# Singleton engine (loaded once)
_engine: Glicko2Engine | None = None


def get_engine() -> Glicko2Engine:
    """Get or load the Glicko-2 engine."""
    global _engine
    if _engine is None:
        _engine = Glicko2Engine.load("ml/saved_models/elo_atp_ratings.joblib")
    return _engine


@router.get("/elo/ratings")
async def get_ratings(
    sport: str = Query(default="tennis"),
    surface: str = Query(default="overall"),
    limit: int = Query(default=50, le=200),
):
    """Get top players by Elo rating."""
    engine = get_engine()
    players = engine.get_top_players(n=limit, surface=surface)

    return {
        "sport": sport,
        "surface": surface,
        "players": players,
        "total_rated": len(engine.ratings),
    }


@router.get("/elo/player/{player_name}")
async def get_player_rating(
    player_name: str,
    surface: str = Query(default="overall"),
):
    """Get detailed rating for a specific player."""
    engine = get_engine()

    # Try exact match first, then fuzzy
    rating = engine.get_player_rating(player_name, surface)
    if not rating:
        # Try case-insensitive match
        for name in engine.ratings:
            if name.lower() == player_name.lower():
                rating = engine.get_player_rating(name, surface)
                break

    if not rating:
        return {"error": f"Player '{player_name}' not found in ratings database"}

    return rating


@router.get("/elo/predict/tennis")
async def predict_tennis_match(
    player_a: str = Query(..., description="First player name"),
    player_b: str = Query(..., description="Second player name"),
    surface: str = Query(default="hard", description="Playing surface"),
):
    """Predict win probability for a tennis H2H matchup.

    Uses Glicko-2 ratings with surface-specific blending.
    """
    engine = get_engine()

    # Resolve player names (case-insensitive)
    resolved_a = _resolve_player(engine, player_a)
    resolved_b = _resolve_player(engine, player_b)

    if not resolved_a:
        return {"error": f"Player '{player_a}' not found"}
    if not resolved_b:
        return {"error": f"Player '{player_b}' not found"}

    prob_a, confidence = engine.win_probability(resolved_a, resolved_b, surface)

    rating_a = engine.get_player_rating(resolved_a, surface)
    rating_b = engine.get_player_rating(resolved_b, surface)

    return {
        "player_a": resolved_a,
        "player_b": resolved_b,
        "surface": surface,
        "prob_a_wins": round(prob_a, 4),
        "prob_b_wins": round(1 - prob_a, 4),
        "confidence": round(confidence, 4),
        "rating_a": rating_a["mu"] if rating_a else None,
        "rating_b": rating_b["mu"] if rating_b else None,
        "rd_a": rating_a["phi"] if rating_a else None,
        "rd_b": rating_b["phi"] if rating_b else None,
        "rating_diff": round(
            (rating_a["mu"] if rating_a else 1500) - (rating_b["mu"] if rating_b else 1500), 1
        ),
    }


@router.get("/elo/edges")
async def get_elo_edges(
    session: AsyncSession = Depends(get_session),
):
    """Get currently active Elo-based edge signals (mispriced markets)."""
    edges = await get_active_edges(session)
    return {
        "edges": edges,
        "count": len(edges),
    }


@router.post("/elo/scan")
async def trigger_edge_scan(
    session: AsyncSession = Depends(get_session),
):
    """Manually trigger an Elo edge scan on active sports markets."""
    engine = get_engine()
    edges = await scan_for_edges(session, engine)
    return {
        "edges_found": len(edges),
        "edges": edges,
    }


def _resolve_player(engine: Glicko2Engine, name: str) -> str | None:
    """Resolve a player name (case-insensitive + partial match)."""
    # Exact match
    if name in engine.ratings:
        return name

    # Case-insensitive
    name_lower = name.lower()
    for known in engine.ratings:
        if known.lower() == name_lower:
            return known

    # Partial match (last name)
    name_parts = name_lower.split()
    if name_parts:
        last_name = name_parts[-1]
        matches = [k for k in engine.ratings if last_name in k.lower()]
        if len(matches) == 1:
            return matches[0]

    return None
