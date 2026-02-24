"""Enhanced sports features for tennis and UFC prediction models.

Research basis:
- Serve/return separation improves tennis prediction accuracy 20-30%
  over combined ratings (Sharp-9 / Betfair research).
- Head-to-head records matter, especially on specific surfaces.
- Recent form (last N matches) captures fitness and confidence momentum.
- Tournament level weighting: Grand Slam performances are more predictive.
- Fatigue indicators: match load and rest days affect performance.
- TU Dortmund (2025): Enhanced covariates including Elo, age, and ability
  parameters improve classification rate.

This module computes features FROM the existing Glicko2Engine ratings
and historical match data to feed into gradient-boosted models that
layer on top of raw Elo predictions.
"""

import math
import logging
from collections import defaultdict
from datetime import date, timedelta
from typing import Optional

from ml.models.elo_sports import Glicko2Engine, PlayerRating, MatchResult

logger = logging.getLogger(__name__)

# Tournament level weights for form calculation (Grand Slams matter more)
TOURNEY_WEIGHTS = {
    "G": 3.0,   # Grand Slam
    "F": 2.5,   # Tour Finals
    "M": 2.0,   # Masters 1000
    "A": 1.0,   # ATP 500/250
    "D": 0.8,   # Davis Cup
    "C": 0.5,   # Challenger
}


class EnhancedSportsFeatures:
    """Compute rich features from historical match data and Elo ratings."""

    def __init__(self):
        self.match_history: list[MatchResult] = []
        self._h2h_cache: dict[tuple[str, str], dict] = {}
        self._player_matches: dict[str, list[MatchResult]] = defaultdict(list)

    def load_matches(self, matches: list[MatchResult]) -> None:
        """Load historical match data for feature computation."""
        self.match_history = sorted(matches, key=lambda m: m.match_date)
        self._player_matches.clear()
        self._h2h_cache.clear()

        for m in self.match_history:
            self._player_matches[m.winner].append(m)
            self._player_matches[m.loser].append(m)

        logger.info(
            f"Loaded {len(matches)} matches for "
            f"{len(self._player_matches)} players"
        )

    def compute_features(
        self,
        player_a: str,
        player_b: str,
        surface: str,
        match_date: date,
        engine: Glicko2Engine,
    ) -> dict[str, float]:
        """Compute full feature vector for a matchup.

        Returns dict of features suitable for gradient-boosted model input.
        """
        features = {}

        # 1. Core Elo features
        elo_features = self._elo_features(player_a, player_b, surface, engine)
        features.update(elo_features)

        # 2. Head-to-head features
        h2h = self._h2h_features(player_a, player_b, surface, match_date)
        features.update(h2h)

        # 3. Recent form features
        form_a = self._recent_form(player_a, surface, match_date)
        form_b = self._recent_form(player_b, surface, match_date)
        features.update({f"form_a_{k}": v for k, v in form_a.items()})
        features.update({f"form_b_{k}": v for k, v in form_b.items()})

        # 4. Fatigue features
        fatigue_a = self._fatigue_features(player_a, match_date)
        fatigue_b = self._fatigue_features(player_b, match_date)
        features.update({f"fatigue_a_{k}": v for k, v in fatigue_a.items()})
        features.update({f"fatigue_b_{k}": v for k, v in fatigue_b.items()})

        # 5. Surface affinity
        surf_a = self._surface_affinity(player_a, surface, match_date)
        surf_b = self._surface_affinity(player_b, surface, match_date)
        features["surface_affinity_a"] = surf_a
        features["surface_affinity_b"] = surf_b
        features["surface_affinity_diff"] = surf_a - surf_b

        return features

    def _elo_features(
        self,
        player_a: str,
        player_b: str,
        surface: str,
        engine: Glicko2Engine,
    ) -> dict[str, float]:
        """Extract features from Glicko-2 ratings."""
        ratings_a = engine.ratings.get(player_a, {})
        ratings_b = engine.ratings.get(player_b, {})

        overall_a = ratings_a.get("overall", PlayerRating())
        overall_b = ratings_b.get("overall", PlayerRating())
        surface_a = ratings_a.get(surface, PlayerRating())
        surface_b = ratings_b.get(surface, PlayerRating())

        # Win probability from Elo
        prob_a, confidence = engine.win_probability(player_a, player_b, surface)

        return {
            "elo_prob_a": prob_a,
            "elo_confidence": confidence,
            "elo_mu_diff": overall_a.mu - overall_b.mu,
            "elo_mu_a": overall_a.mu,
            "elo_mu_b": overall_b.mu,
            "elo_rd_a": overall_a.phi,
            "elo_rd_b": overall_b.phi,
            "elo_surface_mu_diff": surface_a.mu - surface_b.mu,
            "elo_sigma_a": overall_a.sigma,
            "elo_sigma_b": overall_b.sigma,
            "match_count_a": float(overall_a.match_count),
            "match_count_b": float(overall_b.match_count),
        }

    def _h2h_features(
        self,
        player_a: str,
        player_b: str,
        surface: str,
        as_of: date,
    ) -> dict[str, float]:
        """Head-to-head record features."""
        cache_key = (player_a, player_b)
        if cache_key not in self._h2h_cache:
            self._h2h_cache[cache_key] = self._compute_h2h(player_a, player_b)

        h2h = self._h2h_cache[cache_key]

        # Filter to before match date
        a_wins = sum(1 for m in h2h.get("matches", [])
                     if m.winner == player_a and m.match_date < as_of)
        b_wins = sum(1 for m in h2h.get("matches", [])
                     if m.winner == player_b and m.match_date < as_of)
        total = a_wins + b_wins

        # Surface-specific H2H
        a_surf_wins = sum(1 for m in h2h.get("matches", [])
                         if m.winner == player_a and m.surface == surface and m.match_date < as_of)
        b_surf_wins = sum(1 for m in h2h.get("matches", [])
                         if m.winner == player_b and m.surface == surface and m.match_date < as_of)
        surf_total = a_surf_wins + b_surf_wins

        return {
            "h2h_total": float(total),
            "h2h_a_wins": float(a_wins),
            "h2h_b_wins": float(b_wins),
            "h2h_win_rate_a": a_wins / max(total, 1),
            "h2h_surface_total": float(surf_total),
            "h2h_surface_win_rate_a": a_surf_wins / max(surf_total, 1),
        }

    def _compute_h2h(self, player_a: str, player_b: str) -> dict:
        """Find all historical matches between two players."""
        matches = []
        for m in self._player_matches.get(player_a, []):
            if m.winner == player_b or m.loser == player_b:
                matches.append(m)
        return {"matches": matches}

    def _recent_form(
        self,
        player: str,
        surface: str,
        as_of: date,
        n_matches: int = 10,
    ) -> dict[str, float]:
        """Recent form features (last N matches before as_of)."""
        recent = [
            m for m in self._player_matches.get(player, [])
            if m.match_date < as_of
        ][-n_matches:]

        if not recent:
            return {
                "win_rate": 0.5,
                "surface_win_rate": 0.5,
                "weighted_win_rate": 0.5,
                "streak": 0.0,
                "matches_played": 0.0,
            }

        wins = sum(1 for m in recent if m.winner == player)
        total = len(recent)
        win_rate = wins / max(total, 1)

        # Surface-specific recent form
        surface_matches = [m for m in recent if m.surface == surface]
        surface_wins = sum(1 for m in surface_matches if m.winner == player)
        surface_wr = surface_wins / max(len(surface_matches), 1) if surface_matches else 0.5

        # Tournament-weighted win rate (Grand Slam wins count more)
        weighted_wins = 0.0
        weighted_total = 0.0
        for m in recent:
            weight = TOURNEY_WEIGHTS.get(m.tourney_level, 1.0)
            weighted_total += weight
            if m.winner == player:
                weighted_wins += weight
        weighted_wr = weighted_wins / max(weighted_total, 1)

        # Current streak (positive = winning streak, negative = losing)
        streak = 0
        for m in reversed(recent):
            if m.winner == player:
                if streak >= 0:
                    streak += 1
                else:
                    break
            else:
                if streak <= 0:
                    streak -= 1
                else:
                    break

        return {
            "win_rate": win_rate,
            "surface_win_rate": surface_wr,
            "weighted_win_rate": weighted_wr,
            "streak": float(streak),
            "matches_played": float(total),
        }

    def _fatigue_features(
        self,
        player: str,
        as_of: date,
    ) -> dict[str, float]:
        """Fatigue indicators: match load and rest days."""
        recent = [
            m for m in self._player_matches.get(player, [])
            if m.match_date < as_of and m.match_date >= as_of - timedelta(days=30)
        ]

        if not recent:
            return {
                "matches_last_30d": 0.0,
                "matches_last_7d": 0.0,
                "days_since_last_match": 30.0,
            }

        matches_30d = len(recent)
        matches_7d = sum(1 for m in recent if m.match_date >= as_of - timedelta(days=7))

        last_match = max(m.match_date for m in recent)
        days_since = (as_of - last_match).days

        return {
            "matches_last_30d": float(matches_30d),
            "matches_last_7d": float(matches_7d),
            "days_since_last_match": float(days_since),
        }

    def _surface_affinity(
        self,
        player: str,
        surface: str,
        as_of: date,
        lookback_years: int = 3,
    ) -> float:
        """Compute player's win rate on a specific surface (last N years).

        Returns win rate on surface minus overall win rate (positive = surface specialist).
        """
        cutoff = as_of - timedelta(days=lookback_years * 365)
        recent = [
            m for m in self._player_matches.get(player, [])
            if cutoff <= m.match_date < as_of
        ]

        if not recent:
            return 0.0

        total_wins = sum(1 for m in recent if m.winner == player)
        total_matches = len(recent)
        overall_wr = total_wins / max(total_matches, 1)

        surface_matches = [m for m in recent if m.surface == surface]
        if not surface_matches:
            return 0.0

        surface_wins = sum(1 for m in surface_matches if m.winner == player)
        surface_wr = surface_wins / max(len(surface_matches), 1)

        return surface_wr - overall_wr
