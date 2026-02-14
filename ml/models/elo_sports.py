"""Glicko-2 rating engine for individual sports (tennis MVP).

Ratings are stored in familiar Elo scale (mu=1500, phi=350) but ALL
computations are done in Glicko-2 internal scale:
    mu_g2 = (mu - 1500) / 173.7178
    phi_g2 = phi / 173.7178

Key differences from basic Elo:
- Rating Deviation (phi/RD): measures uncertainty. New/inactive players have high RD.
- Volatility (sigma): how consistently a player performs.
- No K-factor: Glicko-2 uses tau, rd_inflation, rd_floor/ceiling instead.
- RD grows with inactivity, shrinks with each rated match.

References:
- Glicko-2 spec: http://www.glicko.net/glicko/glicko2.pdf
- Tennis Elo accuracy: ~72% (Sackmann), ~74% with surface-specific ratings
"""

import math
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)

# Scale conversion constant: Glicko-2 internal scale factor
SCALE = 173.7178


@dataclass
class SportConfig:
    """Configuration for a sport's Glicko-2 parameters."""
    sport: str = "tennis"
    tau: float = 0.5  # Volatility constraint (0.3-1.2 typical, lower = more stable)
    rd_inflation_per_day: float = 0.5  # RD growth when inactive (Glicko-2 scale)
    rd_floor: float = 30.0  # Minimum RD (display scale) — very active player
    rd_ceiling: float = 350.0  # Maximum RD (display scale) — brand new / very inactive
    default_mu: float = 1500.0  # Starting rating (display scale)
    default_phi: float = 350.0  # Starting RD (display scale)
    default_sigma: float = 0.06  # Starting volatility
    convergence_tolerance: float = 1e-6  # For volatility iteration
    surfaces: list[str] = field(default_factory=lambda: ["hard", "clay", "grass"])
    surface_blend: float = 0.5  # Weight of surface-specific vs overall rating
    min_surface_matches: int = 5  # Min matches on surface before using surface rating


@dataclass
class PlayerRating:
    """A player's Glicko-2 rating (stored in display scale)."""
    mu: float = 1500.0  # Rating (display scale)
    phi: float = 350.0  # Rating deviation (display scale)
    sigma: float = 0.06  # Volatility
    last_match_date: Optional[date] = None
    match_count: int = 0

    def to_glicko2(self) -> tuple[float, float]:
        """Convert to Glicko-2 internal scale."""
        return (self.mu - 1500.0) / SCALE, self.phi / SCALE

    @staticmethod
    def from_glicko2(mu_g2: float, phi_g2: float) -> tuple[float, float]:
        """Convert from Glicko-2 internal scale to display scale."""
        return mu_g2 * SCALE + 1500.0, phi_g2 * SCALE


@dataclass
class MatchResult:
    """A single match result for rating updates."""
    winner: str
    loser: str
    match_date: date
    surface: str = "hard"
    tourney_level: str = ""  # 'G' (Grand Slam), 'M' (Masters), etc.


class Glicko2Engine:
    """Glicko-2 rating engine with surface-specific ratings for tennis."""

    def __init__(self, config: SportConfig | None = None):
        self.config = config or SportConfig()
        # {player_name: {"overall": PlayerRating, "hard": PlayerRating, ...}}
        self.ratings: dict[str, dict[str, PlayerRating]] = {}

    def get_or_create_player(self, name: str) -> dict[str, PlayerRating]:
        """Get existing or create new player ratings."""
        if name not in self.ratings:
            self.ratings[name] = {
                "overall": PlayerRating(
                    mu=self.config.default_mu,
                    phi=self.config.default_phi,
                    sigma=self.config.default_sigma,
                ),
            }
            for surface in self.config.surfaces:
                self.ratings[name][surface] = PlayerRating(
                    mu=self.config.default_mu,
                    phi=self.config.default_phi,
                    sigma=self.config.default_sigma,
                )
        return self.ratings[name]

    def _inflate_rd(self, rating: PlayerRating, current_date: date) -> None:
        """Increase RD due to inactivity (Glicko-2 step 6)."""
        if rating.last_match_date is None:
            return

        days_inactive = (current_date - rating.last_match_date).days
        if days_inactive <= 0:
            return

        # Inflate phi in Glicko-2 scale
        _, phi_g2 = rating.to_glicko2()
        inflation = self.config.rd_inflation_per_day * days_inactive / SCALE
        phi_g2_new = math.sqrt(phi_g2 ** 2 + inflation ** 2)

        # Convert back and cap at ceiling
        _, new_phi = PlayerRating.from_glicko2(0, phi_g2_new)
        rating.phi = min(new_phi, self.config.rd_ceiling)

    def _g_function(self, phi_g2: float) -> float:
        """Glicko-2 g(phi) function — reduces impact of uncertain opponents."""
        return 1.0 / math.sqrt(1.0 + 3.0 * phi_g2 ** 2 / (math.pi ** 2))

    def _expected_score(self, mu_g2: float, opp_mu_g2: float, opp_phi_g2: float) -> float:
        """Expected score (win probability) in Glicko-2 scale."""
        g = self._g_function(opp_phi_g2)
        return 1.0 / (1.0 + math.exp(-g * (mu_g2 - opp_mu_g2)))

    def win_probability(self, player_a: str, player_b: str, surface: str = "hard") -> tuple[float, float]:
        """Compute win probability and confidence for player A vs player B.

        Returns:
            (prob_a_wins, confidence) where confidence is 0-1 based on RD.
        """
        ratings_a = self.get_or_create_player(player_a)
        ratings_b = self.get_or_create_player(player_b)

        # Blend surface-specific and overall ratings
        ra = self._blend_rating(ratings_a, surface)
        rb = self._blend_rating(ratings_b, surface)

        # Convert to Glicko-2 scale for probability calculation
        mu_a_g2, phi_a_g2 = ra.to_glicko2()
        mu_b_g2, phi_b_g2 = rb.to_glicko2()

        # Win probability
        prob_a = self._expected_score(mu_a_g2, mu_b_g2, phi_b_g2)

        # Confidence based on combined RD (exponential decay)
        combined_rd = math.sqrt(ra.phi ** 2 + rb.phi ** 2)
        # Calibrated so that typical RD (60-100) maps to ~0.6-0.8
        confidence = math.exp(-combined_rd / 200.0)

        return prob_a, confidence

    def _blend_rating(self, ratings: dict[str, PlayerRating], surface: str) -> PlayerRating:
        """Blend surface-specific and overall ratings."""
        overall = ratings["overall"]
        surface_rating = ratings.get(surface)

        if surface_rating is None or surface_rating.match_count < self.config.min_surface_matches:
            return overall

        w = self.config.surface_blend
        blended_mu = w * surface_rating.mu + (1 - w) * overall.mu
        blended_phi = math.sqrt(
            (w * surface_rating.phi) ** 2 + ((1 - w) * overall.phi) ** 2
        )

        return PlayerRating(mu=blended_mu, phi=blended_phi, sigma=overall.sigma)

    def _compute_new_volatility(
        self, sigma: float, phi_g2: float, delta: float, v: float
    ) -> float:
        """Compute new volatility using Illinois algorithm (Glicko-2 step 5)."""
        tau = self.config.tau
        a = math.log(sigma ** 2)

        def f(x: float) -> float:
            ex = math.exp(x)
            d2 = delta ** 2
            p2 = phi_g2 ** 2
            num1 = ex * (d2 - p2 - v - ex)
            denom1 = 2.0 * (p2 + v + ex) ** 2
            return num1 / denom1 - (x - a) / (tau ** 2)

        # Set initial bounds
        A = a
        if delta ** 2 > phi_g2 ** 2 + v:
            B = math.log(delta ** 2 - phi_g2 ** 2 - v)
        else:
            k = 1
            while f(a - k * tau) < 0:
                k += 1
            B = a - k * tau

        # Illinois algorithm
        fA = f(A)
        fB = f(B)

        for _ in range(100):  # Max iterations
            if abs(B - A) < self.config.convergence_tolerance:
                break
            C = A + (A - B) * fA / (fB - fA)
            fC = f(C)

            if fC * fB <= 0:
                A = B
                fA = fB
            else:
                fA /= 2.0

            B = C
            fB = fC

        return math.exp(A / 2.0)

    def update_ratings(self, result: MatchResult) -> None:
        """Update ratings for a single match result.

        Updates BOTH overall and surface-specific ratings.
        """
        winner_ratings = self.get_or_create_player(result.winner)
        loser_ratings = self.get_or_create_player(result.loser)

        # Update both overall and surface-specific ratings
        rating_keys = ["overall"]
        if result.surface in self.config.surfaces:
            rating_keys.append(result.surface)

        for key in rating_keys:
            w_rating = winner_ratings[key]
            l_rating = loser_ratings[key]

            # Inflate RD for inactivity
            self._inflate_rd(w_rating, result.match_date)
            self._inflate_rd(l_rating, result.match_date)

            # Perform Glicko-2 update
            self._update_single_pair(w_rating, l_rating, score_a=1.0, score_b=0.0)

            # Update metadata
            w_rating.last_match_date = result.match_date
            l_rating.last_match_date = result.match_date
            w_rating.match_count += 1
            l_rating.match_count += 1

    def _update_single_pair(
        self, rating_a: PlayerRating, rating_b: PlayerRating,
        score_a: float, score_b: float,
    ) -> None:
        """Update a pair of ratings for a single game outcome."""
        # Convert to Glicko-2 scale
        mu_a, phi_a = rating_a.to_glicko2()
        mu_b, phi_b = rating_b.to_glicko2()

        # Compute quantities for player A
        g_b = self._g_function(phi_b)
        E_a = self._expected_score(mu_a, mu_b, phi_b)
        v_a = 1.0 / (g_b ** 2 * E_a * (1.0 - E_a))
        delta_a = v_a * g_b * (score_a - E_a)

        # Compute quantities for player B
        g_a = self._g_function(phi_a)
        E_b = self._expected_score(mu_b, mu_a, phi_a)
        v_b = 1.0 / (g_a ** 2 * E_b * (1.0 - E_b))
        delta_b = v_b * g_a * (score_b - E_b)

        # Update volatility
        new_sigma_a = self._compute_new_volatility(rating_a.sigma, phi_a, delta_a, v_a)
        new_sigma_b = self._compute_new_volatility(rating_b.sigma, phi_b, delta_b, v_b)

        # Update phi (RD)
        phi_star_a = math.sqrt(phi_a ** 2 + new_sigma_a ** 2)
        phi_star_b = math.sqrt(phi_b ** 2 + new_sigma_b ** 2)

        new_phi_a = 1.0 / math.sqrt(1.0 / phi_star_a ** 2 + 1.0 / v_a)
        new_phi_b = 1.0 / math.sqrt(1.0 / phi_star_b ** 2 + 1.0 / v_b)

        # Update mu
        new_mu_a = mu_a + new_phi_a ** 2 * g_b * (score_a - E_a)
        new_mu_b = mu_b + new_phi_b ** 2 * g_a * (score_b - E_b)

        # Convert back to display scale
        rating_a.mu, rating_a.phi = PlayerRating.from_glicko2(new_mu_a, new_phi_a)
        rating_b.mu, rating_b.phi = PlayerRating.from_glicko2(new_mu_b, new_phi_b)
        rating_a.sigma = new_sigma_a
        rating_b.sigma = new_sigma_b

        # Enforce RD floor/ceiling
        rating_a.phi = max(self.config.rd_floor, min(self.config.rd_ceiling, rating_a.phi))
        rating_b.phi = max(self.config.rd_floor, min(self.config.rd_ceiling, rating_b.phi))

    def process_matches(self, matches: list[MatchResult]) -> dict:
        """Process a chronological list of matches. Returns stats."""
        correct = 0
        total = 0
        brier_sum = 0.0

        for match in matches:
            # Predict before updating
            prob_winner, conf = self.win_probability(
                match.winner, match.loser, match.surface,
            )

            if total > 0:  # Skip first match (no ratings yet)
                if prob_winner > 0.5:
                    correct += 1
                brier_sum += (1.0 - prob_winner) ** 2

            total += 1

            # Update ratings with this match result
            self.update_ratings(match)

        accuracy = correct / max(1, total - 1)
        brier = brier_sum / max(1, total - 1)

        return {
            "total_matches": total,
            "accuracy": accuracy,
            "brier_score": brier,
            "unique_players": len(self.ratings),
        }

    def get_top_players(self, n: int = 20, surface: str = "overall") -> list[dict]:
        """Get top N players by rating."""
        players = []
        for name, ratings in self.ratings.items():
            r = ratings.get(surface, ratings["overall"])
            if r.match_count >= 10:  # Minimum matches to be ranked
                players.append({
                    "name": name,
                    "mu": round(r.mu, 1),
                    "phi": round(r.phi, 1),
                    "sigma": round(r.sigma, 4),
                    "matches": r.match_count,
                    "last_match": str(r.last_match_date) if r.last_match_date else None,
                })
        players.sort(key=lambda x: x["mu"], reverse=True)
        return players[:n]

    def get_player_rating(self, name: str, surface: str = "overall") -> Optional[dict]:
        """Get a single player's rating details."""
        if name not in self.ratings:
            return None
        ratings = self.ratings[name]
        r = ratings.get(surface, ratings["overall"])
        return {
            "name": name,
            "surface": surface,
            "mu": round(r.mu, 1),
            "phi": round(r.phi, 1),
            "sigma": round(r.sigma, 4),
            "matches": r.match_count,
            "last_match": str(r.last_match_date) if r.last_match_date else None,
            "surfaces": {
                s: {
                    "mu": round(ratings[s].mu, 1),
                    "phi": round(ratings[s].phi, 1),
                    "matches": ratings[s].match_count,
                }
                for s in self.config.surfaces
                if s in ratings
            },
        }

    def save(self, path: str = "ml/saved_models/elo_sports_ratings.joblib") -> None:
        """Save ratings to disk."""
        import joblib
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "ratings": self.ratings,
            "config": self.config,
        }
        joblib.dump(data, path)
        logger.info(f"Saved {len(self.ratings)} player ratings to {path}")

    @classmethod
    def load(cls, path: str = "ml/saved_models/elo_sports_ratings.joblib") -> "Glicko2Engine":
        """Load ratings from disk."""
        import joblib
        from pathlib import Path

        if not Path(path).exists():
            logger.warning(f"No saved ratings at {path}, starting fresh")
            return cls()

        data = joblib.load(path)
        engine = cls(config=data.get("config", SportConfig()))
        engine.ratings = data.get("ratings", {})
        logger.info(f"Loaded {len(engine.ratings)} player ratings from {path}")
        return engine
