"""Historical NBA game results collector for Glicko-2 team ratings.

Data source: NBA Stats API via nba_api Python package (free, no API key).
Falls back to Basketball Reference CSV export format if nba_api is unavailable.

Returns MatchResult objects compatible with the existing Glicko2Engine so that
NBA team ratings can be built and backtested in exactly the same way as Tennis
and UFC ratings.

Each NBA game is treated as a 1v1 matchup between two teams.
"""

import logging
import time
from datetime import date, datetime
from typing import Optional

from ml.models.elo_sports import MatchResult

logger = logging.getLogger(__name__)

# ── NBA team abbreviation → full name ──────────────────────────────────────
NBA_TEAM_ABBR: dict[str, str] = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "BRK": "Brooklyn Nets",
    "NJN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHH": "Charlotte Hornets",
    "CHO": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "LA Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NOH": "New Orleans Pelicans",
    "NOK": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "SEA": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "PHO": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
    "WSB": "Washington Wizards",
}

# Reverse map: full name → abbreviation
_NAME_TO_ABBR: dict[str, str] = {v.lower(): k for k, v in NBA_TEAM_ABBR.items()}
# Deduplicate to canonical abbr (keep most recent)
_CANONICAL_ABBR: dict[str, str] = {
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "New Orleans Pelicans": "NOP",
    "Oklahoma City Thunder": "OKC",
    "Phoenix Suns": "PHX",
}
for full, abbr in _CANONICAL_ABBR.items():
    _NAME_TO_ABBR[full.lower()] = abbr


def team_name_to_abbr(name: str) -> Optional[str]:
    """Convert a full team name to its 3-letter abbreviation."""
    return _NAME_TO_ABBR.get(name.lower().strip())


def abbr_to_team_name(abbr: str) -> Optional[str]:
    """Convert a 3-letter abbreviation to canonical team name."""
    return NBA_TEAM_ABBR.get(abbr.upper())


def load_nba_from_nba_api(
    start_season: int = 2000,
    end_season: int = 2025,
    delay: float = 0.6,
) -> list[MatchResult]:
    """Fetch all NBA regular season + playoff game results via nba_api.

    Args:
        start_season: First season year (e.g. 2000 = 2000-01 season)
        end_season: Last season year (e.g. 2025 = 2024-25 season)
        delay: Seconds between API calls to avoid rate limiting

    Returns:
        Chronologically sorted list of MatchResult objects
    """
    try:
        from nba_api.stats.endpoints import leaguegamelog
        from nba_api.stats.library.parameters import SeasonType
    except ImportError:
        raise ImportError(
            "nba_api is not installed. Run: pip install nba_api"
        )

    all_matches: list[MatchResult] = []
    seen_game_ids: set[str] = set()

    for season_year in range(start_season, end_season + 1):
        season_str = f"{season_year}-{str(season_year + 1)[-2:]}"

        for season_type in ["Regular Season", "Playoffs"]:
            try:
                log = leaguegamelog.LeagueGameLog(
                    season=season_str,
                    season_type_all_star=season_type,
                    league_id="00",
                    timeout=30,
                )
                df = log.get_data_frames()[0]
                time.sleep(delay)

                if df.empty:
                    continue

                matches = _parse_nba_api_dataframe(df, season_type, seen_game_ids)
                all_matches.extend(matches)
                logger.info(
                    f"NBA {season_str} {season_type}: {len(matches)} games"
                )

            except Exception as e:
                logger.warning(f"Failed {season_str} {season_type}: {e}")
                time.sleep(delay * 2)
                continue

    all_matches.sort(key=lambda m: m.match_date)
    logger.info(
        f"Total NBA games loaded: {len(all_matches)} "
        f"({start_season}-{end_season}), "
        f"{len(set(m.winner for m in all_matches) | set(m.loser for m in all_matches))} teams"
    )
    return all_matches


def _parse_nba_api_dataframe(df, season_type: str, seen_game_ids: set) -> list[MatchResult]:
    """Parse nba_api LeagueGameLog DataFrame into MatchResult objects.

    Each row is one team's view of a game. We pair home vs away by GAME_ID,
    then create one MatchResult (winner=higher-scoring team).
    """
    matches = []

    # Group by GAME_ID to pair the two teams
    grouped = {}
    for _, row in df.iterrows():
        game_id = str(row.get("GAME_ID", ""))
        if not game_id:
            continue
        if game_id not in grouped:
            grouped[game_id] = []
        grouped[game_id].append(row)

    for game_id, rows in grouped.items():
        if game_id in seen_game_ids:
            continue
        if len(rows) != 2:
            continue

        row_a, row_b = rows[0], rows[1]

        try:
            # Parse date
            date_str = str(row_a.get("GAME_DATE", ""))
            if not date_str:
                continue
            game_date = datetime.strptime(date_str[:10], "%Y-%m-%d").date()

            # Get team names
            team_a = str(row_a.get("TEAM_NAME", "")).strip()
            team_b = str(row_b.get("TEAM_NAME", "")).strip()
            if not team_a or not team_b:
                continue

            # Determine winner by points
            pts_a = float(row_a.get("PTS", 0) or 0)
            pts_b = float(row_b.get("PTS", 0) or 0)
            if pts_a == pts_b:
                continue  # Overtime tie shouldn't happen in NBA, skip

            winner = team_a if pts_a > pts_b else team_b
            loser = team_b if pts_a > pts_b else team_a

            tourney_level = "playoff" if season_type == "Playoffs" else "regular"

            matches.append(MatchResult(
                winner=winner,
                loser=loser,
                match_date=game_date,
                surface="court",
                tourney_level=tourney_level,
            ))
            seen_game_ids.add(game_id)

        except Exception:
            continue

    return matches


def load_nba_from_bref_csv(csv_path: str) -> list[MatchResult]:
    """Load NBA games from a Basketball Reference-style CSV.

    Expected columns (any order, case-insensitive):
        date, home_team (or visitor_team), visitor_team, home_pts, visitor_pts
    OR Basketball Reference game log format:
        Date, Home/Neutral, Visitor/Neutral, PTS (home), PTS (visitor)

    This serves as a fallback if nba_api is unavailable.
    """
    import csv as csv_module

    matches = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv_module.DictReader(f)
        headers = [h.lower().strip() for h in (reader.fieldnames or [])]

        for row in reader:
            norm = {k.lower().strip(): v for k, v in row.items()}
            try:
                # Parse date
                date_str = norm.get("date", "")
                if not date_str or date_str.startswith("date"):
                    continue
                game_date = _parse_nba_date(date_str)
                if game_date is None:
                    continue

                # Team names (basketball-reference uses Visitor/Home)
                home = (norm.get("home/neutral") or norm.get("home_team") or "").strip()
                visitor = (norm.get("visitor/neutral") or norm.get("visitor_team") or "").strip()
                if not home or not visitor:
                    continue

                # Points
                home_pts = float(norm.get("pts", norm.get("home_pts", 0)) or 0)
                vis_pts = float(norm.get("pts.1", norm.get("visitor_pts", 0)) or 0)
                if home_pts == vis_pts:
                    continue

                winner = home if home_pts > vis_pts else visitor
                loser = visitor if home_pts > vis_pts else home

                matches.append(MatchResult(
                    winner=winner,
                    loser=loser,
                    match_date=game_date,
                    surface="court",
                    tourney_level="regular",
                ))
            except Exception:
                continue

    matches.sort(key=lambda m: m.match_date)
    logger.info(f"Loaded {len(matches)} NBA games from CSV: {csv_path}")
    return matches


def _parse_nba_date(date_str: str) -> Optional[date]:
    """Parse various NBA date formats."""
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    return None


async def fetch_all_nba_matches(
    start_season: int = 2000,
    end_season: int = 2025,
    csv_path: Optional[str] = None,
    delay: float = 0.6,
) -> list[MatchResult]:
    """Main entry point: load NBA game results.

    Priority:
    1. csv_path (if provided)
    2. nba_api (requires: pip install nba_api)

    Args:
        start_season: First season year
        end_season: Last season year
        csv_path: Optional path to Basketball Reference CSV
        delay: Seconds between nba_api calls

    Returns:
        Chronologically sorted list of MatchResult objects
    """
    if csv_path:
        return load_nba_from_bref_csv(csv_path)

    return load_nba_from_nba_api(
        start_season=start_season,
        end_season=end_season,
        delay=delay,
    )
