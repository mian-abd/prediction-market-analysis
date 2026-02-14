"""Historical sports results collector — Tennis (ATP/WTA) via Jeff Sackmann's GitHub CSVs.

Data source: https://github.com/JeffSackmann/tennis_atp (free, updated regularly)
Contains match-level data from 1968-present including surface, tournament level, etc.

This collector fetches CSV data directly from GitHub (no API key needed).
"""

import csv
import io
import logging
from datetime import date, datetime
from typing import Optional

import httpx

from ml.models.elo_sports import MatchResult

logger = logging.getLogger(__name__)

# Jeff Sackmann's tennis_atp GitHub raw CSV base URL
ATP_BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
WTA_BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master"

# Surface normalization
SURFACE_MAP = {
    "Hard": "hard",
    "Clay": "clay",
    "Grass": "grass",
    "Carpet": "hard",  # Carpet is similar to hard, rare now
}

# Tournament level mapping
TOURNEY_LEVEL_MAP = {
    "G": "grand_slam",    # Grand Slam
    "M": "masters",       # Masters 1000
    "A": "atp_500_250",   # ATP 500/250
    "D": "davis_cup",     # Davis Cup
    "F": "tour_finals",   # ATP Finals
    "C": "challenger",    # Challenger
}


async def fetch_tennis_matches(
    year: int,
    tour: str = "atp",
    client: httpx.AsyncClient | None = None,
) -> list[MatchResult]:
    """Fetch all tennis matches for a given year from Sackmann's CSV.

    Args:
        year: Year to fetch (e.g. 2024)
        tour: "atp" or "wta"
        client: Optional shared httpx client

    Returns:
        List of MatchResult objects sorted by date
    """
    base_url = ATP_BASE_URL if tour == "atp" else WTA_BASE_URL
    url = f"{base_url}/{tour}_matches_{year}.csv"

    should_close = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=30)

    try:
        response = await client.get(url)
        if response.status_code == 404:
            logger.warning(f"No data for {tour} {year} (404)")
            return []
        response.raise_for_status()

        matches = _parse_csv(response.text, tour)
        logger.info(f"Fetched {len(matches)} {tour.upper()} matches for {year}")
        return matches
    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch {tour} {year}: {e}")
        return []
    finally:
        if should_close:
            await client.aclose()


async def fetch_all_tennis_matches(
    start_year: int = 2015,
    end_year: int = 2026,
    tour: str = "atp",
) -> list[MatchResult]:
    """Fetch multiple years of tennis matches.

    Args:
        start_year: First year to fetch
        end_year: Last year to fetch (inclusive)
        tour: "atp" or "wta"

    Returns:
        Chronologically sorted list of MatchResult objects
    """
    all_matches = []
    async with httpx.AsyncClient(timeout=30) as client:
        for year in range(start_year, end_year + 1):
            matches = await fetch_tennis_matches(year, tour, client)
            all_matches.extend(matches)

    # Sort chronologically (critical for Elo — no future data leakage)
    all_matches.sort(key=lambda m: m.match_date)

    logger.info(
        f"Total: {len(all_matches)} {tour.upper()} matches "
        f"({start_year}-{end_year}), "
        f"{len(set(m.winner for m in all_matches) | set(m.loser for m in all_matches))} unique players"
    )
    return all_matches


def _parse_csv(csv_text: str, tour: str) -> list[MatchResult]:
    """Parse Sackmann CSV format into MatchResult objects."""
    matches = []
    reader = csv.DictReader(io.StringIO(csv_text))

    for row in reader:
        try:
            # Skip walkovers, retirements, defaults (incomplete matches)
            score = row.get("score", "")
            if not score or any(x in score.upper() for x in ["W/O", "RET", "DEF", "UNP", "ABN"]):
                continue

            # Parse date
            tourney_date_str = row.get("tourney_date", "")
            if not tourney_date_str or len(tourney_date_str) < 8:
                continue
            match_date = _parse_date(tourney_date_str)
            if match_date is None:
                continue

            # Get player names
            winner = row.get("winner_name", "").strip()
            loser = row.get("loser_name", "").strip()
            if not winner or not loser:
                continue

            # Parse surface
            raw_surface = row.get("surface", "Hard")
            surface = SURFACE_MAP.get(raw_surface, "hard")

            # Tournament level
            tourney_level = row.get("tourney_level", "A")

            matches.append(MatchResult(
                winner=winner,
                loser=loser,
                match_date=match_date,
                surface=surface,
                tourney_level=tourney_level,
            ))

        except Exception as e:
            # Skip malformed rows silently
            continue

    return matches


def _parse_date(date_str: str) -> Optional[date]:
    """Parse YYYYMMDD date string."""
    try:
        return datetime.strptime(date_str[:8], "%Y%m%d").date()
    except (ValueError, IndexError):
        return None
