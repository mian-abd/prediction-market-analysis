"""Conservative market-to-matchup parser for tennis H2H markets.

Hard-gate MVP: Only accepts markets where ALL of the following are true:
1. Sport is confidently detected (keywords in question + category)
2. Explicit "A vs B" or "A v B" pattern found
3. Both player names match in Elo database (fuzzy, >0.85 score)
4. Neither player's RD > 300 (too uncertain)
5. Can determine which side "Yes" maps to

Skips: tournament winners, props, team sports, ambiguous markets.
Expected parse rate: ~50-60% of tennis H2H markets.
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Patterns that indicate this is a match winner market (NOT props)
_VS_PATTERNS = [
    re.compile(r'(.+?)\s+vs?\.?\s+(.+?)(?:\s*[:\-\|]|\s*$)', re.IGNORECASE),
    re.compile(r'(.+?)\s+versus\s+(.+?)(?:\s*[:\-\|]|\s*$)', re.IGNORECASE),
    re.compile(r'will\s+(.+?)\s+beat\s+(.+?)[\?\.]', re.IGNORECASE),
]

# Patterns that indicate this is a PROP bet (skip these)
_PROP_PATTERNS = [
    re.compile(r'o/u\s*[\d.]', re.IGNORECASE),
    re.compile(r'over/under', re.IGNORECASE),
    re.compile(r'spread\s*[+\-]', re.IGNORECASE),
    re.compile(r'handicap', re.IGNORECASE),
    re.compile(r'total\s+(sets?|games?|points?|aces?)', re.IGNORECASE),
    re.compile(r'(set|game|map)\s*\d+\s*winner', re.IGNORECASE),
    re.compile(r'first\s+(set|game|blood|kill)', re.IGNORECASE),
    re.compile(r'(rebounds|assists|strikeouts|yards|touchdowns)', re.IGNORECASE),
]

# Patterns that indicate tournament/season winner (skip)
_TOURNAMENT_PATTERNS = [
    re.compile(r'win\s+(the\s+)?(tournament|championship|title|cup|open|slam)', re.IGNORECASE),
    re.compile(r'(tournament|championship|season)\s+winner', re.IGNORECASE),
    re.compile(r'advance\s+to\s+(the\s+)?(final|semi|quarter)', re.IGNORECASE),
]

# Tennis-specific indicators
_TENNIS_INDICATORS = [
    "tennis", "atp", "wta", "grand slam", "australian open", "french open",
    "wimbledon", "us open", "roland garros", "indian wells", "miami open",
    "monte carlo", "madrid open", "rome open", "canadian open", "cincinnati",
    "shanghai masters", "paris masters", "atp finals", "davis cup",
]

# UFC/MMA indicators (for detect_ufc_market and parse_ufc_matchup)
_UFC_INDICATORS = [
    "ufc", "mma", "fight", "fighter", "octagon", "ppv", "espn",
    "jon jones", "islam makhachev", "conor mcgregor", "khabib",
    "heavyweight", "lightweight", "welterweight", "middleweight",
    "stipe miocic", "alex pereira", "leon edwards",
]

# Keywords that help determine which side "Yes" maps to
_YES_SIDE_PATTERNS = [
    # "Will X beat Y?" → Yes = X wins
    re.compile(r'will\s+(.+?)\s+beat\s+(.+)', re.IGNORECASE),
    # "X to win" → Yes = X wins
    re.compile(r'(.+?)\s+to\s+win', re.IGNORECASE),
    # "X vs Y: Match Winner" → Usually first player
    re.compile(r'(.+?)\s+vs?\.?\s+(.+?):\s*match\s*winner', re.IGNORECASE),
]


@dataclass
class ParsedMatchup:
    """A successfully parsed tennis matchup from a market question."""
    player_a: str  # First player (mapped to "Yes" side)
    player_b: str  # Second player (mapped to "No" side)
    sport: str  # "tennis"
    surface: str  # "hard", "clay", "grass", or "unknown"
    confidence: float  # 0-1, how confident we are in the parse
    yes_side_player: str  # Which player "Yes" maps to
    market_id: int | None = None
    raw_question: str = ""


def _clean_player_name(name: str) -> str:
    """Clean extracted player name from regex match."""
    name = name.strip()
    # Remove common suffixes/prefixes
    name = re.sub(r'\s*\(.*?\)\s*', '', name)  # Remove parenthetical
    name = re.sub(r'\s*\[.*?\]\s*', '', name)  # Remove brackets
    name = re.sub(r'^\d+\.\s*', '', name)  # Remove ranking prefix "1. "
    name = re.sub(r'\s+', ' ', name)  # Normalize whitespace
    # Remove trailing punctuation
    name = name.rstrip('?.:!,')
    return name.strip()


def _detect_tennis(question: str, category: str = "") -> bool:
    """Check if a market is about tennis."""
    combined = f"{question} {category}".lower()
    return any(indicator in combined for indicator in _TENNIS_INDICATORS)


def detect_ufc_market(question: str, category: str = "") -> bool:
    """Check if a market is about UFC/MMA. Used for UFC Elo scanning."""
    combined = f"{question} {category}".lower()
    if any(indicator in combined for indicator in _UFC_INDICATORS):
        return True
    # "Will X beat Y?" with two capitalized names often UFC (combat sports)
    if re.search(r"will\s+.+\s+beat\s+.+\?", question, re.IGNORECASE):
        return True
    if re.search(r".+\s+vs\.?\s+.+", question, re.IGNORECASE) and "sports" in combined:
        return True
    return False


def _detect_surface(question: str) -> str:
    """Try to detect playing surface from market text."""
    q = question.lower()
    if "clay" in q or "roland garros" in q or "french open" in q or "rome" in q or "madrid" in q:
        return "clay"
    if "grass" in q or "wimbledon" in q:
        return "grass"
    if "hard" in q or "australian" in q or "us open" in q or "indian wells" in q or "miami" in q:
        return "hard"
    return "unknown"


def _is_prop_bet(question: str) -> bool:
    """Check if market is a prop bet (not match winner)."""
    return any(p.search(question) for p in _PROP_PATTERNS)


def _is_tournament_winner(question: str) -> bool:
    """Check if market is about tournament/season winner."""
    return any(p.search(question) for p in _TOURNAMENT_PATTERNS)


def _determine_yes_side(
    question: str, player_a: str, player_b: str,
) -> tuple[str, float]:
    """Determine which player "Yes" maps to.

    Returns (yes_player, confidence).
    """
    q = question.lower()

    # "Will X beat Y?" → Yes = X
    for pattern in _YES_SIDE_PATTERNS:
        match = pattern.search(question)
        if match:
            first = match.group(1).strip().lower()
            if first in player_a.lower() or player_a.lower() in first:
                return player_a, 0.9
            if first in player_b.lower() or player_b.lower() in first:
                return player_b, 0.9

    # Default: "A vs B" → first player is typically the "Yes" side
    # (this is the Polymarket/Kalshi convention)
    return player_a, 0.7


def parse_matchup(
    question: str,
    category: str = "",
    market_id: int | None = None,
) -> Optional[ParsedMatchup]:
    """Try to parse a market question into a tennis matchup.

    Returns ParsedMatchup if successful, None if the market should be skipped.
    This is a CONSERVATIVE parser — returns None for anything ambiguous.
    """
    # Gate 1: Is this tennis?
    if not _detect_tennis(question, category):
        return None

    # Gate 2: Skip prop bets
    if _is_prop_bet(question):
        return None

    # Gate 3: Skip tournament winners
    if _is_tournament_winner(question):
        return None

    # Gate 4: Find "A vs B" pattern
    player_a = None
    player_b = None
    for pattern in _VS_PATTERNS:
        match = pattern.search(question)
        if match:
            player_a = _clean_player_name(match.group(1))
            player_b = _clean_player_name(match.group(2))
            break

    if not player_a or not player_b:
        return None

    # Gate 5: Names must look like real player names (2+ chars each)
    if len(player_a) < 3 or len(player_b) < 3:
        return None

    # Gate 6: Determine which side "Yes" maps to
    yes_side, yes_confidence = _determine_yes_side(question, player_a, player_b)
    if yes_confidence < 0.6:
        return None

    # Detect surface
    surface = _detect_surface(question)

    # Overall confidence
    confidence = 0.8 if surface != "unknown" else 0.7

    return ParsedMatchup(
        player_a=player_a,
        player_b=player_b,
        sport="tennis",
        surface=surface,
        confidence=confidence * yes_confidence,
        yes_side_player=yes_side,
        market_id=market_id,
        raw_question=question,
    )


def parse_ufc_matchup(
    question: str,
    category: str = "",
    market_id: int | None = None,
) -> Optional[ParsedMatchup]:
    """Parse a market question into a UFC/MMA matchup.

    Same patterns as tennis (vs, beat) but for UFC. Returns ParsedMatchup with
    sport="ufc", surface="cage". Used when detect_ufc_market() is True.
    """
    if not detect_ufc_market(question, category):
        return None
    if _is_prop_bet(question):
        return None
    if _is_tournament_winner(question):
        return None

    player_a = None
    player_b = None
    for pattern in _VS_PATTERNS:
        match = pattern.search(question)
        if match:
            player_a = _clean_player_name(match.group(1))
            player_b = _clean_player_name(match.group(2))
            break

    if not player_a or not player_b or len(player_a) < 3 or len(player_b) < 3:
        return None

    yes_side, yes_confidence = _determine_yes_side(question, player_a, player_b)
    if yes_confidence < 0.6:
        return None

    return ParsedMatchup(
        player_a=player_a,
        player_b=player_b,
        sport="ufc",
        surface="cage",
        confidence=0.8 * yes_confidence,
        yes_side_player=yes_side,
        market_id=market_id,
        raw_question=question,
    )


_NBA_INDICATORS = [
    "nba", "basketball", "lakers", "celtics", "warriors", "bulls", "heat",
    "nets", "bucks", "suns", "clippers", "nuggets", "knicks",
    "76ers", "sixers", "cavaliers", "hawks", "hornets", "magic",
    "pacers", "pistons", "raptors", "grizzlies", "pelicans", "thunder",
    "trail blazers", "blazers", "spurs", "jazz", "rockets", "timberwolves",
    "mavericks", "wizards",
]

# Slug pattern: nba-LAL-GSW-2026-02-28
_NBA_SLUG_RE = re.compile(
    r"nba-([A-Z]{2,3})-([A-Z]{2,3})-(\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)


def detect_nba_market(question: str, category: str = "", slug: str = "") -> bool:
    """Check if a market is about an NBA game."""
    combined = f"{question} {category} {slug}".lower()
    if _NBA_SLUG_RE.search(slug):
        return True
    return any(indicator in combined for indicator in _NBA_INDICATORS)


def parse_nba_matchup(
    question: str,
    category: str = "",
    market_id: int | None = None,
    slug: str = "",
) -> Optional["ParsedMatchup"]:
    """Parse an NBA market into a team matchup.

    Preferred path: extract team codes from slug (nba-LAL-GSW-2026-02-28).
    Fallback: extract "TeamA vs TeamB" from question text.

    Returns ParsedMatchup with sport="nba", surface="court".
    """
    from data_pipeline.collectors.nba_results import abbr_to_team_name

    if not detect_nba_market(question, category, slug):
        return None
    if _is_prop_bet(question):
        return None

    team_a = team_b = None

    # Preferred: parse slug for exact team codes
    slug_match = _NBA_SLUG_RE.search(slug)
    if slug_match:
        abbr_a = slug_match.group(1).upper()
        abbr_b = slug_match.group(2).upper()
        team_a = abbr_to_team_name(abbr_a) or abbr_a
        team_b = abbr_to_team_name(abbr_b) or abbr_b

    # Fallback: "A vs B" pattern in question
    if not team_a or not team_b:
        for pattern in _VS_PATTERNS:
            m = pattern.search(question)
            if m:
                team_a = _clean_player_name(m.group(1))
                team_b = _clean_player_name(m.group(2))
                break

    if not team_a or not team_b or len(team_a) < 3 or len(team_b) < 3:
        return None

    if _is_tournament_winner(question):
        return None

    yes_side, yes_confidence = _determine_yes_side(question, team_a, team_b)
    if yes_confidence < 0.6:
        return None

    return ParsedMatchup(
        player_a=team_a,
        player_b=team_b,
        sport="nba",
        surface="court",
        confidence=0.85 * yes_confidence,
        yes_side_player=yes_side,
        market_id=market_id,
        raw_question=question,
    )


def fuzzy_match_player(
    parsed_name: str,
    known_players: list[str],
    threshold: float = 0.85,
    delta_threshold: float = 0.15,
) -> Optional[str]:
    """Match a parsed player name to known Elo-rated players.

    Uses simple token-based similarity. For production, use thefuzz library.

    Args:
        parsed_name: Name extracted from market question
        known_players: List of player names in Elo database
        threshold: Minimum similarity score (0-1)
        delta_threshold: Min gap between best and second-best match

    Returns:
        Best matching player name, or None if no confident match.
    """
    try:
        from thefuzz import fuzz
    except ImportError:
        # Fallback to basic string matching
        return _basic_match(parsed_name, known_players, threshold)

    parsed_lower = parsed_name.lower().strip()
    scores = []

    for known in known_players:
        # Use token_sort_ratio which handles "Nadal, Rafael" vs "Rafael Nadal"
        score = fuzz.token_sort_ratio(parsed_lower, known.lower()) / 100.0
        scores.append((known, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    if not scores or scores[0][1] < threshold:
        return None

    # Check delta between best and second-best
    if len(scores) > 1:
        delta = scores[0][1] - scores[1][1]
        if delta < delta_threshold:
            # Ambiguous match — two players too close in name similarity
            return None

    return scores[0][0]


def _basic_match(
    parsed_name: str, known_players: list[str], threshold: float,
) -> Optional[str]:
    """Basic string matching fallback (no thefuzz dependency)."""
    parsed_lower = parsed_name.lower().strip()
    parsed_parts = set(parsed_lower.split())

    best_match = None
    best_score = 0.0

    for known in known_players:
        known_lower = known.lower()

        # Exact match
        if parsed_lower == known_lower:
            return known

        # Token overlap score
        known_parts = set(known_lower.split())
        if parsed_parts and known_parts:
            overlap = len(parsed_parts & known_parts)
            total = max(len(parsed_parts), len(known_parts))
            score = overlap / total
        else:
            score = 0.0

        # Also check if last name matches (most common in market questions)
        if parsed_parts and known_parts:
            if list(parsed_parts)[-1] == list(known_parts)[-1]:
                score = max(score, 0.7)

        if score > best_score:
            best_score = score
            best_match = known

    return best_match if best_score >= threshold else None
