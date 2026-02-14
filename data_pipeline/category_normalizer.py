"""Category normalization - maps raw API categories + question keywords to clean categories.

Two-level normalization:
1. Sports pattern detection (catches "o/u 2.5", "spread -7.5", "match winner", etc.)
2. Direct mapping + keyword fallback for everything else
"""

import re

# Clean target categories
CATEGORIES = [
    "politics", "sports", "crypto", "economics", "technology",
    "science", "weather", "entertainment", "culture", "other",
]

# Direct mapping of known raw category strings
RAW_MAP: dict[str, str] = {
    "politics": "politics", "us politics": "politics", "elections": "politics",
    "midterms": "politics", "trump": "politics", "biden": "politics",
    "congress": "politics", "presidential": "politics", "senate": "politics",
    "governor": "politics", "democrat": "politics", "republican": "politics",
    "geopolitics": "politics", "iran": "politics", "government": "politics",

    "crypto": "crypto", "bitcoin": "crypto", "ethereum": "crypto",
    "btc": "crypto", "eth": "crypto", "defi": "crypto", "solana": "crypto",
    "xrp": "crypto", "web3": "crypto",

    "sports": "sports", "nfl": "sports", "nba": "sports", "mlb": "sports",
    "nhl": "sports", "soccer": "sports", "football": "sports",
    "basketball": "sports", "baseball": "sports", "ufc": "sports",
    "mma": "sports", "tennis": "sports", "f1": "sports", "formula 1": "sports",
    "premier league": "sports", "champions league": "sports",
    "world cup": "sports", "super bowl": "sports", "olympics": "sports",
    "golf": "sports", "boxing": "sports", "cricket": "sports",
    "rugby": "sports", "table tennis": "sports", "volleyball": "sports",
    "cycling": "sports", "swimming": "sports", "athletics": "sports",
    "esports": "sports", "dota": "sports", "csgo": "sports", "cs2": "sports",
    "valorant": "sports", "league of legends": "sports",
    "atp": "sports", "wta": "sports", "pga": "sports", "lpga": "sports",
    "serie a": "sports", "la liga": "sports", "bundesliga": "sports",
    "ligue 1": "sports", "eredivisie": "sports", "mls": "sports",
    "copa america": "sports", "euro 2024": "sports", "euro 2028": "sports",

    "science": "science", "space": "science", "nasa": "science",
    "ai": "technology", "artificial intelligence": "technology",
    "tech": "technology", "technology": "technology",

    "climate": "weather", "weather": "weather", "hurricane": "weather",
    "temperature": "weather",

    "finance": "economics", "economics": "economics", "economy": "economics",
    "fed": "economics", "interest rate": "economics", "inflation": "economics",
    "gdp": "economics", "stock": "economics", "recession": "economics",
    "financials": "economics", "companies": "economics", "earnings": "economics",

    "entertainment": "entertainment", "movies": "entertainment",
    "oscars": "entertainment", "music": "entertainment",
    "grammys": "entertainment", "tv": "entertainment",

    "culture": "culture", "pop culture": "culture", "social media": "culture",
    "mentions": "culture",
}

# Compiled patterns for sports detection from junk categories
# These catch categories like "o/u 2.5", "spread -7.5", "match winner", etc.
_SPORTS_PATTERNS = [
    # Betting line formats: "o/u 2.5", "spread -7.5", "total 3.5", "handicap +1.5"
    re.compile(r'^o/u\s*[\d.]'),
    re.compile(r'^spread\s*[+\-\d.]'),
    re.compile(r'^total\s*[\d.]'),
    re.compile(r'^handicap\s*[+\-\d.]'),
    re.compile(r'^moneyline$'),
    # Match/game/set winner patterns
    re.compile(r'(match|game|set|map|round)\s*(winner|\d)'),
    # Tournament/event headers: "dallas open: ...", "atp 500: ...", "uefa: ..."
    re.compile(r'(open|cup|league|tournament|masters|grand prix|grand slam|trophy|championship)\s*:'),
    re.compile(r'^(atp|wta|pga|uefa|fifa|nba|nfl|mlb|nhl|ufc)\s*([\d:]|tour|cup|league)'),
    # Esports formats
    re.compile(r'bo[1-9]$|^bo[1-9]\b'),
    # Player stat props: "rebounds", "assists", "points", "strikeouts", "yards"
    re.compile(r'(rebounds|assists|strikeouts|touchdowns|yards|goals|saves|aces)\s*(o/u)?'),
    # vs pattern in category itself
    re.compile(r'\bvs\.?\b'),
]

# Sport names used for question/description-based detection
_SPORT_NAMES = [
    "tennis", "basketball", "football", "soccer", "baseball", "hockey",
    "golf", "boxing", "mma", "ufc", "cricket", "rugby", "volleyball",
    "table tennis", "badminton", "cycling", "swimming", "athletics",
    "formula 1", "f1", "nascar", "darts", "snooker", "wrestling",
]

# League/tournament names that definitively indicate sports
_SPORTS_LEAGUES = [
    "nba", "nfl", "mlb", "nhl", "mls", "premier league", "la liga",
    "bundesliga", "serie a", "ligue 1", "champions league", "europa league",
    "atp", "wta", "pga", "lpga", "australian open", "french open",
    "wimbledon", "us open", "roland garros", "indian wells", "miami open",
    "copa america", "world cup", "super bowl", "stanley cup", "world series",
    "march madness", "ncaa", "college football", "college basketball",
    "six nations", "test match", "ipl", "big bash", "ashes",
]

# Keyword patterns for question-based classification (fallback)
KEYWORD_PATTERNS: list[tuple[str, list[str]]] = [
    ("politics", ["president", "election", "vote", "congress", "senate", "governor",
                  "democrat", "republican", "trump", "biden", "political", "legislation",
                  "impeach", "supreme court", "house of representatives", "gop",
                  "primary", "nominee", "cabinet", "veto", "pardon", "deport",
                  "government shutdown", "executive order", "indictment"]),
    ("sports", ["championship", "mvp", "playoff", "season", "nfl", "nba",
                "mlb", "nhl", "ufc", "premier league", "world series", "super bowl",
                "grand slam", "olympics", "medal", "vs.", "vs ",
                "match", "game", "tournament", "seed", "draft", "roster",
                "coach", "quarterback", "pitcher", "goalkeeper",
                "slam dunk", "home run", "touchdown", "goal",
                "halftime", "overtime", "penalty", "foul",
                "standings", "rankings", "relegation", "promotion"]),
    ("crypto", ["bitcoin", "btc", "ethereum", "eth", "crypto", "blockchain", "token",
                "solana", "sol", "defi", "nft", "web3", "mining", "halving",
                "altcoin", "stablecoin", "binance", "coinbase"]),
    ("economics", ["gdp", "inflation", "fed", "interest rate", "unemployment", "cpi",
                   "treasury", "yield", "recession", "s&p", "nasdaq", "dow",
                   "stock market", "tariff", "jobs report", "bps", "fomc",
                   "federal reserve"]),
    ("technology", ["ai", "openai", "chatgpt", "artificial intelligence",
                    "spacex", "tesla", "apple", "google", "meta", "microsoft",
                    "gta", "release", "launch"]),
    ("science", ["study", "research", "nasa", "space", "mars", "vaccine", "fda",
                 "clinical trial", "discovery", "quantum"]),
    ("weather", ["hurricane", "temperature", "climate", "flood", "tornado",
                 "wildfire", "drought", "storm"]),
    ("entertainment", ["oscar", "grammy", "emmy", "movie", "film", "album",
                       "box office", "netflix", "disney", "rotten tomatoes"]),
    ("culture", ["tiktok", "twitter", "viral", "meme", "influencer", "youtube",
                 "celebrity", "podcast", "say", "cocaine"]),
]

# Pattern to detect junk categories (numbers, dates, dollar amounts)
_JUNK_RE = re.compile(r'^[\d,$.\-\s%+]+$|^\w+ \d+$|^\d+[\-–]\d+$')


def _is_sports_by_pattern(raw: str, question: str = "", description: str = "") -> bool:
    """Check if category/question/description matches known sports patterns.

    This catches the 1,072+ sports markets scattered across 109+ junk categories
    like "o/u 2.5", "spread -7.5", "match winner", etc.
    """
    # Check category against sports regex patterns
    for pattern in _SPORTS_PATTERNS:
        if pattern.search(raw):
            return True

    # Check combined text for league/tournament names
    combined = f"{raw} {question} {description}".lower()
    for league in _SPORTS_LEAGUES:
        if league in combined:
            return True

    # Check question for sport names + competitive context
    q_lower = question.lower()
    for sport in _SPORT_NAMES:
        if sport in q_lower:
            return True

    return False


def normalize_category(
    raw_category: str | None,
    question: str = "",
    description: str = "",
) -> str:
    """Map raw category string + question text to a clean canonical category.

    Args:
        raw_category: Original category string from API
        question: Market question text
        description: Market description text (optional, for better detection)

    Returns:
        One of CATEGORIES: politics, sports, crypto, economics, technology,
        science, weather, entertainment, culture, other
    """
    raw = (raw_category or "").strip().lower()
    q = question.strip().lower()
    desc = (description or "").strip().lower()

    # 1. Direct match on RAW_MAP (trusted — these always win)
    if raw in RAW_MAP:
        return RAW_MAP[raw]

    # 2. Substring match (e.g. "us politics 2025" contains "politics")
    for key, mapped in RAW_MAP.items():
        if len(key) >= 3 and (key in raw or raw in key):
            return mapped

    # 3. Sports pattern detection (for unknown/junk categories)
    # Catches "o/u 2.5", "spread -7.5", "match winner", tournament names, etc.
    if _is_sports_by_pattern(raw, q, desc):
        return "sports"

    # 4. Keyword match on combined text
    combined = f"{raw} {q} {desc}"
    best_cat = None
    best_hits = 0
    for cat, keywords in KEYWORD_PATTERNS:
        hits = sum(1 for kw in keywords if kw in combined)
        if hits > best_hits:
            best_hits = hits
            best_cat = cat

    if best_cat and best_hits >= 1:
        return best_cat

    # 5. If raw looks like junk, default to "other"
    if _JUNK_RE.match(raw) or not raw:
        return "other"

    return "other"
