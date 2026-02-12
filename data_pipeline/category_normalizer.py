"""Category normalization - maps raw API categories + question keywords to clean categories."""

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

# Keyword patterns for question-based classification (fallback)
KEYWORD_PATTERNS: list[tuple[str, list[str]]] = [
    ("politics", ["president", "election", "vote", "congress", "senate", "governor",
                  "democrat", "republican", "trump", "biden", "political", "legislation",
                  "impeach", "supreme court", "house of representatives", "gop",
                  "primary", "nominee", "cabinet", "veto", "pardon", "deport",
                  "government shutdown", "executive order", "indictment"]),
    ("sports", ["win", "championship", "mvp", "playoff", "season", "nfl", "nba",
                "mlb", "nhl", "ufc", "premier league", "world series", "super bowl",
                "grand slam", "olympics", "score", "medal", "vs.", "game",
                "spurs", "warriors", "lakers", "celtics", "yankees"]),
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


def normalize_category(raw_category: str | None, question: str = "") -> str:
    """Map raw category string + question text to a clean canonical category."""
    raw = (raw_category or "").strip().lower()
    q = question.strip().lower()

    # 1. Direct match
    if raw in RAW_MAP:
        return RAW_MAP[raw]

    # 2. Substring match (e.g. "us politics 2025" contains "politics")
    for key, mapped in RAW_MAP.items():
        if len(key) >= 3 and (key in raw or raw in key):
            return mapped

    # 3. Keyword match on combined text
    combined = f"{raw} {q}"
    best_cat = None
    best_hits = 0
    for cat, keywords in KEYWORD_PATTERNS:
        hits = sum(1 for kw in keywords if kw in combined)
        if hits > best_hits:
            best_hits = hits
            best_cat = cat

    if best_cat and best_hits >= 1:
        return best_cat

    # 4. If raw looks like junk, default to "other"
    if _JUNK_RE.match(raw) or not raw:
        return "other"

    return "other"
