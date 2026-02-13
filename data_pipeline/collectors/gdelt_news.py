"""GDELT News API collector - real-time event context for prediction markets.

GDELT (Global Database of Events, Language, and Tone) provides:
- Real-time news coverage worldwide
- Sentiment analysis (tone)
- Event categorization

Free API, no auth required.
"""

import httpx
import logging
from datetime import datetime, timedelta
from urllib.parse import quote

from config.settings import settings

logger = logging.getLogger(__name__)

# GDELT API endpoints
GDELT_API_BASE = settings.gdelt_api_url or "https://api.gdeltproject.org/api/v2"


async def search_news(
    query: str,
    max_records: int = 10,
    timespan: str = "3d",  # 3 days lookback
) -> list[dict]:
    """Search GDELT for news articles matching query.

    Args:
        query: Search keywords (e.g., "Trump election", "Bitcoin price")
        max_records: Number of articles to return (default 10)
        timespan: Lookback period - "1d", "3d", "7d", "30d"

    Returns:
        List of article dicts with: {url, title, tone, domain, publish_date}
    """
    # GDELT Doc API: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
    url = f"{GDELT_API_BASE}/doc/doc"

    params = {
        "query": query,
        "mode": "ArtList",  # Article list mode
        "maxrecords": min(max_records, 250),  # GDELT max is 250
        "timespan": timespan,
        "format": "json",
        "sort": "DateDesc",  # Most recent first
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        articles = data.get("articles", [])
        logger.info(f"GDELT search '{query}' returned {len(articles)} articles")

        # Parse GDELT response
        parsed_articles = []
        for art in articles:
            parsed_articles.append({
                "url": art.get("url"),
                "title": art.get("title"),
                "domain": art.get("domain"),
                "language": art.get("language", "en"),
                "publish_date": art.get("seendate"),  # YYYYMMDDHHMMSS format
                "tone": float(art.get("tone", 0)),  # -10 (negative) to +10 (positive)
                "social_image": art.get("socialimage"),
            })

        return parsed_articles

    except Exception as e:
        logger.error(f"GDELT search failed for '{query}': {e}")
        return []


async def get_category_news(category: str, max_articles: int = 5) -> list[dict]:
    """Get recent news for a market category.

    Args:
        category: Market category (politics, crypto, sports, etc.)
        max_articles: Number of articles to fetch

    Returns:
        List of relevant news articles
    """
    # Map categories to search queries
    category_queries = {
        "politics": "election OR president OR congress OR senate",
        "crypto": "Bitcoin OR Ethereum OR cryptocurrency",
        "sports": "NFL OR NBA OR soccer OR championship",
        "economics": "economy OR inflation OR recession OR GDP",
        "tech": "technology OR AI OR startup",
        "science": "research OR study OR discovery",
        "weather": "weather OR climate OR hurricane",
        "entertainment": "movie OR music OR celebrity",
        "culture": "culture OR society",
        "other": "news OR events",
    }

    query = category_queries.get(category.lower(), category)
    return await search_news(query, max_records=max_articles, timespan="3d")


async def get_market_news(question: str, category: str | None = None) -> list[dict]:
    """Get news relevant to a specific market.

    Args:
        question: Market question (e.g., "Will Trump win 2024?")
        category: Optional category for fallback search

    Returns:
        Up to 5 most relevant articles
    """
    # Extract keywords from question
    # Simple approach: remove common words and use first 3-5 keywords
    stop_words = {"will", "the", "a", "an", "be", "by", "in", "on", "at", "to", "of"}
    words = question.lower().split()
    keywords = [w for w in words if w not in stop_words and len(w) > 3][:5]

    query = " ".join(keywords)

    articles = await search_news(query, max_records=5, timespan="7d")

    # If no results, fallback to category news
    if not articles and category:
        logger.info(f"No articles for market query, trying category '{category}'")
        articles = await get_category_news(category, max_articles=3)

    return articles
