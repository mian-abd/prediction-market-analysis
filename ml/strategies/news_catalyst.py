"""News-Catalyst Speed Trading — detect market-moving events before markets adjust.

Research basis:
- Kalshi research: prediction markets adjust to breaking news within seconds to
  minutes, but many smaller/thinner markets are slower to react.
- MANA-Net (2024): dynamic news weighting improves Sharpe by 0.252.
- Event-based sentiment (Journal of Prediction Markets): events, not just news,
  are the correct construct for predicting market movements.

Strategy:
1. Monitor real-time news via GDELT (free, no auth, already integrated)
2. Match news articles to active prediction markets using keyword overlap + LLM
3. Score sentiment impact (positive/negative for the YES outcome)
4. Generate signals when strong news hasn't been priced in yet

The key insight: GDELT updates every 15 minutes. Many prediction markets update
their prices more slowly than that (thin markets, few active traders).
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Market, NewsEvent
from data_pipeline.collectors.gdelt_news import search_news, get_market_news
from ml.strategies.ensemble_edge_detector import (
    POLYMARKET_FEE_RATE, SLIPPAGE_BUFFER, compute_kelly,
)

logger = logging.getLogger(__name__)

MIN_VOLUME_TOTAL = 15_000
MIN_NET_EDGE = 0.03
STRONG_TONE_THRESHOLD = 5.0  # GDELT tone: -10 to +10; |5|+ = very strong
MODERATE_TONE_THRESHOLD = 2.5
MIN_ARTICLES_FOR_SIGNAL = 2  # Need >=2 articles pointing same direction
RECENCY_BONUS_HOURS = 3  # Articles within 3 hours get extra weight


def _extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from text, filtering stop words."""
    stop_words = {
        "will", "the", "a", "an", "be", "by", "in", "on", "at", "to", "of",
        "is", "it", "for", "and", "or", "but", "not", "no", "with", "this",
        "that", "from", "has", "have", "had", "was", "were", "are", "been",
        "being", "do", "does", "did", "if", "can", "could", "would", "should",
        "may", "might", "shall", "than", "before", "after", "about", "more",
        "what", "when", "where", "who", "how", "which", "their", "there",
    }
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return {w for w in words if w not in stop_words}


def _compute_keyword_overlap(market_keywords: set[str], article_title: str) -> float:
    """Compute keyword overlap score between market question and article title."""
    article_keywords = _extract_keywords(article_title)
    if not market_keywords or not article_keywords:
        return 0.0
    overlap = market_keywords & article_keywords
    # Jaccard-like similarity but weighted toward market keywords
    return len(overlap) / max(len(market_keywords), 1)


def _estimate_sentiment_impact(
    tone: float,
    overlap_score: float,
    is_recent: bool,
) -> tuple[float, str]:
    """Estimate how much a news article should move the YES price.

    Returns (impact_magnitude, direction) where:
    - impact_magnitude: 0.0-1.0 (strength of the signal)
    - direction: "bullish_yes" or "bearish_yes"
    """
    abs_tone = abs(tone)
    direction = "bullish_yes" if tone > 0 else "bearish_yes"

    # Base impact from tone strength
    if abs_tone >= STRONG_TONE_THRESHOLD:
        base_impact = 0.8
    elif abs_tone >= MODERATE_TONE_THRESHOLD:
        base_impact = 0.5
    else:
        base_impact = 0.2

    # Scale by keyword relevance
    impact = base_impact * min(overlap_score * 2, 1.0)

    # Recency bonus: fresh news is more likely not yet priced in
    if is_recent:
        impact *= 1.3

    return min(1.0, impact), direction


async def analyze_market_news(
    market: Market,
    articles: list[dict],
) -> Optional[dict]:
    """Analyze news articles for a specific market and compute sentiment signal.

    Returns signal dict if strong enough sentiment is detected.
    """
    if not articles:
        return None

    market_keywords = _extract_keywords(market.question or "")
    market_price = market.price_yes or 0.5

    bullish_score = 0.0
    bearish_score = 0.0
    bullish_articles = 0
    bearish_articles = 0
    max_tone = 0.0
    now = datetime.utcnow()

    for article in articles:
        title = article.get("title", "")
        tone = float(article.get("tone", 0))
        overlap = _compute_keyword_overlap(market_keywords, title)

        if overlap < 0.15:
            continue  # Not relevant enough

        # Check recency
        publish_str = article.get("publish_date", "")
        is_recent = False
        if publish_str and len(publish_str) >= 14:
            try:
                pub_time = datetime.strptime(publish_str[:14], "%Y%m%d%H%M%S")
                is_recent = (now - pub_time) < timedelta(hours=RECENCY_BONUS_HOURS)
            except ValueError:
                pass

        impact, direction = _estimate_sentiment_impact(tone, overlap, is_recent)

        if direction == "bullish_yes":
            bullish_score += impact
            bullish_articles += 1
        else:
            bearish_score += impact
            bearish_articles += 1

        if abs(tone) > abs(max_tone):
            max_tone = tone

    # Need consensus: multiple articles pointing the same direction
    if bullish_score > bearish_score and bullish_articles >= MIN_ARTICLES_FOR_SIGNAL:
        net_sentiment = bullish_score - bearish_score
        # Bullish news → YES price should be higher than current
        estimated_impact = min(0.15, net_sentiment * 0.05)
        implied_prob = min(0.95, market_price + estimated_impact)
        trade_direction = "buy_yes"
    elif bearish_score > bullish_score and bearish_articles >= MIN_ARTICLES_FOR_SIGNAL:
        net_sentiment = bearish_score - bullish_score
        estimated_impact = min(0.15, net_sentiment * 0.05)
        implied_prob = max(0.05, market_price - estimated_impact)
        trade_direction = "buy_no"
    else:
        return None

    # Compute fee-aware edge
    p = implied_prob
    q = market_price
    if trade_direction == "buy_yes":
        fee = p * POLYMARKET_FEE_RATE * (1 - q) + SLIPPAGE_BUFFER
        net_ev = p * (1 - q) - (1 - p) * q - fee
    else:
        fee = (1 - p) * POLYMARKET_FEE_RATE * q + SLIPPAGE_BUFFER
        net_ev = (1 - p) * q - p * (1 - q) - fee

    if net_ev < MIN_NET_EDGE:
        return None

    kelly = compute_kelly(trade_direction, implied_prob, market_price, fee)

    # Confidence is based on article count, tone strength, and recency
    article_count_factor = min(1.0, max(bullish_articles, bearish_articles) / 5.0)
    tone_factor = min(1.0, abs(max_tone) / 8.0)
    confidence = 0.3 + 0.4 * article_count_factor + 0.3 * tone_factor

    return {
        "market_id": market.id,
        "strategy": "news_catalyst",
        "question": market.question,
        "direction": trade_direction,
        "implied_prob": round(implied_prob, 4),
        "market_price": round(market_price, 4),
        "raw_edge": round(abs(implied_prob - market_price), 4),
        "net_ev": round(net_ev, 4),
        "fee_cost": round(fee, 4),
        "kelly_fraction": round(kelly, 4),
        "confidence": round(confidence, 3),
        "sentiment": {
            "bullish_score": round(bullish_score, 2),
            "bearish_score": round(bearish_score, 2),
            "bullish_articles": bullish_articles,
            "bearish_articles": bearish_articles,
            "max_tone": round(max_tone, 2),
            "net_direction": "bullish" if bullish_score > bearish_score else "bearish",
        },
    }


async def scan_news_catalysts(
    session: AsyncSession,
    max_markets: int = 100,
) -> list[dict]:
    """Scan active markets for news-driven trading opportunities.

    Fetches recent news for top markets and matches sentiment to prices.
    """
    result = await session.execute(
        select(Market).where(
            Market.is_active == True,  # noqa: E711
            Market.price_yes != None,  # noqa: E711
            Market.volume_total >= MIN_VOLUME_TOTAL,
        ).order_by(Market.volume_24h.desc()).limit(max_markets)
    )
    markets = result.scalars().all()

    if not markets:
        return []

    # Group markets by category for batch news fetching
    category_markets: dict[str, list[Market]] = {}
    for m in markets:
        cat = (m.normalized_category or m.category or "other").lower()
        category_markets.setdefault(cat, []).append(m)

    edges_found = []

    for category, cat_markets in category_markets.items():
        # Fetch news for this category once
        try:
            from data_pipeline.collectors.gdelt_news import get_category_news
            category_articles = await get_category_news(category, max_articles=20)
        except Exception as e:
            logger.debug(f"Category news fetch failed for {category}: {e}")
            continue

        # Also fetch market-specific news for the top markets in this category
        for market in cat_markets[:10]:
            try:
                market_articles = await get_market_news(
                    market.question or "", category
                )
                all_articles = market_articles + category_articles
            except Exception:
                all_articles = category_articles

            signal = await analyze_market_news(market, all_articles)
            if signal:
                edges_found.append(signal)
                logger.info(
                    f"News catalyst: {market.question[:50]}... | "
                    f"Sentiment: {signal['sentiment']['net_direction']} | "
                    f"EV: {signal['net_ev']:.1%}"
                )

    # Store relevant news events
    for edge in edges_found:
        try:
            news_event = NewsEvent(
                category=edge.get("question", "")[:100],
                title=f"News catalyst signal for market {edge['market_id']}",
                tone=edge["sentiment"]["max_tone"],
                fetched_at=datetime.utcnow(),
            )
            session.add(news_event)
        except Exception:
            pass

    if edges_found:
        await session.commit()

    logger.info(f"News catalyst scan: {len(markets)} markets, {len(edges_found)} edges")
    return edges_found
