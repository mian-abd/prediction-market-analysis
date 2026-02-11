"""Cross-platform market matching using TF-IDF cosine similarity.
Identifies the same event traded on Polymarket AND Kalshi."""

import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Market, Platform, CrossPlatformMatch

logger = logging.getLogger(__name__)

MIN_SIMILARITY = 0.45  # Threshold for considering a match


def compute_similarity_matrix(
    poly_questions: list[str],
    kalshi_questions: list[str],
) -> list[list[float]]:
    """Compute TF-IDF cosine similarity between two sets of market questions."""
    all_questions = poly_questions + kalshi_questions
    if not all_questions:
        return []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
    )
    tfidf_matrix = vectorizer.fit_transform(all_questions)

    n_poly = len(poly_questions)
    poly_vectors = tfidf_matrix[:n_poly]
    kalshi_vectors = tfidf_matrix[n_poly:]

    sim_matrix = cosine_similarity(poly_vectors, kalshi_vectors)
    return sim_matrix.tolist()


async def find_cross_platform_matches(
    session: AsyncSession,
    min_similarity: float = MIN_SIMILARITY,
) -> list[dict]:
    """Find matching markets between Polymarket and Kalshi."""
    # Get platform IDs
    poly_result = await session.execute(
        select(Platform).where(Platform.name == "polymarket")
    )
    kalshi_result = await session.execute(
        select(Platform).where(Platform.name == "kalshi")
    )
    poly_platform = poly_result.scalar_one_or_none()
    kalshi_platform = kalshi_result.scalar_one_or_none()

    if not poly_platform or not kalshi_platform:
        logger.warning("Missing platform records for matching")
        return []

    # Get active markets from each platform
    poly_markets_result = await session.execute(
        select(Market)
        .where(Market.platform_id == poly_platform.id, Market.is_active == True)  # noqa
        .order_by(Market.volume_24h.desc())
        .limit(500)
    )
    kalshi_markets_result = await session.execute(
        select(Market)
        .where(Market.platform_id == kalshi_platform.id, Market.is_active == True)  # noqa
        .order_by(Market.volume_24h.desc())
        .limit(500)
    )

    poly_markets = list(poly_markets_result.scalars().all())
    kalshi_markets = list(kalshi_markets_result.scalars().all())

    if not poly_markets or not kalshi_markets:
        logger.info("No markets to match")
        return []

    logger.info(
        f"Matching {len(poly_markets)} Polymarket vs {len(kalshi_markets)} Kalshi markets"
    )

    poly_questions = [m.question for m in poly_markets]
    kalshi_questions = [m.question for m in kalshi_markets]

    sim_matrix = compute_similarity_matrix(poly_questions, kalshi_questions)

    matches = []
    for i, poly_market in enumerate(poly_markets):
        best_j = -1
        best_score = 0.0
        for j in range(len(kalshi_markets)):
            score = sim_matrix[i][j]
            if score > best_score:
                best_score = score
                best_j = j

        if best_score >= min_similarity and best_j >= 0:
            kalshi_market = kalshi_markets[best_j]

            # Check if match already exists
            existing = await session.execute(
                select(CrossPlatformMatch).where(
                    CrossPlatformMatch.market_id_a == poly_market.id,
                    CrossPlatformMatch.market_id_b == kalshi_market.id,
                )
            )
            if not existing.scalar_one_or_none():
                match = CrossPlatformMatch(
                    market_id_a=poly_market.id,
                    market_id_b=kalshi_market.id,
                    similarity_score=best_score,
                    match_method="tfidf",
                    is_confirmed=best_score > 0.8,
                )
                session.add(match)

            matches.append({
                "poly_id": poly_market.id,
                "kalshi_id": kalshi_market.id,
                "poly_question": poly_market.question,
                "kalshi_question": kalshi_market.question,
                "similarity": best_score,
                "poly_price_yes": poly_market.price_yes,
                "kalshi_price_yes": kalshi_market.price_yes,
            })

    await session.commit()
    logger.info(f"Found {len(matches)} cross-platform matches")
    return matches
