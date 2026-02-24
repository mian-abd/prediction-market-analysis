"""Market Correlation Clustering — find statistical arbitrage across related markets.

Research basis:
- "Semantic Trading" (arxiv, 2024): LLM-based market clustering generates ~20%
  returns over weekly horizons by identifying correlated/anti-correlated markets.
- Binary outcome correlations can be fully characterized via Pearson's coefficient
  and first three moments (MPRA research).
- Price disagreements between correlated markets represent statistical arbitrage.

Strategy:
1. Cluster markets by semantic similarity (same topic/event family)
2. Compute price correlations within clusters using price history
3. When correlated markets diverge in price, bet on convergence
4. When anti-correlated markets converge, bet on divergence

Example: "Trump wins GOP primary" at 75% but "Trump wins general" at 55%.
If historically correlated (conditional probability link), the spread may represent
a mispricing opportunity.
"""

import logging
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Market, PriceSnapshot, MarketRelationship
from ml.strategies.ensemble_edge_detector import (
    POLYMARKET_FEE_RATE, SLIPPAGE_BUFFER, compute_kelly,
)

logger = logging.getLogger(__name__)

MIN_VOLUME_TOTAL = 10_000
MIN_NET_EDGE = 0.03
MIN_CORRELATION = 0.5  # Minimum correlation to consider markets related
MIN_CLUSTER_SIZE = 2
KEYWORD_SIMILARITY_THRESHOLD = 0.3  # Jaccard similarity for keyword clustering
PRICE_HISTORY_DAYS = 7
MIN_PRICE_POINTS = 10  # Need >=10 overlapping price points for correlation


def _extract_entity_keywords(text: str) -> set[str]:
    """Extract entity-level keywords (proper nouns, key terms) from market question."""
    stop_words = {
        "will", "the", "a", "an", "be", "by", "in", "on", "at", "to", "of",
        "is", "it", "for", "and", "or", "but", "not", "no", "with", "this",
        "that", "from", "has", "have", "had", "before", "after", "more",
        "than", "yes", "win", "above", "below", "over", "under", "reach",
    }
    words = re.findall(r'\b[A-Za-z]{3,}\b', text)
    return {w.lower() for w in words if w.lower() not in stop_words}


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def cluster_markets_by_keywords(markets: list[Market]) -> dict[str, list[Market]]:
    """Cluster markets by keyword similarity (greedy agglomerative).

    Groups markets that share enough keywords to be about the same topic.
    """
    market_keywords = {}
    for m in markets:
        keywords = _extract_entity_keywords(m.question or "")
        if len(keywords) >= 2:
            market_keywords[m.id] = (m, keywords)

    clusters: dict[str, list[Market]] = {}
    assigned: set[int] = set()
    cluster_id = 0

    market_ids = list(market_keywords.keys())
    for i, mid_a in enumerate(market_ids):
        if mid_a in assigned:
            continue

        m_a, kw_a = market_keywords[mid_a]
        cluster_name = f"cluster_{cluster_id}"
        clusters[cluster_name] = [m_a]
        assigned.add(mid_a)

        for mid_b in market_ids[i+1:]:
            if mid_b in assigned:
                continue
            _, kw_b = market_keywords[mid_b]
            sim = _jaccard_similarity(kw_a, kw_b)
            if sim >= KEYWORD_SIMILARITY_THRESHOLD:
                clusters[cluster_name].append(market_keywords[mid_b][0])
                assigned.add(mid_b)

        if len(clusters[cluster_name]) >= MIN_CLUSTER_SIZE:
            cluster_id += 1
        else:
            # Single-market cluster, remove it
            for m in clusters[cluster_name]:
                assigned.discard(m.id)
            del clusters[cluster_name]

    return clusters


async def compute_price_correlation(
    session: AsyncSession,
    market_a_id: int,
    market_b_id: int,
    days_back: int = PRICE_HISTORY_DAYS,
) -> Optional[float]:
    """Compute price correlation between two markets over recent history."""
    cutoff = datetime.utcnow() - timedelta(days=days_back)

    prices_a = await session.execute(
        select(PriceSnapshot.timestamp, PriceSnapshot.price_yes)
        .where(PriceSnapshot.market_id == market_a_id, PriceSnapshot.timestamp >= cutoff)
        .order_by(PriceSnapshot.timestamp)
    )
    prices_b = await session.execute(
        select(PriceSnapshot.timestamp, PriceSnapshot.price_yes)
        .where(PriceSnapshot.market_id == market_b_id, PriceSnapshot.timestamp >= cutoff)
        .order_by(PriceSnapshot.timestamp)
    )

    a_points = {row[0].strftime("%Y-%m-%d %H"): row[1] for row in prices_a.all()}
    b_points = {row[0].strftime("%Y-%m-%d %H"): row[1] for row in prices_b.all()}

    # Find overlapping timestamps (hourly resolution)
    common_times = sorted(set(a_points.keys()) & set(b_points.keys()))
    if len(common_times) < MIN_PRICE_POINTS:
        return None

    a_vals = np.array([a_points[t] for t in common_times])
    b_vals = np.array([b_points[t] for t in common_times])

    # Pearson correlation
    if np.std(a_vals) < 1e-6 or np.std(b_vals) < 1e-6:
        return None

    correlation = float(np.corrcoef(a_vals, b_vals)[0, 1])
    return correlation


async def find_correlation_edges(
    session: AsyncSession,
    cluster: list[Market],
) -> list[dict]:
    """Find mispricing opportunities within a market cluster.

    When two correlated markets have divergent prices relative to their
    historical correlation, one is likely mispriced.
    """
    if len(cluster) < 2:
        return []

    edges = []

    for i in range(len(cluster)):
        for j in range(i + 1, len(cluster)):
            m_a, m_b = cluster[i], cluster[j]
            price_a = m_a.price_yes or 0.5
            price_b = m_b.price_yes or 0.5

            correlation = await compute_price_correlation(session, m_a.id, m_b.id)
            if correlation is None:
                continue

            abs_corr = abs(correlation)
            if abs_corr < MIN_CORRELATION:
                continue

            # For positively correlated markets:
            # If A is much higher than B, B may be underpriced (or A overpriced)
            if correlation > 0:
                price_diff = price_a - price_b
                if abs(price_diff) < 0.08:
                    continue  # Not enough divergence

                # The cheaper market is likely underpriced
                if price_diff > 0.08:
                    underpriced = m_b
                    overpriced = m_a
                    direction = "buy_yes"
                    implied_adjustment = price_diff * abs_corr * 0.3
                elif price_diff < -0.08:
                    underpriced = m_a
                    overpriced = m_b
                    direction = "buy_yes"
                    implied_adjustment = abs(price_diff) * abs_corr * 0.3
                else:
                    continue

                target_market = underpriced
                target_price = target_market.price_yes or 0.5
                implied_prob = min(0.95, target_price + implied_adjustment)

            else:
                # Anti-correlated: if both are high, at least one is wrong
                if price_a > 0.6 and price_b > 0.6:
                    # At least one should be lower
                    target_market = m_b if price_b > price_a else m_a
                    direction = "buy_no"
                    target_price = target_market.price_yes or 0.5
                    implied_prob = max(0.05, target_price - abs(correlation) * 0.1)
                elif price_a < 0.4 and price_b < 0.4:
                    target_market = m_a if price_a < price_b else m_b
                    direction = "buy_yes"
                    target_price = target_market.price_yes or 0.5
                    implied_prob = min(0.95, target_price + abs(correlation) * 0.1)
                else:
                    continue

            p = implied_prob
            q = target_price
            if direction == "buy_yes":
                fee = p * POLYMARKET_FEE_RATE * (1 - q) + SLIPPAGE_BUFFER
                net_ev = p * (1 - q) - (1 - p) * q - fee
            else:
                fee = (1 - p) * POLYMARKET_FEE_RATE * q + SLIPPAGE_BUFFER
                net_ev = (1 - p) * q - p * (1 - q) - fee

            if net_ev < MIN_NET_EDGE:
                continue

            kelly = compute_kelly(direction, implied_prob, target_price, fee)

            confidence = 0.3 + 0.4 * abs_corr  # Higher correlation = more confident

            edges.append({
                "market_id": target_market.id,
                "strategy": "market_clustering",
                "question": target_market.question,
                "direction": direction,
                "implied_prob": round(implied_prob, 4),
                "market_price": round(target_price, 4),
                "raw_edge": round(abs(implied_prob - target_price), 4),
                "net_ev": round(net_ev, 4),
                "fee_cost": round(fee, 4),
                "kelly_fraction": round(kelly, 4),
                "confidence": round(confidence, 3),
                "correlation": round(correlation, 3),
                "related_market_id": overpriced.id if correlation > 0 else (m_a.id if target_market == m_b else m_b.id),
                "related_market_question": (overpriced.question if correlation > 0
                                           else (m_a.question if target_market == m_b else m_b.question)),
            })

    return edges


async def scan_market_clusters(
    session: AsyncSession,
) -> list[dict]:
    """Full scan: cluster markets, compute correlations, find edges."""
    result = await session.execute(
        select(Market).where(
            Market.is_active == True,  # noqa: E711
            Market.price_yes != None,  # noqa: E711
            Market.volume_total >= MIN_VOLUME_TOTAL,
        ).order_by(Market.volume_24h.desc()).limit(500)
    )
    markets = result.scalars().all()

    if not markets:
        return []

    # Step 1: Cluster by keywords
    clusters = cluster_markets_by_keywords(markets)
    logger.info(f"Market clustering: {len(clusters)} clusters from {len(markets)} markets")

    # Step 2: Find correlation edges within each cluster
    all_edges = []
    for cluster_name, cluster_markets in clusters.items():
        edges = await find_correlation_edges(session, cluster_markets)
        all_edges.extend(edges)

        if edges:
            logger.info(
                f"Cluster '{cluster_name}' ({len(cluster_markets)} markets): "
                f"{len(edges)} correlation edges"
            )

    # Step 3: Store relationships for future use
    for edge in all_edges:
        try:
            rel = MarketRelationship(
                market_id_a=edge["market_id"],
                market_id_b=edge["related_market_id"],
                relationship_type="price_correlation",
                confidence=abs(edge["correlation"]),
                source="keyword_clustering",
            )
            session.add(rel)
        except Exception:
            pass

    if all_edges:
        await session.commit()

    logger.info(f"Market clustering scan: {len(all_edges)} total edges found")
    return all_edges
