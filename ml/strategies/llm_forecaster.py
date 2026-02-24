"""LLM Superforecaster — automated structured probability forecasting.

Uses the CHAMP framework (Comparison classes, Historical base rates,
Adjustment for evidence, Multiple scenarios, Probability assignment)
to generate calibrated probability estimates for prediction markets.

Research basis:
- "Wisdom of the Silicon Crowd" (2024, Science Advances): LLM ensembles
  rival human crowd accuracy on binary forecasting questions.
- When given market prices as Bayesian priors, accuracy improves 17-28%.
- Superforecaster methodology (Tetlock) improves LLM calibration.

This strategy covers ALL market types (politics, crypto, sports, science,
weather) — the biggest gap in the current system since the ensemble ML
model only recalibrates existing prices rather than adding new information.
"""

import logging
import hashlib
import json
from datetime import datetime, timedelta
from typing import Optional

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from db.models import Market, AIAnalysis
from ml.strategies.ensemble_edge_detector import (
    compute_kelly, POLYMARKET_FEE_RATE, SLIPPAGE_BUFFER,
)

logger = logging.getLogger(__name__)

MIN_VOLUME_TOTAL = 10_000
MIN_NET_EDGE = 0.04
CACHE_HOURS = 6
MAX_MARKETS_PER_SCAN = 50
COST_PER_CALL_USD = 0.03  # Estimated cost per Haiku call

SYSTEM_PROMPT = """You are an expert superforecaster trained in the methodology of Philip Tetlock's Good Judgment Project. You produce well-calibrated probability estimates for binary questions about future events.

Your methodology (CHAMP Framework):
1. COMPARISON CLASSES: Identify reference classes of similar past events. What is the base rate?
2. HISTORICAL BASE RATES: Use specific historical data. How often do events like this actually happen?
3. ADJUSTMENT: Adjust from the base rate using specific evidence unique to this situation.
4. MULTIPLE SCENARIOS: Consider at least 3 scenarios (likely, unlikely, surprise).
5. PROBABILITY: Assign a final probability between 1% and 99%.

Calibration rules:
- Events you rate at 70% should happen ~70% of the time.
- Avoid extremes (below 5% or above 95%) unless the evidence is overwhelming.
- When uncertain, stay closer to the base rate.
- The current market price reflects collective wisdom — deviate only with strong reasoning.

You MUST respond with valid JSON and nothing else."""

FORECAST_PROMPT_TEMPLATE = """Forecast the following prediction market question:

QUESTION: {question}

DESCRIPTION: {description}

CATEGORY: {category}
CURRENT MARKET PRICE: {market_price:.1%} (this is what the crowd currently believes)
TIME TO RESOLUTION: {time_to_resolution}
VOLUME TRADED: ${volume_total:,.0f}

{news_context}

Apply the CHAMP framework:

1. What are the relevant comparison classes and base rates?
2. What specific evidence adjusts the probability up or down from the base rate?
3. What are 3 plausible scenarios and their rough probabilities?
4. Considering all factors, including the current market price as a Bayesian prior, what is your calibrated probability?

Respond with ONLY this JSON structure:
{{
  "probability": <float between 0.01 and 0.99>,
  "confidence": <float between 0.0 and 1.0, how confident you are in your estimate>,
  "base_rate": <float, the historical base rate you used>,
  "adjustment_direction": "<string: 'up' or 'down' or 'none'>",
  "adjustment_magnitude": <float, how much you adjusted from base rate>,
  "key_factors": ["<factor 1>", "<factor 2>", "<factor 3>"],
  "scenarios": [
    {{"scenario": "<description>", "probability": <float>}},
    {{"scenario": "<description>", "probability": <float>}},
    {{"scenario": "<description>", "probability": <float>}}
  ],
  "reasoning_summary": "<2-3 sentence summary of your reasoning>"
}}"""


def _format_time_to_resolution(market) -> str:
    if not market.end_date:
        return "Unknown"
    now = datetime.utcnow()
    end = market.end_date.replace(tzinfo=None) if market.end_date.tzinfo else market.end_date
    delta = end - now
    if delta.total_seconds() < 0:
        return "Past due"
    days = delta.days
    hours = delta.seconds // 3600
    if days > 30:
        return f"{days // 30} months"
    if days > 0:
        return f"{days} days, {hours} hours"
    return f"{hours} hours"


def _build_prompt_hash(question: str, market_price: float) -> str:
    price_bucket = round(market_price, 2)
    raw = f"llm_forecast_v2:{question}:{price_bucket}"
    return hashlib.sha256(raw.encode()).hexdigest()


async def _get_news_context(question: str, category: str) -> str:
    """Fetch recent news for context (uses existing GDELT collector)."""
    try:
        from data_pipeline.collectors.gdelt_news import get_market_news
        articles = await get_market_news(question, category)
        if not articles:
            return "RECENT NEWS: No recent news articles found."

        lines = ["RECENT NEWS:"]
        for art in articles[:5]:
            tone_label = "positive" if art["tone"] > 2 else "negative" if art["tone"] < -2 else "neutral"
            lines.append(f"- [{tone_label}] {art['title']}")
        return "\n".join(lines)
    except Exception as e:
        logger.debug(f"News fetch failed: {e}")
        return "RECENT NEWS: Unavailable."


async def forecast_market(
    market: Market,
    session: AsyncSession,
    news_context: str = "",
) -> Optional[dict]:
    """Generate LLM probability forecast for a single market.

    Returns dict with probability, confidence, edge info, or None if
    API unavailable / cached result exists with no edge.
    """
    if not settings.anthropic_api_key:
        return None

    market_price = market.price_yes or 0.5
    prompt_hash = _build_prompt_hash(market.question, market_price)

    # Check cache
    cached = await session.execute(
        select(AIAnalysis).where(
            AIAnalysis.prompt_hash == prompt_hash,
            AIAnalysis.created_at > datetime.utcnow() - timedelta(hours=CACHE_HOURS),
        )
    )
    existing = cached.scalar_one_or_none()
    if existing and existing.structured_result:
        return existing.structured_result

    time_str = _format_time_to_resolution(market)
    if not news_context:
        news_context = await _get_news_context(
            market.question, market.normalized_category or market.category or "other"
        )

    prompt = FORECAST_PROMPT_TEMPLATE.format(
        question=market.question,
        description=(market.description or "No additional description.")[:500],
        category=market.normalized_category or market.category or "unknown",
        market_price=market_price,
        time_to_resolution=time_str,
        volume_total=market.volume_total or 0,
        news_context=news_context,
    )

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": settings.anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-3-5-haiku-20241022",
                    "max_tokens": 800,
                    "system": SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            resp.raise_for_status()
            data = resp.json()

        response_text = data["content"][0]["text"]
        input_tokens = data.get("usage", {}).get("input_tokens", 0)
        output_tokens = data.get("usage", {}).get("output_tokens", 0)

        # Parse JSON response
        result = json.loads(response_text)
        llm_prob = float(result["probability"])
        llm_prob = max(0.01, min(0.99, llm_prob))
        llm_confidence = float(result.get("confidence", 0.5))

        # Compute edge
        direction, net_ev, fee_cost = _compute_directional_ev(llm_prob, market_price)
        kelly = compute_kelly(direction, llm_prob, market_price, fee_cost)
        raw_edge = abs(llm_prob - market_price)

        signal = {
            "market_id": market.id,
            "strategy": "llm_forecast",
            "llm_probability": round(llm_prob, 4),
            "market_price": round(market_price, 4),
            "raw_edge": round(raw_edge, 4),
            "direction": direction,
            "net_ev": round(net_ev, 4),
            "fee_cost": round(fee_cost, 4),
            "kelly_fraction": round(kelly, 4),
            "confidence": round(llm_confidence, 3),
            "base_rate": result.get("base_rate"),
            "key_factors": result.get("key_factors", []),
            "scenarios": result.get("scenarios", []),
            "reasoning": result.get("reasoning_summary", ""),
        }

        # Cache in AI analyses table
        analysis = AIAnalysis(
            market_id=market.id,
            analysis_type="llm_forecast",
            prompt_hash=prompt_hash,
            prompt_text=prompt[:2000],
            response_text=response_text,
            structured_result=signal,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=COST_PER_CALL_USD,
            model_used="claude-3-5-haiku-20241022",
        )
        session.add(analysis)

        return signal

    except json.JSONDecodeError as e:
        logger.warning(f"LLM forecast JSON parse failed for market {market.id}: {e}")
        return None
    except httpx.HTTPStatusError as e:
        logger.warning(f"LLM API error for market {market.id}: {e.response.status_code}")
        return None
    except Exception as e:
        logger.error(f"LLM forecast failed for market {market.id}: {e}")
        return None


def _compute_directional_ev(
    llm_prob: float,
    market_price: float,
) -> tuple[Optional[str], float, float]:
    """Compute fee-aware directional EV (same logic as ensemble detector)."""
    p = llm_prob
    q = market_price

    fee_yes = p * POLYMARKET_FEE_RATE * (1 - q) + SLIPPAGE_BUFFER
    ev_yes = p * (1 - q) - (1 - p) * q - fee_yes

    fee_no = (1 - p) * POLYMARKET_FEE_RATE * q + SLIPPAGE_BUFFER
    ev_no = (1 - p) * q - p * (1 - q) - fee_no

    if ev_yes > ev_no and ev_yes > 0:
        return "buy_yes", ev_yes, fee_yes
    elif ev_no > 0:
        return "buy_no", ev_no, fee_no
    return None, 0.0, fee_yes if ev_yes >= ev_no else fee_no


async def scan_llm_forecasts(
    session: AsyncSession,
    max_markets: int = MAX_MARKETS_PER_SCAN,
    min_volume: float = MIN_VOLUME_TOTAL,
) -> list[dict]:
    """Scan active markets and generate LLM forecasts for those most likely mispriced.

    Prioritizes:
    1. Markets with high volume (more liquid = more tradeable)
    2. Non-sports markets (sports covered by Elo; LLM adds most value elsewhere)
    3. Markets not recently forecasted
    """
    if not settings.anthropic_api_key:
        logger.info("LLM forecaster: no API key configured, skipping")
        return []

    # Get candidate markets
    result = await session.execute(
        select(Market).where(
            Market.is_active == True,  # noqa: E711
            Market.price_yes != None,  # noqa: E711
            Market.volume_total >= min_volume,
        ).order_by(Market.volume_24h.desc()).limit(200)
    )
    markets = result.scalars().all()

    if not markets:
        return []

    # Filter out recently forecasted markets
    recent_cutoff = datetime.utcnow() - timedelta(hours=CACHE_HOURS)
    recent_forecasts = await session.execute(
        select(AIAnalysis.market_id).where(
            AIAnalysis.analysis_type == "llm_forecast",
            AIAnalysis.created_at > recent_cutoff,
        )
    )
    recently_done = {row[0] for row in recent_forecasts.all()}

    candidates = [
        m for m in markets
        if m.id not in recently_done
        and 0.05 < (m.price_yes or 0.5) < 0.95  # Skip near-resolved
    ]

    # Prioritize non-sports (LLM adds most value on politics, crypto, science)
    non_sports = [m for m in candidates if (m.normalized_category or "") != "sports"]
    sports = [m for m in candidates if (m.normalized_category or "") == "sports"]
    ordered = non_sports + sports

    edges_found = []
    cost_total = 0.0

    for market in ordered[:max_markets]:
        signal = await forecast_market(market, session)
        if not signal:
            continue

        cost_total += COST_PER_CALL_USD

        if (signal.get("direction")
                and signal.get("net_ev", 0) >= MIN_NET_EDGE
                and signal.get("confidence", 0) >= 0.4):
            edges_found.append(signal)
            logger.info(
                f"LLM edge: {market.question[:60]}... | "
                f"LLM: {signal['llm_probability']:.1%} vs Market: {signal['market_price']:.1%} | "
                f"Net EV: {signal['net_ev']:.1%}"
            )

    await session.commit()

    logger.info(
        f"LLM forecast scan: {len(ordered[:max_markets])} markets analyzed, "
        f"{len(edges_found)} edges found, cost: ${cost_total:.2f}"
    )
    return edges_found
