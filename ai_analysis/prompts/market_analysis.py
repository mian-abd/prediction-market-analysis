"""Prompt templates for Claude market analysis."""

MARKET_ANALYSIS_SYSTEM = """You are a quantitative prediction market analyst. You combine:
1. Probabilistic reasoning (base rates, Bayesian updating)
2. Current events knowledge (via web search)
3. Market microstructure understanding (liquidity, volume, spreads)

Your job is to analyze a prediction market and provide:
- Your independent probability estimate
- Whether the market appears mispriced
- Key factors driving the outcome
- Risk assessment

Be rigorous. Show your reasoning. Cite sources when using web search.

OUTPUT FORMAT (respond with valid JSON):
{
    "ai_probability": 0.XX,
    "market_price": 0.XX,
    "calibrated_price": 0.XX,
    "delta": 0.XX,
    "direction": "OVERPRICED" | "UNDERPRICED" | "FAIR",
    "confidence": "LOW" | "MEDIUM" | "HIGH",
    "key_factors": [
        {"factor": "description", "direction": "bullish|bearish|neutral", "weight": "high|medium|low"}
    ],
    "reasoning": "2-3 paragraph analysis",
    "risk_factors": ["risk 1", "risk 2"],
    "recommendation": "BUY_YES" | "BUY_NO" | "HOLD" | "AVOID",
    "edge_estimate_pct": 0.0
}"""

MARKET_ANALYSIS_USER = """Analyze this prediction market:

**Question:** {question}
**Description:** {description}
**Category:** {category}

**Current Pricing:**
- Market YES price: {price_yes:.4f} ({price_yes_pct:.1f}%)
- Market NO price: {price_no:.4f} ({price_no_pct:.1f}%)
- ML Calibrated probability: {calibrated_price:.4f} ({calibrated_pct:.1f}%)
- Calibration bias: {calibration_bias:+.1f} percentage points

**Market Stats:**
- 24h Volume: ${volume_24h:,.0f}
- Total Volume: ${volume_total:,.0f}
- Liquidity: ${liquidity:,.0f}
- End Date: {end_date}
- Time to Resolution: {time_to_resolution}

**Recent Price History:**
{price_history_summary}

Please search the web for the latest information about this topic, then provide your analysis."""


def format_analysis_prompt(market: dict, calibration: dict, price_history: list) -> tuple[str, str]:
    """Format the analysis prompt with market data."""
    # Price history summary
    if price_history:
        prices = [p["price_yes"] for p in price_history[-10:]]
        price_summary = f"Last 10 prices: {', '.join(f'{p:.4f}' for p in prices)}"
        if len(prices) >= 2:
            trend = "trending UP" if prices[-1] > prices[0] else "trending DOWN" if prices[-1] < prices[0] else "stable"
            price_summary += f" ({trend})"
    else:
        price_summary = "No recent price history available"

    # Time to resolution
    end_date = market.get("end_date", "Unknown")
    time_str = market.get("time_to_resolution", "Unknown")

    user_prompt = MARKET_ANALYSIS_USER.format(
        question=market.get("question", "Unknown"),
        description=market.get("description", "No description")[:500],
        category=market.get("category", "Unknown"),
        price_yes=market.get("price_yes", 0),
        price_yes_pct=market.get("price_yes", 0) * 100,
        price_no=market.get("price_no", 0),
        price_no_pct=market.get("price_no", 0) * 100,
        calibrated_price=calibration.get("calibrated_price", 0),
        calibrated_pct=calibration.get("calibrated_price", 0) * 100,
        calibration_bias=calibration.get("delta_pct", 0),
        volume_24h=market.get("volume_24h", 0),
        volume_total=market.get("volume_total", 0),
        liquidity=market.get("liquidity", 0),
        end_date=end_date,
        time_to_resolution=time_str,
        price_history_summary=price_summary,
    )

    return MARKET_ANALYSIS_SYSTEM, user_prompt
