"""Claude API client wrapper with cost tracking and response caching.
User-triggered only. Every response cached in SQLite to avoid redundant calls."""

import hashlib
import json
import logging
from datetime import datetime

import anthropic
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from db.models import AIAnalysis

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (Opus 4.6 estimated)
INPUT_COST_PER_1M = 15.0
OUTPUT_COST_PER_1M = 75.0


class ClaudeClient:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.total_cost = 0.0
        self.call_count = 0

    def _hash_prompt(self, system: str, user: str) -> str:
        """SHA-256 hash of prompt for cache dedup."""
        content = f"{system}|||{user}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens / 1_000_000 * INPUT_COST_PER_1M +
                output_tokens / 1_000_000 * OUTPUT_COST_PER_1M)

    async def get_cached(
        self,
        session: AsyncSession,
        prompt_hash: str,
    ) -> AIAnalysis | None:
        """Check if we have a cached response for this prompt."""
        result = await session.execute(
            select(AIAnalysis).where(AIAnalysis.prompt_hash == prompt_hash).limit(1)
        )
        return result.scalar_one_or_none()

    async def analyze(
        self,
        session: AsyncSession,
        market_id: int,
        system_prompt: str,
        user_prompt: str,
        analysis_type: str = "deep_analysis",
        use_thinking: bool = True,
        use_web_search: bool = True,
        model: str = "claude-opus-4-6",
    ) -> dict:
        """Run Claude analysis with caching.

        Returns cached result if available, otherwise calls API.
        """
        prompt_hash = self._hash_prompt(system_prompt, user_prompt)

        # Check cache first
        cached = await self.get_cached(session, prompt_hash)
        if cached:
            logger.info(f"Cache hit for market {market_id} ({analysis_type})")
            return {
                "cached": True,
                "response_text": cached.response_text,
                "structured_result": cached.structured_result,
                "cost": 0.0,
                "created_at": cached.created_at.isoformat(),
            }

        # Build API call kwargs
        kwargs = {
            "model": model,
            "max_tokens": 16000,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }

        # Add extended thinking if requested
        if use_thinking:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": 10000,
            }

        # Add web search tool if requested
        if use_web_search:
            kwargs["tools"] = [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 5,
                }
            ]

        # Call Claude API
        logger.info(f"Calling Claude for market {market_id} ({analysis_type})...")
        try:
            response = self.client.messages.create(**kwargs)
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return {"error": str(e), "cached": False, "cost": 0.0}

        # Extract response text
        response_text = ""
        thinking_text = ""
        for block in response.content:
            if block.type == "thinking":
                thinking_text += block.thinking
            elif block.type == "text":
                response_text += block.text

        # Parse structured JSON if present
        structured_result = None
        try:
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                structured_result = json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            pass

        # Cost tracking
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = self._estimate_cost(input_tokens, output_tokens)
        self.total_cost += cost
        self.call_count += 1

        # Cache in DB
        analysis = AIAnalysis(
            market_id=market_id,
            analysis_type=analysis_type,
            prompt_hash=prompt_hash,
            prompt_text=user_prompt[:5000],  # Truncate for storage
            response_text=response_text,
            structured_result=structured_result,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=cost,
            model_used=model,
        )
        session.add(analysis)
        await session.commit()

        logger.info(
            f"Claude analysis complete. Cost: ${cost:.4f} "
            f"(in={input_tokens}, out={output_tokens})"
        )

        return {
            "cached": False,
            "response_text": response_text,
            "thinking_text": thinking_text,
            "structured_result": structured_result,
            "cost": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "created_at": datetime.utcnow().isoformat(),
        }

    def get_cost_summary(self) -> dict:
        return {
            "total_cost_usd": self.total_cost,
            "total_calls": self.call_count,
            "avg_cost_per_call": self.total_cost / self.call_count if self.call_count > 0 else 0,
        }
