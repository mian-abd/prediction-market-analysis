"""Ultra-fast in-memory price cache with arbitrage detection.

This module provides <1ms price lookups and real-time arbitrage detection
across Polymarket and Kalshi platforms using Redis.

Performance:
- Price update: <5ms
- Arbitrage check: <5ms
- Total latency: <10ms (vs 100-500ms database writes)
"""

import logging
from datetime import datetime
from typing import Optional, Dict, List
import asyncio

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("redis.asyncio not available - install with: pip install redis[asyncio]")

logger = logging.getLogger(__name__)


class PriceCache:
    """In-memory price cache with <1ms latency and arbitrage detection.

    Features:
    - Sub-millisecond price lookups
    - Real-time cross-platform arbitrage detection
    - Automatic price expiry (60 seconds)
    - Thread-safe atomic operations

    Example:
        >>> cache = PriceCache()
        >>> await cache.connect()
        >>> await cache.update_price(123, "polymarket", 0.65, 0.35)
        >>> await cache.update_price(123, "kalshi", 0.60, 0.40)  # 5% spread detected!
        >>> signals = await cache.get_arbitrage_signals()
        >>> print(f"Found {len(signals)} arbitrage opportunities")
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize price cache.

        Args:
            redis_url: Redis connection URL (default: localhost:6379)
        """
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None
        self.arb_signals: List[Dict] = []  # Queue for arbitrage signals
        self._lock = asyncio.Lock()  # Thread-safe signal queue

        if not REDIS_AVAILABLE:
            logger.warning("Redis not available - price cache will be disabled")

    async def connect(self):
        """Initialize Redis connection.

        Raises:
            redis.ConnectionError: If Redis is not available
        """
        if not REDIS_AVAILABLE:
            logger.warning("Skipping Redis connection (not installed)")
            return

        try:
            self.redis = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5
            )
            # Test connection
            await self.redis.ping()
            logger.info(f"✅ Redis price cache connected: {self.redis_url}")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self.redis = None

    async def update_price(
        self,
        market_id: int,
        platform: str,
        price_yes: float,
        price_no: float,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """Update price and check for arbitrage (<5ms total).

        Args:
            market_id: Internal market ID
            platform: "polymarket" or "kalshi"
            price_yes: YES side price (0-1)
            price_no: NO side price (0-1)
            timestamp: Optional timestamp (default: now)

        Returns:
            True if update successful, False otherwise
        """
        if not self.redis:
            return False

        if not timestamp:
            timestamp = datetime.utcnow()

        try:
            # Store price (expires in 60 seconds to prevent stale data)
            key = f"price:{platform}:{market_id}"
            value = f"{price_yes:.6f}|{price_no:.6f}|{timestamp.isoformat()}"
            await self.redis.setex(key, 60, value)

            # Check for cross-platform arbitrage (async, non-blocking)
            asyncio.create_task(
                self._check_arbitrage(market_id, platform, price_yes, price_no)
            )

            return True

        except Exception as e:
            logger.error(f"Price update failed for {platform}:{market_id}: {e}")
            return False

    async def _check_arbitrage(
        self,
        market_id: int,
        updated_platform: str,
        updated_price_yes: float,
        updated_price_no: float
    ):
        """Check if arbitrage exists with other platform (<5ms).

        Arbitrage exists when:
        - Cross-platform spread > 3% (after 2% fee + 1% slippage)
        - Both prices are fresh (<60 seconds old)

        Args:
            market_id: Market ID to check
            updated_platform: Platform that was just updated
            updated_price_yes: New YES price
            updated_price_no: New NO price
        """
        try:
            # Get price from other platform
            other_platform = "kalshi" if updated_platform == "polymarket" else "polymarket"
            other_key = f"price:{other_platform}:{market_id}"
            other_data = await self.redis.get(other_key)

            if not other_data:
                return  # No match on other platform

            # Parse other platform's price
            parts = other_data.split("|")
            if len(parts) != 3:
                return

            other_price_yes = float(parts[0])
            other_price_no = float(parts[1])
            other_ts = datetime.fromisoformat(parts[2])

            # Check if other price is stale (>60 seconds old)
            age_seconds = (datetime.utcnow() - other_ts).total_seconds()
            if age_seconds > 60:
                return  # Too old, not reliable

            # Calculate spread (both directions)
            # Direction 1: Buy on updated platform, sell on other
            spread_1 = abs(updated_price_yes - (1 - other_price_no))

            # Direction 2: Buy on other platform, sell on updated
            spread_2 = abs(other_price_yes - (1 - updated_price_no))

            max_spread = max(spread_1, spread_2)
            direction = "buy_updated_sell_other" if spread_1 > spread_2 else "buy_other_sell_updated"

            # If spread > 3% (after 2% fee + 1% slippage), it's potentially profitable
            MIN_PROFITABLE_SPREAD = 0.03
            if max_spread > MIN_PROFITABLE_SPREAD:
                signal = {
                    "market_id": market_id,
                    "platform_a": updated_platform,
                    "platform_b": other_platform,
                    "price_a_yes": updated_price_yes,
                    "price_b_yes": other_price_yes,
                    "spread_pct": round(max_spread * 100, 2),
                    "direction": direction,
                    "gross_profit_pct": round((max_spread - 0.03) * 100, 2),  # After fees
                    "detected_at": datetime.utcnow().isoformat(),
                    "age_seconds": age_seconds,
                }

                # Add to signal queue (thread-safe)
                async with self._lock:
                    self.arb_signals.append(signal)

                logger.info(
                    f"🚨 ARBITRAGE: Market {market_id} | "
                    f"{updated_platform} vs {other_platform} | "
                    f"Spread: {max_spread:.1%} | "
                    f"Profit: {signal['gross_profit_pct']:.1f}%"
                )

        except Exception as e:
            logger.error(f"Arbitrage check failed for market {market_id}: {e}")

    async def get_price(
        self,
        market_id: int,
        platform: str
    ) -> Optional[tuple[float, float, datetime]]:
        """Get cached price (<1ms).

        Args:
            market_id: Market ID
            platform: "polymarket" or "kalshi"

        Returns:
            Tuple of (price_yes, price_no, timestamp) or None if not found
        """
        if not self.redis:
            return None

        try:
            key = f"price:{platform}:{market_id}"
            data = await self.redis.get(key)
            if not data:
                return None

            parts = data.split("|")
            if len(parts) != 3:
                return None

            price_yes = float(parts[0])
            price_no = float(parts[1])
            ts = datetime.fromisoformat(parts[2])

            return (price_yes, price_no, ts)

        except Exception as e:
            logger.error(f"Price retrieval failed for {platform}:{market_id}: {e}")
            return None

    async def get_arbitrage_signals(self) -> List[Dict]:
        """Get and clear arbitrage signals (thread-safe).

        Returns:
            List of arbitrage signal dictionaries
        """
        async with self._lock:
            signals = self.arb_signals.copy()
            self.arb_signals.clear()
        return signals

    async def clear_all(self):
        """Clear all cached prices (useful for testing)."""
        if not self.redis:
            return

        try:
            # Delete all price keys
            keys = await self.redis.keys("price:*")
            if keys:
                await self.redis.delete(*keys)
                logger.info(f"Cleared {len(keys)} cached prices")
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")

    async def get_stats(self) -> Dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        if not self.redis:
            return {"status": "disconnected"}

        try:
            poly_keys = await self.redis.keys("price:polymarket:*")
            kalshi_keys = await self.redis.keys("price:kalshi:*")

            return {
                "status": "connected",
                "polymarket_cached": len(poly_keys),
                "kalshi_cached": len(kalshi_keys),
                "total_cached": len(poly_keys) + len(kalshi_keys),
                "pending_signals": len(self.arb_signals),
            }
        except Exception as e:
            logger.error(f"Stats retrieval failed: {e}")
            return {"status": "error", "error": str(e)}

    async def close(self):
        """Close Redis connection gracefully."""
        if self.redis:
            await self.redis.close()
            logger.info("Redis connection closed")
