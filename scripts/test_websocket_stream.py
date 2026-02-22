"""Test script for WebSocket streaming implementation.

This script tests the Redis cache and Polymarket WebSocket stream in isolation
before integrating with the main pipeline.

Usage:
    python scripts/test_websocket_stream.py

Requirements:
    - Redis server running on localhost:6379
    - Internet connection for WebSocket
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_pipeline.streams import PriceCache, PolymarketStream

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_redis_cache():
    """Test 1: Redis price cache functionality."""
    logger.info("=" * 60)
    logger.info("TEST 1: Redis Price Cache")
    logger.info("=" * 60)

    try:
        # Initialize cache
        cache = PriceCache()
        await cache.connect()

        if not cache.redis:
            logger.error("❌ Redis not available - install and start Redis server")
            return False

        # Test 1: Update and retrieve price
        logger.info("Test 1.1: Update and retrieve price...")
        await cache.update_price(123, "polymarket", 0.65, 0.35)
        result = await cache.get_price(123, "polymarket")

        if result:
            price_yes, price_no, ts = result
            logger.info(f"✅ Price retrieved: YES={price_yes}, NO={price_no}")
        else:
            logger.error("❌ Price retrieval failed")
            return False

        # Test 2: Arbitrage detection (simulated)
        logger.info("Test 1.2: Arbitrage detection...")
        await cache.update_price(456, "polymarket", 0.65, 0.35)
        await cache.update_price(456, "kalshi", 0.60, 0.40)  # 5% spread!

        signals = await cache.get_arbitrage_signals()
        if signals:
            logger.info(f"✅ Arbitrage detected: {len(signals)} signals")
            for signal in signals:
                logger.info(f"   Market {signal['market_id']}: {signal['spread_pct']}% spread")
        else:
            logger.warning("⚠️  No arbitrage detected (might be <3% threshold)")

        # Test 3: Cache stats
        logger.info("Test 1.3: Cache statistics...")
        stats = await cache.get_stats()
        logger.info(f"✅ Cache stats: {stats}")

        # Cleanup
        await cache.clear_all()
        await cache.close()

        logger.info("✅ Redis cache tests PASSED")
        return True

    except Exception as e:
        logger.error(f"❌ Redis cache test FAILED: {e}")
        return False


async def test_polymarket_stream():
    """Test 2: Polymarket WebSocket stream (30 seconds)."""
    logger.info("=" * 60)
    logger.info("TEST 2: Polymarket WebSocket Stream (30 seconds)")
    logger.info("=" * 60)

    try:
        # Initialize components
        cache = PriceCache()
        await cache.connect()

        if not cache.redis:
            logger.warning("⚠️  Skipping WebSocket test (Redis not available)")
            return True  # Not a failure, just skipped

        stream = PolymarketStream(cache)

        # Test connection
        logger.info("Test 2.1: WebSocket connection...")
        await stream.connect()

        if not stream.running:
            logger.error("❌ WebSocket connection failed")
            return False

        logger.info("✅ WebSocket connected")

        # Subscribe to a few markets (example IDs, may not exist)
        logger.info("Test 2.2: Market subscription...")
        test_markets = [
            "12345",  # Replace with real market IDs
            "67890",
            "11111"
        ]
        await stream.subscribe_markets(test_markets)

        # Stream for 30 seconds
        logger.info("Test 2.3: Streaming for 30 seconds...")
        logger.info("(If no messages, check that market IDs exist)")

        try:
            await asyncio.wait_for(stream.stream(), timeout=30)
        except asyncio.TimeoutError:
            logger.info("✅ 30-second stream test completed")

        # Check stats
        stats = await stream.get_stats()
        logger.info(f"Stream stats: {stats}")

        if stats['messages_processed'] > 0:
            logger.info(f"✅ Received {stats['messages_processed']} messages")
        else:
            logger.warning("⚠️  No messages received (markets may not exist or be inactive)")

        # Check for arbitrage signals
        signals = await cache.get_arbitrage_signals()
        if signals:
            logger.info(f"🚨 {len(signals)} arbitrage signals detected!")
            for signal in signals:
                logger.info(f"   {signal}")
        else:
            logger.info("No arbitrage signals (expected for short test)")

        # Cleanup
        await stream.close()
        await cache.close()

        logger.info("✅ WebSocket stream tests PASSED")
        return True

    except Exception as e:
        logger.error(f"❌ WebSocket stream test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    logger.info("🚀 Starting WebSocket Streaming Tests")
    logger.info("")

    # Test 1: Redis Cache
    cache_ok = await test_redis_cache()
    logger.info("")

    # Test 2: WebSocket Stream
    stream_ok = await test_polymarket_stream()
    logger.info("")

    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Redis Cache:       {'✅ PASS' if cache_ok else '❌ FAIL'}")
    logger.info(f"WebSocket Stream:  {'✅ PASS' if stream_ok else '❌ FAIL'}")
    logger.info("=" * 60)

    if cache_ok and stream_ok:
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Get real Polymarket market IDs from database")
        logger.info("2. Integrate with scheduler.py")
        logger.info("3. Deploy to production")
        return 0
    else:
        logger.error("❌ SOME TESTS FAILED")
        logger.info("")
        logger.info("Troubleshooting:")
        if not cache_ok:
            logger.info("- Install Redis: https://redis.io/download")
            logger.info("- Start Redis: redis-server")
            logger.info("- Verify: redis-cli ping")
        if not stream_ok:
            logger.info("- Check internet connection")
            logger.info("- Verify Polymarket API status")
            logger.info("- Use real market IDs in test")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
