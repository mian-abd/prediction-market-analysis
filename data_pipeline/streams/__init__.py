"""Real-time streaming data pipeline for ultra-low-latency arbitrage detection.

This module provides WebSocket-based streaming for Polymarket and Kalshi,
with Redis-backed price caching for sub-100ms arbitrage signal generation.

Components:
- PriceCache: <1ms in-memory price lookups with Redis
- PolymarketStream: Real-time Polymarket orderbook updates via WebSocket
- (Future) KalshiStream: Real-time Kalshi orderbook updates via WebSocket

Performance Targets:
- Price update latency: <20ms (vs 20-60s polling)
- Arbitrage detection: <50ms (vs 5-10min batch processing)
- End-to-end signal: <100ms (vs 5-10min)

Example:
    >>> from data_pipeline.streams import PriceCache, PolymarketStream
    >>>
    >>> # Initialize components
    >>> cache = PriceCache()
    >>> await cache.connect()
    >>>
    >>> stream = PolymarketStream(cache)
    >>> await stream.connect()
    >>> await stream.subscribe_markets(["market_id_1", "market_id_2"])
    >>>
    >>> # Start streaming (runs forever)
    >>> await stream.stream()
"""

from .price_cache import PriceCache
from .polymarket_ws import PolymarketStream

__all__ = [
    "PriceCache",
    "PolymarketStream",
]

__version__ = "0.1.0"
