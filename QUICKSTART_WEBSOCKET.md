# Quick Start: WebSocket Streaming Implementation

## Goal
Transform from 20-60s polling → <20ms real-time streaming for arbitrage detection.

**Expected Impact**: Capture arbitrage opportunities currently being missed (100% miss rate → 50%+ capture rate)

---

## Step 1: Install Dependencies (5 minutes)

```bash
# In your virtual environment
pip install redis websockets aioredis msgpack

# Start Redis locally (Windows)
# Download from: https://github.com/microsoftarchive/redis/releases
# Or use WSL: wsl -d Ubuntu sudo service redis-server start

# Verify Redis is running
redis-cli ping
# Should return: PONG
```

---

## Step 2: Create Redis Price Cache (30 minutes)

**File**: `data_pipeline/streams/price_cache.py`

```python
"""Ultra-fast in-memory price cache with arbitrage detection."""
import redis.asyncio as redis
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

class PriceCache:
    """In-memory price cache with <1ms latency."""

    def __init__(self):
        self.redis = None
        self.arb_signals = []  # Queue for arbitrage signals

    async def connect(self):
        """Initialize Redis connection."""
        self.redis = await redis.from_url(
            "redis://localhost:6379",
            encoding="utf-8",
            decode_responses=True
        )
        logger.info("Redis price cache connected")

    async def update_price(
        self,
        market_id: int,
        platform: str,
        price_yes: float,
        price_no: float,
        timestamp: datetime = None
    ):
        """Update price and check for arbitrage (<5ms total)."""
        if not timestamp:
            timestamp = datetime.utcnow()

        # Store price (expires in 60 seconds)
        key = f"price:{platform}:{market_id}"
        value = f"{price_yes}|{price_no}|{timestamp.isoformat()}"
        await self.redis.setex(key, 60, value)

        # Check for cross-platform arbitrage
        await self._check_arbitrage(market_id, platform, price_yes, price_no)

    async def _check_arbitrage(
        self,
        market_id: int,
        updated_platform: str,
        updated_price_yes: float,
        updated_price_no: float
    ):
        """Check if arbitrage exists with other platform (<5ms)."""
        # Get price from other platform
        other_platform = "kalshi" if updated_platform == "polymarket" else "polymarket"
        other_key = f"price:{other_platform}:{market_id}"
        other_data = await self.redis.get(other_key)

        if not other_data:
            return  # No match on other platform

        # Parse other platform's price
        other_price_yes, other_price_no, other_ts = other_data.split("|")
        other_price_yes = float(other_price_yes)
        other_price_no = float(other_price_no)

        # Calculate spread (both directions)
        spread_buy_poly_sell_kalshi = abs(updated_price_yes - (1 - other_price_no))
        spread_buy_kalshi_sell_poly = abs(other_price_yes - (1 - updated_price_no))
        max_spread = max(spread_buy_poly_sell_kalshi, spread_buy_kalshi_sell_poly)

        # If spread > 3% (after 2% fee + 1% slippage), it's profitable
        if max_spread > 0.03:
            signal = {
                "market_id": market_id,
                "poly_price_yes": updated_price_yes if updated_platform == "polymarket" else other_price_yes,
                "kalshi_price_yes": other_price_yes if updated_platform == "polymarket" else updated_price_yes,
                "spread": max_spread,
                "detected_at": datetime.utcnow().isoformat(),
            }
            self.arb_signals.append(signal)
            logger.info(f"🚨 ARBITRAGE DETECTED: Market {market_id}, spread {max_spread:.1%}")

    async def get_price(self, market_id: int, platform: str) -> Optional[tuple]:
        """Get cached price (<1ms)."""
        key = f"price:{platform}:{market_id}"
        data = await self.redis.get(key)
        if not data:
            return None

        price_yes, price_no, ts = data.split("|")
        return float(price_yes), float(price_no), datetime.fromisoformat(ts)

    async def get_arbitrage_signals(self) -> list:
        """Get and clear arbitrage signals."""
        signals = self.arb_signals.copy()
        self.arb_signals.clear()
        return signals
```

**Test it**:
```python
import asyncio

async def test_cache():
    cache = PriceCache()
    await cache.connect()

    # Simulate price update
    await cache.update_price(123, "polymarket", 0.65, 0.35)
    await cache.update_price(123, "kalshi", 0.60, 0.40)  # 5% spread!

    # Check for signals
    signals = await cache.get_arbitrage_signals()
    print(f"Detected {len(signals)} arbitrage signals")
    print(signals)

asyncio.run(test_cache())
```

---

## Step 3: Polymarket WebSocket Stream (2 hours)

**File**: `data_pipeline/streams/polymarket_ws.py`

```python
"""Polymarket WebSocket stream for real-time price updates."""
import asyncio
import websockets
import json
import logging
from typing import Set
from .price_cache import PriceCache

logger = logging.getLogger(__name__)

class PolymarketStream:
    """WebSocket connection to Polymarket CLOB for real-time orderbook updates."""

    def __init__(self, price_cache: PriceCache):
        self.ws = None
        self.price_cache = price_cache
        self.subscribed_markets: Set[str] = set()
        self.running = False

    async def connect(self, api_key: str = None):
        """Connect to Polymarket WebSocket."""
        # Note: Free tier has rate limits, premium ($99/mo) recommended for production
        uri = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self.ws = await websockets.connect(uri, extra_headers=headers)
        logger.info("Connected to Polymarket WebSocket")
        self.running = True

    async def subscribe_markets(self, market_ids: list[str]):
        """Subscribe to orderbook updates for specific markets."""
        for market_id in market_ids:
            if market_id not in self.subscribed_markets:
                await self.ws.send(json.dumps({
                    "type": "subscribe",
                    "market": market_id,
                    "assets_ids": [market_id]  # Subscribe to this market
                }))
                self.subscribed_markets.add(market_id)
                logger.info(f"Subscribed to Polymarket market {market_id}")

    async def stream(self):
        """Main streaming loop - process incoming messages."""
        try:
            async for message in self.ws:
                try:
                    data = json.loads(message)
                    await self._process_message(data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON: {message[:100]}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed, reconnecting...")
            await asyncio.sleep(5)
            await self.reconnect()

    async def _process_message(self, data: dict):
        """Process orderbook update message (<10ms)."""
        if data.get("type") != "book":
            return  # Ignore non-orderbook messages

        market_id = data.get("market")
        if not market_id:
            return

        # Extract best bid/ask
        bids = data.get("bids", [])
        asks = data.get("asks", [])

        if not bids or not asks:
            return

        # Best bid = highest buy price, Best ask = lowest sell price
        best_bid = float(bids[0]["price"]) if bids else 0.0
        best_ask = float(asks[0]["price"]) if asks else 1.0

        # Mid-price approximation
        price_yes = (best_bid + best_ask) / 2
        price_no = 1.0 - price_yes

        # Update cache (triggers arbitrage check)
        await self.price_cache.update_price(
            market_id=int(market_id),
            platform="polymarket",
            price_yes=price_yes,
            price_no=price_no
        )

    async def reconnect(self):
        """Reconnect with exponential backoff."""
        retry_delay = 5
        while not self.running:
            try:
                await self.connect()
                # Re-subscribe to all markets
                await self.subscribe_markets(list(self.subscribed_markets))
                logger.info("Reconnected and resubscribed")
                break
            except Exception as e:
                logger.error(f"Reconnect failed: {e}, retrying in {retry_delay}s")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)  # Cap at 60s

    async def close(self):
        """Close WebSocket connection."""
        self.running = False
        if self.ws:
            await self.ws.close()
```

**Test it**:
```python
async def test_stream():
    cache = PriceCache()
    await cache.connect()

    stream = PolymarketStream(cache)
    await stream.connect()

    # Subscribe to a few markets
    await stream.subscribe_markets(["12345", "67890"])  # Replace with real market IDs

    # Stream for 60 seconds
    await asyncio.wait_for(stream.stream(), timeout=60)

asyncio.run(test_stream())
```

---

## Step 4: Integration with Scheduler (1 hour)

**File**: `data_pipeline/scheduler.py` (add new function)

```python
# Add to imports
from data_pipeline.streams.price_cache import PriceCache
from data_pipeline.streams.polymarket_ws import PolymarketStream

# Add global instances
price_cache = None
poly_stream = None

async def init_realtime_streams():
    """Initialize WebSocket streams for real-time arbitrage detection."""
    global price_cache, poly_stream

    logger.info("Initializing real-time price streams...")

    # 1. Start Redis cache
    price_cache = PriceCache()
    await price_cache.connect()

    # 2. Start Polymarket stream
    poly_stream = PolymarketStream(price_cache)
    await poly_stream.connect()

    # 3. Subscribe to top 100 markets by volume
    async with async_session() as session:
        result = await session.execute(
            select(Market.token_id_yes)
            .where(Market.platform == "polymarket")
            .order_by(Market.volume_total.desc())
            .limit(100)
        )
        top_markets = [row[0] for row in result.all() if row[0]]

    await poly_stream.subscribe_markets(top_markets)
    logger.info(f"Subscribed to {len(top_markets)} top Polymarket markets")

    # 4. Start streaming task (runs in background)
    asyncio.create_task(poly_stream.stream())
    logger.info("Real-time streams active")

async def check_arbitrage_signals():
    """Check for new arbitrage signals from stream (run every 5 seconds)."""
    if not price_cache:
        return

    signals = await price_cache.get_arbitrage_signals()
    if signals:
        logger.info(f"🚨 {len(signals)} arbitrage signals detected!")
        for signal in signals:
            # Log to database or execute trade
            logger.info(f"  Market {signal['market_id']}: {signal['spread']:.1%} spread")
            # TODO: Execute arbitrage trade
```

**Add to lifespan**:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... existing startup code ...

    # NEW: Start real-time streams
    await init_realtime_streams()

    # NEW: Add arbitrage check to pipeline
    async def arbitrage_loop():
        while True:
            await check_arbitrage_signals()
            await asyncio.sleep(5)  # Check every 5 seconds

    arbitrage_task = asyncio.create_task(arbitrage_loop())

    yield

    # ... existing shutdown code ...
    arbitrage_task.cancel()
```

---

## Step 5: Monitor & Validate (30 minutes)

**Check WebSocket is working**:
```bash
# In production logs, you should see:
grep "Subscribed to Polymarket market" logs/app.log
grep "ARBITRAGE DETECTED" logs/app.log

# Check Redis cache
redis-cli
> KEYS price:*
> GET price:polymarket:123
```

**Performance metrics to track**:
1. WebSocket message rate (should be 10-100 messages/second)
2. Cache update latency (should be <5ms)
3. Arbitrage signals detected (expect 5-20 per day)
4. False positive rate (should be <10%)

---

## Expected Results (48 hours)

✅ **WebSocket connected**: Real-time price updates every 100-500ms
✅ **Redis cache**: <1ms price lookups (vs 100-500ms database)
✅ **First arbitrage signal**: Within 24-48 hours
✅ **Latency improvement**: 20-60s → 20ms (120× faster)

**ROI**: Capture 1 arbitrage trade = $5-20 profit → pays for 1 month of Redis + VPS

---

## Next Steps

After Phase 1 is working:
1. **Add Kalshi WebSocket** (similar to Polymarket, 2 hours)
2. **Deploy to NYC VPS** (8-12ms latency vs 50-200ms home, $48/month)
3. **Add pre-signed order pool** (10ms execution vs 100ms, 4 hours)
4. **Enable auto-execution** (currently just detection, 2 hours)

**Total to production-ready arbitrage bot**: ~15 hours over 2 weeks
