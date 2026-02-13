"""Test if Kalshi API has resolved markets with outcomes."""
import asyncio
import httpx

async def test_kalshi_resolved():
    url = "https://api.elections.kalshi.com/trade-api/v2/markets"
    params = {
        "status": "settled",  # settled = resolved
        "limit": 5,
    }

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

    markets = data.get("markets", [])
    print(f"Found {len(markets)} settled Kalshi markets\n")

    for m in markets[:2]:
        ticker = m.get("ticker")
        question = m.get("subtitle", "")[:60]
        status = m.get("status")
        result = m.get("result")  # Should have outcome!
        result_price = m.get("result_price")

        print(f"Ticker: {ticker}")
        print(f"Question: {question}...")
        print(f"Status: {status}")
        print(f"Result: {result}")
        print(f"Result Price: {result_price}")
        print()

asyncio.run(test_kalshi_resolved())
