"""Test if Polymarket API returns resolved markets."""
import asyncio
import httpx

async def test_resolved_markets():
    url = "https://gamma-api.polymarket.com/markets"
    params = {
        "closed": "true",
        "limit": 5,
        "offset": 0,
    }

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        markets = resp.json()

    print(f"Found {len(markets)} resolved markets\n")

    if markets:
        print("First market full structure:")
        print(markets[0])
        print("\n" + "="*80 + "\n")

    for m in markets[:3]:
        question = m.get("question", "N/A")[:60]
        outcome_prices = m.get("outcomePrices", [])
        volume = m.get("volume", 0)
        closed = m.get("closed", False)
        # Check for resolution fields
        outcomes = m.get("outcomes", [])
        condition_id = m.get("conditionId")

        print(f"Question: {question}...")
        print(f"  Closed: {closed}")
        print(f"  Volume: ${float(volume) if volume else 0:,.0f}")
        print(f"  Outcome Prices: {outcome_prices}")
        print(f"  Outcomes: {outcomes}")
        print(f"  Condition ID: {condition_id}")
        print()

asyncio.run(test_resolved_markets())
