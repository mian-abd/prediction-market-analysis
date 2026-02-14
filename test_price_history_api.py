"""Test if Polymarket CLOB price history API works for active vs resolved markets."""
import sys
sys.path.insert(0, '.')

import asyncio
from data_pipeline.collectors.polymarket_clob import fetch_price_history
from db.database import init_db, async_session
from db.models import Market
from sqlalchemy import select


async def test():
    await init_db()
    
    async with async_session() as session:
        # Test 3 active markets
        result = await session.execute(
            select(Market).where(Market.is_active == True).limit(3)
        )
        active_markets = result.scalars().all()
        
        print(f"\n=== Testing {len(active_markets)} ACTIVE markets ===")
        for m in active_markets:
            print(f"\nMarket {m.id}: {m.question[:70]}...")
            print(f"  token_id_yes: {m.token_id_yes}")
            
            history = await fetch_price_history(m.token_id_yes, interval='1d', fidelity=3600)
            if history:
                print(f"  ✓ SUCCESS: {len(history)} data points")
                print(f"  Sample: {history[0]}")
                return True
            else:
                print(f"  ✗ FAILED: No data returned")
        
        return False


if __name__ == "__main__":
    success = asyncio.run(test())
    if success:
        print("\n✓ API WORKS: Can backfill active markets (then wait for resolution)")
    else:
        print("\n✗ API BROKEN: prices-history endpoint not working")
