import asyncio
from sqlalchemy import select
from db.database import async_session
from db.models import Market, PriceSnapshot

async def test_price_endpoint_logic():
    """Test the same logic the API endpoint uses."""
    market_id = 1
    interval = "5m"
    limit = 500

    async with async_session() as session:
        # Check if market exists
        market = await session.get(Market, market_id)
        if not market:
            print(f"Market {market_id} not found")
            return

        print(f"Market {market_id}: {market.question[:80]}...")
        print(f"Current price_yes: {market.price_yes}")

        # Fetch price snapshots (same as API)
        result = await session.execute(
            select(PriceSnapshot)
            .where(PriceSnapshot.market_id == market_id)
            .order_by(PriceSnapshot.timestamp.asc())
        )
        snapshots = result.scalars().all()

        print(f"\nTotal price snapshots for market {market_id}: {len(snapshots)}")

        if snapshots:
            print(f"First snapshot: {snapshots[0].timestamp}, price_yes={snapshots[0].price_yes:.4f}")
            print(f"Last snapshot:  {snapshots[-1].timestamp}, price_yes={snapshots[-1].price_yes:.4f}")

            # Show some sample data
            print("\nSample snapshots (first 5):")
            for s in snapshots[:5]:
                print(f"  {s.timestamp} - YES: {s.price_yes:.4f}, NO: {s.price_no:.4f}, VOL: {s.volume}")

asyncio.run(test_price_endpoint_logic())
