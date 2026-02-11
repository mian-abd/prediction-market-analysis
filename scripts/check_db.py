"""Quick script to check database state."""
import asyncio
import sys
sys.path.insert(0, ".")

async def main():
    from db.database import async_session
    from sqlalchemy import select, func
    from db.models import Market, Platform, PriceSnapshot

    async with async_session() as session:
        platforms = (await session.execute(select(Platform))).scalars().all()
        for p in platforms:
            count = (await session.execute(
                select(func.count(Market.id)).where(
                    Market.platform_id == p.id, Market.is_active == True
                )
            )).scalar()
            print(f"{p.name}: {count} active markets")

        # Top Polymarket markets
        poly = (await session.execute(
            select(Platform).where(Platform.name == "polymarket")
        )).scalar()
        if poly:
            top = (await session.execute(
                select(Market)
                .where(Market.platform_id == poly.id, Market.is_active == True, Market.price_yes != None)
                .order_by(Market.volume_24h.desc())
                .limit(5)
            )).scalars().all()
            print("\nTop 5 Polymarket markets by 24h volume:")
            for m in top:
                py = m.price_yes or 0
                pn = m.price_no or 0
                v = m.volume_24h or 0
                print(f"  YES={py:.4f} NO={pn:.4f} vol24h=${v:,.0f} | {m.question[:65]}")

        snap_count = (await session.execute(select(func.count(PriceSnapshot.id)))).scalar()
        print(f"\nTotal price snapshots: {snap_count}")

if __name__ == "__main__":
    asyncio.run(main())
