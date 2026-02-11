"""Analyze market prices to understand arbitrage landscape."""
import asyncio
import sys
sys.path.insert(0, ".")

async def main():
    from db.database import async_session
    from sqlalchemy import select
    from db.models import Market

    async with async_session() as session:
        result = await session.execute(
            select(Market).where(
                Market.is_active == True,
                Market.price_yes != None,
                Market.price_no != None,
                Market.price_yes > 0,
                Market.price_no > 0,
            )
        )
        markets = result.scalars().all()

        print(f"Markets with valid YES+NO prices: {len(markets)}\n")

        # Analyze YES + NO distribution
        totals = []
        under_one = []
        over_one = []

        for m in markets:
            total = m.price_yes + m.price_no
            totals.append((total, m))
            if total < 1.0:
                under_one.append((total, m))
            elif total > 1.0:
                over_one.append((total, m))

        print(f"YES + NO < 1.00 (potential arb): {len(under_one)}")
        print(f"YES + NO = 1.00 (fair): {len(markets) - len(under_one) - len(over_one)}")
        print(f"YES + NO > 1.00 (overpriced): {len(over_one)}")

        # Show closest to arb (smallest totals)
        under_one.sort(key=lambda x: x[0])
        print(f"\nTop 10 closest to arbitrage (smallest YES+NO):")
        for total, m in under_one[:10]:
            profit_gross = (1.0 - total) / total * 100
            print(f"  {total:.4f} (gross {profit_gross:.2f}%) | YES={m.price_yes:.4f} NO={m.price_no:.4f} | {m.question[:55]}")

        # Show overpriced markets
        over_one.sort(key=lambda x: x[0], reverse=True)
        print(f"\nTop 10 most overpriced (largest YES+NO):")
        for total, m in over_one[:10]:
            overprice = (total - 1.0) * 100
            print(f"  {total:.4f} (+{overprice:.2f}%) | YES={m.price_yes:.4f} NO={m.price_no:.4f} | {m.question[:55]}")

        # Statistical summary
        import statistics
        vals = [t for t, _ in totals]
        print(f"\nStatistics (YES+NO total):")
        print(f"  Mean: {statistics.mean(vals):.4f}")
        print(f"  Median: {statistics.median(vals):.4f}")
        print(f"  Stdev: {statistics.stdev(vals):.4f}")
        print(f"  Min: {min(vals):.4f}")
        print(f"  Max: {max(vals):.4f}")

if __name__ == "__main__":
    asyncio.run(main())
