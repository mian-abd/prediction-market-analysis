"""Test arbitrage scanner on real data."""
import asyncio
import sys
sys.path.insert(0, ".")

async def main():
    from db.database import async_session
    from arbitrage.engine import run_full_scan, get_summary

    async with async_session() as session:
        print("Running full arbitrage scan...")
        opps = await run_full_scan(session)

        print(f"\nFound {len(opps)} total opportunities\n")

        # Show top 10
        for i, opp in enumerate(opps[:10]):
            print(f"#{i+1} [{opp['strategy_type']}] Net: {opp['net_profit_pct']:.2f}%")
            if opp['strategy_type'] == 'single_market':
                print(f"   Market: {opp['market_question'][:70]}")
                print(f"   YES={opp['yes_price']:.4f} NO={opp['no_price']:.4f} Total={opp['total_cost']:.4f}")
                print(f"   Gross: {opp['gross_spread']:.2f}% Fees: ${opp['total_fees']:.4f} Net: {opp['net_profit_pct']:.2f}%")
                print(f"   Est profit per $100: ${opp['estimated_profit_usd']:.2f}")
            elif opp['strategy_type'] == 'cross_platform':
                print(f"   Buy: {opp['buy_market_question'][:50]} on {opp['buy_platform']}")
                print(f"   Sell: {opp['sell_market_question'][:50]} on {opp['sell_platform']}")
                print(f"   Spread: {opp['raw_spread_pct']:.2f}% Similarity: {opp['similarity_score']:.2f}")
            print()

        # Summary
        summary = await get_summary(session)
        print("=== SUMMARY ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    asyncio.run(main())
