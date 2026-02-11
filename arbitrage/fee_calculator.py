"""Platform-specific fee computation.
Every arbitrage opportunity MUST pass through fee validation."""


class FeeCalculator:

    @staticmethod
    def polymarket_standard(entry_cost: float, payout: float = 1.0) -> float:
        """Polymarket standard markets:
        - 0% taker/maker fee
        - 2% on NET WINNINGS (payout - cost), only if you win
        """
        if payout <= entry_cost:
            return 0.0
        net_winnings = payout - entry_cost
        return net_winnings * 0.02

    @staticmethod
    def polymarket_15min_crypto(amount: float, price: float) -> float:
        """15-minute crypto markets have price-dependent taker fees.
        Fee peaks around 50% price (up to ~1.56% effective).
        We flag these and generally SKIP them for arbitrage.
        """
        # Fee formula: fee_rate = 2 * price * (1 - price) * base_rate
        # base_rate is ~3.15% max
        fee_rate = 2 * price * (1 - price) * 0.0315
        return amount * fee_rate

    @staticmethod
    def kalshi_fee(num_contracts: int, price: float) -> float:
        """Kalshi fee: min($0.01/contract, price * 0.07).
        Typically ~0.7% average.
        """
        fee_per = min(0.01, price * 0.07)
        return num_contracts * fee_per

    @staticmethod
    def single_market_arb_fees(
        yes_price: float,
        no_price: float,
        platform: str = "polymarket",
        quantity: float = 1.0,
    ) -> dict:
        """Calculate fees for single-market rebalancing arb.
        Buy YES + NO, guaranteed $1 payout.
        """
        entry_cost = (yes_price + no_price) * quantity
        payout = 1.0 * quantity
        gross_profit = payout - entry_cost

        if platform == "polymarket":
            # 2% on net winnings (guaranteed to win since we hold both sides)
            fee = FeeCalculator.polymarket_standard(entry_cost, payout)
        elif platform == "kalshi":
            # Kalshi charges per contract on both legs
            fee = FeeCalculator.kalshi_fee(quantity, yes_price) + FeeCalculator.kalshi_fee(quantity, no_price)
        else:
            fee = 0.0

        net_profit = gross_profit - fee
        net_pct = (net_profit / entry_cost * 100) if entry_cost > 0 else 0

        return {
            "entry_cost": entry_cost,
            "payout": payout,
            "gross_profit": gross_profit,
            "gross_pct": (gross_profit / entry_cost * 100) if entry_cost > 0 else 0,
            "fees": fee,
            "net_profit": net_profit,
            "net_pct": net_pct,
            "profitable": net_profit > 0,
        }

    @staticmethod
    def cross_platform_arb_fees(
        price_buy: float,
        price_sell_complement: float,
        platform_buy: str,
        platform_sell: str,
        quantity: float = 1.0,
    ) -> dict:
        """Calculate fees for cross-platform arb.
        Buy YES on cheap platform, buy NO on expensive platform.
        One side guaranteed to pay $1.
        """
        entry_cost = (price_buy + price_sell_complement) * quantity
        payout = 1.0 * quantity
        gross_profit = payout - entry_cost

        # Fees on both platforms
        fee_buy = 0.0
        fee_sell = 0.0

        if platform_buy == "polymarket":
            # Winning side pays 2% on winnings
            fee_buy = FeeCalculator.polymarket_standard(price_buy * quantity, 1.0 * quantity)
        elif platform_buy == "kalshi":
            fee_buy = FeeCalculator.kalshi_fee(quantity, price_buy)

        if platform_sell == "polymarket":
            fee_sell = FeeCalculator.polymarket_standard(price_sell_complement * quantity, 1.0 * quantity)
        elif platform_sell == "kalshi":
            fee_sell = FeeCalculator.kalshi_fee(quantity, price_sell_complement)

        total_fees = fee_buy + fee_sell
        net_profit = gross_profit - total_fees
        net_pct = (net_profit / entry_cost * 100) if entry_cost > 0 else 0

        return {
            "entry_cost": entry_cost,
            "payout": payout,
            "gross_profit": gross_profit,
            "gross_pct": (gross_profit / entry_cost * 100) if entry_cost > 0 else 0,
            "fee_buy": fee_buy,
            "fee_sell": fee_sell,
            "total_fees": total_fees,
            "net_profit": net_profit,
            "net_pct": net_pct,
            "profitable": net_profit > 0,
        }
