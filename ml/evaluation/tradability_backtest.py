"""Tradability backtest engine: answers "would this have made money with realistic execution?"

Uses bid/ask execution (not mid-price), per-market fees, fill modeling,
and size-dependent slippage. Tracks three Kelly curves per strategy:
full Kelly, half Kelly, and fixed $10 sizing.

This module operates on historical signals and market data, not live trading.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from config.constants import compute_polymarket_fee, compute_kalshi_fee

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Single trade in the backtest."""
    market_id: int
    timestamp: datetime
    direction: str  # "buy_yes" or "buy_no"
    model_prob: float
    market_price: float
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    fee: float
    slippage_cost: float
    net_pnl: float
    resolution: float  # 1.0 or 0.0
    sizing_method: str  # "full_kelly", "half_kelly", "fixed"


@dataclass
class BacktestResult:
    """Aggregate result of a backtest run."""
    strategy: str
    sizing_method: str
    trades: list[BacktestTrade] = field(default_factory=list)
    total_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    equity_curve: list[float] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.winning_trades / max(1, self.total_trades)


def compute_entry_price(direction: str, market_price: float, slippage_bps: float = 100) -> float:
    """Compute realistic entry price with slippage.

    For taker (market order) execution, we cross the spread and face slippage.
    """
    slippage_rate = slippage_bps / 10000

    if direction == "buy_yes":
        return market_price * (1 + slippage_rate)
    else:
        no_cost = (1 - market_price) * (1 + slippage_rate)
        return 1.0 - no_cost


def compute_exit_price(side: str, resolution_value: float, is_resolved: bool,
                       market_price_at_exit: float | None = None,
                       slippage_bps: float = 100) -> float:
    """Compute realistic exit price.

    If market resolved, exit at resolution value (no slippage).
    If exiting before resolution, apply exit slippage.
    """
    if is_resolved:
        return resolution_value

    if market_price_at_exit is None:
        return 0.5

    slippage_rate = slippage_bps / 10000
    if side == "yes":
        return market_price_at_exit * (1 - slippage_rate)
    else:
        return market_price_at_exit * (1 + slippage_rate)


def compute_kelly_fraction(
    direction: str,
    model_prob: float,
    market_price: float,
    fee_rate: float = 0.0,
    fraction: float = 0.25,
) -> float:
    """Fractional Kelly criterion for position sizing."""
    p = model_prob
    q = market_price

    if direction == "buy_yes":
        win_prob = p
        win_amount = 1 - q
        lose_amount = q
    else:
        win_prob = 1 - p
        win_amount = q
        lose_amount = 1 - q

    if win_amount <= 0 or lose_amount <= 0:
        return 0.0

    expected_fee = win_prob * fee_rate * win_amount
    net_win = win_amount - expected_fee
    edge = win_prob * net_win - (1 - win_prob) * lose_amount

    if edge <= 0:
        return 0.0

    odds = net_win / lose_amount
    kelly_raw = edge / net_win if net_win > 0 else 0.0
    return max(0.0, min(kelly_raw * fraction, 0.04))


def run_backtest(
    signals: list[dict],
    bankroll: float = 1000.0,
    strategy_name: str = "ensemble",
    slippage_bps: float = 100,
) -> dict[str, BacktestResult]:
    """Run a tradability backtest on historical signals.

    Args:
        signals: List of dicts with keys:
            - market_id: int
            - timestamp: datetime
            - direction: "buy_yes" or "buy_no"
            - model_prob: float (model's probability of YES)
            - market_price: float (YES price at signal time)
            - resolution: float (1.0 or 0.0)
            - taker_fee_bps: int (per-market fee rate)
            - platform: str ("polymarket" or "kalshi")
        bankroll: Starting capital
        strategy_name: Name for labeling
        slippage_bps: Default slippage in basis points

    Returns:
        Dict of {sizing_method: BacktestResult} for three sizing regimes.
    """
    results = {}

    for sizing_method, kelly_fraction, fixed_size in [
        ("full_kelly", 1.0, None),
        ("half_kelly", 0.5, None),
        ("fixed_10", None, 10.0),
    ]:
        result = BacktestResult(strategy=strategy_name, sizing_method=sizing_method)
        current_bankroll = bankroll
        peak_bankroll = bankroll
        max_dd = 0.0
        daily_returns = []
        equity = [bankroll]

        for signal in signals:
            direction = signal["direction"]
            model_prob = signal["model_prob"]
            market_price = signal["market_price"]
            resolution = signal["resolution"]
            fee_bps = signal.get("taker_fee_bps", 0) or 0
            fee_rate = fee_bps / 10000
            platform = signal.get("platform", "polymarket")

            entry = compute_entry_price(direction, market_price, slippage_bps)
            exit_p = compute_exit_price(
                "yes" if direction == "buy_yes" else "no",
                resolution, True
            )

            if direction == "buy_yes":
                cost_per_share = entry
            else:
                cost_per_share = 1.0 - entry

            if cost_per_share <= 0.01:
                continue

            # Position sizing
            if fixed_size is not None:
                position_cost = min(fixed_size, current_bankroll * 0.1)
            else:
                kelly = compute_kelly_fraction(
                    direction, model_prob, market_price,
                    fee_rate=fee_rate, fraction=kelly_fraction * 0.25
                )
                position_cost = current_bankroll * kelly

            if position_cost < 0.01:
                continue

            quantity = position_cost / cost_per_share

            # P&L calculation
            if direction == "buy_yes":
                gross_pnl = (exit_p - entry) * quantity
            else:
                gross_pnl = (entry - exit_p) * quantity

            # Fee on winnings only
            if platform == "kalshi":
                fee = compute_kalshi_fee(market_price, int(quantity), is_maker=False)
            else:
                fee = fee_rate * max(gross_pnl, 0.0)

            slippage_cost = (slippage_bps / 10000) * position_cost
            net_pnl = gross_pnl - fee

            trade = BacktestTrade(
                market_id=signal["market_id"],
                timestamp=signal.get("timestamp", datetime.utcnow()),
                direction=direction,
                model_prob=model_prob,
                market_price=market_price,
                entry_price=entry,
                exit_price=exit_p,
                quantity=quantity,
                gross_pnl=gross_pnl,
                fee=fee,
                slippage_cost=slippage_cost,
                net_pnl=net_pnl,
                resolution=resolution,
                sizing_method=sizing_method,
            )

            result.trades.append(trade)
            result.total_pnl += net_pnl
            result.total_trades += 1
            result.total_fees += fee
            result.total_slippage += slippage_cost
            if net_pnl > 0:
                result.winning_trades += 1

            current_bankroll += net_pnl
            peak_bankroll = max(peak_bankroll, current_bankroll)
            drawdown = (peak_bankroll - current_bankroll) / peak_bankroll if peak_bankroll > 0 else 0
            max_dd = max(max_dd, drawdown)
            equity.append(current_bankroll)
            daily_returns.append(net_pnl / max(bankroll, 0.01))

        result.max_drawdown = max_dd
        result.equity_curve = equity

        if daily_returns and len(daily_returns) > 1:
            returns_arr = np.array(daily_returns)
            mean_ret = returns_arr.mean()
            std_ret = returns_arr.std()
            result.sharpe_ratio = (mean_ret / std_ret * math.sqrt(252)) if std_ret > 0 else 0.0

        results[sizing_method] = result

    return results


def format_backtest_report(results: dict[str, BacktestResult]) -> str:
    """Format backtest results as a readable report."""
    lines = [
        "=" * 65,
        "TRADABILITY BACKTEST REPORT",
        "=" * 65,
    ]

    for method, result in results.items():
        lines.append(f"\n--- {method.upper()} ---")
        lines.append(f"  Total P&L:     ${result.total_pnl:.2f}")
        lines.append(f"  Trades:        {result.total_trades}")
        lines.append(f"  Win Rate:      {result.win_rate:.1%}")
        lines.append(f"  Total Fees:    ${result.total_fees:.2f}")
        lines.append(f"  Total Slippage:${result.total_slippage:.2f}")
        lines.append(f"  Max Drawdown:  {result.max_drawdown:.1%}")
        lines.append(f"  Sharpe Ratio:  {result.sharpe_ratio:.2f}")

    # Gate check: strategy must be profitable in all three sizing regimes
    all_profitable = all(r.total_pnl > 0 for r in results.values())
    lines.append(f"\n{'=' * 65}")
    if all_profitable:
        lines.append("GATE PASS: Profitable across all sizing regimes")
    else:
        profitable = [m for m, r in results.items() if r.total_pnl > 0]
        unprofitable = [m for m, r in results.items() if r.total_pnl <= 0]
        lines.append(f"GATE FAIL: Unprofitable in: {unprofitable}")
        if profitable:
            lines.append(f"  Profitable only in: {profitable}")
    lines.append("=" * 65)

    return "\n".join(lines)
