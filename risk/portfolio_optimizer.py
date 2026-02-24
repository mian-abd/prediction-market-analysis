"""Portfolio-level optimization — correlation-aware sizing, drawdown constraints,
and strategy allocation.

Research basis:
- Kelly criterion for multiple correlated bets (Stanford, Boyd et al.):
  Convex optimization for drawdown-constrained growth maximization.
- Multi-hypothesis ensemble diversification (arxiv, 2025):
  Predictor diversity links to portfolio diversification.
- Prediction market correlations: binary outcome correlations are fully
  characterized by Pearson coefficient and marginals.

This module provides:
1. Cross-strategy allocation: how much capital to allocate to each strategy
2. Correlation-aware position sizing: reduce Kelly when positions are correlated
3. Drawdown constraints: reduce exposure when approaching max drawdown
4. Strategy performance tracking: weight strategies by their realized Sharpe ratio
"""

import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import PortfolioPosition, MarketRelationship

logger = logging.getLogger(__name__)

# Target allocation by strategy (initial weights, updated by realized Sharpe)
DEFAULT_STRATEGY_ALLOCATION = {
    "auto_ensemble": 0.25,
    "auto_elo": 0.15,
    "auto_longshot_bias": 0.15,
    "auto_llm_forecast": 0.15,
    "auto_resolution_convergence": 0.10,
    "auto_news_catalyst": 0.05,
    "auto_orderflow": 0.05,
    "auto_smart_money": 0.05,
    "auto_market_clustering": 0.05,
}

MAX_DRAWDOWN_THRESHOLD = 0.08   # 8% max drawdown before reducing exposure
DRAWDOWN_REDUCTION_FACTOR = 0.3  # Reduce Kelly by 70% when near max drawdown
CORRELATION_PENALTY_FACTOR = 0.5  # Reduce Kelly by 50% for each correlated open position
MAX_POSITIONS_PER_STRATEGY = 5
MAX_TOTAL_POSITIONS = 20

# Sharpe-based allocation bounds
MAX_STRATEGY_ALLOCATION = 0.35   # No strategy gets > 35% of capital
MIN_STRATEGY_ALLOCATION = 0.02   # Keep all strategies collecting data (2% floor)

# How many closed trades before updating a strategy's allocation
MIN_TRADES_FOR_UPDATE = 3        # Faster feedback loop (was 5)


async def compute_correlated_positions(
    session: AsyncSession,
    market_id: int,
    open_positions: list,
) -> int:
    """Count open positions that are correlated with the given market.

    Uses the MarketRelationship table populated by the market_clustering strategy.
    Returns the number of open correlated positions (for penalty calculation).
    """
    if not open_positions:
        return 0

    open_market_ids = {p.market_id for p in open_positions if p.market_id != market_id}
    if not open_market_ids:
        return 0

    try:
        result = await session.execute(
            select(MarketRelationship).where(
                or_(
                    MarketRelationship.market_id_a == market_id,
                    MarketRelationship.market_id_b == market_id,
                ),
                MarketRelationship.confidence >= 0.5,  # Only strong relationships
            )
        )
        relationships = result.scalars().all()

        correlated_count = 0
        for rel in relationships:
            other_id = (
                rel.market_id_b if rel.market_id_a == market_id
                else rel.market_id_a
            )
            if other_id in open_market_ids:
                correlated_count += 1

        return correlated_count

    except Exception as e:
        logger.debug(f"Correlation check failed for market {market_id}: {e}")
        return 0


class PortfolioOptimizer:
    """Optimize position sizing across multiple strategies."""

    def __init__(self):
        self.strategy_performance: dict[str, dict] = {}
        self.strategy_allocations: dict[str, float] = dict(DEFAULT_STRATEGY_ALLOCATION)
        self._trade_count_at_last_update: dict[str, int] = {}

    async def load_performance(self, session: AsyncSession, days_back: int = 30) -> None:
        """Load strategy performance data for allocation optimization."""
        cutoff = datetime.utcnow() - timedelta(days=days_back)

        result = await session.execute(
            select(PortfolioPosition).where(
                PortfolioPosition.portfolio_type == "auto",
                PortfolioPosition.exit_time.isnot(None),
                PortfolioPosition.exit_time >= cutoff,
            )
        )
        positions = result.scalars().all()

        strategy_stats: dict[str, dict] = defaultdict(lambda: {
            "total_trades": 0,
            "winning_trades": 0,
            "total_pnl": 0.0,
            "pnl_values": [],
        })

        for pos in positions:
            strategy = pos.strategy or "unknown"
            stats = strategy_stats[strategy]
            pnl = pos.realized_pnl or 0.0
            stats["total_trades"] += 1
            if pnl > 0:
                stats["winning_trades"] += 1
            stats["total_pnl"] += pnl
            stats["pnl_values"].append(pnl)

        for strategy, stats in strategy_stats.items():
            n = max(stats["total_trades"], 1)
            stats["win_rate"] = stats["winning_trades"] / n
            stats["avg_pnl"] = stats["total_pnl"] / n

            # Sharpe ratio: mean PnL / std PnL (higher = more consistent edge)
            pnl_vals = stats["pnl_values"]
            if len(pnl_vals) >= 2:
                mean_p = stats["avg_pnl"]
                variance = sum((x - mean_p) ** 2 for x in pnl_vals) / len(pnl_vals)
                std_p = math.sqrt(variance)
                stats["sharpe"] = mean_p / max(std_p, 1e-8)
            else:
                stats["sharpe"] = 0.0

        self.strategy_performance = dict(strategy_stats)
        self._update_allocations()

    def _update_allocations(self) -> None:
        """Update strategy allocations based on realized Sharpe ratios.

        Uses a Bayesian-like approach: start with prior allocations,
        update based on Sharpe ratio (risk-adjusted return), not just win rate.
        Requires MIN_TRADES_FOR_UPDATE closed trades before updating.

        Constraints:
        - No strategy exceeds MAX_STRATEGY_ALLOCATION (35%)
        - No strategy falls below MIN_STRATEGY_ALLOCATION (2% floor)
        """
        if not self.strategy_performance:
            return

        prior = dict(DEFAULT_STRATEGY_ALLOCATION)
        posterior = {}

        for strategy, default_weight in prior.items():
            stats = self.strategy_performance.get(strategy, {})
            trades = stats.get("total_trades", 0)

            if trades < MIN_TRADES_FOR_UPDATE:
                # Not enough data: keep prior
                posterior[strategy] = default_weight
                continue

            sharpe = stats.get("sharpe", 0.0)
            win_rate = stats.get("win_rate", 0.5)
            avg_pnl = stats.get("avg_pnl", 0.0)

            # Combined performance score (Sharpe-weighted)
            # Sharpe: normalize to 0-1 range using soft clipping
            # Sharpe of 0.5 → good, 1.0 → excellent, negative → bad
            sharpe_score = max(0.0, min(1.0, (sharpe + 0.2) / 1.4))

            # Win rate edge: 50% = 0.0, 70% = 0.4, 100% = 1.0
            wr_edge = max(0.0, (win_rate - 0.5) * 2)

            # PnL score: center at 0, scale by expected range
            pnl_score = max(0.0, min(1.0, avg_pnl * 10 + 0.5))

            # Combined score: 50% Sharpe, 30% win rate, 20% avg PnL
            performance = 0.50 * sharpe_score + 0.30 * wr_edge + 0.20 * pnl_score

            # Bayesian update: scale prior by (0.4 + performance)
            # performance=0 → weight * 0.4, performance=1 → weight * 1.4
            posterior[strategy] = default_weight * (0.4 + performance)

        # Apply allocation bounds and normalize
        for k in posterior:
            posterior[k] = max(MIN_STRATEGY_ALLOCATION * sum(prior.values()),
                               posterior[k])

        total = sum(posterior.values())
        if total > 0:
            normalized = {k: v / total for k, v in posterior.items()}

            # Enforce max cap with redistribution
            capped = {}
            excess = 0.0
            uncapped_keys = []
            for k, v in normalized.items():
                if v > MAX_STRATEGY_ALLOCATION:
                    excess += v - MAX_STRATEGY_ALLOCATION
                    capped[k] = MAX_STRATEGY_ALLOCATION
                else:
                    uncapped_keys.append(k)

            # Redistribute excess proportionally to uncapped strategies
            if excess > 0 and uncapped_keys:
                uncapped_total = sum(normalized[k] for k in uncapped_keys)
                for k in uncapped_keys:
                    share = normalized[k] / max(uncapped_total, 1e-9)
                    capped[k] = normalized[k] + excess * share
            else:
                capped.update({k: normalized[k] for k in uncapped_keys})

            self.strategy_allocations = {
                k: max(MIN_STRATEGY_ALLOCATION, round(v, 4))
                for k, v in capped.items()
            }

    def should_refresh_for_strategy(self, strategy: str) -> bool:
        """Check if enough new trades have closed to warrant a re-allocation."""
        stats = self.strategy_performance.get(strategy, {})
        current_trades = stats.get("total_trades", 0)
        last_count = self._trade_count_at_last_update.get(strategy, 0)
        new_trades = current_trades - last_count
        if new_trades >= MIN_TRADES_FOR_UPDATE:
            self._trade_count_at_last_update[strategy] = current_trades
            return True
        return False

    def compute_adjusted_kelly(
        self,
        raw_kelly: float,
        strategy: str,
        current_open_positions: int,
        current_drawdown_pct: float,
        correlated_position_count: int = 0,
    ) -> float:
        """Compute portfolio-adjusted Kelly fraction.

        Applies multiple adjustments:
        1. Strategy allocation cap (Sharpe-weighted)
        2. Drawdown constraint
        3. Correlation penalty (now uses real MarketRelationship data)
        4. Position count limit
        """
        if raw_kelly <= 0:
            return 0.0

        # 1. Strategy allocation cap (relative to 25% base weight)
        allocation = self.strategy_allocations.get(strategy, 0.05)
        kelly = raw_kelly * min(1.0, allocation / 0.25)

        # 2. Drawdown constraint: reduce exposure as we approach max drawdown
        if current_drawdown_pct > MAX_DRAWDOWN_THRESHOLD * 0.5:
            drawdown_ratio = current_drawdown_pct / MAX_DRAWDOWN_THRESHOLD
            reduction = 1.0 - (drawdown_ratio - 0.5) * 2 * (1 - DRAWDOWN_REDUCTION_FACTOR)
            kelly *= max(DRAWDOWN_REDUCTION_FACTOR, reduction)

        if current_drawdown_pct >= MAX_DRAWDOWN_THRESHOLD:
            return 0.0  # Full stop: max drawdown reached

        # 3. Correlation penalty using real relationship data
        if correlated_position_count > 0:
            corr_penalty = 1.0 / (1.0 + correlated_position_count * CORRELATION_PENALTY_FACTOR)
            kelly *= corr_penalty

        # 4. Position count limits
        if current_open_positions >= MAX_TOTAL_POSITIONS:
            return 0.0

        strategy_positions = min(current_open_positions, MAX_POSITIONS_PER_STRATEGY)
        if strategy_positions >= MAX_POSITIONS_PER_STRATEGY:
            kelly *= 0.3  # Heavily reduced if strategy is at max positions

        return max(0.0, kelly)

    def get_portfolio_stats(self) -> dict:
        """Get current portfolio optimization state."""
        return {
            "strategy_allocations": self.strategy_allocations,
            "strategy_performance": {
                k: {kk: vv for kk, vv in v.items() if kk != "pnl_values"}
                for k, v in self.strategy_performance.items()
            },
            "max_drawdown_threshold": MAX_DRAWDOWN_THRESHOLD,
            "max_positions_per_strategy": MAX_POSITIONS_PER_STRATEGY,
            "max_total_positions": MAX_TOTAL_POSITIONS,
        }


# Module-level singleton
_optimizer: Optional[PortfolioOptimizer] = None


async def get_optimizer(session: AsyncSession) -> PortfolioOptimizer:
    """Get or initialize the portfolio optimizer singleton."""
    global _optimizer
    if _optimizer is None:
        _optimizer = PortfolioOptimizer()
        await _optimizer.load_performance(session)
    return _optimizer


async def refresh_optimizer(session: AsyncSession) -> None:
    """Refresh optimizer with latest performance data."""
    global _optimizer
    if _optimizer is None:
        _optimizer = PortfolioOptimizer()
    await _optimizer.load_performance(session)
