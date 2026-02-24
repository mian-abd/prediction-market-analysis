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
4. Strategy performance tracking: weight strategies by their realized performance
"""

import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import PortfolioPosition

logger = logging.getLogger(__name__)

# Target allocation by strategy (initial weights, updated by performance)
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

MAX_DRAWDOWN_THRESHOLD = 0.08  # 8% max drawdown before reducing exposure
DRAWDOWN_REDUCTION_FACTOR = 0.3  # Reduce Kelly by 70% when near max drawdown
CORRELATION_PENALTY_FACTOR = 0.5  # Reduce Kelly by 50% for correlated positions
MAX_POSITIONS_PER_STRATEGY = 5
MAX_TOTAL_POSITIONS = 20


class PortfolioOptimizer:
    """Optimize position sizing across multiple strategies."""

    def __init__(self):
        self.strategy_performance: dict[str, dict] = {}
        self.strategy_allocations: dict[str, float] = dict(DEFAULT_STRATEGY_ALLOCATION)

    async def load_performance(self, session: AsyncSession, days_back: int = 30) -> None:
        """Load strategy performance data for allocation optimization."""
        cutoff = datetime.utcnow() - timedelta(days=days_back)

        result = await session.execute(
            select(PortfolioPosition).where(
                PortfolioPosition.portfolio_type == "auto",
                PortfolioPosition.exit_time != None,  # noqa
                PortfolioPosition.exit_time >= cutoff,
            )
        )
        positions = result.scalars().all()

        strategy_stats: dict[str, dict] = defaultdict(lambda: {
            "total_trades": 0, "winning_trades": 0,
            "total_pnl": 0.0, "max_drawdown": 0.0,
        })

        for pos in positions:
            strategy = pos.strategy or "unknown"
            stats = strategy_stats[strategy]
            pnl = pos.realized_pnl or 0.0
            stats["total_trades"] += 1
            if pnl > 0:
                stats["winning_trades"] += 1
            stats["total_pnl"] += pnl

        for strategy, stats in strategy_stats.items():
            stats["win_rate"] = (
                stats["winning_trades"] / max(stats["total_trades"], 1)
            )
            stats["avg_pnl"] = stats["total_pnl"] / max(stats["total_trades"], 1)

        self.strategy_performance = dict(strategy_stats)
        self._update_allocations()

    def _update_allocations(self) -> None:
        """Update strategy allocations based on realized performance.

        Uses a Bayesian-like approach: start with prior allocations,
        update based on realized win rate and PnL.
        """
        if not self.strategy_performance:
            return

        prior = dict(DEFAULT_STRATEGY_ALLOCATION)
        posterior = {}

        for strategy, default_weight in prior.items():
            stats = self.strategy_performance.get(strategy, {})
            trades = stats.get("total_trades", 0)

            if trades < 5:
                # Not enough data: keep prior
                posterior[strategy] = default_weight
                continue

            win_rate = stats.get("win_rate", 0.5)
            avg_pnl = stats.get("avg_pnl", 0)

            # Performance score: weighted average of win rate edge and PnL
            wr_edge = max(0, (win_rate - 0.5) * 2)  # 0 at 50%, 1 at 100%
            pnl_score = max(0, min(1, avg_pnl * 10 + 0.5))  # Centered at 0.5
            performance = 0.6 * wr_edge + 0.4 * pnl_score

            # Bayesian update: prior * likelihood
            # Higher performance → higher allocation
            posterior[strategy] = default_weight * (0.5 + performance)

        # Normalize to sum to 1.0
        total = sum(posterior.values())
        if total > 0:
            self.strategy_allocations = {
                k: round(v / total, 4) for k, v in posterior.items()
            }

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
        1. Strategy allocation cap
        2. Drawdown constraint
        3. Correlation penalty
        4. Position count limit
        """
        if raw_kelly <= 0:
            return 0.0

        # 1. Strategy allocation cap
        allocation = self.strategy_allocations.get(strategy, 0.05)
        kelly = raw_kelly * min(1.0, allocation / 0.25)  # Scale relative to 25% base

        # 2. Drawdown constraint: reduce exposure as we approach max drawdown
        if current_drawdown_pct > MAX_DRAWDOWN_THRESHOLD * 0.5:
            drawdown_ratio = current_drawdown_pct / MAX_DRAWDOWN_THRESHOLD
            reduction = 1.0 - (drawdown_ratio - 0.5) * 2 * (1 - DRAWDOWN_REDUCTION_FACTOR)
            kelly *= max(DRAWDOWN_REDUCTION_FACTOR, reduction)

        if current_drawdown_pct >= MAX_DRAWDOWN_THRESHOLD:
            return 0.0  # Full stop: max drawdown reached

        # 3. Correlation penalty: reduce when many correlated positions open
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
            "strategy_performance": self.strategy_performance,
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
