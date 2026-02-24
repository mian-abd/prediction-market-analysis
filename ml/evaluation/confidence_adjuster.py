"""Adaptive confidence adjustment based on realized performance.

Tracks realized hit rates by tier, direction, and price zone, then adjusts
confidence scores to reflect actual edge quality in each segment.

Formula: adjusted_conf = base_conf × (realized_hit_rate / expected_hit_rate)

Example:
    - Base confidence: 0.70 (70% expected)
    - Realized hit rate in Zone 2 (0.20-0.40): 0.85 (85%)
    - Adjustment multiplier: 0.85 / 0.70 = 1.21
    - Adjusted confidence: 0.70 × 1.21 = 0.847 (84.7%)

Uses exponential moving average (EMA) with 50-signal window for stability.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import PortfolioPosition, EnsembleEdgeSignal, EloEdgeSignal

logger = logging.getLogger(__name__)

# Minimum samples before applying adjustments (lower = learn from fewer trades)
MIN_SAMPLES = 4

# EMA smoothing factor (~20-signal window for faster adaptation to recent performance)
EMA_ALPHA = 0.095


@dataclass
class PerformanceSegment:
    """Performance statistics for a market segment."""
    total_trades: int
    winning_trades: int
    realized_hit_rate: float
    adjustment_multiplier: float
    last_updated: datetime


class ConfidenceAdjuster:
    """Adjusts confidence scores based on realized performance by segment.

    Segments tracked:
    - Tier: high/medium/low confidence
    - Direction: buy_yes/buy_no
    - Price zone: 0.05-0.20, 0.20-0.40, 0.40-0.60, 0.60-0.80, 0.80-0.98

    Adjustment formula:
        adjusted_conf = base_conf × (realized_hit_rate / expected_hit_rate)

    Where:
        - base_conf = original confidence from ensemble
        - realized_hit_rate = % of winning trades in segment (EMA smoothed)
        - expected_hit_rate = average base confidence in segment
    """

    def __init__(self):
        """Initialize confidence adjuster with empty performance tracking."""
        # Key: (tier, direction, price_zone) -> PerformanceSegment
        self.segments: dict[tuple[str, str, str], PerformanceSegment] = {}

        # Cache for adjustment multipliers (avoid recomputing every call)
        self._multiplier_cache: dict[tuple[str, str, str], float] = {}
        self._cache_updated: datetime | None = None

    async def load_performance_stats(self, session: AsyncSession, lookback_days: int = 30):
        """Load realized performance from paper trades (last N days).

        Args:
            session: Database session
            lookback_days: Number of days of history to analyze
        """
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)

        # Get closed positions with their signal confidence (join with signal tables)
        # For ensemble positions
        ensemble_result = await session.execute(
            select(
                PortfolioPosition.entry_price,
                PortfolioPosition.side,
                PortfolioPosition.realized_pnl,
                PortfolioPosition.entry_time,
                EnsembleEdgeSignal.confidence
            )
            .join(EnsembleEdgeSignal, PortfolioPosition.market_id == EnsembleEdgeSignal.market_id)
            .where(
                and_(
                    PortfolioPosition.exit_time.isnot(None),  # Closed positions
                    PortfolioPosition.entry_time >= cutoff,
                    PortfolioPosition.strategy == "auto_ensemble",
                    EnsembleEdgeSignal.confidence.isnot(None),
                )
            )
        )
        ensemble_trades = ensemble_result.all()

        # For ELO positions
        elo_result = await session.execute(
            select(
                PortfolioPosition.entry_price,
                PortfolioPosition.side,
                PortfolioPosition.realized_pnl,
                PortfolioPosition.entry_time,
                EloEdgeSignal.elo_confidence.label('confidence')
            )
            .join(EloEdgeSignal, PortfolioPosition.market_id == EloEdgeSignal.market_id)
            .where(
                and_(
                    PortfolioPosition.exit_time.isnot(None),  # Closed positions
                    PortfolioPosition.entry_time >= cutoff,
                    PortfolioPosition.strategy == "auto_elo",
                    EloEdgeSignal.elo_confidence.isnot(None),
                )
            )
        )
        elo_trades = elo_result.all()

        # Combine trades
        trades = list(ensemble_trades) + list(elo_trades)

        if not trades:
            logger.info("No closed trades found for confidence adjustment")
            return

        # Group trades by segment
        segment_trades = defaultdict(list)

        for trade in trades:
            # Unpack tuple: (entry_price, side, realized_pnl, entry_time, confidence)
            entry_price, side, realized_pnl, entry_time, confidence = trade

            # Determine tier from confidence
            if confidence >= 0.70:
                tier = "high"
            elif confidence >= 0.55:
                tier = "medium"
            else:
                tier = "low"

            # Direction (convert side to direction format)
            direction = f"buy_{side.lower()}"  # "YES" -> "buy_yes", "NO" -> "buy_no"

            # Price zone (entry price)
            if entry_price < 0.20:
                zone = "zone1"
            elif entry_price < 0.40:
                zone = "zone2"
            elif entry_price < 0.60:
                zone = "zone3"
            elif entry_price < 0.80:
                zone = "zone4"
            else:
                zone = "zone5"

            segment_key = (tier, direction, zone)
            # Store as dict for easier access
            segment_trades[segment_key].append({
                'confidence': confidence,
                'realized_pnl': realized_pnl,
                'entry_price': entry_price,
            })

        # Compute statistics for each segment
        for segment_key, seg_trades in segment_trades.items():
            if len(seg_trades) < MIN_SAMPLES:
                logger.debug(f"Skipping segment {segment_key}: only {len(seg_trades)} samples")
                continue

            total = len(seg_trades)
            winning = sum(1 for t in seg_trades if t['realized_pnl'] and t['realized_pnl'] > 0)
            realized_rate = winning / total if total > 0 else 0.0

            # Expected hit rate = average confidence in this segment
            expected_rate = sum(t['confidence'] for t in seg_trades) / total

            # Adjustment multiplier
            if expected_rate > 0.01:  # Avoid division by zero
                multiplier = realized_rate / expected_rate
                # Clamp to reasonable range (0.5 to 1.5)
                multiplier = max(0.5, min(1.5, multiplier))
            else:
                multiplier = 1.0

            self.segments[segment_key] = PerformanceSegment(
                total_trades=total,
                winning_trades=winning,
                realized_hit_rate=realized_rate,
                adjustment_multiplier=multiplier,
                last_updated=datetime.utcnow(),
            )

            logger.info(
                f"Segment {segment_key}: {total} trades, "
                f"{realized_rate:.1%} realized vs {expected_rate:.1%} expected, "
                f"multiplier={multiplier:.3f}"
            )

        # Update cache
        self._multiplier_cache = {
            k: v.adjustment_multiplier for k, v in self.segments.items()
        }
        self._cache_updated = datetime.utcnow()

        logger.info(f"Loaded performance stats for {len(self.segments)} segments")

    def adjust_confidence(
        self,
        base_confidence: float,
        direction: str,
        entry_price: float,
    ) -> tuple[float, str]:
        """Adjust confidence based on realized performance in similar scenarios.

        Args:
            base_confidence: Original confidence from ensemble (0-1)
            direction: Trade direction ("buy_yes" or "buy_no")
            entry_price: Entry price (0-1) for price zone classification

        Returns:
            (adjusted_confidence, reason): Adjusted confidence and explanation
        """
        # Determine tier
        if base_confidence >= 0.70:
            tier = "high"
        elif base_confidence >= 0.55:
            tier = "medium"
        else:
            tier = "low"

        # Determine price zone
        if entry_price < 0.20:
            zone = "zone1"
        elif entry_price < 0.40:
            zone = "zone2"
        elif entry_price < 0.60:
            zone = "zone3"
        elif entry_price < 0.80:
            zone = "zone4"
        else:
            zone = "zone5"

        segment_key = (tier, direction, zone)

        # Check if we have performance data for this segment
        if segment_key not in self._multiplier_cache:
            return base_confidence, "no_adjustment_data"

        multiplier = self._multiplier_cache[segment_key]
        adjusted = base_confidence * multiplier

        # Clamp to valid probability range
        adjusted = max(0.01, min(0.99, adjusted))

        # Reason
        segment = self.segments[segment_key]
        reason = (
            f"{tier}_{direction}_{zone}: "
            f"{segment.realized_hit_rate:.1%} realized "
            f"({segment.total_trades} trades)"
        )

        return adjusted, reason

    def get_stats(self) -> dict:
        """Get summary statistics for monitoring."""
        if not self.segments:
            return {
                "total_segments": 0,
                "total_trades_analyzed": 0,
                "cache_age_minutes": None,
            }

        total_trades = sum(s.total_trades for s in self.segments.values())
        cache_age = None
        if self._cache_updated:
            cache_age = (datetime.utcnow() - self._cache_updated).total_seconds() / 60

        return {
            "total_segments": len(self.segments),
            "total_trades_analyzed": total_trades,
            "cache_age_minutes": round(cache_age, 1) if cache_age else None,
            "segments": {
                str(k): {
                    "trades": v.total_trades,
                    "hit_rate": round(v.realized_hit_rate, 3),
                    "multiplier": round(v.adjustment_multiplier, 3),
                }
                for k, v in self.segments.items()
            },
        }


# Global instance (initialized in scheduler)
_confidence_adjuster: ConfidenceAdjuster | None = None


async def init_confidence_adjuster(session: AsyncSession):
    """Initialize global confidence adjuster with historical performance data."""
    global _confidence_adjuster

    _confidence_adjuster = ConfidenceAdjuster()
    await _confidence_adjuster.load_performance_stats(session, lookback_days=60)

    logger.info("Confidence adjuster initialized")


async def refresh_confidence_adjuster(session: AsyncSession):
    """Refresh performance statistics (call periodically)."""
    if _confidence_adjuster:
        await _confidence_adjuster.load_performance_stats(session, lookback_days=60)
        logger.info("Confidence adjuster refreshed")


def adjust_confidence(
    base_confidence: float,
    direction: str,
    entry_price: float,
) -> tuple[float, str]:
    """Adjust confidence using global adjuster instance.

    Returns:
        (adjusted_confidence, reason): If no adjuster, returns (base_confidence, "not_initialized")
    """
    if not _confidence_adjuster:
        return base_confidence, "not_initialized"

    return _confidence_adjuster.adjust_confidence(base_confidence, direction, entry_price)


def get_adjuster_stats() -> dict:
    """Get confidence adjuster statistics for monitoring."""
    if not _confidence_adjuster:
        return {"status": "not_initialized"}

    return _confidence_adjuster.get_stats()
