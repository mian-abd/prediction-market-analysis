"""Auto paper-trade from high-confidence signals (ensemble + Elo).

Config-driven multi-strategy executor. Each strategy reads its thresholds
from AutoTradingConfig in the DB, so they can be changed at runtime via API.
All auto positions get portfolio_type="auto" for isolated risk accounting.
"""

import logging
import math
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import (
    PortfolioPosition, Market, AutoTradingConfig,
    EnsembleEdgeSignal, EloEdgeSignal,
)
from risk.risk_manager import check_risk_limits

logger = logging.getLogger(__name__)

# Edge decay: signals lose half their edge every 2 hours (exponential decay)
EDGE_DECAY_HALF_LIFE_HOURS = 2.0


def compute_execution_cost(
    direction: str,
    market_price: float,
    slippage_bps: float = 200,  # 2% default
    taker_fee_bps: int = 0,
) -> dict:
    """Model realistic execution costs for paper trading.

    Returns execution price with slippage. Caller uses this to compute
    quantity based on Kelly allocation.
    """
    slippage_rate = slippage_bps / 10000

    if direction == "buy_yes":
        # Pay slippage above market for YES
        execution_price = market_price * (1 + slippage_rate)
    else:  # buy_no
        # Pay (1 - market_price) + slippage for NO
        # NO position benefits when YES price drops, so execution_price is market_price adjusted down
        execution_price = market_price * (1 - slippage_rate)

    return {
        "execution_price": execution_price,
        "slippage_bps": slippage_bps,
        "market_price": market_price,
    }


async def _get_auto_config(session: AsyncSession, strategy: str) -> AutoTradingConfig | None:
    """Load AutoTradingConfig row for a strategy. Returns None if missing or disabled."""
    result = await session.execute(
        select(AutoTradingConfig).where(AutoTradingConfig.strategy == strategy)
    )
    config = result.scalar_one_or_none()
    if config and not config.is_enabled:
        return None
    return config


async def execute_paper_trades(session: AsyncSession) -> list[int]:
    """Orchestrator: run both ensemble and Elo auto-trading."""
    created = []
    created.extend(await _execute_ensemble_trades(session))
    created.extend(await _execute_elo_trades(session))
    return created


async def _execute_ensemble_trades(session: AsyncSession) -> list[int]:
    """Auto-open paper positions for new high-confidence ensemble signals."""
    config = await _get_auto_config(session, "ensemble")
    if not config:
        return []

    # Build quality tier filter: include target tier and all tiers above it
    tier_hierarchy = ["low", "medium", "high", "speculative"]
    min_idx = tier_hierarchy.index(config.min_quality_tier) if config.min_quality_tier in tier_hierarchy else 0
    accepted_tiers = tier_hierarchy[min_idx:]

    from sqlalchemy import func as sql_func, case
    from db.models import Market as MarketModel

    # Bias toward markets ending soon (for demo):
    # Calculate days until close, boost EV score for near-term markets
    now = datetime.utcnow()

    result = await session.execute(
        select(EnsembleEdgeSignal, MarketModel)
        .join(MarketModel, EnsembleEdgeSignal.market_id == MarketModel.id)
        .where(
            EnsembleEdgeSignal.expired_at == None,  # noqa
            EnsembleEdgeSignal.quality_tier.in_(accepted_tiers),
            EnsembleEdgeSignal.confidence >= config.min_confidence,
            EnsembleEdgeSignal.net_ev >= config.min_net_ev,
        )
        .limit(50)  # Get more candidates for sorting
    )
    signal_market_pairs = result.all()

    # Sort by edge-decayed EV: discount stale signals, skip near-resolution markets
    def urgency_score(signal, market):
        base_ev = signal.net_ev
        # Edge decay: exponential decay based on signal age
        if signal.detected_at:
            signal_age_hours = (now - signal.detected_at).total_seconds() / 3600
            decay = math.exp(-0.693 * signal_age_hours / EDGE_DECAY_HALF_LIFE_HOURS)
            base_ev *= decay
        # Skip markets too close to resolution (no time for stop-loss)
        if market.end_date:
            hours_until = (market.end_date - now).total_seconds() / 3600
            if hours_until < 2:
                return -1
        return base_ev

    # Filter out markets too close to resolution
    sorted_pairs = sorted(signal_market_pairs, key=lambda p: urgency_score(p[0], p[1]), reverse=True)
    sorted_pairs = [(s, m) for s, m in sorted_pairs if urgency_score(s, m) > 0]
    signals = [pair[0] for pair in sorted_pairs[:10]]

    if not signals:
        return []

    # Markets with existing open auto positions (ANY strategy in auto portfolio)
    # Prevent duplicate positions on same market across all auto-strategies
    open_markets = set()
    open_result = await session.execute(
        select(PortfolioPosition.market_id)
        .where(
            PortfolioPosition.exit_time == None,  # noqa
            PortfolioPosition.portfolio_type == "auto",
        )
    )
    for row in open_result.all():
        open_markets.add(row[0])

    created_ids = []

    for signal in signals:
        if signal.market_id in open_markets:
            continue

        market = await session.get(Market, signal.market_id)
        if not market or not market.is_active:
            continue

        # Skip miscategorized markets ("other") — no reliable features for prediction
        category = (market.category or "").lower()
        normalized_cat = (market.normalized_category or "").lower() if hasattr(market, 'normalized_category') else ""
        if category == "other" or normalized_cat == "other":
            continue

        # Skip short-term crypto price-at-point-in-time markets (essentially random)
        question_lower = (market.question or "").lower()
        if category == "crypto" and market.end_date:
            hours_left = (market.end_date - now).total_seconds() / 3600
            is_price_prediction = any(kw in question_lower for kw in [
                "price of", "above $", "below $", "between $", "up or down",
                "reach $", "dip to $",
            ])
            if is_price_prediction and hours_left < 48:
                continue  # Skip short-term crypto price bets

        kelly = signal.kelly_fraction or 0.0
        kelly = min(kelly, config.max_kelly_fraction)
        # Apply edge decay to Kelly (stale signals get smaller positions)
        if signal.detected_at:
            signal_age_hours = (now - signal.detected_at).total_seconds() / 3600
            decay = math.exp(-0.693 * signal_age_hours / EDGE_DECAY_HALF_LIFE_HOURS)
            kelly *= decay
        if kelly <= 0:
            continue

        # Reduce position size for short-duration markets (< 24h)
        # These have less time for stop-loss to protect and higher binary risk
        if market.end_date:
            hours_until = (market.end_date - now).total_seconds() / 3600
            if hours_until < 6:
                kelly *= 0.3  # 30% of normal size for very short markets
            elif hours_until < 24:
                kelly *= 0.5  # 50% of normal size for < 1 day markets

        # Sanity check: don't open if market is effectively decided
        # Positions at 0.001 or 0.999 are too late — market already converged
        if market.price_yes is not None and (market.price_yes <= 0.01 or market.price_yes >= 0.99):
            logger.warning(f"Skipping {market.question[:60]}: effectively decided (price={market.price_yes:.4f})")
            continue

        # Model execution costs (2% slippage)
        exec_result = compute_execution_cost(
            direction=signal.direction,
            market_price=signal.market_price,
            slippage_bps=200,
        )
        entry_price = exec_result["execution_price"]

        # Calculate quantity based on Kelly with execution price
        if signal.direction == "buy_yes":
            cost_per_share = entry_price
        else:
            cost_per_share = 1.0 - entry_price

        quantity = (config.bankroll * kelly) / max(cost_per_share, 0.01)
        position_cost = cost_per_share * quantity

        risk_check = await check_risk_limits(session, position_cost, "auto_ensemble", portfolio_type="auto", strategy="auto_ensemble")
        if not risk_check.allowed:
            logger.info(f"Ensemble auto-trade blocked: {risk_check.reason}")
            continue

        position = PortfolioPosition(
            user_id="auto_ensemble",
            market_id=signal.market_id,
            platform_id=market.platform_id,
            side="yes" if signal.direction == "buy_yes" else "no",
            entry_price=entry_price,  # Store YES price (with slippage)
            quantity=round(quantity, 2),
            entry_time=datetime.utcnow(),
            strategy="auto_ensemble",
            portfolio_type="auto",
            is_simulated=True,
        )
        session.add(position)
        await session.flush()
        created_ids.append(position.id)
        open_markets.add(signal.market_id)

        slippage_usd = abs(entry_price - signal.market_price) * quantity
        logger.info(
            f"Auto ensemble: market {signal.market_id} | {signal.direction} @ {entry_price:.3f} "
            f"(mkt {signal.market_price:.3f}, slip ${slippage_usd:.2f}) | "
            f"qty={quantity:.2f} | Kelly={kelly:.4f}"
        )

    if created_ids:
        await session.commit()
        logger.info(f"Ensemble executor: {len(created_ids)} new auto positions")

    return created_ids


async def _execute_elo_trades(session: AsyncSession) -> list[int]:
    """Auto-open paper positions for high-confidence Elo edge signals."""
    config = await _get_auto_config(session, "elo")
    if not config:
        return []

    now = datetime.utcnow()

    result = await session.execute(
        select(EloEdgeSignal)
        .where(
            EloEdgeSignal.expired_at == None,  # noqa
            EloEdgeSignal.elo_confidence >= config.min_confidence,
            EloEdgeSignal.net_edge >= config.min_net_ev,
        )
        .order_by(EloEdgeSignal.net_edge.desc())
        .limit(10)
    )
    signals = result.scalars().all()

    if not signals:
        return []

    # Markets with existing open auto positions (ANY strategy in auto portfolio)
    # Prevent duplicate positions on same market across all auto-strategies
    open_markets = set()
    open_result = await session.execute(
        select(PortfolioPosition.market_id)
        .where(
            PortfolioPosition.exit_time == None,  # noqa
            PortfolioPosition.portfolio_type == "auto",
        )
    )
    for row in open_result.all():
        open_markets.add(row[0])

    created_ids = []

    for signal in signals:
        if signal.market_id in open_markets:
            continue

        market = await session.get(Market, signal.market_id)
        if not market or not market.is_active:
            continue

        # Sanity check: don't open if market is effectively decided
        if market.price_yes is not None and (market.price_yes <= 0.01 or market.price_yes >= 0.99):
            logger.warning(f"Skipping {market.question[:60]}: effectively decided (price={market.price_yes:.4f})")
            continue

        # Derive direction from Elo probability
        elo_prob_yes = signal.elo_prob_a if signal.yes_side_player == signal.player_a else (1 - signal.elo_prob_a)
        direction = "buy_yes" if elo_prob_yes > signal.market_price_yes else "buy_no"

        kelly = signal.kelly_fraction or 0.0
        kelly = min(kelly, config.max_kelly_fraction)
        # Apply edge decay to Kelly (stale signals get smaller positions)
        if signal.detected_at:
            signal_age_hours = (now - signal.detected_at).total_seconds() / 3600
            decay = math.exp(-0.693 * signal_age_hours / EDGE_DECAY_HALF_LIFE_HOURS)
            kelly *= decay
        if kelly <= 0:
            continue

        if direction == "buy_yes":
            cost_per_share = signal.market_price_yes
        else:
            cost_per_share = 1.0 - signal.market_price_yes
        quantity = (config.bankroll * kelly) / max(cost_per_share, 0.01)
        entry_price = signal.market_price_yes  # always YES price

        position_cost = cost_per_share * quantity
        risk_check = await check_risk_limits(session, position_cost, "auto_elo", portfolio_type="auto", strategy="auto_elo")
        if not risk_check.allowed:
            logger.info(f"Elo auto-trade blocked: {risk_check.reason}")
            continue

        position = PortfolioPosition(
            user_id="auto_elo",
            market_id=signal.market_id,
            platform_id=market.platform_id,
            side="yes" if direction == "buy_yes" else "no",
            entry_price=entry_price,
            quantity=round(quantity, 2),
            entry_time=datetime.utcnow(),
            strategy="auto_elo",
            portfolio_type="auto",
            is_simulated=True,
        )
        session.add(position)
        await session.flush()
        created_ids.append(position.id)
        open_markets.add(signal.market_id)

        logger.info(
            f"Auto elo trade: market {signal.market_id} | "
            f"{direction} @ {entry_price:.3f} | "
            f"qty={quantity:.2f} | Kelly={kelly:.4f} | edge={signal.net_edge:.3f}"
        )

    if created_ids:
        await session.commit()
        logger.info(f"Elo executor: {len(created_ids)} new auto positions")

    return created_ids
