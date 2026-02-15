"""Auto paper-trade from high-confidence signals (ensemble + Elo).

Config-driven multi-strategy executor. Each strategy reads its thresholds
from AutoTradingConfig in the DB, so they can be changed at runtime via API.
All auto positions get portfolio_type="auto" for isolated risk accounting.
"""

import logging
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import (
    PortfolioPosition, Market, AutoTradingConfig,
    EnsembleEdgeSignal, EloEdgeSignal,
)
from risk.risk_manager import check_risk_limits

logger = logging.getLogger(__name__)


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

    result = await session.execute(
        select(EnsembleEdgeSignal)
        .where(
            EnsembleEdgeSignal.expired_at == None,  # noqa
            EnsembleEdgeSignal.quality_tier == config.min_quality_tier,
            EnsembleEdgeSignal.confidence >= config.min_confidence,
            EnsembleEdgeSignal.net_ev >= config.min_net_ev,
        )
        .order_by(EnsembleEdgeSignal.net_ev.desc())
        .limit(10)
    )
    signals = result.scalars().all()

    if not signals:
        return []

    # Markets with existing open auto positions (ensemble)
    open_markets = set()
    open_result = await session.execute(
        select(PortfolioPosition.market_id)
        .where(
            PortfolioPosition.exit_time == None,  # noqa
            PortfolioPosition.strategy == "auto_ensemble",
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

        kelly = signal.kelly_fraction or 0.0
        kelly = min(kelly, config.max_kelly_fraction)
        if kelly <= 0:
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

        risk_check = await check_risk_limits(session, position_cost, "auto_ensemble", portfolio_type="auto")
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

    # Markets with existing open auto positions (elo)
    open_markets = set()
    open_result = await session.execute(
        select(PortfolioPosition.market_id)
        .where(
            PortfolioPosition.exit_time == None,  # noqa
            PortfolioPosition.strategy == "auto_elo",
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

        # Derive direction from Elo probability
        elo_prob_yes = signal.elo_prob_a if signal.yes_side_player == signal.player_a else (1 - signal.elo_prob_a)
        direction = "buy_yes" if elo_prob_yes > signal.market_price_yes else "buy_no"

        kelly = signal.kelly_fraction or 0.0
        kelly = min(kelly, config.max_kelly_fraction)
        if kelly <= 0:
            continue

        if direction == "buy_yes":
            cost_per_share = signal.market_price_yes
        else:
            cost_per_share = 1.0 - signal.market_price_yes
        quantity = (config.bankroll * kelly) / max(cost_per_share, 0.01)
        entry_price = signal.market_price_yes  # always YES price

        position_cost = cost_per_share * quantity
        risk_check = await check_risk_limits(session, position_cost, "auto_elo", portfolio_type="auto")
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
