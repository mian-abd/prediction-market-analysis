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
    EnsembleEdgeSignal, EloEdgeSignal, FavoriteLongshotEdgeSignal,
    StrategySignal, OrderbookSnapshot,
)
from risk.risk_manager import check_risk_limits

logger = logging.getLogger(__name__)

# Per-strategy edge decay half-lives (hours). Short-lived signal types decay faster;
# structural edges (longshot bias, market clustering) persist much longer.
STRATEGY_DECAY_HOURS: dict[str, float] = {
    "ensemble": 2.0,
    "elo": 2.0,
    "longshot_bias": 24.0,        # Structural bias persists for days
    "resolution_convergence": 12.0,  # Theta decays as market nears resolution
    "llm_forecast": 6.0,           # LLM analysis stays valid for several hours
    "news_catalyst": 1.0,          # News stales very quickly
    "orderflow": 0.5,              # Flow signals are extremely short-lived
    "smart_money": 4.0,            # On-chain positions shift over hours
    "market_clustering": 12.0,     # Correlation-based edges are slow-moving
    "consensus": 6.0,              # Multi-strategy agreement
}
_DEFAULT_DECAY_HOURS = 2.0


def _get_decay_hours(strategy: str) -> float:
    """Return edge decay half-life for a strategy (strips 'auto_' prefix)."""
    name = strategy.removeprefix("auto_")
    return STRATEGY_DECAY_HOURS.get(name, _DEFAULT_DECAY_HOURS)


async def _get_market_slippage_bps(session: AsyncSession, market: Market) -> float:
    """Compute realistic slippage from orderbook spread or liquidity tier.

    Priority:
    1. Recent orderbook snapshot → half-spread as taker slippage
    2. Fallback → liquidity-tiered flat rate
    """
    try:
        result = await session.execute(
            select(OrderbookSnapshot.bid_ask_spread)
            .where(OrderbookSnapshot.market_id == market.id)
            .order_by(OrderbookSnapshot.timestamp.desc())
            .limit(1)
        )
        spread = result.scalar_one_or_none()
        if spread is not None and spread > 0:
            # Half-spread = typical taker cost; convert to basis points
            slippage_bps = (spread / 2) * 10_000
            return min(500.0, max(20.0, slippage_bps))  # Cap 0.2% – 5%
    except Exception:
        pass

    liquidity = float(market.liquidity or 0)
    if liquidity >= 50_000:
        return 50.0    # 0.5%
    elif liquidity >= 10_000:
        return 100.0   # 1.0%
    elif liquidity >= 5_000:
        return 200.0   # 2.0%
    return 300.0       # 3.0%

# Markets with these keywords are blocked from auto-trading (no edge, pure noise)
BLOCKED_KEYWORDS = [
    "jesus christ", "gta vi", "gta 6", "bridgerton", "will.*die before",
    "alien", "rapture", "second coming", "fdv above", "fdv below",
    "mcap above", "market cap above.*one day", "before gta",
    "nothing ever happens",
]
# Categories blocked from auto-trading
BLOCKED_CATEGORIES = {"entertainment", "novelty"}

import re
_BLOCKED_PATTERNS = [re.compile(kw, re.IGNORECASE) for kw in BLOCKED_KEYWORDS]


def _is_blocked_market(question: str, category: str) -> bool:
    """Check if a market should be blocked from auto-trading."""
    if category.lower() in BLOCKED_CATEGORIES:
        return True
    for pattern in _BLOCKED_PATTERNS:
        if pattern.search(question):
            return True
    return False


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
    """Orchestrator: run ensemble, Elo, favorite-longshot, AND new strategy auto-trading."""
    created = []
    created.extend(await _execute_ensemble_trades(session))
    created.extend(await _execute_elo_trades(session))
    created.extend(await _execute_favorite_longshot_trades(session))
    created.extend(await _execute_new_strategy_trades(session))
    return created


async def _execute_ensemble_trades(session: AsyncSession) -> list[int]:
    """Auto-open paper positions for new high-confidence ensemble signals."""
    config = await _get_auto_config(session, "ensemble")
    if not config:
        return []

    # Build quality tier filter: include target tier and tiers above it
    # "speculative" is now allowed — the payoff asymmetry factor in edge detection
    # already suppresses speculative signals on expensive contracts (where they're noise)
    # while allowing them on cheap contracts (where large edges = real alpha)
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
    ensemble_decay = _get_decay_hours("ensemble")

    def urgency_score(signal, market):
        base_ev = signal.net_ev
        # Edge decay: exponential decay based on signal age
        if signal.detected_at:
            signal_age_hours = (now - signal.detected_at).total_seconds() / 3600
            decay = math.exp(-0.693 * signal_age_hours / ensemble_decay)
            base_ev *= decay
        # Skip markets too close to resolution (no time for stop-loss)
        if market.end_date:
            hours_until = (market.end_date - now).total_seconds() / 3600
            if hours_until < 2:
                return -1
        return base_ev

    # Filter out markets too close to resolution, sort by decayed EV (cache score to avoid double-call)
    scored = [(s, m, urgency_score(s, m)) for s, m in signal_market_pairs]
    scored = [(s, m, sc) for s, m, sc in scored if sc > 0]
    scored.sort(key=lambda t: t[2], reverse=True)
    signals = [s for s, m, sc in scored[:10]]

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

        # Skip blocked markets (meme, novelty, entertainment — no edge possible)
        question = market.question or ""
        if _is_blocked_market(question, category):
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
        # Apply strategy-specific edge decay to Kelly (stale signals get smaller positions)
        if signal.detected_at:
            signal_age_hours = (now - signal.detected_at).total_seconds() / 3600
            decay = math.exp(-0.693 * signal_age_hours / ensemble_decay)
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

        # Dynamic slippage from orderbook depth or liquidity tier
        slippage_bps = await _get_market_slippage_bps(session, market)
        exec_result = compute_execution_cost(
            direction=signal.direction,
            market_price=signal.market_price,
            slippage_bps=slippage_bps,
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
        # Apply strategy-specific edge decay to Kelly
        elo_decay = _get_decay_hours("elo")
        if signal.detected_at:
            signal_age_hours = (now - signal.detected_at).total_seconds() / 3600
            decay = math.exp(-0.693 * signal_age_hours / elo_decay)
            kelly *= decay
        if kelly <= 0:
            continue

        # Penalty zone: prices 0.40-0.60 are most efficiently priced
        if signal.market_price_yes is not None and 0.40 <= signal.market_price_yes <= 0.60:
            kelly *= 0.5

        # Dynamic slippage from orderbook depth or liquidity tier
        slippage_bps = await _get_market_slippage_bps(session, market)
        exec_result = compute_execution_cost(
            direction=direction,
            market_price=signal.market_price_yes,
            slippage_bps=slippage_bps,
        )
        entry_price = exec_result["execution_price"]

        if direction == "buy_yes":
            cost_per_share = entry_price
        else:
            cost_per_share = 1.0 - entry_price
        quantity = (config.bankroll * kelly) / max(cost_per_share, 0.01)

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


async def _execute_favorite_longshot_trades(session: AsyncSession) -> list[int]:
    """Auto-open paper positions for favorite-longshot bias signals (FavoriteLongshotEdgeSignal table)."""
    config = await _get_auto_config(session, "favorite_longshot")
    if not config:
        return []

    now = datetime.utcnow()

    result = await session.execute(
        select(FavoriteLongshotEdgeSignal)
        .where(
            FavoriteLongshotEdgeSignal.expired_at == None,  # noqa
            FavoriteLongshotEdgeSignal.net_ev >= config.min_net_ev,
        )
        .order_by(FavoriteLongshotEdgeSignal.net_ev.desc())
        .limit(10)
    )
    signals = result.scalars().all()

    if not signals:
        return []

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

        category = (market.normalized_category or market.category or "").lower()
        if _is_blocked_market(market.question or "", category):
            continue

        if market.price_yes is not None and (market.price_yes <= 0.01 or market.price_yes >= 0.99):
            continue

        kelly = signal.kelly_fraction or 0.0
        kelly = min(kelly, config.max_kelly_fraction)
        decay_hours = _get_decay_hours("favorite_longshot")
        if signal.detected_at:
            signal_age_hours = (now - signal.detected_at).total_seconds() / 3600
            decay = math.exp(-0.693 * signal_age_hours / decay_hours)
            kelly *= decay
        if kelly <= 0:
            continue

        slippage_bps = await _get_market_slippage_bps(session, market)
        exec_result = compute_execution_cost(
            direction=signal.direction,
            market_price=signal.market_price,
            slippage_bps=slippage_bps,
        )
        entry_price = exec_result["execution_price"]

        if signal.direction == "buy_yes":
            cost_per_share = entry_price
        else:
            cost_per_share = 1.0 - entry_price
        quantity = (config.bankroll * kelly) / max(cost_per_share, 0.01)
        position_cost = cost_per_share * quantity

        risk_check = await check_risk_limits(
            session, position_cost, "auto_favorite_longshot",
            portfolio_type="auto", strategy="auto_favorite_longshot"
        )
        if not risk_check.allowed:
            logger.info(f"Favorite-longshot auto-trade blocked: {risk_check.reason}")
            continue

        position = PortfolioPosition(
            user_id="auto_favorite_longshot",
            market_id=signal.market_id,
            platform_id=market.platform_id,
            side="yes" if signal.direction == "buy_yes" else "no",
            entry_price=entry_price,
            quantity=round(quantity, 2),
            entry_time=datetime.utcnow(),
            strategy="auto_favorite_longshot",
            portfolio_type="auto",
            is_simulated=True,
        )
        session.add(position)
        await session.flush()
        created_ids.append(position.id)
        open_markets.add(signal.market_id)

        logger.info(
            f"Auto favorite-longshot: market {signal.market_id} | {signal.direction} @ {entry_price:.3f} | "
            f"qty={quantity:.2f} | Kelly={kelly:.4f} | EV={signal.net_ev:.3f}"
        )

    if created_ids:
        await session.commit()
        logger.info(f"Favorite-longshot executor: {len(created_ids)} new auto positions")

    return created_ids


async def _execute_new_strategy_trades(session: AsyncSession) -> list[int]:
    """Auto-open paper positions for high-confidence signals from new strategies.

    Supports: llm_forecast, longshot_bias, news_catalyst, resolution_convergence,
    orderflow, smart_money, market_clustering.

    Uses a unified auto-trading config keyed by 'new_strategies'.
    Falls back to ensemble config thresholds if no dedicated config exists.
    """
    # Try dedicated config, fall back to ensemble config thresholds
    config = await _get_auto_config(session, "new_strategies")
    if not config:
        config = await _get_auto_config(session, "ensemble")
    if not config:
        return []

    now = datetime.utcnow()

    result = await session.execute(
        select(StrategySignal)
        .where(
            StrategySignal.expired_at == None,  # noqa
            StrategySignal.direction != None,  # noqa
            StrategySignal.net_ev >= config.min_net_ev,
        )
        .order_by(StrategySignal.net_ev.desc())
        .limit(20)
    )
    signals = result.scalars().all()

    if not signals:
        return []

    # Filter by confidence (use slightly lower bar since these are diversified strategies)
    min_conf = max(0.3, config.min_confidence - 0.1)
    signals = [s for s in signals if (s.confidence or 0) >= min_conf]

    # Markets with existing open auto positions
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

        # Skip effectively decided markets
        if market.price_yes is not None and (market.price_yes <= 0.01 or market.price_yes >= 0.99):
            continue

        # Skip blocked markets
        question = market.question or ""
        category = (market.normalized_category or market.category or "").lower()
        if _is_blocked_market(question, category):
            continue

        kelly = signal.kelly_fraction or 0.0
        kelly = min(kelly, config.max_kelly_fraction)

        # Strategy-specific edge decay (uses signal.strategy as key)
        strategy = signal.strategy or ""
        signal_decay_hours = _get_decay_hours(strategy)
        if signal.detected_at:
            signal_age_hours = (now - signal.detected_at).total_seconds() / 3600
            decay = math.exp(-0.693 * signal_age_hours / signal_decay_hours)
            kelly *= decay
        if kelly <= 0:
            continue

        # Strategy-specific Kelly adjustments
        if strategy == "consensus":
            kelly *= 1.5  # Multi-strategy agreement = highest conviction
        elif strategy == "resolution_convergence":
            kelly *= 1.2  # Slightly more aggressive for high win-rate strategy
        elif strategy == "longshot_bias":
            kelly *= 1.0  # Standard — well-documented structural edge
        elif strategy in ("news_catalyst", "orderflow"):
            kelly *= 0.7  # More conservative — shorter-lived signals
        elif strategy in ("smart_money", "market_clustering"):
            kelly *= 0.8  # Upgraded: now uses real on-chain data

        # Dynamic slippage from orderbook depth or liquidity tier
        slippage_bps = await _get_market_slippage_bps(session, market)
        exec_result = compute_execution_cost(
            direction=signal.direction,
            market_price=signal.market_price,
            slippage_bps=slippage_bps,
        )
        entry_price = exec_result["execution_price"]

        if signal.direction == "buy_yes":
            cost_per_share = entry_price
        else:
            cost_per_share = 1.0 - entry_price

        quantity = (config.bankroll * kelly) / max(cost_per_share, 0.01)
        position_cost = cost_per_share * quantity

        user_id = f"auto_{strategy}"
        risk_check = await check_risk_limits(
            session, position_cost, user_id,
            portfolio_type="auto", strategy=user_id
        )
        if not risk_check.allowed:
            logger.info(f"{strategy} auto-trade blocked: {risk_check.reason}")
            continue

        position = PortfolioPosition(
            user_id=user_id,
            market_id=signal.market_id,
            platform_id=market.platform_id,
            side="yes" if signal.direction == "buy_yes" else "no",
            entry_price=entry_price,
            quantity=round(quantity, 2),
            entry_time=datetime.utcnow(),
            strategy=user_id,
            portfolio_type="auto",
            is_simulated=True,
        )
        session.add(position)
        await session.flush()
        created_ids.append(position.id)
        open_markets.add(signal.market_id)

        logger.info(
            f"Auto {strategy}: market {signal.market_id} | "
            f"{signal.direction} @ {entry_price:.3f} | "
            f"qty={quantity:.2f} | Kelly={kelly:.4f} | EV={signal.net_ev:.3f}"
        )

    if created_ids:
        await session.commit()
        logger.info(f"New strategy executor: {len(created_ids)} auto positions")

    return created_ids
