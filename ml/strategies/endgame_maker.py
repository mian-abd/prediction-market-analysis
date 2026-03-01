"""Resolution Convergence ("Endgame") Maker Strategy.

Capitalizes on the convergence of prediction market prices toward 0 or 1
as resolution approaches. Instead of *taking* the edge (and paying fees),
this strategy *makes* by posting aggressive limit orders that get filled
by impatient takers fleeing the wrong side.

Core thesis:
  When the ML model is highly confident (e.g. 95% YES) but the market
  price is lagging (e.g. 0.85), there is a ~10c edge that will collapse
  to zero at resolution. Rather than buy YES at 0.85 as a taker, we post
  a BUY YES limit at best_bid + 1c, earning the spread and potentially
  a maker rebate on fill.

Why this works near resolution:
  1. Directional edge is highest when model confidence is strong AND the
     market hasn't fully priced in the outcome.
  2. Urgency is on the wrong side: holders of NO contracts facing a YES
     resolution will aggressively sell (= hit our bid), rather than wait
     for a maker to come along.
  3. Maker economics: no taker fees, possible rebate on fee-enabled markets.
  4. Short duration: positions are held hours, not days, reducing variance.

Risks:
  - Model is wrong → binary loss on the wrong side of resolution.
  - Resolution surprise (e.g. last-minute news reversal).
  - Limit order doesn't fill → no trade, no loss, no gain.
  - Too close to resolution → exchange may halt trading.

This module is strategy logic only — no live trading or exchange interaction.
"""

import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Maker rebate assumption: on fee-enabled Polymarket markets, makers earn
# ~20% of the taker fee as a rebate. This is a conservative estimate.
# ---------------------------------------------------------------------------
DEFAULT_MAKER_REBATE_FRAC = 0.20


@dataclass
class EndgameConfig:
    min_model_confidence: float = 0.90
    min_edge_cents: float = 0.03
    max_hours_to_resolution: float = 72.0
    min_hours_to_resolution: float = 2.0
    max_position_usd: float = 200.0
    price_improvement_cents: float = 0.01
    resolution_convergence_rate: float = 0.8
    maker_rebate_frac: float = DEFAULT_MAKER_REBATE_FRAC
    max_portfolio_exposure_usd: float = 1000.0
    max_concurrent_markets: int = 10


@dataclass
class EndgameOpportunity:
    market_id: int
    direction: str  # "buy_yes" or "buy_no"
    model_prob: float
    market_price: float
    edge: float
    hours_to_resolution: float
    recommended_price: float
    recommended_size: float
    expected_profit: float
    expected_rebate: float
    confidence_score: float  # 0-1 composite combining model confidence, edge, time


class EndgameMaker:
    """Post aggressive maker orders on markets approaching resolution.

    When the ML model is highly confident about the outcome and the market
    price hasn't fully converged, post limit orders that are likely to get
    filled by takers fleeing the wrong side.
    """

    def __init__(self, config: EndgameConfig | None = None):
        self.config = config or EndgameConfig()

    # ------------------------------------------------------------------
    # 1. Opportunity evaluation
    # ------------------------------------------------------------------

    def evaluate_opportunity(
        self,
        market_id: int,
        model_prob: float,
        market_price: float,
        hours_to_resolution: float,
        taker_fee_bps: int = 0,
        best_bid: float | None = None,
        best_ask: float | None = None,
    ) -> EndgameOpportunity | None:
        """Evaluate whether a market qualifies for the endgame strategy.

        Args:
            market_id: Internal market identifier.
            model_prob: ML model's predicted P(YES = 1), range [0, 1].
            market_price: Current market YES price, range (0, 1).
            hours_to_resolution: Hours remaining until resolution.
            taker_fee_bps: Taker fee in basis points (0 for fee-free markets).
            best_bid: Current best bid on the YES order book.
            best_ask: Current best ask on the YES order book.

        Returns:
            An EndgameOpportunity if the market qualifies, otherwise None.
        """
        cfg = self.config

        # --- Gate: time window ---
        if hours_to_resolution > cfg.max_hours_to_resolution:
            logger.debug(
                "Market %d: %.1fh to resolution > %.1fh max — skipping",
                market_id, hours_to_resolution, cfg.max_hours_to_resolution,
            )
            return None
        if hours_to_resolution < cfg.min_hours_to_resolution:
            logger.debug(
                "Market %d: %.1fh to resolution < %.1fh min — too risky",
                market_id, hours_to_resolution, cfg.min_hours_to_resolution,
            )
            return None

        # --- Gate: model confidence ---
        model_confidence = max(model_prob, 1.0 - model_prob)
        if model_confidence < cfg.min_model_confidence:
            logger.debug(
                "Market %d: model confidence %.2f < %.2f threshold",
                market_id, model_confidence, cfg.min_model_confidence,
            )
            return None

        # --- Determine direction ---
        # model_prob > market_price ⇒ YES is underpriced ⇒ buy YES
        # model_prob < market_price ⇒ NO is underpriced  ⇒ buy NO
        if model_prob > market_price:
            direction = "buy_yes"
            raw_edge = model_prob - market_price
        else:
            direction = "buy_no"
            raw_edge = market_price - model_prob

        # --- Gate: minimum edge ---
        if raw_edge < cfg.min_edge_cents:
            logger.debug(
                "Market %d: edge %.3f < %.3f minimum",
                market_id, raw_edge, cfg.min_edge_cents,
            )
            return None

        # --- Compute recommended limit price ---
        if direction == "buy_yes":
            base = best_bid if best_bid is not None else market_price - 0.01
            recommended_price = min(
                base + cfg.price_improvement_cents,
                model_prob - cfg.min_edge_cents / 2,
            )
            recommended_price = math.floor(recommended_price * 100) / 100.0
        else:
            base = best_ask if best_ask is not None else market_price + 0.01
            recommended_price = max(
                base - cfg.price_improvement_cents,
                (1.0 - model_prob) + cfg.min_edge_cents / 2,
            )
            # For buy_no, the cost is (1 - recommended_price) per share when
            # expressed in YES-price terms.  Snap to tick (ceiling here because
            # higher YES price = cheaper NO).
            recommended_price = math.ceil(recommended_price * 100) / 100.0

        recommended_price = float(np.clip(recommended_price, 0.01, 0.99))

        # --- Position sizing ---
        # Size in USD, capped by config.  Scale down when edge is thin to
        # limit exposure on marginal trades.
        edge_ratio = raw_edge / max(cfg.min_edge_cents, 0.01)
        size_scalar = min(1.0, edge_ratio)
        if direction == "buy_yes":
            cost_per_share = recommended_price
        else:
            cost_per_share = 1.0 - recommended_price
        cost_per_share = max(cost_per_share, 0.01)
        max_shares = cfg.max_position_usd / cost_per_share
        recommended_size = round(max_shares * size_scalar, 2)

        # --- Expected profit ---
        # At resolution, winning shares pay $1.  Expected PnL per share:
        #   buy_yes: model_prob * (1 - entry) - (1 - model_prob) * entry
        #   buy_no:  (1 - model_prob) * (1 - (1 - entry)) - model_prob * (1 - entry)
        #          = (1 - model_prob) * entry - model_prob * (1 - entry)  [entry in YES terms]
        if direction == "buy_yes":
            ev_per_share = model_prob * (1.0 - recommended_price) - (1.0 - model_prob) * recommended_price
        else:
            ev_per_share = (1.0 - model_prob) * recommended_price - model_prob * (1.0 - recommended_price)

        # Discount by convergence rate: not all edge is realized (some markets
        # resolve with residual pricing noise).
        ev_per_share *= cfg.resolution_convergence_rate

        expected_profit = round(ev_per_share * recommended_size, 4)

        # --- Maker rebate ---
        fee_rate = taker_fee_bps / 10000.0
        if fee_rate > 0 and cfg.maker_rebate_frac > 0:
            # Rebate applies when our resting order gets filled.  Estimate
            # rebate assuming full fill and win.
            win_prob = model_prob if direction == "buy_yes" else (1.0 - model_prob)
            winnings_per_share = (1.0 - recommended_price) if direction == "buy_yes" else recommended_price
            taker_fee_per_share = fee_rate * winnings_per_share
            rebate_per_share = taker_fee_per_share * cfg.maker_rebate_frac
            expected_rebate = round(rebate_per_share * recommended_size * win_prob, 4)
        else:
            expected_rebate = 0.0

        # --- Composite confidence score ---
        confidence_score = self._compute_confidence_score(
            model_confidence, raw_edge, hours_to_resolution,
        )

        opp = EndgameOpportunity(
            market_id=market_id,
            direction=direction,
            model_prob=round(model_prob, 4),
            market_price=round(market_price, 4),
            edge=round(raw_edge, 4),
            hours_to_resolution=round(hours_to_resolution, 2),
            recommended_price=recommended_price,
            recommended_size=recommended_size,
            expected_profit=expected_profit,
            expected_rebate=expected_rebate,
            confidence_score=round(confidence_score, 4),
        )

        logger.info(
            "Market %d: endgame %s | edge=%.3f | price=%.2f | "
            "size=%.1f | E[profit]=$%.2f | conf=%.3f | %.1fh left",
            market_id, direction, raw_edge, recommended_price,
            recommended_size, expected_profit, confidence_score,
            hours_to_resolution,
        )
        return opp

    # ------------------------------------------------------------------
    # 2. Order computation
    # ------------------------------------------------------------------

    def compute_maker_orders(
        self,
        opportunity: EndgameOpportunity,
        best_bid: float,
        best_ask: float,
        taker_fee_bps: int = 0,
    ) -> list[dict]:
        """Compute specific limit orders for an endgame opportunity.

        If the model says YES is underpriced (buy_yes):
          → Post BUY YES limit at best_bid + price_improvement.
        If the model says NO is underpriced (buy_no):
          → Post SELL YES limit above model value (equivalently, BUY NO).

        Returns a list of order dicts ready for submission (sans exchange details).
        """
        cfg = self.config
        orders: list[dict] = []

        if opportunity.direction == "buy_yes":
            limit_price = min(
                best_bid + cfg.price_improvement_cents,
                opportunity.model_prob - cfg.min_edge_cents / 2,
            )
            limit_price = math.floor(limit_price * 100) / 100.0
            limit_price = float(np.clip(limit_price, 0.01, 0.99))

            orders.append({
                "market_id": opportunity.market_id,
                "side": "buy",
                "token": "yes",
                "price": limit_price,
                "size": opportunity.recommended_size,
                "order_type": "GTC",
                "strategy": "endgame_maker",
                "direction": "buy_yes",
                "edge": opportunity.edge,
                "confidence": opportunity.confidence_score,
            })

        else:  # buy_no → post SELL YES limit (or equivalently BUY NO)
            limit_price = max(
                best_ask - cfg.price_improvement_cents,
                opportunity.model_prob + cfg.min_edge_cents / 2,
            )
            limit_price = math.ceil(limit_price * 100) / 100.0
            limit_price = float(np.clip(limit_price, 0.01, 0.99))

            # Express as both SELL YES and BUY NO for flexibility.
            orders.append({
                "market_id": opportunity.market_id,
                "side": "sell",
                "token": "yes",
                "price": limit_price,
                "size": opportunity.recommended_size,
                "order_type": "GTC",
                "strategy": "endgame_maker",
                "direction": "buy_no",
                "edge": opportunity.edge,
                "confidence": opportunity.confidence_score,
            })

        maker_rebate_note = ""
        if taker_fee_bps > 0:
            rebate_bps = taker_fee_bps * cfg.maker_rebate_frac
            maker_rebate_note = f" (est. rebate: {rebate_bps:.1f} bps)"

        for o in orders:
            logger.info(
                "Endgame order: market %d %s %s @ %.2f x %.1f%s",
                o["market_id"], o["side"], o["token"],
                o["price"], o["size"], maker_rebate_note,
            )

        return orders

    # ------------------------------------------------------------------
    # 3. Portfolio scoring
    # ------------------------------------------------------------------

    def score_portfolio(
        self,
        opportunities: list[EndgameOpportunity],
    ) -> list[EndgameOpportunity]:
        """Rank opportunities by expected profit per hour and apply limits.

        Returns the filtered, sorted list of opportunities that fit within
        portfolio-level constraints.
        """
        cfg = self.config

        if not opportunities:
            return []

        # Compute profit-per-hour for ranking (Sharpe-like: reward / time-at-risk)
        scored: list[tuple[float, EndgameOpportunity]] = []
        for opp in opportunities:
            hours_at_risk = max(opp.hours_to_resolution, 0.5)
            profit_per_hour = (opp.expected_profit + opp.expected_rebate) / hours_at_risk
            scored.append((profit_per_hour, opp))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Apply portfolio constraints
        selected: list[EndgameOpportunity] = []
        total_exposure = 0.0

        for profit_per_hour, opp in scored:
            if len(selected) >= cfg.max_concurrent_markets:
                logger.debug(
                    "Portfolio cap: %d markets reached, skipping market %d",
                    cfg.max_concurrent_markets, opp.market_id,
                )
                break

            if opp.direction == "buy_yes":
                position_cost = opp.recommended_price * opp.recommended_size
            else:
                position_cost = (1.0 - opp.recommended_price) * opp.recommended_size

            if total_exposure + position_cost > cfg.max_portfolio_exposure_usd:
                logger.debug(
                    "Portfolio exposure cap: $%.0f + $%.0f > $%.0f, skipping market %d",
                    total_exposure, position_cost, cfg.max_portfolio_exposure_usd,
                    opp.market_id,
                )
                continue

            total_exposure += position_cost
            selected.append(opp)

            logger.info(
                "Portfolio select: market %d | $/hr=%.4f | exposure=$%.0f cumul=$%.0f",
                opp.market_id, profit_per_hour, position_cost, total_exposure,
            )

        logger.info(
            "Portfolio scored: %d candidates → %d selected | total exposure $%.0f",
            len(opportunities), len(selected), total_exposure,
        )
        return selected

    # ------------------------------------------------------------------
    # 4. Backtest on resolved markets
    # ------------------------------------------------------------------

    def backtest_on_resolved(self, markets_data: list[dict]) -> dict:
        """Simple backtest on historically resolved markets.

        For each market, check whether the endgame strategy would have
        entered. If so, assume fill at our limit price and settle at the
        binary resolution outcome (0 or 1).

        Args:
            markets_data: List of dicts, each containing:
                - market_id (int)
                - model_prob (float): ML model's P(YES) at entry time
                - market_price (float): market YES price at entry time
                - hours_to_resolution (float)
                - resolution_outcome (int): 1 if YES won, 0 if NO won
                - taker_fee_bps (int, optional): defaults to 0
                - best_bid (float, optional)
                - best_ask (float, optional)

        Returns:
            Dict with aggregate P&L statistics.
        """
        results: list[dict] = []
        total_profit = 0.0
        total_rebate = 0.0
        wins = 0
        losses = 0

        for m in markets_data:
            market_id = m["market_id"]
            resolution = m["resolution_outcome"]
            taker_fee_bps = m.get("taker_fee_bps", 0)

            opp = self.evaluate_opportunity(
                market_id=market_id,
                model_prob=m["model_prob"],
                market_price=m["market_price"],
                hours_to_resolution=m["hours_to_resolution"],
                taker_fee_bps=taker_fee_bps,
                best_bid=m.get("best_bid"),
                best_ask=m.get("best_ask"),
            )

            if opp is None:
                results.append({
                    "market_id": market_id,
                    "entered": False,
                    "pnl": 0.0,
                })
                continue

            # Settle at binary outcome
            if opp.direction == "buy_yes":
                cost_per_share = opp.recommended_price
                payout_per_share = 1.0 if resolution == 1 else 0.0
                pnl_per_share = payout_per_share - cost_per_share
            else:
                # buy_no: we sold YES at recommended_price
                revenue_per_share = opp.recommended_price
                liability_per_share = 1.0 if resolution == 1 else 0.0
                pnl_per_share = revenue_per_share - liability_per_share

            pnl = pnl_per_share * opp.recommended_size
            rebate = opp.expected_rebate if pnl > 0 else 0.0

            total_profit += pnl
            total_rebate += rebate
            if pnl > 0:
                wins += 1
            else:
                losses += 1

            results.append({
                "market_id": market_id,
                "entered": True,
                "direction": opp.direction,
                "entry_price": opp.recommended_price,
                "size": opp.recommended_size,
                "resolution": resolution,
                "pnl": round(pnl, 4),
                "rebate": round(rebate, 4),
                "edge": opp.edge,
                "confidence": opp.confidence_score,
                "hours_to_resolution": opp.hours_to_resolution,
            })

        n_entered = wins + losses
        win_rate = wins / max(n_entered, 1)
        avg_pnl = total_profit / max(n_entered, 1)

        # Compute Sharpe-like ratio on per-trade PnL
        pnls = [r["pnl"] for r in results if r["entered"]]
        pnl_std = float(np.std(pnls)) if len(pnls) > 1 else 1.0

        summary = {
            "n_markets": len(markets_data),
            "n_entered": n_entered,
            "n_skipped": len(markets_data) - n_entered,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 4),
            "total_pnl": round(total_profit, 4),
            "total_rebates": round(total_rebate, 4),
            "net_pnl": round(total_profit + total_rebate, 4),
            "avg_pnl_per_trade": round(avg_pnl, 4),
            "pnl_std": round(pnl_std, 4),
            "sharpe": round(avg_pnl / max(pnl_std, 0.001), 4),
            "trades": results,
        }

        logger.info(
            "Endgame backtest: %d markets, %d entered, %d wins / %d losses | "
            "PnL=$%.2f | rebates=$%.2f | net=$%.2f | Sharpe=%.2f",
            len(markets_data), n_entered, wins, losses,
            total_profit, total_rebate, total_profit + total_rebate,
            summary["sharpe"],
        )
        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_confidence_score(
        self,
        model_confidence: float,
        edge: float,
        hours_to_resolution: float,
    ) -> float:
        """Composite confidence score combining model confidence, edge, time.

        Components (equally weighted):
          1. Model confidence: linear ramp from min_confidence threshold to 1.0.
          2. Edge magnitude: sigmoid-like scaling — rapidly increasing up to
             ~10c, then diminishing returns.
          3. Time factor: closer to resolution = higher confidence (the market
             has less time to move against us), but penalise below min_hours.

        Returns a value in [0, 1].
        """
        cfg = self.config

        # Model confidence component
        conf_floor = cfg.min_model_confidence
        conf_component = (model_confidence - conf_floor) / (1.0 - conf_floor)
        conf_component = float(np.clip(conf_component, 0.0, 1.0))

        # Edge component (sigmoid-like: quick ramp to 1.0 around 8-10c edge)
        edge_component = 1.0 - math.exp(-edge / 0.06)
        edge_component = float(np.clip(edge_component, 0.0, 1.0))

        # Time component: prefer 4-24h window (sweet spot)
        # Linearly ramp from min_hours to ~6h, plateau 6-24h, gentle decay beyond
        if hours_to_resolution < 6.0:
            time_component = hours_to_resolution / 6.0
        elif hours_to_resolution <= 24.0:
            time_component = 1.0
        else:
            time_component = max(0.3, 1.0 - (hours_to_resolution - 24.0) / cfg.max_hours_to_resolution)
        time_component = float(np.clip(time_component, 0.0, 1.0))

        composite = (
            conf_component * 0.40
            + edge_component * 0.35
            + time_component * 0.25
        )
        return float(np.clip(composite, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Batch simulation entry point
# ---------------------------------------------------------------------------

def simulate_endgame_batch(
    markets: list[dict],
    config: EndgameConfig | None = None,
) -> dict:
    """Run the endgame strategy simulation on a batch of markets.

    Convenience wrapper around EndgameMaker.backtest_on_resolved() that
    also runs portfolio scoring to report which markets would have been
    selected under portfolio constraints.

    Args:
        markets: List of market dicts.  Required keys per dict:
            - market_id (int)
            - model_prob (float)
            - market_price (float)
            - hours_to_resolution (float)
            - resolution_outcome (int): 1 = YES won, 0 = NO won
          Optional:
            - taker_fee_bps (int)
            - best_bid (float)
            - best_ask (float)
        config: Strategy configuration (uses defaults if None).

    Returns:
        Dict with 'backtest' (full results) and 'portfolio' (scored subset).
    """
    maker = EndgameMaker(config)

    # Phase 1: evaluate all opportunities (ignoring resolution for scoring)
    opportunities: list[EndgameOpportunity] = []
    for m in markets:
        opp = maker.evaluate_opportunity(
            market_id=m["market_id"],
            model_prob=m["model_prob"],
            market_price=m["market_price"],
            hours_to_resolution=m["hours_to_resolution"],
            taker_fee_bps=m.get("taker_fee_bps", 0),
            best_bid=m.get("best_bid"),
            best_ask=m.get("best_ask"),
        )
        if opp is not None:
            opportunities.append(opp)

    # Phase 2: portfolio scoring
    portfolio = maker.score_portfolio(opportunities)
    portfolio_ids = {opp.market_id for opp in portfolio}

    # Phase 3: backtest (full set, not just portfolio — for comparison)
    backtest = maker.backtest_on_resolved(markets)

    # Tag portfolio trades in backtest results
    for trade in backtest.get("trades", []):
        trade["in_portfolio"] = trade["market_id"] in portfolio_ids

    # Compute portfolio-only P&L
    portfolio_trades = [t for t in backtest["trades"] if t.get("in_portfolio") and t.get("entered")]
    portfolio_pnl = sum(t["pnl"] for t in portfolio_trades)
    portfolio_rebates = sum(t.get("rebate", 0) for t in portfolio_trades)

    result = {
        "backtest": backtest,
        "portfolio": {
            "n_selected": len(portfolio),
            "n_traded": len(portfolio_trades),
            "portfolio_pnl": round(portfolio_pnl, 4),
            "portfolio_rebates": round(portfolio_rebates, 4),
            "portfolio_net": round(portfolio_pnl + portfolio_rebates, 4),
            "selected_market_ids": sorted(portfolio_ids),
        },
    }

    logger.info(
        "Endgame batch: %d markets → %d opps → %d portfolio | "
        "full PnL=$%.2f | portfolio PnL=$%.2f",
        len(markets), len(opportunities), len(portfolio),
        backtest["net_pnl"], result["portfolio"]["portfolio_net"],
    )
    return result
