"""Paper-trade the full maker strategy stack.

Combines:
1. ML-Informed Market Making (Avellaneda-Stoikov with model skewing)
2. Endgame Resolution Convergence (aggressive maker orders near resolution)
3. CLOB Manager in dry-run mode (logs orders without posting)

Runs a single cycle: fetches live market data, generates quotes, logs
what it would do. Designed to be run periodically (e.g., every 5 minutes
via cron or a loop) to build a paper P&L track record.

Usage:
    python scripts/run_paper_mm.py              # single cycle
    python scripts/run_paper_mm.py --loop 300   # loop every 300 seconds
"""

import argparse
import asyncio
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ml.strategies.ml_informed_mm import MLInformedMarketMaker, MLMMConfig, select_markets_for_mm
from ml.strategies.endgame_maker import EndgameMaker, EndgameConfig
from execution.clob_manager import CLOBManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("paper_mm")

DB_PATH = project_root / "data" / "markets.db"
PAPER_LOG = project_root / "data" / "paper_trades.jsonl"
MODEL_CARD_PATH = project_root / "ml" / "saved_models" / "model_card.json"


def load_live_markets(max_markets: int = 30) -> list[dict]:
    """Load active markets with recent orderbook data from DB."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    query = """
        SELECT
            m.id,
            m.question,
            m.token_id_yes,
            m.token_id_no,
            m.price_yes,
            m.end_date,
            m.volume_24h,
            m.liquidity,
            m.taker_fee_bps,
            m.is_active
        FROM markets m
        WHERE m.is_active = 1
          AND m.resolved_at IS NULL
          AND m.token_id_yes IS NOT NULL
          AND m.price_yes IS NOT NULL
          AND m.price_yes > 0.05
          AND m.price_yes < 0.95
        ORDER BY m.volume_total DESC
        LIMIT ?
    """
    rows = conn.execute(query, (max_markets,)).fetchall()
    markets = [dict(r) for r in rows]

    now = datetime.now(timezone.utc)
    for m in markets:
        if m.get("end_date"):
            try:
                end = datetime.fromisoformat(m["end_date"])
                if end.tzinfo is None:
                    end = end.replace(tzinfo=timezone.utc)
                m["tau_hours"] = max((end - now).total_seconds() / 3600, 0.1)
            except (ValueError, TypeError):
                m["tau_hours"] = 720.0
        else:
            m["tau_hours"] = 720.0

    conn.close()
    return markets


def load_price_history(market_id: int, limit: int = 50) -> list[float]:
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT price_yes FROM price_snapshots "
        "WHERE market_id = ? ORDER BY timestamp DESC LIMIT ?",
        (market_id, limit),
    ).fetchall()
    conn.close()
    return [r[0] for r in reversed(rows)] if rows else []


def get_model_prediction(price_yes: float) -> tuple[float, float]:
    """Get a model prediction. For now, uses the calibration model heuristic.

    Returns (model_fair_value, confidence).
    In production, this would load the saved ensemble and run inference.
    """
    try:
        import joblib
        cal_path = project_root / "ml" / "saved_models" / "post_calibrator.joblib"
        if cal_path.exists():
            cal = joblib.load(cal_path)
            pred = cal.predict([price_yes])[0]
            confidence = 1.0 - 2.0 * abs(pred - 0.5)
            return float(pred), float(max(confidence, 0.1))
    except Exception:
        pass

    return float(price_yes), 0.5


def run_cycle(
    mm: MLInformedMarketMaker,
    endgame: EndgameMaker,
    clob: CLOBManager,
    max_markets: int = 30,
) -> dict:
    """Run a single paper trading cycle."""
    cycle_start = time.time()
    now_str = datetime.now(timezone.utc).isoformat()

    markets = load_live_markets(max_markets)
    logger.info(f"Loaded {len(markets)} active markets")

    mm_candidates = select_markets_for_mm(markets)
    logger.info(f"  {len(mm_candidates)} qualify for market making")

    mm_quotes = []
    endgame_opps = []
    paper_orders = []

    for m in mm_candidates[:15]:
        market_id = m["id"]
        price_yes = m["price_yes"]
        tau_hours = m.get("tau_hours", 720.0)
        fee_bps = m.get("taker_fee_bps", 0) or 0

        price_history = load_price_history(market_id)
        if len(price_history) < 3:
            continue

        model_fv, model_conf = get_model_prediction(price_yes)

        spread_info = clob.get_spread(m["token_id_yes"])
        best_bid = spread_info.get("best_bid")
        best_ask = spread_info.get("best_ask")
        mid = (best_bid + best_ask) / 2.0 if best_bid and best_ask else price_yes

        bid_q, ask_q, action, debug = mm.compute_skewed_quotes(
            market_id=market_id,
            mid=mid,
            price_history=price_history,
            tau_hours=tau_hours,
            model_fair_value=model_fv,
            model_confidence=model_conf,
            best_bid=best_bid,
            best_ask=best_ask,
            bid_depth_usd=m.get("liquidity", 500) / 2,
            ask_depth_usd=m.get("liquidity", 500) / 2,
            taker_fee_bps=fee_bps,
        )

        if action.value == "post" and (bid_q or ask_q):
            order_ids = clob.quotes_from_mm_engine(
                bid_q, ask_q,
                token_id_yes=m["token_id_yes"],
                token_id_no=m.get("token_id_no", ""),
            )

            entry = {
                "timestamp": now_str,
                "strategy": "ml_mm",
                "market_id": market_id,
                "question": m["question"][:80],
                "mid": round(mid, 4),
                "model_fv": round(model_fv, 4),
                "bid": bid_q.price if bid_q else None,
                "ask": ask_q.price if ask_q else None,
                "bid_size": round(bid_q.size, 2) if bid_q else 0,
                "ask_size": round(ask_q.size, 2) if ask_q else 0,
                "skew": debug.get("skew_applied", 0),
                "tau_hours": round(tau_hours, 1),
                "action": action.value,
                "dry_run": clob.dry_run,
                "order_ids": order_ids,
            }
            mm_quotes.append(entry)
            paper_orders.append(entry)

    for m in markets:
        price_yes = m["price_yes"]
        tau_hours = m.get("tau_hours", 720.0)
        model_fv, model_conf = get_model_prediction(price_yes)
        fee_bps = m.get("taker_fee_bps", 0) or 0

        spread_info = clob.get_spread(m["token_id_yes"])
        best_bid = spread_info.get("best_bid")
        best_ask = spread_info.get("best_ask")

        opp = endgame.evaluate_opportunity(
            market_id=m["id"],
            model_prob=model_fv,
            market_price=price_yes,
            hours_to_resolution=tau_hours,
            best_bid=best_bid or (price_yes - 0.01),
            best_ask=best_ask or (price_yes + 0.01),
            taker_fee_bps=fee_bps,
        )
        if opp is not None:
            endgame_opps.append(opp)

    if endgame_opps:
        ranked = endgame.score_portfolio(endgame_opps)
        for opp in ranked[:5]:
            orders = endgame.compute_maker_orders(
                opp,
                best_bid=opp.recommended_price - 0.01,
                best_ask=opp.recommended_price + 0.01,
                taker_fee_bps=0,
            )
            entry = {
                "timestamp": now_str,
                "strategy": "endgame",
                "market_id": opp.market_id,
                "direction": opp.direction,
                "model_prob": round(opp.model_prob, 4),
                "market_price": round(opp.market_price, 4),
                "edge": round(opp.edge, 4),
                "recommended_price": round(opp.recommended_price, 4),
                "expected_profit": round(opp.expected_profit, 4),
                "tau_hours": round(opp.hours_to_resolution, 1),
                "confidence_score": round(opp.confidence_score, 4),
                "n_orders": len(orders),
                "dry_run": True,
            }
            endgame_opps_log = entry
            paper_orders.append(entry)

    if paper_orders:
        PAPER_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(PAPER_LOG, "a") as f:
            for order in paper_orders:
                f.write(json.dumps(order) + "\n")

    elapsed = time.time() - cycle_start
    summary = {
        "timestamp": now_str,
        "markets_scanned": len(markets),
        "mm_quotes_posted": len(mm_quotes),
        "endgame_opportunities": len(endgame_opps),
        "total_paper_orders": len(paper_orders),
        "elapsed_seconds": round(elapsed, 2),
    }
    logger.info(
        f"Cycle complete: {len(mm_quotes)} MM quotes, "
        f"{len(endgame_opps)} endgame opps, "
        f"{elapsed:.1f}s elapsed"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Paper-trade maker strategy stack")
    parser.add_argument("--loop", type=int, default=0, help="Loop interval in seconds (0=single cycle)")
    parser.add_argument("--max-markets", type=int, default=30, help="Max markets to scan")
    args = parser.parse_args()

    mm = MLInformedMarketMaker(MLMMConfig())
    endgame = EndgameMaker(EndgameConfig())
    clob = CLOBManager()

    logger.info("=" * 60)
    logger.info("PAPER TRADING — MAKER STRATEGY STACK")
    logger.info(f"  Mode: {'DRY-RUN' if clob.dry_run else 'LIVE (!!!)'}")
    logger.info(f"  Max markets: {args.max_markets}")
    logger.info(f"  Loop interval: {args.loop}s" if args.loop else "  Single cycle")
    logger.info("=" * 60)

    if args.loop > 0:
        cycle = 0
        while True:
            cycle += 1
            logger.info(f"\n--- Cycle {cycle} ---")
            try:
                summary = run_cycle(mm, endgame, clob, args.max_markets)
            except Exception as e:
                logger.error(f"Cycle {cycle} failed: {e}", exc_info=True)
            time.sleep(args.loop)
    else:
        summary = run_cycle(mm, endgame, clob, args.max_markets)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
