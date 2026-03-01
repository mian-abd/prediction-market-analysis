"""Backtest the endgame resolution convergence strategy on resolved markets.

For each resolved market, simulates what would have happened if we ran the
endgame maker strategy in the 72 hours before resolution.

Usage:
    python scripts/backtest_endgame.py
    python scripts/backtest_endgame.py --max-markets 200 --min-confidence 0.85
"""

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ml.strategies.endgame_maker import (
    EndgameConfig,
    simulate_endgame_batch,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DB_PATH = project_root / "data" / "markets.db"


def load_resolved_with_snapshots(max_markets: int) -> list[dict]:
    """Load resolved markets with price snapshots near resolution time."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    query = """
        SELECT
            m.id,
            m.question,
            m.resolved_at,
            m.end_date,
            m.resolution_outcome,
            m.resolution_value,
            m.price_yes,
            m.taker_fee_bps,
            m.volume_total,
            m.liquidity
        FROM markets m
        WHERE m.resolved_at IS NOT NULL
          AND m.price_yes IS NOT NULL
        ORDER BY m.volume_total DESC
        LIMIT ?
    """
    rows = conn.execute(query, (max_markets,)).fetchall()
    raw_markets = [dict(r) for r in rows]

    markets_for_backtest = []
    for m in raw_markets:
        resolved_at = m.get("resolved_at")
        if not resolved_at:
            continue

        try:
            if isinstance(resolved_at, str):
                res_dt = datetime.fromisoformat(resolved_at)
            else:
                res_dt = resolved_at
        except (ValueError, TypeError):
            continue

        resolution_val = m.get("resolution_value")
        if resolution_val is not None:
            resolution = float(resolution_val)
        else:
            outcome_raw = m.get("resolution_outcome", "")
            if outcome_raw in ("Yes", "yes", "YES", "1", 1, True):
                resolution = 1.0
            elif outcome_raw in ("No", "no", "NO", "0", 0, False):
                resolution = 0.0
            else:
                continue

        snap_rows = conn.execute(
            """SELECT price_yes, timestamp FROM price_snapshots
               WHERE market_id = ?
               ORDER BY timestamp DESC
               LIMIT 50""",
            (m["id"],),
        ).fetchall()

        if len(snap_rows) < 5:
            continue

        snap_prices = [r[0] for r in snap_rows]
        snap_timestamps = []
        for r in snap_rows:
            ts = r[1]
            if isinstance(ts, str):
                try:
                    snap_timestamps.append(datetime.fromisoformat(ts))
                except ValueError:
                    snap_timestamps.append(None)
            else:
                snap_timestamps.append(ts)

        latest_price = snap_prices[0]
        latest_ts = snap_timestamps[0]

        if latest_ts and res_dt:
            if res_dt.tzinfo is None:
                res_dt = res_dt.replace(tzinfo=timezone.utc)
            if latest_ts.tzinfo is None:
                latest_ts = latest_ts.replace(tzinfo=timezone.utc)
            hours_before = (res_dt - latest_ts).total_seconds() / 3600
        else:
            hours_before = 24.0

        model_prob = latest_price
        try:
            import joblib
            cal_path = project_root / "ml" / "saved_models" / "post_calibrator.joblib"
            if cal_path.exists():
                cal = joblib.load(cal_path)
                model_prob = float(cal.predict([latest_price])[0])
        except Exception:
            pass

        markets_for_backtest.append({
            "market_id": m["id"],
            "question": m["question"],
            "model_prob": model_prob,
            "market_price": latest_price,
            "hours_to_resolution": max(hours_before, 0.5),
            "resolution_outcome": resolution,
            "taker_fee_bps": m.get("taker_fee_bps") or 0,
            "best_bid": max(latest_price - 0.01, 0.01),
            "best_ask": min(latest_price + 0.01, 0.99),
            "volume": m.get("volume_total", 0),
        })

    conn.close()
    return markets_for_backtest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-markets", type=int, default=500)
    parser.add_argument("--min-confidence", type=float, default=0.90)
    parser.add_argument("--max-hours", type=float, default=72.0)
    args = parser.parse_args()

    config = EndgameConfig(
        min_model_confidence=args.min_confidence,
        max_hours_to_resolution=args.max_hours,
    )

    logger.info("Loading resolved markets with price snapshots...")
    markets = load_resolved_with_snapshots(args.max_markets)
    logger.info(f"Loaded {len(markets)} markets for endgame backtest")

    if not markets:
        logger.warning("No qualifying markets found")
        return

    results = simulate_endgame_batch(markets, config)

    print("\n" + "=" * 80)
    print("ENDGAME RESOLUTION CONVERGENCE BACKTEST")
    print("=" * 80)

    bt = results["backtest"]
    print(f"\n  Markets scanned:     {bt.get('total_markets', len(markets))}")
    print(f"  Trades entered:      {bt.get('trades_entered', 0)}")
    print(f"  Win rate:            {bt.get('win_rate', 0):.1%}")
    print(f"  Total P&L:           ${bt.get('total_pnl', 0):,.2f}")
    print(f"  Avg P&L / trade:     ${bt.get('avg_pnl_per_trade', 0):,.4f}")
    print(f"  Total rebates:       ${bt.get('total_rebates', 0):,.4f}")
    print(f"  Sharpe ratio:        {bt.get('sharpe', 0):.3f}")
    print(f"  Best trade:          ${bt.get('best_trade_pnl', 0):,.4f}")
    print(f"  Worst trade:         ${bt.get('worst_trade_pnl', 0):,.4f}")

    opps = results.get("opportunities", [])
    print(f"\n  Opportunities found: {len(opps)}")
    if opps:
        buy_yes = sum(1 for o in opps if o.direction == "buy_yes")
        buy_no = sum(1 for o in opps if o.direction == "buy_no")
        print(f"    BUY YES:  {buy_yes}")
        print(f"    BUY NO:   {buy_no}")
        avg_edge = np.mean([o.edge for o in opps])
        avg_conf = np.mean([o.confidence_score for o in opps])
        print(f"    Avg edge: {avg_edge:.4f}")
        print(f"    Avg confidence: {avg_conf:.4f}")

    portfolio = results.get("portfolio", [])
    print(f"\n  Portfolio (scored):   {len(portfolio)} markets")

    print("\n" + "=" * 80)

    out_path = project_root / "data" / "endgame_backtest_results.json"
    serializable = {
        "backtest": bt,
        "n_opportunities": len(opps),
        "n_portfolio": len(portfolio),
        "config": {
            "min_model_confidence": config.min_model_confidence,
            "min_edge_cents": config.min_edge_cents,
            "max_hours_to_resolution": config.max_hours_to_resolution,
        },
    }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
