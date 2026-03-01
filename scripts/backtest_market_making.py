"""Backtest the Avellaneda-Stoikov market making engine on historical data.

Queries resolved markets from the SQLite DB that have sufficient price and
orderbook snapshots, runs the MM backtest on each, and produces an aggregate
performance summary comparing fee-free vs fee-enabled regimes.

Usage:
    python scripts/backtest_market_making.py                # default 50 markets
    python scripts/backtest_market_making.py --max-markets 200
"""

import argparse
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ml.strategies.market_making import MarketMakingConfig, run_mm_backtest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DB_PATH = project_root / "data" / "markets.db"
RESULTS_PATH = project_root / "data" / "mm_backtest_results.json"
FEE_BPS = 156


def find_qualifying_markets(conn: sqlite3.Connection, max_markets: int) -> list[dict]:
    """Find resolved markets with enough price snapshots for backtesting.

    Orderbook data is synthesized from price data when real orderbook
    snapshots are unavailable (most resolved markets don't have OB data
    since collection only started recently).
    """
    query = """
        SELECT
            m.id,
            m.question,
            m.resolved_at,
            m.resolution_outcome AS outcome,
            m.volume_total,
            m.liquidity,
            ps_count.cnt  AS price_count
        FROM markets m
        INNER JOIN (
            SELECT market_id, COUNT(*) AS cnt
            FROM price_snapshots
            GROUP BY market_id
            HAVING cnt > 30
        ) ps_count ON ps_count.market_id = m.id
        WHERE m.resolved_at IS NOT NULL
        ORDER BY ps_count.cnt DESC
        LIMIT ?
    """
    cur = conn.execute(query, (max_markets,))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def load_price_series(conn: sqlite3.Connection, market_id: int) -> tuple[list[float], list[datetime]]:
    cur = conn.execute(
        "SELECT price_yes, timestamp FROM price_snapshots "
        "WHERE market_id = ? ORDER BY timestamp",
        (market_id,),
    )
    rows = cur.fetchall()
    prices = [r[0] for r in rows]
    timestamps = [datetime.fromisoformat(r[1]) if isinstance(r[1], str) else r[1] for r in rows]
    return prices, timestamps


def load_orderbook_series(conn: sqlite3.Connection, market_id: int) -> list[dict]:
    cur = conn.execute(
        "SELECT best_bid, best_ask, bid_depth_total, ask_depth_total, timestamp "
        "FROM orderbook_snapshots WHERE market_id = ? ORDER BY timestamp",
        (market_id,),
    )
    rows = cur.fetchall()
    return [
        {
            "best_bid": r[0],
            "best_ask": r[1],
            "bid_depth_usd": r[2] or 500.0,
            "ask_depth_usd": r[3] or 500.0,
            "timestamp": r[4],
        }
        for r in rows
    ]


def align_orderbooks_to_prices(
    price_timestamps: list[datetime],
    ob_series: list[dict],
) -> list[dict]:
    """Map each price timestamp to the nearest preceding orderbook snapshot.

    The MM engine expects len(orderbook_series) >= len(price_series).
    When orderbook coverage is sparser than price coverage, we forward-fill
    the most recent orderbook observation.
    """
    if not ob_series:
        return [{}] * len(price_timestamps)

    ob_ts = [
        datetime.fromisoformat(o["timestamp"]) if isinstance(o["timestamp"], str) else o["timestamp"]
        for o in ob_series
    ]

    aligned = []
    ob_idx = 0
    for pts in price_timestamps:
        while ob_idx < len(ob_ts) - 1 and ob_ts[ob_idx + 1] <= pts:
            ob_idx += 1
        aligned.append(ob_series[ob_idx])

    return aligned


def parse_resolution_time(raw: str | datetime | None) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw
    try:
        return datetime.fromisoformat(raw)
    except (ValueError, TypeError):
        return None


def synthesize_orderbook_from_prices(
    prices: list[float],
    liquidity: float | None = None,
) -> list[dict]:
    """Create synthetic orderbook series from price data.

    Uses a realistic spread model based on observed Polymarket spreads:
    - Spreads are wider near 50% and narrower at extremes
    - Typical spread: 1-3 cents for liquid markets, 3-8 cents for illiquid
    - Depth: proportional to liquidity parameter
    """
    base_depth = max(liquidity or 500.0, 200.0)
    ob = []
    for p in prices:
        p_clip = max(0.02, min(0.98, p))
        spread_factor = 4.0 * p_clip * (1 - p_clip)
        half_spread = 0.005 + 0.015 * spread_factor  # 0.5c - 2c half spread
        ob.append({
            "best_bid": round(p_clip - half_spread, 4),
            "best_ask": round(p_clip + half_spread, 4),
            "bid_depth_usd": base_depth * (0.5 + 0.5 * (1 - spread_factor)),
            "ask_depth_usd": base_depth * (0.5 + 0.5 * (1 - spread_factor)),
        })
    return ob


def run_single_market(
    conn: sqlite3.Connection,
    market: dict,
    taker_fee_bps: int = 0,
    config: MarketMakingConfig | None = None,
) -> dict:
    market_id = market["id"]
    prices, timestamps = load_price_series(conn, market_id)

    ob_raw = load_orderbook_series(conn, market_id)
    if len(ob_raw) > 5:
        ob_aligned = align_orderbooks_to_prices(timestamps, ob_raw)
    else:
        ob_aligned = synthesize_orderbook_from_prices(
            prices, market.get("liquidity")
        )

    resolution_time = parse_resolution_time(market["resolved_at"])
    if resolution_time is None:
        return {"error": "Could not parse resolution_time", "market_id": market_id}

    result = run_mm_backtest(
        price_series=prices,
        timestamps=timestamps,
        orderbook_series=ob_aligned,
        resolution_time=resolution_time,
        config=config,
        taker_fee_bps=taker_fee_bps,
    )

    result["market_id"] = market_id
    result["question"] = market.get("question", "")
    result["outcome"] = market.get("outcome")
    result["price_obs"] = len(prices)
    result["ob_obs"] = len(ob_raw)
    result["taker_fee_bps"] = taker_fee_bps

    # Strip large series from per-market results to keep output manageable
    result.pop("pnl_series", None)
    result.pop("fills", None)
    result.pop("quote_log", None)

    return result


def compute_aggregate_stats(results: list[dict]) -> dict:
    if not results:
        return {}

    pnls = [r["total_pnl"] for r in results if "total_pnl" in r]
    fills = [r.get("n_fills", 0) for r in results]
    spreads = [r.get("spread_captured", 0) for r in results]
    rebates = [r.get("rebates_earned", 0) for r in results]

    pnls_arr = np.array(pnls) if pnls else np.array([0.0])
    profitable = sum(1 for p in pnls if p > 0)

    sharpe = float(np.mean(pnls_arr) / np.std(pnls_arr)) if np.std(pnls_arr) > 0 else 0.0

    return {
        "n_markets": len(results),
        "n_errors": sum(1 for r in results if "error" in r),
        "total_pnl": round(float(np.sum(pnls_arr)), 4),
        "avg_pnl_per_market": round(float(np.mean(pnls_arr)), 4),
        "median_pnl": round(float(np.median(pnls_arr)), 4),
        "std_pnl": round(float(np.std(pnls_arr)), 4),
        "win_rate": round(profitable / max(len(pnls), 1), 4),
        "sharpe": round(sharpe, 4),
        "total_fills": int(np.sum(fills)),
        "avg_fills_per_market": round(float(np.mean(fills)), 1),
        "total_spread_captured": round(float(np.sum(spreads)), 4),
        "total_rebates": round(float(np.sum(rebates)), 4),
        "best_pnl": round(float(np.max(pnls_arr)), 4),
        "worst_pnl": round(float(np.min(pnls_arr)), 4),
    }


def print_summary_table(
    free_results: list[dict],
    fee_results: list[dict],
    free_agg: dict,
    fee_agg: dict,
) -> None:
    header = (
        f"{'Market ID':>10} {'Fills':>6} {'P&L (free)':>12} {'P&L (fee)':>12} "
        f"{'Spread':>10} {'Rebates':>10} {'Question':<50}"
    )
    sep = "-" * len(header)

    print("\n" + sep)
    print("MARKET MAKING BACKTEST RESULTS")
    print(sep)
    print(header)
    print(sep)

    free_map = {r["market_id"]: r for r in free_results if "error" not in r}
    fee_map = {r["market_id"]: r for r in fee_results if "error" not in r}

    all_ids = sorted(set(free_map.keys()) | set(fee_map.keys()))
    for mid in all_ids:
        fr = free_map.get(mid, {})
        fe = fee_map.get(mid, {})
        question = (fr.get("question") or fe.get("question", ""))[:50]
        print(
            f"{mid:>10} "
            f"{fr.get('n_fills', 0):>6} "
            f"{fr.get('total_pnl', 0):>12.4f} "
            f"{fe.get('total_pnl', 0):>12.4f} "
            f"{fr.get('spread_captured', 0):>10.4f} "
            f"{fe.get('rebates_earned', 0):>10.4f} "
            f"{question:<50}"
        )

    print(sep)
    print("\nAGGREGATE STATISTICS")
    print(sep)

    row_fmt = f"{'':>30} {'Fee-Free':>15} {'Fee-Enabled':>15} {'Delta':>15}"
    print(row_fmt)
    print("-" * len(row_fmt))

    metrics = [
        ("Total P&L ($)", "total_pnl"),
        ("Avg P&L / Market ($)", "avg_pnl_per_market"),
        ("Median P&L ($)", "median_pnl"),
        ("Std Dev P&L ($)", "std_pnl"),
        ("Win Rate (%)", "win_rate"),
        ("Sharpe-like Ratio", "sharpe"),
        ("Total Fills", "total_fills"),
        ("Avg Fills / Market", "avg_fills_per_market"),
        ("Total Spread Captured ($)", "total_spread_captured"),
        ("Total Rebates ($)", "total_rebates"),
        ("Best Single Market ($)", "best_pnl"),
        ("Worst Single Market ($)", "worst_pnl"),
    ]

    for label, key in metrics:
        fv = free_agg.get(key, 0)
        ev = fee_agg.get(key, 0)
        delta = ev - fv

        if key == "win_rate":
            print(f"{label:>30} {fv * 100:>14.1f}% {ev * 100:>14.1f}% {delta * 100:>14.1f}%")
        elif isinstance(fv, int):
            print(f"{label:>30} {fv:>15} {ev:>15} {delta:>15}")
        else:
            print(f"{label:>30} {fv:>15.4f} {ev:>15.4f} {delta:>15.4f}")

    print(sep + "\n")


def main():
    parser = argparse.ArgumentParser(description="Backtest MM engine on historical data")
    parser.add_argument("--max-markets", type=int, default=50, help="Max markets to test")
    args = parser.parse_args()

    if not DB_PATH.exists():
        logger.error(f"Database not found at {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    logger.info(f"Finding qualifying markets (limit {args.max_markets})...")
    markets = find_qualifying_markets(conn, args.max_markets)
    logger.info(f"Found {len(markets)} qualifying resolved markets")

    if not markets:
        logger.warning("No qualifying markets found. Check DB for resolved markets with snapshots.")
        conn.close()
        sys.exit(0)

    free_results = []
    fee_results = []
    t0 = time.time()

    for idx, market in enumerate(markets, 1):
        mid = market["id"]
        question = (market.get("question") or "")[:60]
        logger.info(f"[{idx}/{len(markets)}] Market {mid}: {question}")

        # Fee-free run
        try:
            res_free = run_single_market(conn, market, taker_fee_bps=0)
            free_results.append(res_free)
        except Exception:
            logger.exception(f"  Fee-free run failed for market {mid}")
            free_results.append({"market_id": mid, "error": "fee-free run failed"})

        # Fee-enabled run
        try:
            res_fee = run_single_market(conn, market, taker_fee_bps=FEE_BPS)
            fee_results.append(res_fee)
        except Exception:
            logger.exception(f"  Fee-enabled run failed for market {mid}")
            fee_results.append({"market_id": mid, "error": "fee-enabled run failed"})

    elapsed = time.time() - t0
    logger.info(f"Backtests completed in {elapsed:.1f}s ({elapsed / max(len(markets), 1):.2f}s/market)")

    free_agg = compute_aggregate_stats([r for r in free_results if "error" not in r])
    fee_agg = compute_aggregate_stats([r for r in fee_results if "error" not in r])

    print_summary_table(free_results, fee_results, free_agg, fee_agg)

    output = {
        "run_timestamp": datetime.utcnow().isoformat(),
        "n_markets_tested": len(markets),
        "fee_bps_tested": FEE_BPS,
        "free_aggregate": free_agg,
        "fee_aggregate": fee_agg,
        "free_per_market": [r for r in free_results if "error" not in r],
        "fee_per_market": [r for r in fee_results if "error" not in r],
        "errors": [r for r in free_results + fee_results if "error" in r],
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Results saved to {RESULTS_PATH}")

    conn.close()


if __name__ == "__main__":
    main()
