"""Build platform- and category-specific calibration curves from resolved markets.

Reads resolved markets from the DB, buckets them by (platform, category), fits
isotonic-regression-style empirical calibration curves, and writes the result to
ml/saved_models/calibration_curves.json.

This script should be run:
    1. After backfill_price_history.py (so as_of prices are available).
    2. Before or after train_ensemble.py — it's independent of the ensemble.
    3. The Favorite-Longshot detector and calibration_features.py will automatically
       pick up the new file on next load (calibration_lookup.py caches with mtime).

Usage:
    python scripts/build_calibration_curves.py
    python scripts/build_calibration_curves.py --min-samples 100
    python scripts/build_calibration_curves.py --n-bins 20
"""

from __future__ import annotations

import sys
import json
import asyncio
import logging
import argparse
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import select, func
from db.database import init_db, async_session
from db.models import Market, Platform, PriceSnapshot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_PATH = project_root / "ml" / "saved_models" / "calibration_curves.json"


def _empirical_curve(
    prices: list[float],
    outcomes: list[float],
    n_bins: int = 20,
) -> dict:
    """Compute empirical resolution rate per price bin using isotonic regression.

    Returns a dict with:
        buckets  — bin centre prices (sorted)
        rates    — empirical resolution rates per bin
        n_samples — total sample count
    """
    prices_arr = np.array(prices)
    outcomes_arr = np.array(outcomes)

    # Bin prices into n_bins equal-width buckets
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_rates: list[float] = []
    bin_centres_out: list[float] = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (prices_arr >= lo) & (prices_arr <= hi)
        else:
            mask = (prices_arr >= lo) & (prices_arr < hi)
        count = mask.sum()
        if count == 0:
            continue
        rate = float(outcomes_arr[mask].mean())
        bin_centres_out.append(float(round(bin_centres[i], 4)))
        bin_rates.append(round(rate, 4))

    # Isotonic regression to ensure monotonicity
    if len(bin_rates) >= 3:
        try:
            from sklearn.isotonic import IsotonicRegression
            ir = IsotonicRegression(out_of_bounds="clip")
            bin_rates = list(ir.fit_transform(bin_centres_out, bin_rates))
            bin_rates = [round(float(r), 4) for r in bin_rates]
        except ImportError:
            pass  # Use raw empirical rates if sklearn not available

    return {
        "buckets": bin_centres_out,
        "rates": bin_rates,
        "n_samples": int(len(prices)),
    }


async def build_calibration_curves(min_samples: int = 200, n_bins: int = 20) -> dict:
    """Build calibration curves and save to ml/saved_models/calibration_curves.json."""
    await init_db()

    async with async_session() as session:
        # ── 1. Platform map ────────────────────────────────────────────────────
        platforms_result = await session.execute(select(Platform))
        platform_map: dict[int, str] = {p.id: p.name for p in platforms_result.scalars().all()}

        # ── 2. Resolved markets with volume ───────────────────────────────────
        logger.info("Querying resolved markets...")
        result = await session.execute(
            select(Market).where(
                Market.is_resolved == True,  # noqa: E712
                Market.resolution_value != None,  # noqa: E711
                Market.volume_total > 0,
            )
        )
        markets = result.scalars().all()
        logger.info(f"Found {len(markets)} resolved markets")

        if not markets:
            logger.warning("No resolved markets with volume > 0. Nothing to build.")
            return {}

        market_map = {m.id: m for m in markets}
        market_ids = list(market_map.keys())

        # ── 3. Load as_of snapshot prices ─────────────────────────────────────
        logger.info("Loading price snapshots for as_of prices...")
        # For each market: as_of = resolved_at - 24h
        # We want the last snapshot before as_of
        CHUNK = 900
        price_at_asof: dict[int, float] = {}

        for i in range(0, len(market_ids), CHUNK):
            chunk = market_ids[i : i + CHUNK]
            rows = (
                await session.execute(
                    select(
                        PriceSnapshot.market_id,
                        func.max(PriceSnapshot.timestamp).label("max_ts"),
                    )
                    .where(PriceSnapshot.market_id.in_(chunk))
                    .group_by(PriceSnapshot.market_id)
                )
            ).all()

            for row in rows:
                m = market_map.get(row.market_id)
                if not m:
                    continue
                if m.resolved_at:
                    as_of = m.resolved_at - timedelta(hours=24)
                elif m.end_date:
                    as_of = m.end_date - timedelta(hours=24)
                else:
                    continue

                # Fetch the actual last snapshot before as_of
                snap_row = (
                    await session.execute(
                        select(PriceSnapshot.price_yes)
                        .where(
                            PriceSnapshot.market_id == row.market_id,
                            PriceSnapshot.timestamp <= as_of,
                        )
                        .order_by(PriceSnapshot.timestamp.desc())
                        .limit(1)
                    )
                ).scalar_one_or_none()

                if snap_row is not None:
                    price_at_asof[row.market_id] = float(snap_row)

        logger.info(
            f"As-of snapshot prices available for {len(price_at_asof)}/{len(markets)} markets"
        )

    # ── 4. Assemble (price, outcome) pairs by (platform, category) ────────────
    # Use snapshot price if available; fall back to market.price_yes with warning
    leakage_count = 0
    bucket_data: dict[str, dict[str, tuple[list[float], list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: ([], []))
    )

    for m in markets:
        if m.resolution_value is None:
            continue

        pname = platform_map.get(m.platform_id, "unknown")
        cat = (m.normalized_category or m.category or "other").lower().strip()
        outcome = float(m.resolution_value)

        if m.id in price_at_asof:
            price = price_at_asof[m.id]
        else:
            # Fallback: use market.price_yes at resolution time — contains leakage
            # (price at resolution reflects the outcome). We log a count but still
            # include these for global/platform curves since they're still useful
            # for understanding long-run calibration; they're just noisier.
            if m.price_yes is None:
                continue
            price = float(m.price_yes)
            leakage_count += 1

        # Clamp to valid range
        price = max(0.01, min(0.99, price))

        bucket_data[pname][cat][0].append(price)
        bucket_data[pname][cat][1].append(outcome)

    if leakage_count > 0:
        logger.warning(
            f"{leakage_count} markets used market.price_yes (potential leakage) "
            "— run backfill first for cleaner curves."
        )

    # ── 5. Build curves per (platform, category) ──────────────────────────────
    curves: dict = {}

    global_prices: list[float] = []
    global_outcomes: list[float] = []

    total_curves = 0
    for pname, cat_data in bucket_data.items():
        curves[pname] = {}
        platform_prices: list[float] = []
        platform_outcomes: list[float] = []

        for cat, (prices, outcomes) in cat_data.items():
            platform_prices.extend(prices)
            platform_outcomes.extend(outcomes)
            global_prices.extend(prices)
            global_outcomes.extend(outcomes)

            if len(prices) >= min_samples:
                curves[pname][cat] = _empirical_curve(prices, outcomes, n_bins)
                total_curves += 1
                logger.info(
                    f"  {pname}/{cat}: {len(prices)} samples → curve built"
                )
            else:
                logger.debug(
                    f"  {pname}/{cat}: {len(prices)} samples < {min_samples} — skipped (will fall back to platform default)"
                )

        # Platform-level default (all categories combined)
        if platform_prices:
            curves[pname]["default"] = _empirical_curve(platform_prices, platform_outcomes, n_bins)
            curves[pname]["default"]["note"] = "platform aggregate (all categories)"
            total_curves += 1
            logger.info(
                f"  {pname}/default: {len(platform_prices)} samples → curve built"
            )

    # Global fallback
    if global_prices:
        curves["global"] = _empirical_curve(global_prices, global_outcomes, n_bins)
        curves["global"]["note"] = "global aggregate (all platforms)"
        total_curves += 1
        logger.info(f"  global: {len(global_prices)} samples → curve built")

    # ── 6. Metadata ───────────────────────────────────────────────────────────
    output = {
        "_meta": {
            "built_at": datetime.utcnow().isoformat(),
            "min_samples_threshold": min_samples,
            "n_bins": n_bins,
            "total_markets": len(markets),
            "markets_with_asof_price": len(price_at_asof),
            "markets_with_leakage_price": leakage_count,
            "total_curves_built": total_curves,
        },
        **curves,
    }

    # ── 7. Save ───────────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("")
    logger.info("=" * 60)
    logger.info("CALIBRATION CURVES BUILT")
    logger.info("=" * 60)
    logger.info(f"Total markets:            {len(markets):,}")
    logger.info(f"As-of snapshot coverage:  {len(price_at_asof):,}  ({100*len(price_at_asof)/max(1,len(markets)):.1f}%)")
    logger.info(f"Leakage fallbacks:        {leakage_count:,}")
    logger.info(f"Curves built:             {total_curves}")
    logger.info(f"Saved to:                 {OUTPUT_PATH}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("calibration_lookup.py will pick up the new file automatically.")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Build platform-specific calibration curves from resolved market data."
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=200,
        help="Minimum samples required to build a category-level curve (default: 200).",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=20,
        help="Number of equal-width price bins (default: 20).",
    )
    args = parser.parse_args()
    asyncio.run(build_calibration_curves(min_samples=args.min_samples, n_bins=args.n_bins))


if __name__ == "__main__":
    main()
