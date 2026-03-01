"""End-to-end deployment validation script.

Verifies that all critical fixes are in place and the system works correctly.

Usage:
    python scripts/validate_deployment.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import asyncio
from datetime import datetime, timezone

# Color codes for terminal output
GREEN = ''
RED = ''
YELLOW = ''
RESET = ''
BOLD = ''


def print_section(title):
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}{title.center(70)}{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")


def check(name, passed, details=""):
    status = f"{GREEN}[PASS]{RESET}" if passed else f"{RED}[FAIL]{RESET}"
    print(f"{status} {name}")
    if details and not passed:
        print(f"      {YELLOW}{details}{RESET}")
    return passed


async def validate_all():
    """Run all validation checks."""
    all_passed = True

    # ========================================================================
    # 1. MODEL VALIDATION
    # ========================================================================
    print_section("MODEL VALIDATION")

    try:
        from ml.models.ensemble import EnsembleModel
        from ml.features.training_features import ENSEMBLE_FEATURE_NAMES

        # Check ensemble loads
        ensemble = EnsembleModel()
        loaded = ensemble.load_all()
        all_passed &= check("Ensemble model loads", loaded)

        # Check contaminated features excluded
        contaminated = ['volume_volatility', 'volume_trend_7d', 'log_volume_total',
                       'volume_per_day', 'volume_acceleration', 'volume_to_liquidity_ratio']
        present = [f for f in contaminated if f in ENSEMBLE_FEATURE_NAMES]
        all_passed &= check(
            "Contaminated volume features excluded",
            len(present) == 0,
            f"Found: {present}" if present else ""
        )

        # Check clean features present
        all_passed &= check(
            "Clean momentum features present",
            'volatility_20' in ENSEMBLE_FEATURE_NAMES
        )

        # Feature count grows as new features are added; require at least 10
        all_passed &= check(
            f"Feature count reasonable ({len(ENSEMBLE_FEATURE_NAMES)} features)",
            len(ENSEMBLE_FEATURE_NAMES) >= 10,
            f"Expected >=10 features, got {len(ENSEMBLE_FEATURE_NAMES)}"
        )

    except Exception as e:
        all_passed &= check("Model validation", False, str(e))

    # ========================================================================
    # 2. QUALITY GATES VALIDATION
    # ========================================================================
    print_section("QUALITY GATES VALIDATION")

    try:
        from ml.strategies.ensemble_edge_detector import (
            MIN_VOLUME_TOTAL, MIN_VOLUME_24H, MIN_LIQUIDITY,
            MAX_KELLY, KELLY_FRACTION
        )

        all_passed &= check(
            "MIN_VOLUME_TOTAL tightened (>=$10K)",
            MIN_VOLUME_TOTAL >= 10000,
            f"Current: ${MIN_VOLUME_TOTAL:,}"
        )

        all_passed &= check(
            "MIN_VOLUME_24H tightened (>=$1K)",
            MIN_VOLUME_24H >= 1000,
            f"Current: ${MIN_VOLUME_24H:,}"
        )

        all_passed &= check(
            "MIN_LIQUIDITY tightened (>=$5K)",
            MIN_LIQUIDITY >= 5000,
            f"Current: ${MIN_LIQUIDITY:,}"
        )

        # Validate Kelly formula
        from ml.strategies.ensemble_edge_detector import compute_kelly

        # Test case: 4% edge should give ~1.5% Kelly
        kelly_4pct = compute_kelly('buy_yes', 0.54, 0.50, 0.01)
        all_passed &= check(
            "Kelly formula scales with edge (4% edge -> ~1.5%)",
            0.01 <= kelly_4pct <= 0.02,
            f"Got {kelly_4pct:.2%}"
        )

        # Test case: 8% edge should be below MAX_KELLY cap (4%)
        # Formula: EV=0.07, raw_kelly=0.14, capped=min(0.14, MAX_KELLY/KELLY_FRACTION)*KELLY_FRACTION
        kelly_8pct = compute_kelly('buy_yes', 0.58, 0.50, 0.01)
        all_passed &= check(
            f"Kelly formula bounded by MAX_KELLY ({MAX_KELLY:.0%}) - 8pct edge gives {kelly_8pct:.2%}",
            0 < kelly_8pct <= MAX_KELLY,
            f"Got {kelly_8pct:.2%}, expected 0-{MAX_KELLY:.0%}"
        )

    except Exception as e:
        all_passed &= check("Quality gates validation", False, str(e))

    # ========================================================================
    # 3. MODEL CARD VALIDATION
    # ========================================================================
    print_section("MODEL CARD VALIDATION")

    try:
        model_card_path = project_root / "ml/saved_models/model_card.json"

        all_passed &= check(
            "Model card exists",
            model_card_path.exists()
        )

        if model_card_path.exists():
            card = json.load(open(model_card_path))

            # Check training date is recent
            trained_at = datetime.fromisoformat(card['trained_at'])
            # Handle both timezone-aware (from datetime.now(timezone.utc)) and
            # naive (from older datetime.utcnow()) stored datetimes
            now_utc = datetime.now(timezone.utc)
            if trained_at.tzinfo is None:
                trained_at = trained_at.replace(tzinfo=timezone.utc)
            age_hours = (now_utc - trained_at).total_seconds() / 3600
            all_passed &= check(
                "Model trained recently (<24h)",
                age_hours < 24,
                f"Trained {age_hours:.1f}h ago"
            )

            # Check feature count matches
            all_passed &= check(
                "Feature count in model card matches",
                len(card['feature_names']) == len(ensemble._active_features)
            )

            # Check leakage warnings present
            all_passed &= check(
                "Leakage warnings documented",
                'leakage_warnings' in card and len(card['leakage_warnings']) >= 4
            )

            # Check metrics sanity.
            # For standard training (snapshot_only_mode=False), we expect an honest,
            # non-overfit ensemble with Brier >= 0.055 after cleaning. For
            # snapshot-only training (snapshot_only_mode=True), we instead require
            # that Brier is not worse than 0.06, since that regime can legitimately
            # be much easier (near-resolution markets with full price history).
            snap_only_mode = bool(card.get("snapshot_coverage", {}).get("snapshot_only_mode", False))
            if not snap_only_mode:
                all_passed &= check(
                    "Brier score honest (>=0.055 after cleaning)",
                    card['ensemble_brier'] >= 0.055,
                    f"Brier: {card['ensemble_brier']}"
                )
            else:
                all_passed &= check(
                    "Brier score acceptable for snapshot-only (<=0.060)",
                    card['ensemble_brier'] <= 0.060,
                    f"Brier: {card['ensemble_brier']}"
                )

            # Check XGBoost doesn't rely on contaminated features
            if 'xgb_feature_importance' in card:
                top_feat = max(card['xgb_feature_importance'].items(), key=lambda x: x[1])
                all_passed &= check(
                    f"XGBoost top feature is clean ({top_feat[0]})",
                    top_feat[0] not in contaminated
                )

            # ?? Snapshot coverage gate ??????????????????????????????????????
            # Momentum features activate when >= 10% of training markets have
            # real as_of snapshot prices (hard fail below 10%).
            # Note: coverage % shrinks as new NO-market data is added to denominator;
            # Polymarket-specific coverage is ~68% but total training set is 15k+.
            # Target: 30%+ of total training set. Aspirational: 50%+.
            if 'snapshot_coverage' in card:
                cov = card['snapshot_coverage']
                cov_pct = cov.get('coverage_pct', 0.0)
                n_with = cov.get('n_with_snapshots', 0)
                n_total = cov.get('n_total', 0)
                snap_only = cov.get('snapshot_only_mode', False)

                coverage_ok = cov_pct >= 10.0
                all_passed &= check(
                    f"Snapshot coverage >=10% (current: {cov_pct:.1f}%  {n_with}/{n_total})",
                    coverage_ok,
                    "Run: python scripts/analyze_training_universe.py && "
                    "python scripts/prioritize_backfill.py && "
                    "python scripts/backfill_price_history.py --market-ids data/backfill_priority_list.json"
                )
                if coverage_ok and cov_pct < 30.0:
                    print(
                        f"      {YELLOW}[WARN]{RESET} Coverage {cov_pct:.1f}% < 30% target - "
                        "momentum features are sparse. Backfill NO markets to improve."
                    )
                elif coverage_ok and cov_pct < 50.0:
                    print(
                        f"      {YELLOW}[WARN]{RESET} Coverage {cov_pct:.1f}% < 50% aspirational - "
                        "consider more backfill for stronger momentum features."
                    )
                if snap_only:
                    print(
                        f"      {GREEN}[INFO]{RESET} snapshot_only_mode=True: "
                        "training used only clean as_of prices (no leakage)."
                    )
            else:
                print(
                    f"      {YELLOW}[WARN]{RESET} snapshot_coverage not in model card - "
                    "retrain with updated scripts/train_ensemble.py to track coverage."
                )

    except Exception as e:
        all_passed &= check("Model card validation", False, str(e))

    # ========================================================================
    # 4. API VALIDATION
    # ========================================================================
    print_section("API VALIDATION")

    try:
        from api.main import create_app

        app = create_app()
        routes = [r.path for r in app.routes if hasattr(r, 'path') and r.path.startswith('/api/v1')]

        all_passed &= check(
            "API routes registered",
            len(routes) >= 40,
            f"Found {len(routes)} routes"
        )

        required_endpoints = [
            '/api/v1/health',
            '/api/v1/system/stats',
            '/api/v1/markets',
            '/api/v1/predictions/{market_id}',
            '/api/v1/strategies/signals'
        ]

        for endpoint in required_endpoints:
            all_passed &= check(
                f"Endpoint exists: {endpoint}",
                endpoint in routes
            )

    except Exception as e:
        all_passed &= check("API validation", False, str(e))

    # ========================================================================
    # 5. LIVE PREDICTION TEST
    # ========================================================================
    print_section("LIVE PREDICTION TEST")

    try:
        from db.database import init_db, async_session
        from db.models import Market
        from sqlalchemy import select

        await init_db()

        async with async_session() as session:
            result = await session.execute(
                select(Market)
                .where(Market.is_active == True, Market.price_yes != None)
                .order_by(Market.volume_24h.desc())
                .limit(1)
            )
            market = result.scalar_one_or_none()

            if market:
                # Eagerly capture float values before more session IO (avoids SQLAlchemy async expiry)
                market_price_float = float(market.price_yes or 0.5)
                market_id_val = market.id

                from ml.features.training_features import load_serving_context
                from sqlalchemy import inspect as sa_inspect
                price_snaps, ob_snap = await load_serving_context(session, market_id_val)
                # Refresh market to avoid SQLAlchemy async attribute expiry
                await session.refresh(market)
                pred = ensemble.predict_market(
                    market,
                    price_snapshots=price_snaps,
                    orderbook_snapshot=ob_snap,
                    price_yes_override=market_price_float,
                )

                all_passed &= check(
                    "Live prediction works",
                    'ensemble_probability' in pred
                )

                all_passed &= check(
                    "Ensemble active (not calibration-only)",
                    pred.get('ensemble_active', False)
                )

                all_passed &= check(
                    "Uses correct feature count",
                    pred['features_used'] == len(ensemble._active_features)
                )

                print(f"      Sample prediction: Market={market_price_float:.1%}, "
                      f"Ensemble={pred['ensemble_probability']:.1%}, "
                      f"Delta={pred['delta']:+.1%}")
            else:
                all_passed &= check("Live prediction test", False, "No active markets found")

    except Exception as e:
        import traceback
        traceback.print_exc()
        all_passed &= check("Live prediction test", False, str(e))

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print_section("FINAL SUMMARY")

    if all_passed:
        print(f"{GREEN}{BOLD}[OK] ALL CHECKS PASSED{RESET}")
        print(f"\n{GREEN}System is ready for hackathon demo!{RESET}")
        print("\nNext steps:")
        print("  1. Start API: uvicorn api.main:app --reload")
        print("  2. Start frontend: cd frontend && npm run dev")
        print("  3. Review: docs/PRODUCTION_READINESS.md")
        return 0
    else:
        print(f"{RED}{BOLD}[FAIL] SOME CHECKS FAILED{RESET}")
        print(f"\n{RED}System needs fixes before deployment.{RESET}")
        print("Review failures above and fix issues.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(validate_all())
    sys.exit(exit_code)
