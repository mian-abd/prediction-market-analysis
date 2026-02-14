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
from datetime import datetime

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'


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

        # Check feature count
        all_passed &= check(
            f"Feature count reduced (was 25, now {len(ENSEMBLE_FEATURE_NAMES)})",
            len(ENSEMBLE_FEATURE_NAMES) <= 10,
            f"Expected <=10, got {len(ENSEMBLE_FEATURE_NAMES)}"
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
            "MIN_VOLUME_TOTAL tightened (≥$10K)",
            MIN_VOLUME_TOTAL >= 10000,
            f"Current: ${MIN_VOLUME_TOTAL:,}"
        )

        all_passed &= check(
            "MIN_VOLUME_24H tightened (≥$1K)",
            MIN_VOLUME_24H >= 1000,
            f"Current: ${MIN_VOLUME_24H:,}"
        )

        all_passed &= check(
            "MIN_LIQUIDITY tightened (≥$5K)",
            MIN_LIQUIDITY >= 5000,
            f"Current: ${MIN_LIQUIDITY:,}"
        )

        # Validate Kelly formula
        from ml.strategies.ensemble_edge_detector import compute_kelly

        # Test case: 4% edge should give ~1.5% Kelly
        kelly_4pct = compute_kelly('buy_yes', 0.54, 0.50, 0.01)
        all_passed &= check(
            "Kelly formula scales with edge (4% edge → ~1.5%)",
            0.01 <= kelly_4pct <= 0.02,
            f"Got {kelly_4pct:.2%}"
        )

        # Test case: 8% edge should cap at 2%
        kelly_8pct = compute_kelly('buy_yes', 0.58, 0.50, 0.01)
        all_passed &= check(
            "Kelly formula caps at 2% (8% edge)",
            abs(kelly_8pct - 0.02) < 0.001,
            f"Got {kelly_8pct:.2%}"
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
            age_hours = (datetime.utcnow() - trained_at).total_seconds() / 3600
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

            # Check honest metrics (Brier >0.055 after cleaning)
            all_passed &= check(
                "Brier score honest (≥0.055 after cleaning)",
                card['ensemble_brier'] >= 0.055,
                f"Brier: {card['ensemble_brier']}"
            )

            # Check XGBoost doesn't rely on contaminated features
            if 'xgb_feature_importance' in card:
                top_feat = max(card['xgb_feature_importance'].items(), key=lambda x: x[1])
                all_passed &= check(
                    f"XGBoost top feature is clean ({top_feat[0]})",
                    top_feat[0] not in contaminated
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
                pred = ensemble.predict_market(market)

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

                print(f"      Sample prediction: Market={market.price_yes:.1%}, "
                      f"Ensemble={pred['ensemble_probability']:.1%}, "
                      f"Delta={pred['delta']:+.1%}")
            else:
                all_passed &= check("Live prediction test", False, "No active markets found")

    except Exception as e:
        all_passed &= check("Live prediction test", False, str(e))

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print_section("FINAL SUMMARY")

    if all_passed:
        print(f"{GREEN}{BOLD}✓ ALL CHECKS PASSED{RESET}")
        print(f"\n{GREEN}System is ready for hackathon demo!{RESET}")
        print("\nNext steps:")
        print("  1. Start API: uvicorn api.main:app --reload")
        print("  2. Start frontend: cd frontend && npm run dev")
        print("  3. Review: docs/PRODUCTION_READINESS.md")
        return 0
    else:
        print(f"{RED}{BOLD}✗ SOME CHECKS FAILED{RESET}")
        print(f"\n{RED}System needs fixes before deployment.{RESET}")
        print("Review failures above and fix issues.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(validate_all())
    sys.exit(exit_code)
