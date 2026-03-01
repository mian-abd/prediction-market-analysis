"""Deployment gate — checks whether the current model is approved for live trading.

The training pipeline writes ml/saved_models/deployment_status.json after
running the 5 validation gates. This module reads that file and provides
a simple API for the scheduler and API layer to check before executing trades.

Usage:
    from ml.evaluation.deployment_gate import is_model_approved, get_deployment_status

    if is_model_approved():
        # proceed with auto-trading
    else:
        status = get_deployment_status()
        logger.warning("Model not approved: %s", status["action"])
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

DEPLOYMENT_STATUS_PATH = Path("ml/saved_models/deployment_status.json")
MODEL_CARD_PATH = Path("ml/saved_models/model_card.json")
MAX_MODEL_AGE_HOURS = 168  # 7 days — retrain if older


def get_deployment_status() -> dict:
    """Read deployment status from disk. Returns status dict or default (blocked)."""
    if not DEPLOYMENT_STATUS_PATH.exists():
        return {
            "approved": False,
            "timestamp": None,
            "gates_passed": 0,
            "gates_total": 0,
            "action": "no_status_file",
            "reason": "No deployment_status.json found. Run train_ensemble.py first.",
        }

    try:
        with open(DEPLOYMENT_STATUS_PATH) as f:
            status = json.load(f)

        if status.get("timestamp"):
            trained_at = datetime.fromisoformat(status["timestamp"].replace("Z", "+00:00"))
            age_hours = (datetime.now(trained_at.tzinfo) - trained_at).total_seconds() / 3600
            status["model_age_hours"] = round(age_hours, 1)
            if age_hours > MAX_MODEL_AGE_HOURS:
                status["stale"] = True
                status["action"] = "retrain_needed"

        return status
    except (json.JSONDecodeError, KeyError, Exception) as e:
        logger.error("Failed to read deployment status: %s", e)
        return {
            "approved": False,
            "action": "status_file_corrupt",
            "error": str(e),
        }


def is_model_approved(allow_stale: bool = False) -> bool:
    """Check if the current model is approved for live trading.

    Args:
        allow_stale: If True, approve stale models (>7 days old).
                     Default False — stale models are not approved.
    """
    status = get_deployment_status()
    if not status.get("approved"):
        return False
    if not allow_stale and status.get("stale"):
        logger.warning(
            "Model is %.0f hours old (max %d). Retrain recommended.",
            status.get("model_age_hours", 0), MAX_MODEL_AGE_HOURS,
        )
        return False
    return True


def get_model_health_summary() -> dict:
    """Comprehensive model health check for the truth dashboard."""
    status = get_deployment_status()

    model_card = {}
    if MODEL_CARD_PATH.exists():
        try:
            with open(MODEL_CARD_PATH) as f:
                model_card = json.load(f)
        except Exception:
            pass

    return {
        "deployment": {
            "approved": status.get("approved", False),
            "action": status.get("action", "unknown"),
            "gates_passed": status.get("gates_passed", 0),
            "gates_total": status.get("gates_total", 0),
            "model_age_hours": status.get("model_age_hours"),
            "stale": status.get("stale", False),
            "gates": status.get("gates", []),
        },
        "training": {
            "trained_at": model_card.get("trained_at"),
            "n_usable": model_card.get("n_usable"),
            "n_train": model_card.get("n_train"),
            "n_test": model_card.get("n_test"),
            "variable_as_of": model_card.get("snapshot_coverage", {}).get("variable_as_of", False),
        },
        "performance": {
            "ensemble_brier": model_card.get("ensemble_brier"),
            "baseline_brier": model_card.get("baseline_brier"),
            "ensemble_auc": model_card.get("ensemble_auc"),
            "optimal_threshold": model_card.get("optimal_threshold"),
        },
        "features": {
            "active": model_card.get("feature_names", []),
            "dropped": model_card.get("features_dropped", []),
            "models_included": model_card.get("models_included", []),
        },
    }
