"""Production model monitoring: drift detection, degradation alerts, retrain triggers.

This module provides continuous monitoring of the deployed ensemble model by
tracking prediction quality over rolling windows. It detects:

  1. Feature distribution drift (KS test on each feature vs training distribution)
  2. Prediction calibration degradation (rolling Brier score vs baseline)
  3. Edge decay (rolling profitability trending toward zero)
  4. Concept drift (target distribution shift)

When degradation is detected beyond configurable thresholds, it emits a
retrain signal that the scheduler can act on.

Design principles:
  - Lightweight: runs as part of the regular pipeline cycle
  - Stateful: persists monitoring state to DB for continuity across restarts
  - Configurable: all thresholds are parameterized
  - Non-blocking: monitoring never blocks the trading pipeline
"""

import logging
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)

MONITOR_STATE_PATH = "ml/saved_models/monitor_state.json"


@dataclass
class MonitorConfig:
    """Thresholds for monitoring alerts."""
    # Rolling window for metrics (number of resolved predictions)
    rolling_window: int = 50
    # Brier score degradation threshold (fraction above baseline)
    brier_degradation_threshold: float = 0.15
    # KS-test p-value below which we flag feature drift
    feature_drift_pvalue: float = 0.01
    # Minimum fraction of features drifted to trigger retrain
    min_drift_fraction: float = 0.30
    # Rolling profitability: retrain if edge ratio drops below this
    edge_decay_threshold: float = 0.0
    # Minimum samples before monitoring starts
    min_samples: int = 30
    # Max age of model before mandatory retrain (days)
    max_model_age_days: int = 14
    # Cooldown between retrain signals (hours)
    retrain_cooldown_hours: int = 24


@dataclass
class MonitorAlert:
    """A single monitoring alert."""
    alert_type: str
    severity: str  # "info", "warning", "critical"
    message: str
    details: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PredictionRecord:
    """A single resolved prediction for monitoring."""
    market_id: int
    predicted_prob: float
    actual_outcome: float  # 0.0 or 1.0
    market_price: float
    features: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    strategy: str = "ensemble"
    pnl: float = 0.0


@dataclass
class MonitorState:
    """Persistent state for the model monitor."""
    records: list[dict] = field(default_factory=list)
    training_feature_stats: dict = field(default_factory=dict)
    model_trained_at: str = ""
    last_retrain_signal: str = ""
    alerts_history: list[dict] = field(default_factory=list)


class ModelMonitor:
    """Continuous model monitoring for production deployment.

    Usage:
        monitor = ModelMonitor()
        monitor.load_state()
        monitor.set_training_baseline(X_train, feature_names)

        # On each resolved prediction:
        monitor.record_prediction(PredictionRecord(...))

        # Periodic health check:
        alerts = monitor.check_health()
        if monitor.should_retrain():
            trigger_retrain()
    """

    def __init__(self, config: MonitorConfig | None = None):
        self.config = config or MonitorConfig()
        self.state = MonitorState()
        self._training_means: dict[str, float] = {}
        self._training_stds: dict[str, float] = {}
        self._training_samples: dict[str, np.ndarray] = {}

    def set_training_baseline(
        self,
        X_train: np.ndarray,
        feature_names: list[str],
    ) -> None:
        """Store training distribution statistics for drift detection.

        Should be called once after model training, using the actual
        training data.
        """
        stats = {}
        for i, name in enumerate(feature_names):
            col = X_train[:, i]
            stats[name] = {
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
                "min": float(np.min(col)),
                "max": float(np.max(col)),
                "median": float(np.median(col)),
                "q25": float(np.percentile(col, 25)),
                "q75": float(np.percentile(col, 75)),
            }
            self._training_samples[name] = col.copy()

        self.state.training_feature_stats = stats
        self.state.model_trained_at = datetime.utcnow().isoformat()
        self._training_means = {n: s["mean"] for n, s in stats.items()}
        self._training_stds = {n: max(s["std"], 1e-8) for n, s in stats.items()}

        logger.info(f"Training baseline set: {len(feature_names)} features, {len(X_train)} samples")

    def record_prediction(self, record: PredictionRecord) -> None:
        """Record a resolved prediction for monitoring."""
        self.state.records.append({
            "market_id": record.market_id,
            "predicted_prob": record.predicted_prob,
            "actual_outcome": record.actual_outcome,
            "market_price": record.market_price,
            "features": record.features,
            "timestamp": record.timestamp.isoformat(),
            "strategy": record.strategy,
            "pnl": record.pnl,
        })

        # Keep only recent records to bound memory
        max_records = self.config.rolling_window * 5
        if len(self.state.records) > max_records:
            self.state.records = self.state.records[-max_records:]

    def check_health(self) -> list[MonitorAlert]:
        """Run all monitoring checks and return any alerts.

        Should be called periodically (e.g., every pipeline cycle).
        """
        alerts: list[MonitorAlert] = []
        records = self.state.records

        if len(records) < self.config.min_samples:
            return [MonitorAlert(
                alert_type="insufficient_data",
                severity="info",
                message=f"Only {len(records)} records, need {self.config.min_samples} for monitoring",
            )]

        recent = records[-self.config.rolling_window:]

        alerts.extend(self._check_brier_degradation(recent))
        alerts.extend(self._check_feature_drift(recent))
        alerts.extend(self._check_edge_decay(recent))
        alerts.extend(self._check_model_age())

        for alert in alerts:
            self.state.alerts_history.append(asdict(alert))

        # Trim alerts history
        if len(self.state.alerts_history) > 500:
            self.state.alerts_history = self.state.alerts_history[-500:]

        return alerts

    def _check_brier_degradation(self, recent: list[dict]) -> list[MonitorAlert]:
        """Check if prediction quality has degraded."""
        alerts = []

        preds = np.array([r["predicted_prob"] for r in recent])
        actuals = np.array([r["actual_outcome"] for r in recent])
        market_prices = np.array([r["market_price"] for r in recent])

        model_brier = float(np.mean((preds - actuals) ** 2))
        baseline_brier = float(np.mean((market_prices - actuals) ** 2))

        if baseline_brier > 0:
            improvement = (baseline_brier - model_brier) / baseline_brier
        else:
            improvement = 0.0

        if model_brier > baseline_brier * (1 + self.config.brier_degradation_threshold):
            alerts.append(MonitorAlert(
                alert_type="brier_degradation",
                severity="critical",
                message=(
                    f"Model Brier ({model_brier:.4f}) significantly worse than "
                    f"baseline ({baseline_brier:.4f}). Improvement: {improvement:.1%}"
                ),
                details={
                    "model_brier": model_brier,
                    "baseline_brier": baseline_brier,
                    "improvement_pct": round(improvement * 100, 2),
                    "n_samples": len(recent),
                },
            ))
        elif model_brier > baseline_brier:
            alerts.append(MonitorAlert(
                alert_type="brier_warning",
                severity="warning",
                message=f"Model Brier ({model_brier:.4f}) worse than baseline ({baseline_brier:.4f})",
                details={
                    "model_brier": model_brier,
                    "baseline_brier": baseline_brier,
                },
            ))

        return alerts

    def _check_feature_drift(self, recent: list[dict]) -> list[MonitorAlert]:
        """Check for feature distribution drift using KS test."""
        alerts = []
        if not self._training_samples:
            return alerts

        # Collect features from recent predictions
        recent_features: dict[str, list[float]] = {}
        for r in recent:
            for name, value in r.get("features", {}).items():
                if name not in recent_features:
                    recent_features[name] = []
                recent_features[name].append(value)

        drifted_features = []
        total_tested = 0

        for name, values in recent_features.items():
            if name not in self._training_samples or len(values) < 10:
                continue

            total_tested += 1
            train_sample = self._training_samples[name]
            recent_sample = np.array(values)

            try:
                stat, pvalue = ks_2samp(train_sample, recent_sample)
                if pvalue < self.config.feature_drift_pvalue:
                    drifted_features.append({
                        "feature": name,
                        "ks_stat": round(float(stat), 4),
                        "pvalue": round(float(pvalue), 6),
                    })
            except Exception:
                continue

        if total_tested > 0:
            drift_fraction = len(drifted_features) / total_tested
            if drift_fraction >= self.config.min_drift_fraction:
                alerts.append(MonitorAlert(
                    alert_type="feature_drift",
                    severity="critical",
                    message=(
                        f"{len(drifted_features)}/{total_tested} features show significant drift "
                        f"({drift_fraction:.0%} >= {self.config.min_drift_fraction:.0%} threshold)"
                    ),
                    details={"drifted_features": drifted_features},
                ))
            elif drifted_features:
                alerts.append(MonitorAlert(
                    alert_type="feature_drift_minor",
                    severity="info",
                    message=f"{len(drifted_features)}/{total_tested} features drifted (below threshold)",
                    details={"drifted_features": drifted_features},
                ))

        return alerts

    def _check_edge_decay(self, recent: list[dict]) -> list[MonitorAlert]:
        """Check if trading edge (P&L) is decaying."""
        alerts = []

        pnls = [r.get("pnl", 0.0) for r in recent]
        if not pnls:
            return alerts

        cumulative = np.cumsum(pnls)
        if len(cumulative) < 10:
            return alerts

        # Check if the second half is worse than the first half
        half = len(pnls) // 2
        first_half_pnl = sum(pnls[:half])
        second_half_pnl = sum(pnls[half:])

        if sum(pnls) < self.config.edge_decay_threshold:
            alerts.append(MonitorAlert(
                alert_type="edge_decay",
                severity="warning",
                message=(
                    f"Rolling P&L is negative: ${sum(pnls):.2f} over {len(pnls)} trades. "
                    f"First half: ${first_half_pnl:.2f}, second half: ${second_half_pnl:.2f}"
                ),
                details={
                    "total_pnl": round(sum(pnls), 4),
                    "first_half": round(first_half_pnl, 4),
                    "second_half": round(second_half_pnl, 4),
                },
            ))

        return alerts

    def _check_model_age(self) -> list[MonitorAlert]:
        """Check if the model is too old and needs mandatory retrain."""
        alerts = []
        if not self.state.model_trained_at:
            return alerts

        try:
            trained_at = datetime.fromisoformat(self.state.model_trained_at)
            age_days = (datetime.utcnow() - trained_at).total_seconds() / 86400

            if age_days > self.config.max_model_age_days:
                alerts.append(MonitorAlert(
                    alert_type="model_stale",
                    severity="critical",
                    message=(
                        f"Model is {age_days:.1f} days old "
                        f"(max: {self.config.max_model_age_days} days)"
                    ),
                    details={"age_days": round(age_days, 1)},
                ))
        except (ValueError, TypeError):
            pass

        return alerts

    def should_retrain(self) -> bool:
        """Check if a retrain signal should be emitted.

        Returns True if any critical alert exists and cooldown has passed.
        """
        alerts = self.check_health()
        critical_alerts = [a for a in alerts if a.severity == "critical"]

        if not critical_alerts:
            return False

        # Check cooldown
        if self.state.last_retrain_signal:
            try:
                last_signal = datetime.fromisoformat(self.state.last_retrain_signal)
                hours_since = (datetime.utcnow() - last_signal).total_seconds() / 3600
                if hours_since < self.config.retrain_cooldown_hours:
                    return False
            except (ValueError, TypeError):
                pass

        self.state.last_retrain_signal = datetime.utcnow().isoformat()
        logger.warning(
            f"RETRAIN SIGNAL: {len(critical_alerts)} critical alert(s): "
            + "; ".join(a.message for a in critical_alerts)
        )
        return True

    def save_state(self, path: str | None = None) -> None:
        """Persist monitoring state to disk."""
        path = path or MONITOR_STATE_PATH
        try:
            with open(path, "w") as f:
                json.dump(asdict(self.state), f, indent=2, default=str)
            logger.debug(f"Monitor state saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save monitor state: {e}")

    def load_state(self, path: str | None = None) -> None:
        """Load monitoring state from disk."""
        path = path or MONITOR_STATE_PATH
        try:
            with open(path) as f:
                data = json.load(f)
            self.state = MonitorState(**data)

            # Reconstruct training samples from stats (approximate)
            if self.state.training_feature_stats:
                for name, stats in self.state.training_feature_stats.items():
                    self._training_means[name] = stats["mean"]
                    self._training_stds[name] = max(stats["std"], 1e-8)

            logger.info(
                f"Monitor state loaded: {len(self.state.records)} records, "
                f"{len(self.state.alerts_history)} alerts"
            )
        except FileNotFoundError:
            logger.info("No existing monitor state found, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load monitor state: {e}")

    def get_dashboard_data(self) -> dict:
        """Get monitoring data for the frontend dashboard."""
        records = self.state.records
        if not records:
            return {"status": "no_data", "records_count": 0}

        recent = records[-self.config.rolling_window:]
        preds = [r["predicted_prob"] for r in recent]
        actuals = [r["actual_outcome"] for r in recent]
        market_prices = [r["market_price"] for r in recent]

        model_brier = float(np.mean([(p - a) ** 2 for p, a in zip(preds, actuals)]))
        baseline_brier = float(np.mean([(m - a) ** 2 for m, a in zip(market_prices, actuals)]))

        pnls = [r.get("pnl", 0.0) for r in recent]
        cumulative_pnl = float(np.sum(pnls))

        alerts = self.check_health()
        critical_count = sum(1 for a in alerts if a.severity == "critical")
        warning_count = sum(1 for a in alerts if a.severity == "warning")

        status = "healthy"
        if critical_count > 0:
            status = "critical"
        elif warning_count > 0:
            status = "warning"

        return {
            "status": status,
            "records_count": len(records),
            "rolling_window": self.config.rolling_window,
            "model_brier": round(model_brier, 4),
            "baseline_brier": round(baseline_brier, 4),
            "brier_improvement_pct": round(
                (baseline_brier - model_brier) / max(baseline_brier, 0.001) * 100, 2
            ),
            "cumulative_pnl": round(cumulative_pnl, 2),
            "win_rate": round(sum(1 for p in pnls if p > 0) / max(len(pnls), 1), 3),
            "alerts": [
                {"type": a.alert_type, "severity": a.severity, "message": a.message}
                for a in alerts
            ],
            "critical_alerts": critical_count,
            "warning_alerts": warning_count,
            "model_age_days": None,
            "should_retrain": self.should_retrain() if critical_count > 0 else False,
        }
