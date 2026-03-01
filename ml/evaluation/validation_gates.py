"""Strategy Validation Framework — 5 gates before live trading.

Every strategy must pass ALL 5 gates on out-of-sample data before
being deployed with real capital. This prevents deploying noise traders.

Gate 1: Temporal OOS Brier < Market Baseline
Gate 2: Calibration Curve Fit (max deviation per bin)
Gate 3: EV Stability Across Sizing Regimes (full/half/fixed Kelly)
Gate 4: Platform-Specific Friction Stress Test
Gate 5: Minimum Sample Size & Statistical Significance
"""

import logging
import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Result of a single validation gate."""
    gate_name: str
    passed: bool
    value: float
    threshold: float
    detail: str = ""


@dataclass
class ValidationResult:
    """Aggregate result of all 5 gates."""
    strategy: str
    gates: list[GateResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(g.passed for g in self.gates)

    @property
    def n_passed(self) -> int:
        return sum(1 for g in self.gates if g.passed)

    def summary(self) -> str:
        lines = [
            f"{'=' * 60}",
            f"VALIDATION GATES: {self.strategy}",
            f"{'=' * 60}",
        ]
        for g in self.gates:
            status = "PASS" if g.passed else "FAIL"
            lines.append(f"  [{status}] {g.gate_name}: {g.value:.4f} (threshold: {g.threshold:.4f})")
            if g.detail:
                lines.append(f"         {g.detail}")
        lines.append(f"{'=' * 60}")
        lines.append(f"RESULT: {'APPROVED for live trading' if self.all_passed else 'REJECTED — fix failing gates'}")
        lines.append(f"  {self.n_passed}/5 gates passed")
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)


def gate_1_brier_vs_baseline(
    y_true: np.ndarray,
    model_preds: np.ndarray,
    market_prices: np.ndarray,
) -> GateResult:
    """Gate 1: Model Brier score must beat market price baseline on OOS data.

    A model that can't beat the market price as a probability estimate
    has no business trading.
    """
    model_brier = brier_score_loss(y_true, model_preds)
    baseline_brier = brier_score_loss(y_true, market_prices)

    improvement = (baseline_brier - model_brier) / baseline_brier if baseline_brier > 0 else 0

    return GateResult(
        gate_name="Brier vs Market Baseline",
        passed=model_brier < baseline_brier,
        value=model_brier,
        threshold=baseline_brier,
        detail=f"Improvement: {improvement:.1%} (model={model_brier:.4f}, market={baseline_brier:.4f})",
    )


def gate_2_calibration_quality(
    y_true: np.ndarray,
    model_preds: np.ndarray,
    n_bins: int = 10,
    max_bin_deviation: float = 0.10,
) -> GateResult:
    """Gate 2: Calibration curve must not deviate more than max_bin_deviation per bin.

    A miscalibrated model will systematically over- or under-trade in certain
    probability ranges, causing consistent losses.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    max_dev = 0.0
    worst_bin = ""
    n_populated_bins = 0

    for i in range(n_bins):
        mask = (model_preds >= bin_edges[i]) & (model_preds < bin_edges[i + 1])
        n_in_bin = mask.sum()
        if n_in_bin < 3:
            continue
        n_populated_bins += 1
        predicted_mean = model_preds[mask].mean()
        actual_mean = y_true[mask].mean()
        deviation = abs(predicted_mean - actual_mean)
        if deviation > max_dev:
            max_dev = deviation
            worst_bin = f"[{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}]: predicted={predicted_mean:.3f}, actual={actual_mean:.3f}"

    return GateResult(
        gate_name="Calibration Quality",
        passed=max_dev <= max_bin_deviation,
        value=max_dev,
        threshold=max_bin_deviation,
        detail=f"Worst bin: {worst_bin} ({n_populated_bins} populated bins)",
    )


def gate_3_ev_stability(
    backtest_results: dict,
    min_methods_profitable: int = 2,
) -> GateResult:
    """Gate 3: EV must be stable across sizing regimes.

    A strategy that only works at full Kelly is likely noise or taking
    hidden tail risk. Must be profitable in at least 2 of 3 sizing methods.
    """
    profitable_methods = []
    total_pnl = 0.0
    for method, result in backtest_results.items():
        if result.total_pnl > 0:
            profitable_methods.append(method)
        total_pnl += result.total_pnl

    n_profitable = len(profitable_methods)

    return GateResult(
        gate_name="EV Stability Across Sizing",
        passed=n_profitable >= min_methods_profitable,
        value=float(n_profitable),
        threshold=float(min_methods_profitable),
        detail=f"Profitable in: {profitable_methods}. Total P&L: ${total_pnl:.2f}",
    )


def gate_4_friction_stress_test(
    y_true: np.ndarray,
    model_preds: np.ndarray,
    market_prices: np.ndarray,
    platform: str = "polymarket",
    taker_fee_bps: int = 0,
    extra_friction_pct: float = 1.0,
) -> GateResult:
    """Gate 4: Strategy must remain profitable with extra friction added.

    Adds 1% additional friction on top of real fees to stress-test robustness.
    Platform-specific: Polymarket taker fees, Kalshi quadratic fees.
    """
    friction = extra_friction_pct / 100
    fee_rate = taker_fee_bps / 10000

    total_pnl = 0.0
    n_trades = 0

    for i in range(len(y_true)):
        p = model_preds[i]
        q = market_prices[i]
        actual = y_true[i]

        fee_yes = p * fee_rate * (1 - q) + friction
        fee_no = (1 - p) * fee_rate * q + friction
        ev_yes = p * (1 - q) - (1 - p) * q - fee_yes
        ev_no = (1 - p) * q - p * (1 - q) - fee_no

        if ev_yes > ev_no and ev_yes > 0:
            direction = "yes"
            net_ev = ev_yes
        elif ev_no > 0:
            direction = "no"
            net_ev = ev_no
        else:
            continue

        n_trades += 1
        if direction == "yes":
            gross_pnl = (actual - q) - friction
        else:
            gross_pnl = ((1 - actual) - (1 - q)) - friction

        if gross_pnl > 0:
            fee = fee_rate * gross_pnl
            total_pnl += gross_pnl - fee
        else:
            total_pnl += gross_pnl

    pnl_per_trade = total_pnl / max(n_trades, 1)

    return GateResult(
        gate_name=f"Friction Stress Test ({platform}, +{extra_friction_pct}%)",
        passed=total_pnl > 0 and n_trades >= 10,
        value=total_pnl,
        threshold=0.0,
        detail=f"{n_trades} trades, P&L/trade: ${pnl_per_trade:.4f}",
    )


def gate_5_statistical_significance(
    y_true: np.ndarray,
    model_preds: np.ndarray,
    market_prices: np.ndarray,
    min_trades: int = 30,
    min_improvement_pct: float = 2.0,
) -> GateResult:
    """Gate 5: Minimum sample size and statistically significant improvement.

    Requires:
    - At least min_trades predictions
    - Brier improvement > min_improvement_pct vs baseline
    - If scipy available, Wilcoxon signed-rank test p < 0.10
    """
    n = len(y_true)
    if n < min_trades:
        return GateResult(
            gate_name="Statistical Significance",
            passed=False,
            value=float(n),
            threshold=float(min_trades),
            detail=f"Only {n} samples, need {min_trades}",
        )

    model_brier = brier_score_loss(y_true, model_preds)
    baseline_brier = brier_score_loss(y_true, market_prices)
    improvement_pct = (baseline_brier - model_brier) / baseline_brier * 100 if baseline_brier > 0 else 0

    # Statistical test
    p_value = 1.0
    try:
        from scipy.stats import wilcoxon
        model_sq_errors = (y_true - model_preds) ** 2
        baseline_sq_errors = (y_true - market_prices) ** 2
        _, p_value = wilcoxon(model_sq_errors, baseline_sq_errors)
    except (ImportError, ValueError):
        pass

    passed = (
        n >= min_trades
        and improvement_pct >= min_improvement_pct
        and p_value < 0.10
    )

    return GateResult(
        gate_name="Statistical Significance",
        passed=passed,
        value=improvement_pct,
        threshold=min_improvement_pct,
        detail=f"n={n}, improvement={improvement_pct:.1f}%, p-value={p_value:.4f}",
    )


def validate_strategy(
    strategy_name: str,
    y_true: np.ndarray,
    model_preds: np.ndarray,
    market_prices: np.ndarray,
    backtest_results: dict | None = None,
    platform: str = "polymarket",
    taker_fee_bps: int = 0,
) -> ValidationResult:
    """Run all 5 validation gates for a strategy.

    Args:
        strategy_name: Name of the strategy being validated
        y_true: Binary outcomes (1.0 or 0.0)
        model_preds: Model probability predictions
        market_prices: Market prices at time of prediction
        backtest_results: Dict of {sizing_method: BacktestResult} from tradability_backtest
        platform: "polymarket" or "kalshi"
        taker_fee_bps: Per-market taker fee in basis points

    Returns:
        ValidationResult with all gate results
    """
    result = ValidationResult(strategy=strategy_name)

    result.gates.append(gate_1_brier_vs_baseline(y_true, model_preds, market_prices))
    result.gates.append(gate_2_calibration_quality(y_true, model_preds))

    if backtest_results:
        result.gates.append(gate_3_ev_stability(backtest_results))
    else:
        result.gates.append(GateResult(
            gate_name="EV Stability Across Sizing",
            passed=False, value=0.0, threshold=2.0,
            detail="No backtest results provided",
        ))

    result.gates.append(gate_4_friction_stress_test(
        y_true, model_preds, market_prices,
        platform=platform, taker_fee_bps=taker_fee_bps,
    ))

    result.gates.append(gate_5_statistical_significance(y_true, model_preds, market_prices))

    logger.info(result.summary())
    return result
