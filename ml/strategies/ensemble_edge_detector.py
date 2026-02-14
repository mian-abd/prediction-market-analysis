"""Ensemble-based edge detector — finds mispriced markets using ML predictions.

Quality gates filter for tradeable markets. Fee-aware directional EV determines
which side to trade. Fractional Kelly sizes the position.

Two modes:
- Coverage: predict everything, attach quality metadata (dashboard/research)
- Precision: only return markets that pass ALL quality gates (trading signals)
"""

import logging
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# --- Quality Gate Thresholds ---
MIN_VOLUME_TOTAL = 5000     # USD, liquid enough to trade
MIN_VOLUME_24H = 200        # USD, market is actively traded
MIN_LIQUIDITY = 1000        # Open interest, can absorb position
MIN_PRICE = 0.02            # Tradeable range lower bound
MAX_PRICE = 0.98            # Tradeable range upper bound

# --- Edge Thresholds ---
MIN_NET_EDGE = 0.03         # 3% minimum net edge after fees
SLIPPAGE_BUFFER = 0.01      # 1% slippage estimate
MAX_KELLY = 0.02            # Maximum Kelly fraction (2%)
KELLY_FRACTION = 0.25       # Fractional Kelly multiplier (safety)
MIN_CONFIDENCE = 0.5        # Minimum confidence score


@dataclass
class QualityGate:
    """Result of quality gate checks for a market."""
    passes: bool
    reasons: list[str]  # Reasons for failure (empty if passes)
    volume_ok: bool
    volume_24h_ok: bool
    liquidity_ok: bool
    price_ok: bool


@dataclass
class EdgeSignal:
    """Detected edge signal with trading recommendation."""
    market_id: int
    direction: str | None       # "buy_yes", "buy_no", or None
    ensemble_prob: float        # Model's true probability
    market_price: float         # Current market price
    raw_edge: float             # |ensemble_prob - market_price|
    fee_cost: float             # Total fees + slippage
    net_ev: float               # Directional expected value after fees
    kelly_fraction: float       # Position size as fraction of bankroll
    confidence: float           # Multi-signal confidence score
    quality_tier: str           # "high", "medium", "low"
    quality_gate: QualityGate
    model_predictions: dict     # Individual model predictions


def check_quality_gates(market) -> QualityGate:
    """Check if a market passes all quality gates for trading."""
    reasons = []

    vol_total = float(market.volume_total or 0)
    vol_24h = float(market.volume_24h or 0)
    liquidity = float(market.liquidity or 0)
    price = float(market.price_yes or 0.5)

    volume_ok = vol_total >= MIN_VOLUME_TOTAL
    if not volume_ok:
        reasons.append(f"volume_total ${vol_total:.0f} < ${MIN_VOLUME_TOTAL}")

    volume_24h_ok = vol_24h >= MIN_VOLUME_24H
    if not volume_24h_ok:
        reasons.append(f"volume_24h ${vol_24h:.0f} < ${MIN_VOLUME_24H}")

    liquidity_ok = liquidity >= MIN_LIQUIDITY
    if not liquidity_ok:
        reasons.append(f"liquidity ${liquidity:.0f} < ${MIN_LIQUIDITY}")

    price_ok = MIN_PRICE <= price <= MAX_PRICE
    if not price_ok:
        reasons.append(f"price {price:.3f} outside [{MIN_PRICE}, {MAX_PRICE}]")

    return QualityGate(
        passes=volume_ok and volume_24h_ok and liquidity_ok and price_ok,
        reasons=reasons,
        volume_ok=volume_ok,
        volume_24h_ok=volume_24h_ok,
        liquidity_ok=liquidity_ok,
        price_ok=price_ok,
    )


def compute_directional_ev(
    ensemble_prob: float,
    market_price: float,
    taker_fee_bps: int = 0,
) -> tuple[str | None, float, float]:
    """Compute directional expected value for both sides.

    Returns (direction, net_ev, fee_cost)
    """
    fee_cost = (taker_fee_bps / 10000) + SLIPPAGE_BUFFER

    # EV of buying YES at price q when true prob is p
    ev_yes = ensemble_prob - market_price - fee_cost
    # EV of buying NO
    ev_no = market_price - ensemble_prob - fee_cost  # = (1-p) - (1-q) - fees

    if ev_yes > ev_no and ev_yes > 0:
        return "buy_yes", ev_yes, fee_cost
    elif ev_no > 0:
        return "buy_no", ev_no, fee_cost
    else:
        return None, 0.0, fee_cost


def compute_kelly(
    direction: str | None,
    ensemble_prob: float,
    market_price: float,
    fee_cost: float,
) -> float:
    """Compute fractional Kelly criterion for position sizing."""
    if direction is None:
        return 0.0

    if direction == "buy_yes":
        # f* = (p - q - fees) / (1 - q) for buying YES
        kelly_raw = (ensemble_prob - market_price - fee_cost) / max(0.01, 1 - market_price)
    else:
        # f* = (q - p - fees) / q for buying NO
        kelly_raw = (market_price - ensemble_prob - fee_cost) / max(0.01, market_price)

    # Apply fractional Kelly for safety and clip
    return max(0.0, min(kelly_raw * KELLY_FRACTION, MAX_KELLY))


def compute_confidence(
    model_predictions: dict,
    features: dict | None = None,
    active_features: list[str] | None = None,
    net_ev: float = 0.0,
) -> float:
    """Multi-signal confidence score.

    Components:
    1. Model agreement (low std = high agreement)
    2. Feature completeness (how many features had real data vs defaults)
    3. Edge magnitude (stronger edge = more confident)
    """
    # Component 1: Model agreement
    pred_values = [v["probability"] for v in model_predictions.values()]
    if len(pred_values) >= 2:
        model_std = np.std(pred_values)
        agreement_score = max(0.0, min(1.0, 1.0 - model_std * 5))
    else:
        agreement_score = 0.5  # Single model, neutral

    # Component 2: Feature completeness
    completeness_score = 1.0
    if features and active_features:
        # Features that default to 0.0 when no data available
        zero_default_features = {
            "return_1h", "volatility_20", "zscore_24h",
            "obi_level1", "obi_weighted_5", "bid_ask_spread_abs",
            "bid_ask_spread_rel", "bid_depth_usd", "ask_depth_usd",
            "vwap_deviation", "volume_trend_7d", "volume_acceleration",
        }
        n_defaulted = sum(
            1 for name in active_features
            if name in zero_default_features and features.get(name, 0.0) == 0.0
        )
        n_checkable = sum(1 for name in active_features if name in zero_default_features)
        if n_checkable > 0:
            completeness_score = max(0.0, 1.0 - n_defaulted / n_checkable)

    # Component 3: Edge magnitude
    edge_score = min(1.0, abs(net_ev) / 0.10)

    # Weighted combination
    confidence = (
        agreement_score * 0.4 +
        completeness_score * 0.3 +
        edge_score * 0.3
    )
    return round(max(0.0, min(1.0, confidence)), 3)


def classify_quality_tier(net_ev: float, confidence: float) -> str:
    """Classify edge into quality tier."""
    if net_ev >= 0.05 and confidence >= 0.7:
        return "high"
    elif net_ev >= 0.03 and confidence >= 0.4:
        return "medium"
    return "low"


def detect_edge(
    market,
    ensemble_result: dict,
    features: dict | None = None,
    active_features: list[str] | None = None,
) -> EdgeSignal:
    """Full edge detection for a single market.

    Args:
        market: Market ORM object
        ensemble_result: Output from EnsembleModel.predict_market()
        features: Optional feature dict for completeness scoring
        active_features: Optional list of active feature names
    """
    gate = check_quality_gates(market)

    ensemble_prob = ensemble_result["ensemble_probability"]
    market_price = ensemble_result["market_price"]
    taker_fee_bps = getattr(market, "taker_fee_bps", 0) or 0

    direction, net_ev, fee_cost = compute_directional_ev(
        ensemble_prob, market_price, taker_fee_bps
    )

    kelly = compute_kelly(direction, ensemble_prob, market_price, fee_cost)

    confidence = compute_confidence(
        ensemble_result.get("model_predictions", {}),
        features=features,
        active_features=active_features,
        net_ev=net_ev,
    )

    quality_tier = classify_quality_tier(net_ev, confidence)

    return EdgeSignal(
        market_id=market.id,
        direction=direction,
        ensemble_prob=ensemble_prob,
        market_price=market_price,
        raw_edge=abs(ensemble_prob - market_price),
        fee_cost=fee_cost,
        net_ev=net_ev,
        kelly_fraction=kelly,
        confidence=confidence,
        quality_tier=quality_tier,
        quality_gate=gate,
        model_predictions=ensemble_result.get("model_predictions", {}),
    )


def edge_signal_to_dict(signal: EdgeSignal) -> dict:
    """Serialize EdgeSignal to API-friendly dict."""
    return {
        "market_id": signal.market_id,
        "direction": signal.direction,
        "ensemble_prob": round(signal.ensemble_prob, 4),
        "market_price": round(signal.market_price, 4),
        "raw_edge_pct": round(signal.raw_edge * 100, 2),
        "fee_cost_pct": round(signal.fee_cost * 100, 2),
        "net_ev_pct": round(signal.net_ev * 100, 2),
        "kelly_fraction": round(signal.kelly_fraction, 4),
        "confidence": signal.confidence,
        "quality_tier": signal.quality_tier,
        "quality_gate": {
            "passes": signal.quality_gate.passes,
            "reasons": signal.quality_gate.reasons,
        },
        "model_predictions": signal.model_predictions,
    }
