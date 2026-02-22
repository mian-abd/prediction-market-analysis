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

# --- Quality Gate Thresholds (TIGHTENED 2026-02-14 to prevent thin market losses) ---
MIN_VOLUME_TOTAL = 10000    # USD, liquid enough to trade (was 5K, now 2× stricter)
MIN_VOLUME_24H = 1000       # USD, market is actively traded (was 200, now 5× stricter)
MIN_LIQUIDITY = 5000        # Open interest, can absorb position (was 1K, now 5× stricter)
MIN_PRICE = 0.02            # Tradeable range lower bound
MAX_PRICE = 0.98            # Tradeable range upper bound

# --- Edge Thresholds ---
MIN_NET_EDGE = 0.03         # 3% minimum net edge after fees
SLIPPAGE_BUFFER = 0.01      # 1% slippage estimate
POLYMARKET_FEE_RATE = 0.02  # 2% on net winnings (only charged when winning)
MAX_KELLY = 0.02            # Maximum Kelly fraction (2%)
KELLY_FRACTION = 0.25       # Fractional Kelly multiplier (safety)
MIN_CONFIDENCE = 0.5        # Minimum confidence score
MAX_CREDIBLE_EDGE = 0.15    # 15% — edges above this are likely noise/leakage


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

    Polymarket charges 2% on NET WINNINGS only when you win (0% if you lose).
    Expected fee = win_probability * fee_rate * winnings_per_share.

    Direction selection is weighted by payoff asymmetry — we prefer directions
    where risk:reward is favorable (e.g. buy_no on overpriced favorites).

    Returns (direction, net_ev, fee_cost) where fee_cost is for the chosen direction.
    """
    p = ensemble_prob    # true probability of YES
    q = market_price     # market YES price
    extra_fee = (taker_fee_bps / 10000)  # additional platform fees (usually 0)

    # Buy YES: pay q, receive 1 if YES. Winnings = (1-q). Fee only if win.
    fee_yes = p * POLYMARKET_FEE_RATE * (1 - q) + SLIPPAGE_BUFFER + extra_fee * (1 - q)
    ev_yes = p * (1 - q) - (1 - p) * q - fee_yes

    # Buy NO: pay (1-q), receive 1 if NO. Winnings = q. Fee only if win.
    fee_no = (1 - p) * POLYMARKET_FEE_RATE * q + SLIPPAGE_BUFFER + extra_fee * q
    ev_no = (1 - p) * q - p * (1 - q) - fee_no

    # Weight EV by payoff ratio — prefer directions where one win covers multiple losses
    # buy_yes payoff ratio: (1-q)/q — favorable when q is low (cheap YES)
    # buy_no payoff ratio: q/(1-q) — favorable when q is high (overpriced favorite)
    payoff_yes = (1 - q) / max(q, 0.01)  # gain/risk for buy_yes
    payoff_no = q / max(1 - q, 0.01)      # gain/risk for buy_no
    adj_ev_yes = ev_yes * min(payoff_yes, 5.0) if ev_yes > 0 else ev_yes
    adj_ev_no = ev_no * min(payoff_no, 5.0) if ev_no > 0 else ev_no

    if adj_ev_yes > adj_ev_no and ev_yes > 0:
        return "buy_yes", ev_yes, fee_yes
    elif ev_no > 0:
        return "buy_no", ev_no, fee_no
    else:
        return None, 0.0, fee_yes if ev_yes >= ev_no else fee_no


def compute_kelly(
    direction: str | None,
    ensemble_prob: float,
    market_price: float,
    fee_cost: float,
) -> float:
    """Compute fractional Kelly criterion for position sizing.

    Kelly for binary bet with asymmetric fees (fee only on winning):
      Buy YES: f* = EV / (1 - q) where EV already accounts for expected fees
      Buy NO:  f* = EV / q where EV already accounts for expected fees

    fee_cost is the expected fee for this direction (already probability-weighted).
    """
    if direction is None:
        return 0.0

    p = ensemble_prob
    q = market_price

    if direction == "buy_yes":
        # Net EV per dollar of winnings = (p - q - fee_cost)
        # Kelly = net_ev / max_loss_fraction = (p - q - fee_cost) / (1 - q)
        # But fee_cost is now probability-weighted, so use the cleaner form:
        ev = p * (1 - q) - (1 - p) * q - fee_cost
        kelly_raw = ev / max(0.01, 1 - q)
    else:
        ev = (1 - p) * q - p * (1 - q) - fee_cost
        kelly_raw = ev / max(0.01, q)

    # Fractional Kelly: cap raw at 8%, then multiply by 0.25 → 2% max
    kelly_capped = min(kelly_raw, MAX_KELLY / KELLY_FRACTION)
    return max(0.0, kelly_capped * KELLY_FRACTION)


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

    # Component 3: Edge magnitude — any meaningful edge gets full score
    # (Removed penalty on large edges — research shows large edges on cheap markets
    # are where real alpha lives in prediction markets, not noise)
    abs_ev = abs(net_ev)
    if abs_ev >= 0.03:
        edge_score = 1.0  # Any edge above min threshold = full score
    else:
        edge_score = abs_ev / 0.03  # Linear ramp below 3%

    # Weighted combination
    confidence = (
        agreement_score * 0.4 +
        completeness_score * 0.3 +
        edge_score * 0.3
    )
    return round(max(0.0, min(1.0, confidence)), 3)


def classify_quality_tier(net_ev: float, confidence: float, raw_edge: float = 0.0) -> str:
    """Classify edge into quality tier.

    Large edges are now allowed through the asymmetry-adjusted confidence score.
    The payoff asymmetry factor in detect_edge() naturally suppresses large edges
    on expensive contracts (where they're likely noise) while allowing them on
    cheap contracts (where they represent real alpha from favorite-longshot bias).
    """
    if raw_edge > MAX_CREDIBLE_EDGE:
        return "speculative"
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

    # Payoff asymmetry factor: penalize trades where risk:reward is unfavorable
    # buy_yes at 0.80 → payoff_ratio=0.25 → factor=0.50 → confidence halved
    # buy_no at 0.80 → payoff_ratio=4.0 → factor=1.0 → full confidence
    # buy_yes at 0.30 → payoff_ratio=2.33 → factor=1.0 → full confidence
    if direction == "buy_yes":
        payoff_ratio = (1 - market_price) / max(market_price, 0.01)
    elif direction == "buy_no":
        payoff_ratio = market_price / max(1 - market_price, 0.01)
    else:
        payoff_ratio = 1.0
    asymmetry_factor = min(1.0, payoff_ratio / 0.5)  # Full score when gain >= 50% of risk
    confidence = round(confidence * asymmetry_factor, 3)

    raw_edge = abs(ensemble_prob - market_price)
    quality_tier = classify_quality_tier(net_ev, confidence, raw_edge)

    # Warn on extreme edges
    if quality_tier == "speculative" and gate.passes:
        gate.reasons.append(
            f"edge {raw_edge:.1%} exceeds credibility threshold {MAX_CREDIBLE_EDGE:.0%}"
        )

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
