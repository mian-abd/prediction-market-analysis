"""SQLAlchemy ORM models - 12 tables for the prediction market platform."""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text,
    ForeignKey, Index, UniqueConstraint, JSON,
)
from db.database import Base


class Platform(Base):
    __tablename__ = "platforms"

    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    base_url = Column(String(255))
    fee_structure = Column(JSON)
    is_active = Column(Boolean, default=True)


class Market(Base):
    __tablename__ = "markets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    platform_id = Column(Integer, ForeignKey("platforms.id"), nullable=False)
    external_id = Column(String(255), nullable=False)
    condition_id = Column(String(255))
    token_id_yes = Column(String(255))
    token_id_no = Column(String(255))
    question = Column(Text, nullable=False)
    description = Column(Text)
    category = Column(String(100))
    slug = Column(String(255))

    # Current state
    price_yes = Column(Float)
    price_no = Column(Float)
    volume_24h = Column(Float, default=0)
    volume_total = Column(Float, default=0)
    liquidity = Column(Float, default=0)
    open_interest = Column(Float, default=0)

    # Resolution
    is_active = Column(Boolean, default=True)
    is_resolved = Column(Boolean, default=False)
    resolution_outcome = Column(String(10))
    end_date = Column(DateTime)
    resolved_at = Column(DateTime)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_fetched_at = Column(DateTime)

    # Fee info
    taker_fee_bps = Column(Integer, default=0)
    maker_fee_bps = Column(Integer, default=0)
    is_neg_risk = Column(Boolean, default=False)

    __table_args__ = (
        UniqueConstraint("platform_id", "external_id", name="uq_platform_market"),
        Index("ix_market_active", "is_active"),
        Index("ix_market_category", "category"),
        Index("ix_market_end_date", "end_date"),
    )


class PriceSnapshot(Base):
    __tablename__ = "price_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(Integer, ForeignKey("markets.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    price_yes = Column(Float, nullable=False)
    price_no = Column(Float, nullable=False)
    midpoint = Column(Float)
    spread = Column(Float)
    volume = Column(Float, default=0)

    __table_args__ = (
        Index("ix_price_market_time", "market_id", "timestamp"),
    )


class OrderbookSnapshot(Base):
    __tablename__ = "orderbook_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(Integer, ForeignKey("markets.id"), nullable=False)
    side = Column(String(3), nullable=False)
    timestamp = Column(DateTime, nullable=False)

    best_bid = Column(Float)
    best_ask = Column(Float)
    bid_ask_spread = Column(Float)

    bid_depth_total = Column(Float)
    ask_depth_total = Column(Float)

    obi_level1 = Column(Float)
    obi_weighted = Column(Float)
    depth_ratio = Column(Float)

    bids_json = Column(JSON)
    asks_json = Column(JSON)

    __table_args__ = (
        Index("ix_ob_market_time", "market_id", "timestamp"),
    )


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(Integer, ForeignKey("markets.id"), nullable=False)
    external_trade_id = Column(String(255))
    timestamp = Column(DateTime, nullable=False)
    side = Column(String(4), nullable=False)
    outcome = Column(String(3), nullable=False)
    price = Column(Float, nullable=False)
    size = Column(Float, nullable=False)

    __table_args__ = (
        Index("ix_trade_market_time", "market_id", "timestamp"),
    )


class CrossPlatformMatch(Base):
    __tablename__ = "cross_platform_matches"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id_a = Column(Integer, ForeignKey("markets.id"), nullable=False)
    market_id_b = Column(Integer, ForeignKey("markets.id"), nullable=False)
    similarity_score = Column(Float, nullable=False)
    match_method = Column(String(50))
    is_confirmed = Column(Boolean, default=False)
    is_inverted = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("market_id_a", "market_id_b", name="uq_match_pair"),
    )


class MarketRelationship(Base):
    __tablename__ = "market_relationships"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id_a = Column(Integer, ForeignKey("markets.id"), nullable=False)
    market_id_b = Column(Integer, ForeignKey("markets.id"), nullable=False)
    relationship_type = Column(String(50), nullable=False)
    constraint_expression = Column(Text)
    confidence = Column(Float, default=1.0)
    source = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)


class ArbitrageOpportunity(Base):
    __tablename__ = "arbitrage_opportunities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_type = Column(String(50), nullable=False)
    detected_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expired_at = Column(DateTime)

    market_ids = Column(JSON, nullable=False)
    prices_snapshot = Column(JSON)

    gross_spread = Column(Float, nullable=False)
    total_fees = Column(Float, nullable=False)
    net_profit_pct = Column(Float, nullable=False)
    estimated_profit_usd = Column(Float)

    was_executed = Column(Boolean, default=False)
    execution_result = Column(JSON)
    actual_profit_usd = Column(Float)

    __table_args__ = (
        Index("ix_arb_strategy_time", "strategy_type", "detected_at"),
        Index("ix_arb_profit", "net_profit_pct"),
    )


class MLPrediction(Base):
    __tablename__ = "ml_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(Integer, ForeignKey("markets.id"), nullable=False)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50))
    predicted_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    prediction_value = Column(Float, nullable=False)
    prediction_label = Column(String(20))
    confidence = Column(Float)

    market_price_at_prediction = Column(Float)
    calibrated_price = Column(Float)

    features_json = Column(JSON)

    __table_args__ = (
        Index("ix_pred_market_model", "market_id", "model_name", "predicted_at"),
    )


class AIAnalysis(Base):
    __tablename__ = "ai_analyses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(Integer, ForeignKey("markets.id"), nullable=False)
    analysis_type = Column(String(50), nullable=False)

    prompt_hash = Column(String(64), nullable=False)
    prompt_text = Column(Text)

    response_text = Column(Text, nullable=False)
    structured_result = Column(JSON)

    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    estimated_cost_usd = Column(Float)
    model_used = Column(String(100))

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_ai_prompt_hash", "prompt_hash"),
        Index("ix_ai_market", "market_id"),
    )


class PortfolioPosition(Base):
    __tablename__ = "portfolio_positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(Integer, ForeignKey("markets.id"), nullable=False)
    platform_id = Column(Integer, ForeignKey("platforms.id"), nullable=False)
    side = Column(String(3), nullable=False)
    entry_price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    entry_time = Column(DateTime, nullable=False)
    exit_price = Column(Float)
    exit_time = Column(DateTime)
    realized_pnl = Column(Float)
    strategy = Column(String(50))
    is_simulated = Column(Boolean, default=True)

    __table_args__ = (
        Index("ix_position_market", "market_id"),
    )


class SystemMetric(Base):
    __tablename__ = "system_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    metadata_json = Column(JSON)

    __table_args__ = (
        Index("ix_metric_name_time", "metric_name", "timestamp"),
    )
