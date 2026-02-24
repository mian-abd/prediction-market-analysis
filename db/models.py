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
    category = Column(String(100))  # Raw category from API (preserved for debugging)
    normalized_category = Column(String(50))  # Clean category from normalizer (~10 buckets)
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
    resolution_outcome = Column(Text)  # "YES"/"NO" or resolution source URL
    resolution_value = Column(Float)  # Numeric: 1.0=YES, 0.0=NO (for ML training)
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
        UniqueConstraint("market_id", "timestamp", name="uq_price_market_time"),
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


class NewsEvent(Base):
    """GDELT news articles for market context."""
    __tablename__ = "news_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    category = Column(String(100), nullable=False)  # politics, crypto, sports, etc.
    title = Column(Text, nullable=False)
    url = Column(Text)
    domain = Column(String(255))
    language = Column(String(10), default="en")
    publish_date = Column(String(20))  # GDELT format: YYYYMMDDHHMMSS
    tone = Column(Float, default=0.0)  # -10 to +10 sentiment
    social_image = Column(Text)
    fetched_at = Column(DateTime, default=datetime.utcnow)

    # Index for fast category lookups
    __table_args__ = (
        Index("idx_news_category", "category"),
        Index("idx_news_publish_date", "publish_date"),
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
    user_id = Column(String(50), nullable=False)  # Set by caller (header-based)
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
    portfolio_type = Column(String(10), nullable=False, default="manual", server_default="manual")
    is_simulated = Column(Boolean, default=True)

    __table_args__ = (
        Index("ix_position_market", "market_id"),
        Index("ix_position_user_time", "user_id", "entry_time"),
        Index("ix_position_user_market", "user_id", "market_id"),
        Index("ix_position_portfolio_type", "portfolio_type"),
        # Composite index for fast open-position-per-market lookups (duplicate prevention)
        Index("ix_position_open_market", "market_id", "portfolio_type", "exit_time"),
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


class AutoTradingConfig(Base):
    """Per-strategy auto-trading configuration."""
    __tablename__ = "auto_trading_configs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy = Column(String(50), unique=True, nullable=False)  # "ensemble" | "elo"
    is_enabled = Column(Boolean, default=False)

    # Signal filters
    min_quality_tier = Column(String(20), default="high")
    min_confidence = Column(Float, default=0.7)
    min_net_ev = Column(Float, default=0.05)

    # Sizing
    bankroll = Column(Float, default=1000.0)
    max_kelly_fraction = Column(Float, default=0.02)

    # Per-portfolio risk limits
    max_position_usd = Column(Float, default=100.0)
    max_total_exposure_usd = Column(Float, default=500.0)
    max_loss_per_day_usd = Column(Float, default=25.0)
    max_daily_trades = Column(Integer, default=20)

    # Auto-close
    stop_loss_pct = Column(Float, default=0.15)
    close_on_signal_expiry = Column(Boolean, default=True)

    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ============================================================================
# COPY TRADING TABLES
# ============================================================================


class TraderProfile(Base):
    """Trader profiles with performance metrics for copy trading."""
    __tablename__ = "trader_profiles"

    user_id = Column(String(50), primary_key=True)
    display_name = Column(String(100), nullable=False)
    bio = Column(Text)

    # Performance metrics
    total_pnl = Column(Float, default=0.0)
    roi_pct = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    avg_trade_duration_hrs = Column(Float, default=0.0)
    risk_score = Column(Integer, default=5)  # 1-10 scale
    max_drawdown = Column(Float, default=0.0)
    follower_count = Column(Integer, default=0)

    # Settings
    is_public = Column(Boolean, default=True)
    accepts_copiers = Column(Boolean, default=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_trader_public", "is_public"),
        Index("ix_trader_pnl", "total_pnl"),
        Index("ix_trader_win_rate", "win_rate"),
    )


class FollowedTrader(Base):
    """Junction table tracking follower-trader relationships."""
    __tablename__ = "followed_traders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    follower_id = Column(String(50), nullable=False)  # User doing the following
    trader_id = Column(String(50), ForeignKey("trader_profiles.user_id"), nullable=False)

    # Copy settings
    allocation_amount = Column(Float, default=1000.0)  # $ allocated to copying this trader
    copy_percentage = Column(Float, default=1.0)  # 0.5 = 50% of trader's position size
    max_position_size = Column(Float)  # Max $ per copied position
    auto_copy = Column(Boolean, default=True)
    copy_settings = Column(JSON)  # Additional settings (market filters, etc.)

    followed_at = Column(DateTime, default=datetime.utcnow)
    unfollowed_at = Column(DateTime)  # NULL if still following
    is_active = Column(Boolean, default=True)

    __table_args__ = (
        UniqueConstraint("follower_id", "trader_id", name="uq_follower_trader"),
        Index("ix_followed_trader_active", "trader_id", "is_active"),
        Index("ix_followed_follower_active", "follower_id", "is_active"),
    )


class CopyTrade(Base):
    """Records of copied trades linking follower positions to original trader positions."""
    __tablename__ = "copy_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    follower_position_id = Column(Integer, ForeignKey("portfolio_positions.id"), nullable=False)
    trader_position_id = Column(Integer, ForeignKey("portfolio_positions.id"), nullable=False)
    follower_id = Column(String(50), nullable=False)
    trader_id = Column(String(50), ForeignKey("trader_profiles.user_id"), nullable=False)

    copy_ratio = Column(Float, default=1.0)  # Position size ratio applied
    copied_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_copy_follower_position", "follower_position_id"),
        Index("ix_copy_trader_position", "trader_position_id"),
        Index("ix_copy_follower_trader", "follower_id", "trader_id"),
    )


class TraderActivity(Base):
    """Activity feed for trader actions (opens, closes, updates)."""
    __tablename__ = "trader_activities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trader_id = Column(String(50), ForeignKey("trader_profiles.user_id"), nullable=False)
    activity_type = Column(String(50), nullable=False)  # 'open_position', 'close_position', 'update_profile'
    activity_data = Column(JSON)  # Market details, P&L, etc.
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_activity_trader_time", "trader_id", "created_at"),
    )


# ============================================================================
# ML STRATEGY SIGNAL TABLES
# ============================================================================


class EnsembleEdgeSignal(Base):
    """ML ensemble-detected trading edge signals."""
    __tablename__ = "ensemble_edge_signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(Integer, ForeignKey("markets.id"), nullable=False)
    detected_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expired_at = Column(DateTime)  # NULL = still active

    # Ensemble prediction
    direction = Column(String(10), nullable=False)  # "buy_yes" or "buy_no"
    ensemble_prob = Column(Float, nullable=False)
    market_price = Column(Float, nullable=False)

    # Edge calculation (fee-aware)
    raw_edge = Column(Float, nullable=False)
    fee_cost = Column(Float, default=0.0)
    net_ev = Column(Float, nullable=False)
    kelly_fraction = Column(Float)

    # Quality
    confidence = Column(Float, nullable=False)
    quality_tier = Column(String(20), nullable=False)  # "high", "medium", "low", "speculative"
    model_predictions = Column(JSON)  # Individual model probs + weights

    # Outcome tracking
    was_correct = Column(Boolean)  # NULL until resolved
    actual_pnl = Column(Float)

    __table_args__ = (
        Index("ix_ensemble_edge_market", "market_id"),
        Index("ix_ensemble_edge_time", "detected_at"),
        Index("ix_ensemble_edge_tier", "quality_tier", "net_ev"),
    )


# ============================================================================
# ELO / SPORTS RATING TABLES
# ============================================================================


class EloRating(Base):
    """Stored Elo/Glicko-2 ratings for players (snapshots for API serving)."""
    __tablename__ = "elo_ratings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sport = Column(String(50), nullable=False)  # "tennis"
    player_name = Column(String(200), nullable=False)
    surface = Column(String(20), nullable=False, default="overall")  # overall, hard, clay, grass

    mu = Column(Float, nullable=False, default=1500.0)  # Rating (display scale)
    phi = Column(Float, nullable=False, default=350.0)  # Rating deviation
    sigma = Column(Float, nullable=False, default=0.06)  # Volatility
    match_count = Column(Integer, default=0)
    last_match_date = Column(String(10))  # YYYY-MM-DD

    updated_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("sport", "player_name", "surface", name="uq_elo_player_surface"),
        Index("ix_elo_sport_surface", "sport", "surface"),
        Index("ix_elo_player", "player_name"),
        Index("ix_elo_mu", "mu"),
    )


class EloEdgeSignal(Base):
    """Detected mispricing signals from Elo vs market price comparison."""
    __tablename__ = "elo_edge_signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(Integer, ForeignKey("markets.id"), nullable=False)
    sport = Column(String(50), nullable=False)
    detected_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expired_at = Column(DateTime)  # NULL = still active

    # Players
    player_a = Column(String(200), nullable=False)
    player_b = Column(String(200), nullable=False)
    surface = Column(String(20))

    # Elo prediction
    elo_prob_a = Column(Float, nullable=False)  # Elo win probability for player A
    elo_confidence = Column(Float, nullable=False)  # Rating confidence (0-1)

    # Market price
    market_price_yes = Column(Float, nullable=False)  # Current market price for "Yes" side
    yes_side_player = Column(String(200))  # Which player the "Yes" price maps to

    # Edge calculation (fee-aware)
    raw_edge = Column(Float, nullable=False)  # |elo_prob - market_price|
    fee_cost = Column(Float, default=0.0)  # taker_fee_bps / 10000 + slippage
    net_edge = Column(Float, nullable=False)  # raw_edge - fee_cost
    kelly_fraction = Column(Float)  # Kelly criterion bet size (clipped to 2%)

    # Outcome tracking
    was_correct = Column(Boolean)  # NULL until resolved
    actual_pnl = Column(Float)  # NULL until resolved

    __table_args__ = (
        Index("ix_edge_market", "market_id"),
        Index("ix_edge_sport_time", "sport", "detected_at"),
        Index("ix_edge_net_edge", "net_edge"),
    )


class FavoriteLongshotEdgeSignal(Base):
    """Favorite-longshot bias edge signals (research-backed strategy)."""
    __tablename__ = "favorite_longshot_edge_signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(Integer, ForeignKey("markets.id"), nullable=False)
    detected_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expired_at = Column(DateTime)  # NULL = still active

    direction = Column(String(10), nullable=False)  # "buy_yes" or "buy_no"
    calibrated_prob = Column(Float, nullable=False)
    market_price = Column(Float, nullable=False)
    raw_edge = Column(Float, nullable=False)
    fee_cost = Column(Float, default=0.0)
    net_ev = Column(Float, nullable=False)
    kelly_fraction = Column(Float)
    category = Column(String(50))
    efficiency_gap = Column(Float)
    signal_type = Column(String(30))  # underpriced_favorite, overpriced_favorite, fade_longshot

    was_correct = Column(Boolean)
    actual_pnl = Column(Float)

    __table_args__ = (
        Index("ix_fl_edge_market", "market_id"),
        Index("ix_fl_edge_time", "detected_at"),
        Index("ix_fl_edge_net_ev", "net_ev"),
    )
