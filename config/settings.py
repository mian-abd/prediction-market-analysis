from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Paths
    project_root: Path = Path(__file__).parent.parent
    database_url: str = "sqlite+aiosqlite:///./data/markets.db"

    # Anthropic
    anthropic_api_key: str = ""

    # Polymarket
    polymarket_gamma_url: str = "https://gamma-api.polymarket.com"
    polymarket_clob_url: str = "https://clob.polymarket.com"

    # Kalshi
    kalshi_api_url: str = "https://api.elections.kalshi.com/trade-api/v2"

    # GDELT
    gdelt_api_url: str = "https://api.gdeltproject.org/api/v2/doc/doc"

    # Pipeline intervals (seconds)
    price_poll_interval_sec: int = 60
    orderbook_poll_interval_sec: int = 300
    market_refresh_interval_sec: int = 3600

    # Arbitrage thresholds
    min_single_market_profit_pct: float = 0.5
    min_cross_platform_spread_pct: float = 2.5
    max_position_usd: float = 100.0
    max_total_exposure_usd: float = 1000.0
    max_daily_trades: int = 50
    max_loss_per_day_usd: float = 50.0

    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    frontend_url: str = "http://localhost:5173"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
