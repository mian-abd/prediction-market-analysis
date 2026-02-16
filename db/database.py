from pathlib import Path

from sqlalchemy import make_url
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase

from config.settings import settings


def _resolve_sqlite_url(url_str: str) -> str:
    """Ensure SQLite DB path is absolute and its parent directory exists (fixes 'unable to open database file' in production)."""
    url = make_url(url_str)
    if "sqlite" not in url.drivername or not url.database or url.database == ":memory:":
        return url_str
    db_path = Path(url.database)
    if not db_path.is_absolute():
        db_path = Path.cwd() / url.database
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Return URL with absolute path (sqlite wants 3 slashes then path)
    return f"{url.drivername}:///{db_path.as_posix()}"


_db_url = settings.database_url
_is_sqlite = _db_url.strip().startswith("sqlite")
if _is_sqlite:
    _db_url = _resolve_sqlite_url(_db_url)

# SQLite-specific connect_args
_connect_args = {"check_same_thread": False} if _is_sqlite else {}

engine = create_async_engine(
    _db_url,
    echo=False,
    connect_args=_connect_args,
)

async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_session() -> AsyncSession:
    async with async_session() as session:
        yield session


async def init_db():
    """Create all tables and migrate column sizes for PostgreSQL compatibility."""
    from db.models import Base  # noqa: F811
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Widen columns that were too small for PostgreSQL (SQLite ignores VARCHAR limits)
    if not _is_sqlite:
        async with engine.begin() as conn:
            migrations = [
                "ALTER TABLE markets ALTER COLUMN resolution_outcome TYPE TEXT",
                "ALTER TABLE ensemble_edge_signals ALTER COLUMN quality_tier TYPE VARCHAR(20)",
                "ALTER TABLE auto_trading_configs ALTER COLUMN min_quality_tier TYPE VARCHAR(20)",
            ]
            for sql in migrations:
                try:
                    from sqlalchemy import text
                    await conn.execute(text(sql))
                except Exception:
                    pass  # Column already correct type or table doesn't exist yet


async def close_db():
    await engine.dispose()
