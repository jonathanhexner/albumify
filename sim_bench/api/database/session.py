"""Database session management."""

import logging
import os
from pathlib import Path
from typing import Generator

from sqlalchemy import text, inspect
from sqlalchemy.orm import sessionmaker, Session

from sim_bench.api.database.models import Base
from sim_bench.storage.database import DatabaseConfig

logger = logging.getLogger(__name__)


_engine = None
_SessionLocal = None


def get_database_path() -> Path:
    """Get default database path. ``SIM_BENCH_DB`` points the app at another file (spec-116: isolated tests)."""
    override = os.environ.get("SIM_BENCH_DB")
    if override:
        return Path(override)
    db_dir = Path.home() / ".sim_bench"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "sim_bench.db"


def get_database_url(db_path: Path = None) -> str:
    """``$DATABASE_URL`` (spec-117: one database for every Albumify table), else the SQLite file."""
    if db_path is not None:
        return DatabaseConfig.sqlite(db_path).url
    return DatabaseConfig.from_env(get_database_path()).url


_MIGRATIONS = [
    ("pipeline_results", "fc_export_dir"),
    ("pipeline_results", "step_decisions"),
    ("pipeline_runs", "completed_steps"),
]


def _migrate_new_columns(engine) -> None:
    """Idempotent migration: add columns that don't exist yet on existing tables.

    The column type is compiled from the model for the engine's dialect (spec-117 R3), so the same
    migration runs on SQLite and on Postgres."""
    insp = inspect(engine)
    with engine.begin() as conn:
        for table, column in _MIGRATIONS:
            if not insp.has_table(table):
                continue
            existing = {c["name"] for c in insp.get_columns(table)}
            if column not in existing:
                col_type = Base.metadata.tables[table].c[column].type.compile(dialect=engine.dialect)
                conn.execute(text(f'ALTER TABLE {table} ADD COLUMN "{column}" {col_type}'))
                logger.info("Migration: added %s.%s", table, column)


def init_db(db_url: str = None) -> None:
    """Initialize database engine and create tables."""
    global _engine, _SessionLocal

    if db_url is None:
        db_url = get_database_url()

    config = DatabaseConfig(db_url)
    _engine = config.make_engine()
    logger.info("API database: %s", config.redacted())
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

    Base.metadata.create_all(_engine)
    _migrate_new_columns(_engine)


def get_engine():
    """Get the database engine, initializing if needed."""
    global _engine
    if _engine is None:
        init_db()
    return _engine


def get_session() -> Generator[Session, None, None]:
    """Dependency for FastAPI to get a database session."""
    global _SessionLocal
    if _SessionLocal is None:
        init_db()

    session = _SessionLocal()
    try:
        yield session
    finally:
        session.close()


def get_session_direct() -> Session:
    """Get a database session directly (not as a generator)."""
    global _SessionLocal
    if _SessionLocal is None:
        init_db()
    return _SessionLocal()
