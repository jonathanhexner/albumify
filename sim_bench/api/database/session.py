"""Database session management."""

from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from sim_bench.api.database.models import Base


_engine = None
_SessionLocal = None


def get_database_path() -> Path:
    """Get default database path."""
    db_dir = Path.home() / ".sim_bench"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "sim_bench.db"


def get_database_url(db_path: Path = None) -> str:
    """Get SQLite database URL."""
    if db_path is None:
        db_path = get_database_path()
    return f"sqlite:///{db_path}"


def init_db(db_url: str = None) -> None:
    """Initialize database engine and create tables."""
    global _engine, _SessionLocal

    if db_url is None:
        db_url = get_database_url()

    _engine = create_engine(db_url, echo=False, connect_args={"check_same_thread": False})
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

    Base.metadata.create_all(_engine)


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
