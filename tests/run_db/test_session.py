"""spec-059 T011 — session-factory smoke + leak test."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from sim_bench.run_db._schema import SCHEMA_DDL, SCHEMA_VERSION
from sim_bench.run_db._session import make_run_db_engine, make_run_db_sessionmaker
from sim_bench.run_db.models import Face


def _seed_minimal_db(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    db_path = run_dir / "face_clustering.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(SCHEMA_DDL)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
    finally:
        conn.close()


def test_sessionmaker_yields_usable_session(tmp_path: Path) -> None:
    _seed_minimal_db(tmp_path)
    SessionLocal = make_run_db_sessionmaker(tmp_path)
    with SessionLocal() as session:
        # Empty DB; the query must execute without error and return [].
        rows = session.query(Face).all()
        assert rows == []


def test_no_connection_leaks_over_repeated_open_close(tmp_path: Path) -> None:
    """Open and close 100 sessions in a loop. SQLite would error on a
    file-descriptor leak (Windows: WinError 32, Linux: 'too many open
    files'). A clean run is the assertion."""
    _seed_minimal_db(tmp_path)
    SessionLocal = make_run_db_sessionmaker(tmp_path)
    for _ in range(100):
        with SessionLocal() as session:
            session.query(Face).all()


def test_engine_enables_foreign_keys(tmp_path: Path) -> None:
    _seed_minimal_db(tmp_path)
    engine = make_run_db_engine(tmp_path)
    with engine.connect() as conn:
        result = conn.exec_driver_sql("PRAGMA foreign_keys").scalar()
        assert result == 1
    engine.dispose()


def test_engine_dispose_releases_file_handle(tmp_path: Path) -> None:
    """spec-059 F-1: ``engine.dispose()`` releases the OS file handle so the
    DB file can be deleted on Windows.

    Windows refuses to unlink a SQLite file while any process holds an open
    handle (``WinError 32``). The sessionmaker pattern callers use depends
    on dispose returning the file. Regression here would mean Streamlit
    re-runs accumulate handles until the run dir can't be cleaned up.
    """
    _seed_minimal_db(tmp_path)
    engine = make_run_db_engine(tmp_path)
    # Open + close a session to ensure the engine has actually connected.
    SessionLocal = __import__(
        "sqlalchemy.orm", fromlist=["sessionmaker"]
    ).sessionmaker(bind=engine, future=True)
    with SessionLocal() as session:
        session.execute(__import__("sqlalchemy").text("SELECT 1"))
    engine.dispose()
    # After dispose the file must be unlink-able even on Windows.
    db_path = tmp_path / "face_clustering.db"
    db_path.unlink()
    assert not db_path.exists()
