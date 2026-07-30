"""SQLite engine factory with sane defaults.

create_engine_for_path(db_path) returns an Engine with:
- WAL journal mode (concurrent readers + one writer)
- foreign_keys enforcement on
- 5 second busy_timeout (matches legacy connection_timeout_s)
"""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, create_engine, event


def create_engine_for_path(db_path: Path | str) -> Engine:
    url = f"sqlite:///{db_path}"
    engine = create_engine(url, future=True)

    @event.listens_for(engine, "connect")
    def _apply_pragmas(dbapi_conn, _connection_record) -> None:
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.close()

    return engine


__all__ = ["create_engine_for_path"]
