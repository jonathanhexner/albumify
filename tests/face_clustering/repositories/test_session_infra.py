"""Phase 1 — smoke tests for engine + session infrastructure."""
from __future__ import annotations

import pytest
from sqlalchemy import text

from face_cluster.repositories._engine import create_engine_for_path
from face_cluster.repositories._orm_base import Base
from face_cluster.repositories._session import make_sessionmaker, session_scope


def _temp_engine(tmp_path):
    engine = create_engine_for_path(tmp_path / "smoke.db")
    Base.metadata.create_all(engine)
    return engine


def test_engine_pragmas_applied(tmp_path):
    engine = create_engine_for_path(tmp_path / "pragmas.db")
    with engine.connect() as conn:
        assert conn.execute(text("PRAGMA journal_mode")).scalar() == "wal"
        assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1
        assert conn.execute(text("PRAGMA busy_timeout")).scalar() == 5000


def test_session_scope_commits_on_success(tmp_path):
    engine = _temp_engine(tmp_path)
    sm = make_sessionmaker(engine)
    with session_scope(sm) as session:
        session.execute(text("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)"))
        session.execute(text("INSERT INTO t (v) VALUES ('hello')"))

    with engine.connect() as conn:
        row = conn.execute(text("SELECT v FROM t WHERE id=1")).scalar()
    assert row == "hello"


def test_session_scope_rolls_back_on_exception(tmp_path):
    engine = _temp_engine(tmp_path)
    sm = make_sessionmaker(engine)
    with engine.connect() as conn:
        conn.execute(text("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)"))
        conn.commit()

    with pytest.raises(RuntimeError, match="boom"):
        with session_scope(sm) as session:
            session.execute(text("INSERT INTO t (v) VALUES ('should-roll-back')"))
            raise RuntimeError("boom")

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM t")).scalar()
    assert count == 0
