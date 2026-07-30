"""Shared fixtures for face_cluster.repositories tests.

Provides the test-isolation primitives every Repository test reuses:
- fresh_db_engine: engine pointing at an empty temp DB with schema at HEAD
- transactional_session: per-test session wrapped in outer transaction rolled
  back on test exit (clean isolation, no file rewriting per test)
- golden_run_history_db_path: read-only path to the checked-in fixture
- golden_run_history_db_copy: per-test mutable copy of the fixture
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from face_cluster._paths import repo_root
from face_cluster.repositories._engine import create_engine_for_path
from face_cluster.repositories._schema import ensure_schema
from face_cluster.repositories._session import make_sessionmaker

GOLDEN_DB = repo_root() / "tests" / "face_clustering" / "fixtures" / "golden_run_history.db"


@pytest.fixture
def fresh_db_engine(tmp_path):
    db = tmp_path / "fresh.db"
    ensure_schema(db)
    engine = create_engine_for_path(db)
    yield engine
    engine.dispose()


@pytest.fixture
def transactional_session(fresh_db_engine):
    """Session wrapped in an outer transaction. Rolled back on test exit."""
    sm = make_sessionmaker(fresh_db_engine)
    session = sm()
    session.begin_nested()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture(scope="session")
def golden_run_history_db_path() -> Path:
    """Read-only path to the committed golden fixture."""
    if not GOLDEN_DB.exists():
        pytest.fail(
            f"Missing fixture {GOLDEN_DB}. Run "
            "`.venv/Scripts/python -m tests.face_clustering.fixtures.rebuild_golden`"
        )
    return GOLDEN_DB


@pytest.fixture
def golden_run_history_db_copy(tmp_path, golden_run_history_db_path) -> Path:
    """Fresh writable copy of the golden fixture per test."""
    dst = tmp_path / "golden_copy.db"
    shutil.copy(golden_run_history_db_path, dst)
    return dst
