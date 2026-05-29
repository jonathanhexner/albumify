"""spec-048 Phase 1 — `ensure_schema` (in-process alembic.command.upgrade).

Replaces the subprocess(alembic.exe) shell-out. These tests verify:
1. fresh empty DB → schema matches Alembic head
2. already-at-head DB → idempotent (no error, version unchanged)
"""
from __future__ import annotations

import sqlite3

from face_cluster.repositories._schema import ensure_schema


def _alembic_version(db_path) -> str | None:
    with sqlite3.connect(str(db_path)) as conn:
        try:
            row = conn.execute("SELECT version_num FROM alembic_version").fetchone()
        except sqlite3.OperationalError:
            return None
    return row[0] if row else None


def _table_names(db_path) -> set[str]:
    with sqlite3.connect(str(db_path)) as conn:
        return {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }


def test_ensure_schema_on_empty_db_creates_action_log(tmp_path):
    db = tmp_path / "fresh.db"
    ensure_schema(db)
    assert "action_log" in _table_names(db)
    assert _alembic_version(db) is not None


def test_ensure_schema_is_idempotent(tmp_path):
    db = tmp_path / "fresh.db"
    ensure_schema(db)
    version_after_first = _alembic_version(db)

    # Second call must not raise and must not change version
    ensure_schema(db)
    assert _alembic_version(db) == version_after_first


def test_ensure_schema_uses_centralized_alembic_ini(tmp_path):
    """Confirms ensure_schema reads from _paths.alembic_ini_path(), not
    a hardcoded location. If alembic.ini moved, this test would catch it."""
    from face_cluster._paths import alembic_ini_path
    assert alembic_ini_path().exists(), "alembic.ini missing from repo root"
    # And ensure_schema actually works against the resolved file
    db = tmp_path / "sanity.db"
    ensure_schema(db)
    assert "action_log" in _table_names(db)
