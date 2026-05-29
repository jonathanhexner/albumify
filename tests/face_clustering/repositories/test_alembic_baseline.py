"""Phase 3 — baseline migration matches legacy DDL; stamp on populated DB is idempotent."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

from face_cluster._paths import alembic_ini_path
from face_cluster.repositories._schema import ensure_schema
from face_cluster.run_history_db import _CREATE_SQL, _migrate_013


_TYPE_AFFINITY = {"FLOAT": "REAL", "DOUBLE": "REAL", "INT": "INTEGER"}


def _columns(db_path: Path) -> dict[str, dict]:
    """Return {column_name: {type, notnull, pk}} for action_log.

    Normalises SQLite type affinities (FLOAT≡REAL) and the legacy quirk
    where INTEGER PRIMARY KEY can be reported as nullable (SQLite treats
    it as implicitly NOT NULL via the ROWID alias).
    """
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute("PRAGMA table_info(action_log)").fetchall()
    out: dict[str, dict] = {}
    for r in rows:
        col_type = _TYPE_AFFINITY.get(r[2].upper(), r[2].upper())
        notnull = 1 if r[5] == 1 else r[3]
        out[r[1]] = {"type": col_type, "notnull": notnull, "pk": r[5]}
    return out


def _indexes(db_path: Path) -> dict[str, list[str]]:
    """Return {index_name: [column,...]} for action_log (excl. autoindexes)."""
    with sqlite3.connect(str(db_path)) as conn:
        idx_rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='action_log' "
            "AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        result: dict[str, list[str]] = {}
        for (name,) in idx_rows:
            cols = conn.execute(f"PRAGMA index_info({name})").fetchall()
            result[name] = [c[2] for c in cols]
    return result


def _alembic_upgrade(db_path: Path) -> None:
    ensure_schema(db_path)


def _alembic_stamp_head(db_path: Path) -> None:
    cfg = Config(str(alembic_ini_path()))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    command.stamp(cfg, "head")


def _legacy_build(db_path: Path) -> None:
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(_CREATE_SQL)
        _migrate_013(conn)
        conn.commit()


def test_upgrade_head_on_empty_db_matches_legacy_ddl(tmp_path):
    """alembic head schema == legacy _create + _migrate schema (columns + indexes)."""
    legacy_db = tmp_path / "legacy.db"
    alembic_db = tmp_path / "alembic.db"

    _legacy_build(legacy_db)
    _alembic_upgrade(alembic_db)

    assert _columns(alembic_db) == _columns(legacy_db)

    # Index columns must match. Index names match by construction (we kept the
    # legacy names verbatim in the ORM model).
    assert _indexes(alembic_db) == _indexes(legacy_db)


def test_alembic_stamp_on_populated_db_is_idempotent(tmp_path):
    """A pre-existing legacy-built DB with data: stamp head adds alembic_version,
    leaves data intact, doesn't touch the schema."""
    db = tmp_path / "populated.db"
    _legacy_build(db)
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            "INSERT INTO action_log (action_type, status, started_at) "
            "VALUES ('test_run', 'complete', '2026-05-28T10:00:00+00:00')"
        )
        conn.commit()

    cols_before = _columns(db)
    idx_before = _indexes(db)
    with sqlite3.connect(str(db)) as conn:
        row_count_before = conn.execute("SELECT COUNT(*) FROM action_log").fetchone()[0]

    _alembic_stamp_head(db)

    assert _columns(db) == cols_before
    assert _indexes(db) == idx_before
    with sqlite3.connect(str(db)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM action_log").fetchone()[0] == row_count_before
        version = conn.execute("SELECT version_num FROM alembic_version").fetchone()
    assert version is not None
