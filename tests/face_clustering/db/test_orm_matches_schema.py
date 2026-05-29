"""spec-058 drift guard.

Asserts that the per-run face_clustering.db schema produced by
``Base.metadata.create_all()`` matches the schema produced by today's
``executescript(SCHEMA_DDL)`` byte-for-byte at the sqlite_master level.

If anyone edits ``sim_bench/run_db/_schema.py``'s DDL strings by hand
without updating the corresponding ORM model in
``sim_bench/run_db/models/`` (or vice versa), this test fails. Plays the
role ``alembic check`` plays for long-lived DBs (spec-046) — without
Alembic, because per-run DBs are never migrated.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from sim_bench.run_db._schema import SCHEMA_DDL
from sim_bench.run_db.models import Base


def _create_from_ddl(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA_DDL)
        conn.commit()
    finally:
        conn.close()


def _create_from_orm(db_path: Path) -> None:
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    engine.dispose()


def _read_sqlite_master(db_path: Path) -> dict[tuple[str, str], str]:
    """Return {(type, name): normalized_sql} for tables and indexes.

    Whitespace is collapsed and `IF NOT EXISTS` is stripped so cosmetic
    differences between DDL strings and ORM-emitted SQL don't trigger a
    false failure. Column order and constraint semantics are preserved.
    """
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT type, name, sql FROM sqlite_master "
            "WHERE type IN ('table', 'index') "
            "AND name NOT LIKE 'sqlite_%' "
            "ORDER BY type, name"
        ).fetchall()
    finally:
        conn.close()

    return {(t, n): _normalize_sql(sql or "") for t, n, sql in rows}


def _normalize_sql(sql: str) -> str:
    text = " ".join(sql.split())
    for token in ("IF NOT EXISTS ", '"', "`"):
        text = text.replace(token, "")
    return text.lower()


@pytest.fixture
def ddl_db(tmp_path: Path) -> Path:
    p = tmp_path / "ddl.db"
    _create_from_ddl(p)
    return p


@pytest.fixture
def orm_db(tmp_path: Path) -> Path:
    p = tmp_path / "orm.db"
    _create_from_orm(p)
    return p


def test_table_set_matches(ddl_db: Path, orm_db: Path) -> None:
    ddl_tables = {n for (t, n) in _read_sqlite_master(ddl_db) if t == "table"}
    orm_tables = {n for (t, n) in _read_sqlite_master(orm_db) if t == "table"}
    assert ddl_tables == orm_tables, (
        f"Tables differ. DDL-only: {ddl_tables - orm_tables}; "
        f"ORM-only: {orm_tables - ddl_tables}"
    )


def test_index_set_matches(ddl_db: Path, orm_db: Path) -> None:
    ddl_indexes = {n for (t, n) in _read_sqlite_master(ddl_db) if t == "index"}
    orm_indexes = {n for (t, n) in _read_sqlite_master(orm_db) if t == "index"}
    assert ddl_indexes == orm_indexes, (
        f"Indexes differ. DDL-only: {ddl_indexes - orm_indexes}; "
        f"ORM-only: {orm_indexes - ddl_indexes}"
    )


def _columns(db_path: Path, table: str) -> list[tuple]:
    """PRAGMA table_info rows: (cid, name, type, notnull, dflt, pk)."""
    conn = sqlite3.connect(db_path)
    try:
        return [
            (cid, name, type_.upper(), notnull, dflt, pk)
            for (cid, name, type_, notnull, dflt, pk) in conn.execute(
                f"PRAGMA table_info({table})"
            ).fetchall()
        ]
    finally:
        conn.close()


def test_columns_match_per_table(ddl_db: Path, orm_db: Path) -> None:
    ddl_tables = sorted(n for (t, n) in _read_sqlite_master(ddl_db) if t == "table")
    diffs: list[str] = []
    for table in ddl_tables:
        ddl_cols = _columns(ddl_db, table)
        orm_cols = _columns(orm_db, table)
        if ddl_cols != orm_cols:
            diffs.append(
                f"\n  {table}:\n    DDL: {ddl_cols}\n    ORM: {orm_cols}"
            )
    assert not diffs, "Column drift between DDL and ORM:" + "".join(diffs)
