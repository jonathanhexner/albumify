"""In-process Alembic schema upgrade for SQLite databases.

Replaces the spec-046 subprocess shell-out (smell S1) with the standard
Python API. Idempotent — calling multiple times on a head-version DB
is a no-op.

Performance contract: ``ensure_schema`` is called from
``RunHistoryRepository.__init__``. Streamlit reruns the script on every
interaction, so this must be near-instant when the DB is already at
head. The fast path is a single ``SELECT version_num FROM
alembic_version LIMIT 1`` followed by a string compare against the head
revision — no Alembic command invoked, no subprocess, no engine
created. Alembic is only consulted when the schema is actually behind.

Logging: passes ``cfg.attributes["configure_logger"] = False`` so
Alembic's ``env.py`` does NOT call ``fileConfig`` on alembic.ini.
Without this, every upgrade call would reset the host process's root
logger and discard the FileHandler installed by
``sim_bench.logging_setup`` (symptom: ``logs/<ts>/fc_app_v2.log``
empty, Alembic INFO spam on console).

Adoption path: if a pre-existing DB has the legacy ``action_log`` table
but no ``alembic_version`` row, we ``alembic stamp head`` once rather
than re-creating the schema — the standard SQLAlchemy/Alembic recipe
for adopting an existing database.
"""
from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

from face_cluster._paths import alembic_ini_path


def _has_table(db_path: Path, name: str) -> bool:
    if not db_path.exists():
        return False
    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        ).fetchone()
    return row is not None


def _current_version(db_path: Path) -> str | None:
    """Read ``alembic_version.version_num`` or None if table empty/missing."""
    if not _has_table(db_path, "alembic_version"):
        return None
    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute("SELECT version_num FROM alembic_version LIMIT 1").fetchone()
    return row[0] if row else None


def _make_config(db_path: Path) -> Config:
    cfg = Config(str(alembic_ini_path()))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    cfg.attributes["configure_logger"] = False
    return cfg


@lru_cache(maxsize=1)
def _head_revision() -> str:
    """Head migration revision id. Pure function of the on-disk
    ``alembic/versions/`` tree, so safe to cache process-wide."""
    cfg = Config(str(alembic_ini_path()))
    return ScriptDirectory.from_config(cfg).get_current_head()


def ensure_schema(db_path: Path) -> None:
    """Ensure ``db_path`` is at Alembic head. Idempotent and near-instant
    on warm calls (one SELECT)."""
    db_path = Path(db_path)
    current = _current_version(db_path)
    if current == _head_revision():
        return

    cfg = _make_config(db_path)
    if _has_table(db_path, "action_log") and current is None:
        command.stamp(cfg, "head")
    else:
        command.upgrade(cfg, "head")


__all__ = ["ensure_schema"]
