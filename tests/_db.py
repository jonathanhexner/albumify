"""Database URLs for tests (spec-117 R3): SQLite by default, the parity run's Postgres schema when it is given.

``tests/storage/test_neon_parity.py`` (``-m needs_neon``) runs ``tests/api``, ``tests/control`` and
``tests/serving`` in a child pytest with ``ALBUMIFY_TEST_DATABASE_URL`` pointing at a throwaway schema. Each
test then starts from an empty schema, the same as a fresh SQLite file. Not a test module.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from sim_bench.storage.database import DatabaseConfig

TEST_DATABASE_ENV = "ALBUMIFY_TEST_DATABASE_URL"
_MARK = ".postgres_emptied"
_engines: Dict[str, Engine] = {}


def postgres_url() -> str | None:
    return os.environ.get(TEST_DATABASE_ENV) or None


def _empty_postgres(url: str) -> None:
    engine = _engines.get(url)
    if engine is None:
        engine = _engines[url] = DatabaseConfig(url).make_engine()
    with engine.begin() as conn:
        names = inspect(conn).get_table_names()
        if names:
            conn.execute(text("TRUNCATE " + ", ".join(f'"{n}"' for n in names) + " RESTART IDENTITY CASCADE"))


def fresh_database_url(tmp_path: Path, name: str) -> str:
    """One test's database: the parity schema (emptied once per test), else the SQLite file ``tmp_path/name``."""
    url = postgres_url()
    if url is None:
        return f"sqlite:///{(Path(tmp_path) / name).as_posix()}"
    mark = Path(tmp_path) / _MARK
    if not mark.exists():
        _empty_postgres(url)
        mark.write_text("1", encoding="utf-8")
    return url


def fresh_api_engine(tmp_path: Path) -> Engine:
    """Engine with the API tables created, on a fresh database (see :func:`fresh_database_url`)."""
    from sim_bench.api.database.models import Base
    from sim_bench.api.database.session import _migrate_new_columns

    engine = DatabaseConfig(fresh_database_url(tmp_path, "api.db")).make_engine()
    Base.metadata.create_all(engine)
    _migrate_new_columns(engine)
    return engine
