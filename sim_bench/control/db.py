"""Control-plane DB engine (spec-112 A2).

``DATABASE_URL`` picks the backend: unset -> a local SQLite file (Phase A, offline, tests, CI);
a Postgres URL -> Neon (Phase B). Nothing else changes between the two.

Schema creation is ``create_all``, plus ``_add_missing_columns`` for columns added later (spec-114:
``jobs.tag``, ``jobs.worker``). Only additive, nullable columns are allowed that way; anything else (a
rename, a type change, a NOT NULL column) needs a real migration. Alembic in this repo targets
``sim_bench.db`` only (alembic/env.py).
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from sim_bench.control.models import ControlBase

logger = logging.getLogger(__name__)

__all__ = ["ControlDB", "default_url", "ENV_VAR"]

ENV_VAR = "DATABASE_URL"


def default_url() -> str:
    """``$DATABASE_URL`` or ``sqlite:///~/.sim_bench/control.db``."""
    url = os.environ.get(ENV_VAR)
    if url:
        return url
    path = Path.home() / ".sim_bench" / "control.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.as_posix()}"


def _redact(url: str) -> str:
    """Never log a password: keep scheme + host only."""
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    return f"{scheme}://***@{rest.split('@', 1)[1]}"


class ControlDB:
    """Engine + session factory for the control DB. Construct once per process."""

    def __init__(self, url: Optional[str] = None, *, create: bool = True):
        self.url = url or default_url()
        self.is_sqlite = self.url.startswith("sqlite")
        kwargs = {"connect_args": {"check_same_thread": False}} if self.is_sqlite else \
            {"pool_pre_ping": True}   # Neon closes idle connections when it scales to zero
        self.engine: Engine = create_engine(self.url, **kwargs)
        if self.is_sqlite:
            event.listen(self.engine, "connect", _sqlite_fk_on)
        self._sessions = sessionmaker(bind=self.engine, expire_on_commit=False)
        if create:
            ControlBase.metadata.create_all(self.engine)
            _add_missing_columns(self.engine)
        logger.info("control DB: %s", _redact(self.url))

    @contextmanager
    def session(self) -> Iterator[Session]:
        """One unit of work: commit on success, roll back on error."""
        s = self._sessions()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()


def _add_missing_columns(engine: Engine) -> None:
    """Add model columns an older DB lacks. Additive + nullable only; anything else raises."""
    insp = inspect(engine)
    for table in ControlBase.metadata.sorted_tables:
        existing = {c["name"] for c in insp.get_columns(table.name)}
        for col in table.columns:
            if col.name in existing:
                continue
            if not col.nullable:
                raise RuntimeError(f"control DB: {table.name}.{col.name} is NOT NULL and missing; "
                                   "needs a real migration")
            ddl = col.type.compile(dialect=engine.dialect)
            with engine.begin() as conn:
                conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN "{col.name}" {ddl}'))
            logger.info("control DB: added column %s.%s", table.name, col.name)


def _sqlite_fk_on(dbapi_conn, _record) -> None:
    # SQLite ignores ON DELETE CASCADE unless foreign keys are switched on per connection.
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()
