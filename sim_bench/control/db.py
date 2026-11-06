"""Control-plane DB engine (spec-112 A2, spec-117 R1).

``DATABASE_URL`` picks the backend: unset -> a local SQLite file ``control.db`` (offline, tests, CI);
a Postgres URL -> Neon, the same database that holds the API's tables and the feature cache. The engine
comes from :class:`~sim_bench.storage.database.DatabaseConfig`, the one factory every Albumify table uses.

Schema creation is ``create_all``, plus ``_add_missing_columns`` for columns added later (spec-114:
``jobs.tag``, ``jobs.worker``; spec-117: ``jobs.progress`` ...). Only additive, nullable columns are allowed
that way; anything else (a type change, a NOT NULL column) needs a real migration. The one rename so far,
``people`` -> ``job_people`` (spec-117 R2), is done by ``_rename_job_people``. Alembic in this repo targets
``sim_bench.db`` only (alembic/env.py).
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from sim_bench.control.models import ControlBase, Person
from sim_bench.storage.database import ENV_VAR, DatabaseConfig, redact_url

logger = logging.getLogger(__name__)

__all__ = ["ControlDB", "default_url", "ENV_VAR"]


def default_url() -> str:
    """``$DATABASE_URL`` or ``sqlite:///~/.sim_bench/control.db``."""
    return DatabaseConfig.from_env(Path.home() / ".sim_bench" / "control.db").url


_redact = redact_url      # the spec-112 name, kept for its callers


class ControlDB:
    """Engine + session factory for the control DB. Construct once per process."""

    def __init__(self, url: Optional[str] = None, *, create: bool = True):
        self.config = DatabaseConfig(url or default_url())
        self.url = self.config.url
        self.is_sqlite = self.config.is_sqlite
        self.engine: Engine = self.config.make_engine(sqlite_foreign_keys=True)
        self._sessions = sessionmaker(bind=self.engine, expire_on_commit=False)
        if create:
            _rename_job_people(self.engine)
            ControlBase.metadata.create_all(self.engine)
            _add_missing_columns(self.engine)
        logger.info("control DB: %s", self.config.redacted())

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


def _rename_job_people(engine: Engine) -> None:
    """spec-117 R2: the job table ``people`` is now ``job_people`` (the API has its own ``people``). An older
    control.db is renamed in place, rows kept. Only a ``people`` table keyed by ``job_id`` is ours."""
    insp = inspect(engine)
    if not insp.has_table("people") or insp.has_table(Person.__tablename__):
        return
    if "job_id" not in {c["name"] for c in insp.get_columns("people")}:
        return                                   # the API's people table: not ours to touch
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE people RENAME TO {Person.__tablename__}"))
    logger.info("control DB: renamed table people -> %s", Person.__tablename__)


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
