"""Per-run-DB SQLAlchemy session factory (spec-058 / spec-059).

Distinct from ``face_cluster/repositories/_session.py`` (which targets
``~/.sim_bench/sim_bench.db`` — Alembic-managed, long-lived). The per-run
``face_clustering.db`` has a different lifecycle (one DB per run, never
migrated) and uses its own ORM `Base` from
``sim_bench/run_db/models/_base.py``. The two factories must NEVER be
cross-imported.

Per-call sessions: callers do ``with sessionmaker() as session: ...`` and
the engine is held on the construct-once owning object (`RunStore`,
`ClusterAnalysisRepository`). No pooling — matches today's per-call
``sqlite3.connect()`` lifecycle, which Streamlit polling depends on.
"""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker


def make_run_db_engine(run_dir: Path) -> Engine:
    """Build a SQLAlchemy Engine bound to ``run_dir/face_clustering.db``.

    The caller owns the engine — call ``engine.dispose()`` (or let it be
    garbage-collected) when finished. Foreign-key enforcement is enabled
    per connection so reads see the same constraints the writers built.
    """
    db_path = (Path(run_dir) / "face_clustering.db").resolve()
    engine = create_engine(f"sqlite:///{db_path}", future=True)

    @event.listens_for(engine, "connect")
    def _enable_fk(dbapi_conn, _connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.close()

    return engine


def make_run_db_sessionmaker(run_dir: Path) -> sessionmaker[Session]:
    """Return a sessionmaker bound to a fresh per-run-DB engine."""
    engine = make_run_db_engine(run_dir)
    return sessionmaker(bind=engine, future=True, expire_on_commit=False)


__all__ = ["make_run_db_engine", "make_run_db_sessionmaker"]
