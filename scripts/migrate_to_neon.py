"""Copy the Albumify tables from the local SQLite files to the one database (spec-117 W7, R4).

    sim_bench.db   albums, pipeline_runs, pipeline_results, people, config_profiles, universal_cache, user_events,
                   face_overrides, image_metric_rows, face_metric_rows
    control.db     jobs, job_events, photos, faces, people -> job_people, scenes, user_edits

    plan    row counts per table in the sources (and whether the target is empty)
    copy    refuses a target that already holds any of these rows; creates the tables; copies in FK order, in
            batches, through the ORM column types (JSON, DateTime, LargeBinary round-trip on both dialects);
            on Postgres moves the id sequences past the copied ids
    verify  per-table counts, source vs target; raises on any difference

The SQLite files are only read: they stay as the backup. The target URL is never printed (redacted).

    .venv/Scripts/python scripts/migrate_to_neon.py --plan
    .venv/Scripts/python scripts/migrate_to_neon.py --copy --yes      # target = DATABASE_URL from .env
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from sqlalchemy import Column, MetaData, Table, inspect, insert, select, text
from sqlalchemy.engine import Engine

from sim_bench.api.database.models import Base
from sim_bench.control.models import ControlBase, Person as JobPerson
from sim_bench.storage.database import DatabaseConfig

logger = logging.getLogger("migrate_to_neon")

__all__ = ["NeonMigrator", "TablePlan", "NotEmpty", "CountMismatch"]

# where each source table lands (control.db's old ``people`` is ``job_people`` now, spec-117 R2)
_RENAMES = {"control": {"people": JobPerson.__tablename__}}


class NotEmpty(RuntimeError):
    pass


class CountMismatch(RuntimeError):
    pass


@dataclass(frozen=True)
class TablePlan:
    source: str          # "api" | "control"
    source_table: str
    target_table: str
    rows: int


@dataclass
class _Source:
    name: str
    engine: Engine
    metadata: MetaData
    tables: List[Tuple[str, Table]] = field(default_factory=list)    # (name in the source, model table)


class NeonMigrator:
    """Config in ``__init__``: the two SQLite sources and the target URL."""

    def __init__(self, source_urls: Dict[str, str], target_url: str, *, batch: int = 500,
                 skip: Iterable[str] = ()):
        self.source_urls, self.target_url, self.batch = dict(source_urls), target_url, batch
        self.skip = set(skip)
        self._target = DatabaseConfig(target_url)

    # --- plan -----------------------------------------------------------------------------

    def _sources(self) -> List[_Source]:
        out = []
        for name, metadata in (("api", Base.metadata), ("control", ControlBase.metadata)):
            url = self.source_urls.get(name)
            if not url:
                continue
            engine = DatabaseConfig(url).make_engine()
            present = set(inspect(engine).get_table_names())
            src = _Source(name, engine, metadata)
            renames = {v: k for k, v in _RENAMES.get(name, {}).items()}       # target -> source name
            for table in metadata.sorted_tables:
                if table.name in self.skip:
                    continue
                source_name = table.name if table.name in present else renames.get(table.name)
                if source_name in present:
                    src.tables.append((source_name, table))
            out.append(src)
        return out

    def plan(self) -> List[TablePlan]:
        plans = []
        for src in self._sources():
            with src.engine.connect() as c:
                for source_name, table in src.tables:
                    n = c.execute(text(f'SELECT COUNT(*) FROM "{source_name}"')).scalar()
                    plans.append(TablePlan(src.name, source_name, table.name, int(n)))
            src.engine.dispose()
        return plans

    def target_rows(self, engine: Optional[Engine] = None) -> Dict[str, int]:
        own = engine is None
        engine = engine or self._target.make_engine()
        try:
            present = set(inspect(engine).get_table_names())
            names = [t.name for md in (Base.metadata, ControlBase.metadata) for t in md.sorted_tables
                     if t.name in present]
            with engine.connect() as c:
                return {n: int(c.execute(text(f'SELECT COUNT(*) FROM "{n}"')).scalar()) for n in names}
        finally:
            if own:
                engine.dispose()

    # --- copy -----------------------------------------------------------------------------

    def copy(self) -> List[TablePlan]:
        target = self._target.make_engine()
        try:
            filled = {n: c for n, c in self.target_rows(target).items() if c}
            if filled:
                raise NotEmpty(f"target already holds rows in {sorted(filled)}; refusing to copy over them")
            Base.metadata.create_all(target)
            from sim_bench.control.db import ControlDB
            ControlDB(self.target_url).engine.dispose()          # job tables + their column additions
            done = []
            for src in self._sources():
                for source_name, table in src.tables:
                    n = self._copy_table(src, source_name, table, target)
                    done.append(TablePlan(src.name, source_name, table.name, n))
                    logger.info("%s.%s -> %s: %d rows", src.name, source_name, table.name, n)
                src.engine.dispose()
            if not self._target.is_sqlite:
                self._move_sequences(target)
            return done
        finally:
            target.dispose()

    def _copy_table(self, src: _Source, source_name: str, table: Table, target: Engine) -> int:
        have = {c["name"] for c in inspect(src.engine).get_columns(source_name)}
        cols = [c for c in table.columns if c.name in have]              # an older file may lack new columns
        source_table = table if source_name == table.name else _alias(table, source_name)
        n = 0
        with src.engine.connect() as s_conn, target.begin() as t_conn:
            result = s_conn.execute(select(*[source_table.c[c.name] for c in cols]))
            while True:
                rows = result.fetchmany(self.batch)
                if not rows:
                    break
                t_conn.execute(insert(table), [dict(zip([c.name for c in cols], r)) for r in rows])
                n += len(rows)
        return n

    def _move_sequences(self, target: Engine) -> None:
        with target.begin() as conn:
            for md in (Base.metadata, ControlBase.metadata):
                for table in md.sorted_tables:
                    pk = list(table.primary_key.columns)
                    if len(pk) == 1 and pk[0].autoincrement is not False and str(pk[0].type) == "INTEGER":
                        conn.execute(text(
                            f"SELECT setval(pg_get_serial_sequence('{table.name}', '{pk[0].name}'), "
                            f"COALESCE((SELECT MAX({pk[0].name}) FROM {table.name}), 0) + 1, false)"))

    # --- verify ---------------------------------------------------------------------------

    def verify(self) -> Dict[str, Tuple[int, int]]:
        want = {p.target_table: p.rows for p in self.plan()}
        got = self.target_rows()
        counts = {t: (n, got.get(t, 0)) for t, n in want.items()}
        bad = {t: c for t, c in counts.items() if c[0] != c[1]}
        if bad:
            raise CountMismatch(f"row counts differ (source, target): {bad}")
        return counts


def _alias(table: Table, name: str) -> Table:
    """The model's columns (types only, for reading) under the source's old table name (control.db ``people``)."""
    return Table(name, MetaData(), *[Column(c.name, c.type) for c in table.columns])


def main(argv: Optional[List[str]] = None) -> int:
    from sim_bench.storage.database import load_env

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--copy", action="store_true")
    ap.add_argument("--yes", action="store_true", help="really copy")
    ap.add_argument("--api-db", default=str(Path.home() / ".sim_bench" / "sim_bench.db"))
    ap.add_argument("--control-db", default=str(Path.home() / ".sim_bench" / "control.db"))
    ap.add_argument("--skip", nargs="*", default=[], help="tables not to copy (e.g. universal_cache)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_env()
    import os
    target = os.environ.get("DATABASE_URL")
    if not target:
        print("DATABASE_URL is not set (.env); nothing to migrate to")
        return 2
    m = NeonMigrator({"api": DatabaseConfig.sqlite(Path(a.api_db)).url,
                      "control": DatabaseConfig.sqlite(Path(a.control_db)).url}, target, skip=a.skip)
    print(f"target: {DatabaseConfig(target).redacted()}")
    for p in m.plan():
        print(f"  {p.source:8s} {p.source_table:22s} -> {p.target_table:22s} {p.rows:8d} rows")
    if a.copy:
        if not a.yes:
            print("add --yes to copy")
            return 1
        m.copy()
        for t, (src, dst) in m.verify().items():
            print(f"  verified {t:22s} {src:8d} = {dst:8d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
