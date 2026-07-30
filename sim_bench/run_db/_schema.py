"""SQLite schema for the per-run face_clustering.db.

**spec-058: ORM is the source of truth.** Every ``*_DDL`` constant below is
derived from ``Base.metadata`` (sim_bench.run_db.models) at import time.
Editing any DDL string by hand has no effect — change the corresponding
ORM model and the derived constant updates automatically.

Bump ``SCHEMA_VERSION`` for any breaking change (column rename, type
change, constraint that rejects valid prior rows). Additive nullable
columns do NOT need a bump per the spec-033 locked decision.

spec-054: every bump MUST add an entry to ``SCHEMA_HISTORY`` describing
what changed. The arch test ``test_schema_history`` enforces this so the
next person to bump the version is forced to document why.
"""
from __future__ import annotations

from sqlalchemy.dialects import sqlite as _sqlite_dialect
from sqlalchemy.schema import CreateIndex, CreateTable

from sim_bench.run_db.models import Base


# ---------------------------------------------------------------------------
# Schema version history (hand-maintained)
# ---------------------------------------------------------------------------

SCHEMA_HISTORY: dict[int, str] = {
    3: (
        "Initial v3 layout — faces, clusters, cluster_assignments, "
        "run_metadata, merge_decisions."
    ),
    4: (
        "Added spec-030 merge fields — run_metadata.parent_run_id, "
        "iteration counters; merge_decisions table refined."
    ),
    5: (
        "spec-040 Phase 4 — added images table; faces gained area_ratio "
        "+ scene_cluster_id; cluster_assignments linked to scenes."
    ),
}

SCHEMA_VERSION: int = max(SCHEMA_HISTORY)


# Allow-list of files produced by RunExporter.export().  Tests assert
# `listdir(run_dir) == EXPECTED_ARTIFACTS` (FR-012).  `crops` is a directory;
# the rest are files.
EXPECTED_ARTIFACTS = (
    "face_clustering.db",
    "embeddings.npy",
    "embedding_face_ids.npy",
    "pipeline_run.json",
    "crops",
)


# ---------------------------------------------------------------------------
# DDL derivation from ORM models (spec-058 Phase 2)
# ---------------------------------------------------------------------------

_DIALECT = _sqlite_dialect.dialect()


def _compile(stmt) -> str:
    """Compile a CreateTable/CreateIndex statement into a DDL string."""
    return str(stmt.compile(dialect=_DIALECT)).strip() + ";"


def _table_ddl(table_name: str) -> str:
    table = Base.metadata.tables[table_name]
    return _compile(CreateTable(table, if_not_exists=True))


FACES_DDL                     = _table_ddl("faces")
FACE_SCORES_DDL               = _table_ddl("face_scores")
CLUSTERS_DDL                  = _table_ddl("clusters")
CLUSTER_ASSIGNMENTS_DDL       = _table_ddl("cluster_assignments")
MERGE_DECISIONS_DDL           = _table_ddl("merge_decisions")
FILTER_DECISIONS_DDL          = _table_ddl("filter_decisions")
IMAGES_DDL                    = _table_ddl("images")
SCENE_CLUSTERS_DDL            = _table_ddl("scene_clusters")
SCENE_CLUSTER_ASSIGNMENTS_DDL = _table_ddl("scene_cluster_assignments")
RUN_METADATA_DDL              = _table_ddl("run_metadata")


def _all_indexes_ddl() -> str:
    """Concatenate CREATE INDEX statements for every index defined on every model.

    Indexes are emitted in (table-FK-dependency-order, declaration-order) so
    the output is deterministic across runs and platforms.
    """
    lines: list[str] = []
    for table in Base.metadata.sorted_tables:
        for index in table.indexes:
            lines.append(_compile(CreateIndex(index, if_not_exists=True)))
    return "\n".join(lines)


INDEXES_DDL = _all_indexes_ddl()


# Concatenated DDL for executescript().  Tables emitted in
# sorted_tables order — SQLAlchemy resolves FK dependencies (faces before
# face_scores, images before scene_cluster_assignments) automatically.
SCHEMA_DDL = "\n".join(
    [
        "PRAGMA foreign_keys = ON;",
        *(_compile(CreateTable(t, if_not_exists=True)) for t in Base.metadata.sorted_tables),
        INDEXES_DDL,
    ]
)
