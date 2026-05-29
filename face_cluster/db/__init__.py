"""Face-clustering relational store — DDL + validators.

Single source of truth for the per-run SQLite schema (`face_clustering.db`)
and the Pandera DataFrameSchemas that validate reads and writes against it.

Splitting this out from `run_exporter.py` keeps the writer focused on
*writing* and lets readers (`RunStore`, tests, ad-hoc tools) reach the
schema definitions without importing the writer.

Public surface:
    SCHEMA_VERSION       — int, pinned at the application layer's user_version
    EXPECTED_ARTIFACTS   — tuple of filenames each run dir must contain
    SCHEMA_DDL           — full CREATE TABLE script (the value formerly
                           known as `_SCHEMA` in run_exporter)
    FACES_SCHEMA, FACE_SCORES_SCHEMA, FILTER_DECISIONS_SCHEMA
                         — Pandera DataFrameSchemas (spec-033 P-H)
"""
from face_cluster.db.schema import (
    EXPECTED_ARTIFACTS,
    SCHEMA_DDL,
    SCHEMA_HISTORY,
    SCHEMA_VERSION,
)
from face_cluster.db.validators import (
    FACES_SCHEMA,
    FACE_SCORES_SCHEMA,
    FILTER_DECISIONS_SCHEMA,
    IMAGES_SCHEMA,
    SCENE_CLUSTERS_SCHEMA,
    SCENE_CLUSTER_ASSIGNMENTS_SCHEMA,
)

__all__ = [
    "SCHEMA_VERSION",
    "SCHEMA_HISTORY",
    "EXPECTED_ARTIFACTS",
    "SCHEMA_DDL",
    "FACES_SCHEMA",
    "FACE_SCORES_SCHEMA",
    "FILTER_DECISIONS_SCHEMA",
    "IMAGES_SCHEMA",
    "SCENE_CLUSTERS_SCHEMA",
    "SCENE_CLUSTER_ASSIGNMENTS_SCHEMA",
]
