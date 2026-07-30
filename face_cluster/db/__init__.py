"""Face-clustering Pandera validators.

Pandera DataFrameSchemas (spec-033 P-H) that validate reads and writes
against the per-run SQLite tables. The DDL itself lives in
``sim_bench.run_db._schema`` (relocated by spec-056); this package keeps
only the validators, which are face-clustering-domain-specific.
"""
from face_cluster.db.validators import (
    FACES_SCHEMA,
    FACE_SCORES_SCHEMA,
    FILTER_DECISIONS_SCHEMA,
    IMAGES_SCHEMA,
    SCENE_CLUSTERS_SCHEMA,
    SCENE_CLUSTER_ASSIGNMENTS_SCHEMA,
)

__all__ = [
    "FACES_SCHEMA",
    "FACE_SCORES_SCHEMA",
    "FILTER_DECISIONS_SCHEMA",
    "IMAGES_SCHEMA",
    "SCENE_CLUSTERS_SCHEMA",
    "SCENE_CLUSTER_ASSIGNMENTS_SCHEMA",
]
