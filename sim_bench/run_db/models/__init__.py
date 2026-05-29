"""SQLAlchemy ORM models for the per-run face_clustering.db (spec-058).

Source of truth for the per-run schema. The DDL constants in
``sim_bench/run_db/_schema.py`` are kept for backward compat but are derived
from ``Base.metadata`` — see the drift-guard test in
``tests/face_clustering/db/test_orm_matches_schema.py``.
"""
from __future__ import annotations

from sim_bench.run_db.models._base import Base
from sim_bench.run_db.models.cluster import Cluster
from sim_bench.run_db.models.cluster_assignment import ClusterAssignment
from sim_bench.run_db.models.face import Face
from sim_bench.run_db.models.face_scores import FaceScores
from sim_bench.run_db.models.filter_decision import FilterDecision
from sim_bench.run_db.models.image import Image
from sim_bench.run_db.models.merge_decision import MergeDecision
from sim_bench.run_db.models.run_metadata import RunMetadataRow
from sim_bench.run_db.models.scene_cluster import SceneCluster
from sim_bench.run_db.models.scene_cluster_assignment import SceneClusterAssignment

__all__ = [
    "Base",
    "Cluster",
    "ClusterAssignment",
    "Face",
    "FaceScores",
    "FilterDecision",
    "Image",
    "MergeDecision",
    "RunMetadataRow",
    "SceneCluster",
    "SceneClusterAssignment",
]
