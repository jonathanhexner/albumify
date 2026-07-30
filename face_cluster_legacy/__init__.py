"""Legacy FC App runner — strangler-fig staging package.

**DEPRECATED — scheduled for deletion in spec-040 Phase 7.**

This package re-exports the FC-App-specific stage runner from
``face_cluster.pipeline``. Both names resolve to the same objects today;
the rename is virtual until Phase 7 deletes the originals. The new FC App
runner lives at ``face_cluster.fc_app_runner`` (spec-040 Phase 5a, shipped
in commit ``b5ef128``) and is built on the unified ``sim_bench.pipeline``
framework.

NEW callers should use ``face_cluster.fc_app_runner.FCAppRunner``. Old
callers using ``from face_cluster import FaceClusteringPipeline`` keep
working unchanged through Phase 5b; they break at Phase 7.
"""
import warnings

from face_cluster.pipeline import (
    FaceClusteringPipeline,
    PipelineResult,
    PipelineStageError,
)
from face_cluster.config import PipelineConfig

warnings.warn(
    "face_cluster_legacy is deprecated and scheduled for deletion in "
    "spec-040 Phase 7. Migrate to face_cluster.fc_app_runner.FCAppRunner, "
    "which drives the unified sim_bench.pipeline framework and supports "
    "the same step list as the legacy FaceClusteringPipeline.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "FaceClusteringPipeline",
    "PipelineResult",
    "PipelineStageError",
    "PipelineConfig",
]
