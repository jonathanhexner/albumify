"""Legacy FC App runner — strangler-fig staging package.

spec-040 Phase 1: this package re-exports the FC-App-specific stage runner
from its current location in ``face_cluster.pipeline``. Both names resolve to
the same objects today; the rename is virtual until Phase 7 deletes the
originals.

After spec-040 Phase 7 the originals in ``face_cluster.pipeline`` and the
companion dataclass ``face_cluster.config.PipelineConfig`` will be deleted.
The new FC App runner lives at ``face_cluster.fc_app_runner`` (Phase 5)
and is built on the unified ``sim_bench.pipeline`` framework.

NEW callers should use ``face_cluster.fc_app_runner`` (once it lands).
Old callers using ``from face_cluster import FaceClusteringPipeline`` keep
working unchanged through Phase 5; they break at Phase 7.
"""
from face_cluster.pipeline import (
    FaceClusteringPipeline,
    PipelineResult,
    PipelineStageError,
)
from face_cluster.config import PipelineConfig

__all__ = [
    "FaceClusteringPipeline",
    "PipelineResult",
    "PipelineStageError",
    "PipelineConfig",
]
