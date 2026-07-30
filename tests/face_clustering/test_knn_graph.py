"""spec-053 Phase 1.5 — direct unit tests for KNNGraphBuilder.

No dedicated test file existed prior to spec-053. Three minimal cases.
"""
from __future__ import annotations

import numpy as np

from face_cluster.config import PipelineConfig
from face_cluster.knn_graph import KNNGraphBuilder
from face_cluster.types import FaceRecord


def _face(face_id: int, emb: np.ndarray) -> FaceRecord:
    return FaceRecord(
        face_id=face_id, image_id=f"img{face_id}", image_path=f"img{face_id}",
        bbox=(0, 0, 100, 100), landmarks=None, aligned_face=None,
        embedding=emb,
        embedding_normalized=emb / (np.linalg.norm(emb) + 1e-9),
        pose=None, blur_score=100.0, area=10000, is_core=True,
        face_index=face_id,
    )


def _cfg(**over) -> PipelineConfig:
    d = dict(K=2, distance_threshold=0.9, min_cluster_size=1,
             blur_min=0.0, yaw_max=999.0, pitch_max=999.0, roll_max=999.0,
             max_faces_per_image_core=50)
    d.update(over)
    return PipelineConfig(**d)


def test_empty_core_returns_empty_graph() -> None:
    builder = KNNGraphBuilder(_cfg())
    result = builder.build_graph(faces=[], core_indices=[])
    assert list(result.G.nodes()) == []
    assert result.edges == []


def test_two_close_faces_get_an_edge() -> None:
    """Two near-identical embeddings should be mutual-kNN connected."""
    rng = np.random.RandomState(0)
    base = rng.randn(512).astype(np.float32)
    e0 = base + 0.001 * rng.randn(512).astype(np.float32)
    e1 = base + 0.001 * rng.randn(512).astype(np.float32)
    faces = [_face(0, e0), _face(1, e1)]
    builder = KNNGraphBuilder(_cfg(K=1, distance_threshold=0.5))
    result = builder.build_graph(faces, core_indices=[0, 1])
    assert len(result.edges) == 1


def test_far_faces_get_no_edge() -> None:
    """Two random embeddings should NOT be connected under a tight threshold."""
    rng = np.random.RandomState(1)
    e0 = rng.randn(512).astype(np.float32)
    e1 = rng.randn(512).astype(np.float32)
    faces = [_face(0, e0), _face(1, e1)]
    builder = KNNGraphBuilder(_cfg(K=1, distance_threshold=0.01))
    result = builder.build_graph(faces, core_indices=[0, 1])
    assert result.edges == []
