"""spec-053 Phase 1.5 — direct unit tests for D10ExemplarSelector.

No dedicated test file existed prior to spec-053. Three minimal cases.
"""
from __future__ import annotations

import numpy as np

from face_cluster.config import PipelineConfig
from face_cluster.exemplars import D10ExemplarSelector
from face_cluster.knn_graph import KNNGraphBuilder
from face_cluster.types import ClusterResult, FaceRecord


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
             d10_k=2, exemplars_d10_threshold=2.0, N_exemplars_max=3,
             exemplar_suppression_radius=0.05,
             blur_min=0.0, yaw_max=999.0, pitch_max=999.0, roll_max=999.0,
             max_faces_per_image_core=50)
    d.update(over)
    return PipelineConfig(**d)


def _build_graph(faces, cfg):
    return KNNGraphBuilder(cfg).build_graph(faces, list(range(len(faces))))


def test_single_face_cluster_returns_that_face_as_exemplar() -> None:
    cfg = _cfg()
    rng = np.random.RandomState(0)
    faces = [_face(0, rng.randn(512).astype(np.float32))]
    graph = _build_graph(faces, cfg)
    cr = ClusterResult(
        labels=np.array([0]), clusters={0: [0]},
        cluster_stats={0: {}}, exemplars={}, n_clusters=1, n_noise=0,
    )
    result = D10ExemplarSelector(cfg).select_exemplars(cr, graph)
    cr_out, d10_map = result
    assert cr_out.exemplars[0] == [0]


def test_d10_map_populated_for_every_core_node() -> None:
    cfg = _cfg()
    rng = np.random.RandomState(1)
    faces = [_face(i, rng.randn(512).astype(np.float32)) for i in range(4)]
    graph = _build_graph(faces, cfg)
    cr = ClusterResult(
        labels=np.array([0, 0, 0, 0]),
        clusters={0: [0, 1, 2, 3]},
        cluster_stats={0: {}}, exemplars={}, n_clusters=1, n_noise=0,
    )
    _, d10_map = D10ExemplarSelector(cfg).select_exemplars(cr, graph)
    assert set(d10_map.keys()) == {0, 1, 2, 3}


def test_n_exemplars_max_caps_output() -> None:
    """N_exemplars_max=1 should yield at most one exemplar per cluster."""
    cfg = _cfg(N_exemplars_max=1)
    rng = np.random.RandomState(2)
    faces = [_face(i, rng.randn(512).astype(np.float32)) for i in range(5)]
    graph = _build_graph(faces, cfg)
    cr = ClusterResult(
        labels=np.array([0, 0, 0, 0, 0]),
        clusters={0: [0, 1, 2, 3, 4]},
        cluster_stats={0: {}}, exemplars={}, n_clusters=1, n_noise=0,
    )
    cr_out, _ = D10ExemplarSelector(cfg).select_exemplars(cr, graph)
    assert len(cr_out.exemplars[0]) <= 1
