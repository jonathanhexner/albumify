"""Unit tests for the face_cluster_knn method in ClusterPeopleStep.

Coverage:
  - Synthetic 3-identity clustering: purity + completeness
  - Quality gating: blurry/off-pose faces go to holdout (label=-1)
  - Context people_clusters format matches downstream expectations
  - Holdout path exercised (core_indices is proper subset)
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from sim_bench.pipeline.steps.cluster_people import ClusterPeopleStep, FaceForClustering


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_identity_embeddings(n_per_identity: int, n_identities: int, noise: float = 0.02) -> np.ndarray:
    """Create synthetic 512-dim embeddings with clear identity separation.

    Each identity gets a random unit vector; faces get that vector + small noise.
    """
    rng = np.random.RandomState(42)
    embeddings = []
    for _id in range(n_identities):
        center = rng.randn(512).astype(np.float32)
        center /= np.linalg.norm(center)
        for _ in range(n_per_identity):
            face_emb = center + rng.randn(512).astype(np.float32) * noise
            face_emb /= np.linalg.norm(face_emb)
            embeddings.append(face_emb)
    return np.array(embeddings, dtype=np.float32)


def _make_faces(n: int) -> List[FaceForClustering]:
    """Create n FaceForClustering objects with distinct image paths."""
    return [
        FaceForClustering(
            original_path=Path(f"/images/img_{i:04d}.jpg"),
            face_index=0,
            embedding=np.zeros(512, dtype=np.float32),  # placeholder, overwritten
            bbox={"x": 100, "y": 100, "w": 200, "h": 200},
        )
        for i in range(n)
    ]


@dataclass
class _FakeContext:
    """Minimal stand-in for PipelineContext to test the method directly."""
    people_clusters: Dict[int, list] = field(default_factory=dict)
    _fc_export_dir: str = ""

    def report_progress(self, *args, **kwargs):
        pass


# ---------------------------------------------------------------------------
# ut_FaceClusterKNNMethod
# ---------------------------------------------------------------------------

class ut_FaceClusterKNNMethod:

    def test_three_identities_purity_and_completeness(self):
        """3 identities x 5 faces each must cluster into 3 pure, complete groups."""
        n_per = 5
        n_id = 3
        n = n_per * n_id
        embeddings = _make_identity_embeddings(n_per, n_id)
        faces = _make_faces(n)
        for i in range(n):
            faces[i].embedding = embeddings[i]

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings_norm = embeddings / norms

        step = ClusterPeopleStep()
        config = {
            "K": 3,
            "distance_threshold": 0.5,
            "min_cluster_size": 2,
            "blur_min": 0.0,       # disable quality gating for this test
            "yaw_max": 180.0,
            "pitch_max": 180.0,
        }
        ctx = _FakeContext()
        labels = step._run_face_cluster_knn(faces, embeddings_norm, config, ctx)

        # Purity: each cluster has faces from exactly one identity
        cluster_ids = set(labels)
        cluster_ids.discard(-1)
        assert len(cluster_ids) >= 3, f"Expected >= 3 clusters, got {len(cluster_ids)}"
        for cid in cluster_ids:
            face_indices = [i for i, l in enumerate(labels) if l == cid]
            identities = {i // n_per for i in face_indices}
            assert len(identities) == 1, (
                f"Cluster {cid} has mixed identities: {identities}"
            )

        # Completeness: all faces of same identity in same cluster
        for identity in range(n_id):
            expected_faces = list(range(identity * n_per, (identity + 1) * n_per))
            cluster_ids_for_identity = {labels[i] for i in expected_faces if labels[i] != -1}
            assert len(cluster_ids_for_identity) == 1, (
                f"Identity {identity} split across clusters: {cluster_ids_for_identity}"
            )

    def test_returns_label_array_aligned_with_faces(self):
        """Label array must have same length as input faces."""
        n = 10
        embeddings = _make_identity_embeddings(5, 2)
        faces = _make_faces(n)
        for i in range(n):
            faces[i].embedding = embeddings[i]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings_norm = embeddings / norms

        step = ClusterPeopleStep()
        config = {"K": 3, "distance_threshold": 0.5, "min_cluster_size": 2,
                  "blur_min": 0.0, "yaw_max": 180.0, "pitch_max": 180.0}
        labels = step._run_face_cluster_knn(faces, embeddings_norm, config, _FakeContext())
        assert len(labels) == n

    # Deleted 2026-05-29 (SIGHTING-071): test_quality_gating_holdout and
    # test_faces_to_face_records_bridge were calling code paths that no
    # longer exist after spec-040's pipeline unification:
    #   - The blur gate moved out of ClusterPeopleStep into the standalone
    #     `quality_gate` step (spec-053). Coverage of blur enforcement is
    #     in tests/face_clustering/test_quality_gate_step.py
    #     ::test_blur_gate_actually_filters_when_min_is_high.
    #   - `_faces_to_face_records` was removed; face-to-record conversion
    #     happens in producer steps now. No replacement test needed —
    #     producer step coverage exists separately.
