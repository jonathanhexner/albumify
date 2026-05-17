"""spec-040 Phase 6: equivalence test — legacy bridge vs unified clustering chain.

Runs the same input through:
  1. Legacy path: ``face_cluster_bridge.run_face_cluster_knn`` (today's
     production path on Albumify; uses dict-of-dicts).
  2. New path: the 8-step unified clustering chain via ``FCAppRunner``
     (spec-040 Phase 5; reads context.face_records directly).

Asserts pairwise cluster-assignment agreement ≥ 95% (locked bar).

This test is the **merge gate** for spec-040. While the new FC App is
being built, this test runs against synthetic data so the equivalence
contract is tested even before a real labeled album is available.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

from face_cluster.fc_app_runner import FCAppRunner
from face_cluster.types import FaceRecord
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.steps.face_cluster_bridge import run_face_cluster_knn


# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------

@dataclass
class _LegacyFaceForClustering:
    """Minimal duck-typed face matching what the bridge expects.

    Mirrors the shape of sim_bench.pipeline.steps.cluster_people.FaceForClustering
    that the bridge reads in legacy mode.
    """
    original_path: Path
    face_index: int
    embedding: np.ndarray
    bbox: dict = field(default_factory=dict)


def _make_synthetic_faces(
    n_per_identity: int = 5, n_identities: int = 3, dim: int = 64
) -> Tuple[List[_LegacyFaceForClustering], np.ndarray, List[FaceRecord]]:
    """Produce parallel legacy-style and Pydantic-style face lists.

    Same underlying embeddings, just two different representations so the
    two paths receive equivalent inputs.
    """
    legacy: List[_LegacyFaceForClustering] = []
    records: List[FaceRecord] = []
    embeddings = []
    face_id = 0
    for identity in range(n_identities):
        rng = np.random.default_rng(seed=identity * 1000)
        base = np.zeros(dim, dtype=np.float32)
        base[identity % dim] = 1.0
        for k in range(n_per_identity):
            noise = 0.05 * rng.standard_normal(dim).astype(np.float32)
            emb = base + noise
            embeddings.append(emb)
            img_path = Path(f"/fake/identity_{identity}_face_{k}.jpg")
            legacy.append(_LegacyFaceForClustering(
                original_path=img_path,
                face_index=0,
                embedding=emb,
                bbox={"x": 0.0, "y": 0.0, "w": 100.0, "h": 100.0},
            ))
            norm = float(np.linalg.norm(emb))
            records.append(FaceRecord(
                face_id=face_id,
                image_id=img_path.name,
                bbox=(0.0, 0.0, 100.0, 100.0),
                embedding=emb,
                embedding_normalized=emb / norm if norm > 0 else emb,
                area=10000.0,
                blur_score=100.0,
                image_path=str(img_path),
                face_index=0,
            ))
            face_id += 1
    embeddings_arr = np.stack(embeddings)
    # L2-normalize
    norms = np.linalg.norm(embeddings_arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings_norm = embeddings_arr / norms
    return legacy, embeddings_norm, records


# ---------------------------------------------------------------------------
# Equivalence metric: pairwise same-cluster agreement
# ---------------------------------------------------------------------------

def _labels_from_legacy(labels: np.ndarray) -> Dict[int, int]:
    """face_id → cluster_id from the legacy bridge's label array."""
    return {i: int(lab) for i, lab in enumerate(labels)}


def _labels_from_v2(context: PipelineContext) -> Dict[int, int]:
    """face_id → cluster_id from the unified chain's people_clusters."""
    out: Dict[int, int] = {}
    for cid, faces in context.people_clusters.items():
        for face in faces:
            out[face.face_id] = int(cid)
    # noise faces (not in any cluster) get -1
    for i in range(len(context.face_records)):
        out.setdefault(i, -1)
    return out


def _pairwise_agreement(
    labels_a: Dict[int, int], labels_b: Dict[int, int]
) -> float:
    """Fraction of (i,j) pairs where both algorithms agree on
    same-cluster-or-not. Noise (-1) is treated as 'in its own singleton'.

    1.0 = full agreement; 0.0 = total disagreement.
    """
    face_ids = sorted(set(labels_a) | set(labels_b))
    if len(face_ids) < 2:
        return 1.0
    total = 0
    agree = 0
    for i in range(len(face_ids)):
        for j in range(i + 1, len(face_ids)):
            fi, fj = face_ids[i], face_ids[j]
            li_a, lj_a = labels_a.get(fi, -1), labels_a.get(fj, -1)
            li_b, lj_b = labels_b.get(fi, -1), labels_b.get(fj, -1)
            # Same-cluster predicate: same non-noise label.
            same_a = (li_a == lj_a) and (li_a != -1)
            same_b = (li_b == lj_b) and (li_b != -1)
            total += 1
            if same_a == same_b:
                agree += 1
    return agree / total if total else 1.0


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_input():
    return _make_synthetic_faces(n_per_identity=5, n_identities=3)


def test_legacy_and_v2_agree_on_synthetic_fixture(synthetic_input):
    """spec-040 Phase 6 acceptance gate.

    Same synthetic input through legacy bridge AND the unified chain;
    pairwise agreement must be ≥ 95%.
    """
    legacy_faces, embeddings_norm, records = synthetic_input
    config = {
        "K": 3,
        "distance_threshold": 0.6,
        "min_cluster_size": 2,
        "blur_min": 0.0,
        "yaw_max": 999.0,
        "pitch_max": 999.0,
        "roll_max": 999.0,
        "max_faces_per_image_core": 50,
    }

    # --- Legacy path -----------------------------------------------------
    legacy_ctx = PipelineContext()
    legacy_labels_arr, _records_l, _bcr, _mcr, _ci, _ml, _mm, _fc_cfg = run_face_cluster_knn(
        legacy_faces, embeddings_norm, config, legacy_ctx
    )
    legacy_labels = _labels_from_legacy(legacy_labels_arr)

    # --- V2 path ---------------------------------------------------------
    v2_ctx = PipelineContext()
    v2_ctx.face_records = list(records)
    result = FCAppRunner().run(v2_ctx, step_configs={
        name: config for name in [
            "quality_gate_faces", "build_face_knn_graph",
            "cluster_face_components", "select_face_exemplars",
            "merge_face_clusters", "attach_holdout_faces",
            "apply_diameter_cap", "assign_people_clusters",
        ]
    })
    assert result.success, f"v2 runner failed: {result.error_message}"
    v2_labels = _labels_from_v2(v2_ctx)

    # --- Compare ---------------------------------------------------------
    agreement = _pairwise_agreement(legacy_labels, v2_labels)
    assert agreement >= 0.95, (
        f"legacy vs v2 pairwise agreement = {agreement:.3f}, below the 95% bar.\n"
        f"  legacy labels: {legacy_labels}\n"
        f"  v2 labels:     {v2_labels}"
    )


def test_legacy_and_v2_produce_same_cluster_count(synthetic_input):
    """Sanity check: both paths should converge on similar cluster counts.

    Allows ±1 difference (a singleton might be merged or kept).
    """
    legacy_faces, embeddings_norm, records = synthetic_input
    config = {
        "K": 3, "distance_threshold": 0.6, "min_cluster_size": 2,
        "blur_min": 0.0, "yaw_max": 999.0, "pitch_max": 999.0,
        "roll_max": 999.0, "max_faces_per_image_core": 50,
    }

    legacy_ctx = PipelineContext()
    legacy_labels_arr, _, _, _, _, _, _, _ = run_face_cluster_knn(
        legacy_faces, embeddings_norm, config, legacy_ctx
    )
    legacy_n_clusters = len(set(legacy_labels_arr) - {-1})

    v2_ctx = PipelineContext()
    v2_ctx.face_records = list(records)
    result = FCAppRunner().run(v2_ctx, step_configs={
        name: config for name in [
            "quality_gate_faces", "build_face_knn_graph",
            "cluster_face_components", "select_face_exemplars",
            "merge_face_clusters", "attach_holdout_faces",
            "apply_diameter_cap", "assign_people_clusters",
        ]
    })

    diff = abs(legacy_n_clusters - result.n_clusters)
    assert diff <= 1, (
        f"cluster count diverges by {diff}: legacy={legacy_n_clusters}, v2={result.n_clusters}"
    )
