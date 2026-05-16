"""Tests for spec-031: absolute diameter cap step.

Algorithm under test: face_cluster/cluster_diameter_cap.py:apply_diameter_cap
Pipeline integration: face_cluster/pipeline.py:_diameter_cap
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pytest

from face_cluster.cluster_diameter_cap import (
    ClusterCapDecision,
    CapResult,
    apply_diameter_cap,
    decisions_to_dict_list,
    _max_pairwise_cosine,
)
from face_cluster.types import ClusterResult, FaceRecord


# -----------------------------------------------------------------------------
# Test fixture helpers
# -----------------------------------------------------------------------------
def _make_face(face_id: int, embedding: np.ndarray) -> FaceRecord:
    """Build a minimal FaceRecord — only embedding_normalized is used by the cap."""
    e = embedding.astype(np.float32)
    e_norm = e / (np.linalg.norm(e) + 1e-12)
    return FaceRecord(
        face_id=face_id,
        image_id=f"img_{face_id}",
        bbox=(0.0, 0.0, 1.0, 1.0),
        landmarks=None,
        aligned_face=None,
        embedding=e,
        embedding_normalized=e_norm,
        pose=None,
        blur_score=0.0,
        area=0.0,
        is_core=True,
        image_path=f"img_{face_id}.jpg",
        face_index=0,
    )


def _make_cluster_result(labels: list[int], exemplars: dict | None = None) -> ClusterResult:
    labels_arr = np.asarray(labels, dtype=np.int32)
    clusters: dict[int, list[int]] = {}
    for i, c in enumerate(labels):
        clusters.setdefault(c, []).append(i)
    n_clusters = sum(1 for cid in clusters if cid >= 0)
    n_noise = len(clusters.get(-1, []))
    return ClusterResult(
        labels=labels_arr,
        clusters=clusters,
        cluster_stats={cid: {"size": len(v)} for cid, v in clusters.items() if cid >= 0},
        exemplars=exemplars or {cid: v[:3] for cid, v in clusters.items() if cid >= 0},
        n_clusters=n_clusters,
        n_noise=n_noise,
    )


def _orthogonal_basis(n: int, dim: int = 64) -> np.ndarray:
    """Return n random orthogonal-ish unit vectors (pairwise cosine dist ~ 1)."""
    rng = np.random.default_rng(0)
    M = rng.standard_normal((n, dim))
    M -= M.mean(axis=0, keepdims=True)
    # Orthogonalize via QR (gives n vectors with pairwise dot product ~ 0)
    Q, _ = np.linalg.qr(M.T if dim >= n else M)
    # If dim < n the QR yields fewer than n orthonormal columns; fall back
    return Q.T[:n] if Q.shape[1] >= n else (M / np.linalg.norm(M, axis=1, keepdims=True))


# -----------------------------------------------------------------------------
# Unit: _max_pairwise_cosine
# -----------------------------------------------------------------------------
class ut_MaxPairwiseCosine:
    def test_singleton_returns_zero(self):
        embs = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        assert _max_pairwise_cosine(embs) == 0.0

    def test_empty_returns_zero(self):
        embs = np.zeros((0, 3), dtype=np.float32)
        assert _max_pairwise_cosine(embs) == 0.0

    def test_identical_pair_returns_zero(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        embs = np.stack([v, v])
        assert _max_pairwise_cosine(embs) == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_pair_returns_one(self):
        embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        assert _max_pairwise_cosine(embs) == pytest.approx(1.0, abs=1e-6)

    def test_opposite_pair_returns_two(self):
        embs = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
        assert _max_pairwise_cosine(embs) == pytest.approx(2.0, abs=1e-6)

    def test_max_picks_worst_pair(self):
        # Three points: two identical, one far. Max pairwise = dist to the far.
        v = np.array([1.0, 0.0], dtype=np.float32)
        embs = np.stack([v, v, np.array([0.0, 1.0], dtype=np.float32)])
        assert _max_pairwise_cosine(embs) == pytest.approx(1.0, abs=1e-6)


# -----------------------------------------------------------------------------
# Unit: apply_diameter_cap — kept paths
# -----------------------------------------------------------------------------
class ut_DiameterCapKeptPaths:
    def test_tight_cluster_is_kept(self):
        # All 3 faces share the same embedding — diameter 0.
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        faces = [_make_face(i, v) for i in range(3)]
        merged = _make_cluster_result([0, 0, 0])
        base   = _make_cluster_result([0, 0, 0])

        result = apply_diameter_cap(
            merged_result=merged, base_result=base, faces=faces,
            max_full_diameter=1.2, max_exemplar_diameter=0.8,
        )

        assert result.n_kept == 1
        assert result.n_split == 0
        assert result.cluster_result.n_clusters == 1
        assert result.decisions[0].action == "kept"
        assert result.decisions[0].full_diameter == pytest.approx(0.0, abs=1e-6)

    def test_noise_cluster_skipped(self):
        # All faces are noise (cluster_id=-1) — no decisions, no action.
        faces = [_make_face(i, np.eye(3)[i % 3]) for i in range(3)]
        merged = _make_cluster_result([-1, -1, -1])
        base   = _make_cluster_result([-1, -1, -1])

        result = apply_diameter_cap(
            merged_result=merged, base_result=base, faces=faces,
            max_full_diameter=1.2, max_exemplar_diameter=0.8,
        )

        assert result.decisions == []
        assert result.cluster_result.n_noise == 3

    def test_singleton_cluster_skipped(self):
        # n<2 means no pairs to measure. Cluster is kept implicitly.
        faces = [_make_face(0, np.array([1.0, 0.0]))]
        merged = _make_cluster_result([5])
        base   = _make_cluster_result([5])

        result = apply_diameter_cap(
            merged_result=merged, base_result=base, faces=faces,
            max_full_diameter=0.01, max_exemplar_diameter=0.01,
        )

        assert result.decisions == []
        assert result.cluster_result.clusters == {5: [0]}


# -----------------------------------------------------------------------------
# Unit: apply_diameter_cap — split paths (the point of the spec)
# -----------------------------------------------------------------------------
class ut_DiameterCapSplit:
    def test_full_diameter_violation_reverts_to_base(self):
        # 4 orthogonal faces, base says 2 clusters (0,1 vs 2,3), merge combined
        # them into one cluster with full diameter ~1.0. Cap at 0.5 should split.
        e0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        e1 = np.array([0.99, 0.1, 0.0, 0.0], dtype=np.float32)
        e2 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        e3 = np.array([0.0, 0.0, 0.99, 0.1], dtype=np.float32)
        faces = [_make_face(0, e0), _make_face(1, e1),
                 _make_face(2, e2), _make_face(3, e3)]
        base   = _make_cluster_result([10, 10, 20, 20])
        merged = _make_cluster_result([99, 99, 99, 99])  # all in cluster 99

        result = apply_diameter_cap(
            merged_result=merged, base_result=base, faces=faces,
            max_full_diameter=0.5, max_exemplar_diameter=0.5,
        )

        decision = result.decisions[0]
        assert decision.action == "split"
        assert not decision.full_pass
        assert sorted(decision.pre_merge_components) == [10, 20]
        # Reverted labels should match base.
        assert list(result.cluster_result.labels) == [10, 10, 20, 20]
        assert result.cluster_result.n_clusters == 2

    def test_exemplar_violation_with_outlier_tolerant_full(self):
        # 5 faces: 4 tight near e0, 1 far at e1. Base says 4 + 1 separate.
        # If the merge stuffed all 5 in, and exemplars happen to include the
        # outlier, exemplar diameter is high but full diameter might also be high.
        # We construct so exemplar gate fails first.
        e_tight = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        e_far   = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        faces = [
            _make_face(0, e_tight),
            _make_face(1, e_tight),
            _make_face(2, e_tight),
            _make_face(3, e_tight),
            _make_face(4, e_far),
        ]
        base   = _make_cluster_result([10, 10, 10, 10, 20])
        # Merge stuck all into cluster 99. Exemplars include both tight + far.
        merged = ClusterResult(
            labels=np.array([99, 99, 99, 99, 99], dtype=np.int32),
            clusters={99: [0, 1, 2, 3, 4]},
            cluster_stats={99: {"size": 5}},
            exemplars={99: [0, 4]},   # tight + far -> exemplar diam ~ 1.0
            n_clusters=1,
            n_noise=0,
        )

        result = apply_diameter_cap(
            merged_result=merged, base_result=base, faces=faces,
            max_full_diameter=1.5,           # full diam ~1.0 passes
            max_exemplar_diameter=0.5,       # exemplar diam ~1.0 fails
        )

        decision = result.decisions[0]
        assert decision.action == "split"
        assert decision.full_pass         # passed full but
        assert not decision.exemplar_pass  # failed exemplar
        # Revert to base components.
        assert list(result.cluster_result.labels) == [10, 10, 10, 10, 20]

    def test_no_exemplars_means_exemplar_gate_vacuous(self):
        # If a cluster has 0 or 1 exemplars, exemplar gate must pass vacuously
        # and decision must rely on full diameter alone.
        e0 = np.array([1.0, 0.0], dtype=np.float32)
        e1 = np.array([0.0, 1.0], dtype=np.float32)
        faces = [_make_face(0, e0), _make_face(1, e1)]
        base   = _make_cluster_result([10, 20])
        merged = ClusterResult(
            labels=np.array([99, 99], dtype=np.int32),
            clusters={99: [0, 1]},
            cluster_stats={99: {"size": 2}},
            exemplars={99: []},   # zero exemplars
            n_clusters=1,
            n_noise=0,
        )

        result = apply_diameter_cap(
            merged_result=merged, base_result=base, faces=faces,
            max_full_diameter=0.5, max_exemplar_diameter=0.01,
        )
        # Despite exemplar threshold being tight, vacuous pass — only full
        # diameter is checked. Full diameter ~1.0 > 0.5, so it splits.
        decision = result.decisions[0]
        assert decision.action == "split"
        assert decision.exemplar_pass  # vacuous
        assert not decision.full_pass

    def test_split_to_noise_when_base_was_noise(self):
        # Face whose pre-merge label was -1 (noise) should revert to noise on split.
        e0 = np.array([1.0, 0.0], dtype=np.float32)
        e1 = np.array([0.0, 1.0], dtype=np.float32)
        faces = [_make_face(0, e0), _make_face(1, e1)]
        base   = _make_cluster_result([10, -1])
        merged = _make_cluster_result([99, 99])

        result = apply_diameter_cap(
            merged_result=merged, base_result=base, faces=faces,
            max_full_diameter=0.5, max_exemplar_diameter=0.5,
        )

        assert list(result.cluster_result.labels) == [10, -1]
        assert result.cluster_result.n_noise == 1


# -----------------------------------------------------------------------------
# Unit: serialization
# -----------------------------------------------------------------------------
class ut_DecisionSerialization:
    def test_round_trip_keys(self):
        d = ClusterCapDecision(
            cluster_id=4, n_faces=28, full_diameter=0.867,
            exemplar_diameter=0.65, max_full_threshold=1.2,
            max_exemplar_threshold=0.8, full_pass=True, exemplar_pass=True,
            action="kept", reason="ok",
        )
        out = decisions_to_dict_list([d])[0]
        expected_keys = {
            "cluster_id", "n_faces", "full_diameter", "exemplar_diameter",
            "max_full_threshold", "max_exemplar_threshold",
            "full_pass", "exemplar_pass", "action", "reason",
            "pre_merge_components", "pre_merge_sizes",
        }
        assert set(out.keys()) == expected_keys

    def test_split_records_components_and_sizes(self):
        d = ClusterCapDecision(
            cluster_id=4, n_faces=28, full_diameter=0.95,
            exemplar_diameter=0.9, max_full_threshold=1.2,
            max_exemplar_threshold=0.8, full_pass=True, exemplar_pass=False,
            action="split", reason="exemplar 0.9 > 0.8",
            pre_merge_components=[10, 20, 30],
            pre_merge_sizes=[15, 8, 5],
        )
        out = decisions_to_dict_list([d])[0]
        assert out["pre_merge_components"] == [10, 20, 30]
        assert out["pre_merge_sizes"] == [15, 8, 5]
        assert out["action"] == "split"


# -----------------------------------------------------------------------------
# Unit: invariants
# -----------------------------------------------------------------------------
class ut_DiameterCapInvariants:
    def test_one_decision_per_real_cluster(self):
        faces = [_make_face(i, np.eye(4)[i]) for i in range(4)]
        base   = _make_cluster_result([10, 10, 20, 20])
        merged = _make_cluster_result([10, 10, 20, 20])

        result = apply_diameter_cap(
            merged_result=merged, base_result=base, faces=faces,
            max_full_diameter=1.2, max_exemplar_diameter=0.8,
        )
        # Two real clusters (10, 20), expect 2 decisions.
        assert len(result.decisions) == 2

    def test_disabled_via_huge_thresholds_is_no_op(self):
        # With thresholds at 99, nothing should ever split.
        e0 = np.array([1.0, 0.0], dtype=np.float32)
        e1 = np.array([0.0, 1.0], dtype=np.float32)
        faces = [_make_face(0, e0), _make_face(1, e1)]
        base   = _make_cluster_result([10, 20])
        merged = _make_cluster_result([99, 99])

        result = apply_diameter_cap(
            merged_result=merged, base_result=base, faces=faces,
            max_full_diameter=99.0, max_exemplar_diameter=99.0,
        )

        assert result.n_split == 0
        assert list(result.cluster_result.labels) == [99, 99]

    def test_inputs_are_not_mutated(self):
        # The cap step must NOT mutate the inputs in place — returns a new
        # ClusterResult.  Caller relies on this for snapshot/export semantics.
        e0 = np.array([1.0, 0.0], dtype=np.float32)
        e1 = np.array([0.0, 1.0], dtype=np.float32)
        faces = [_make_face(0, e0), _make_face(1, e1)]
        base   = _make_cluster_result([10, 20])
        merged = _make_cluster_result([99, 99])
        before_labels = merged.labels.copy()
        before_clusters = {k: list(v) for k, v in merged.clusters.items()}

        apply_diameter_cap(
            merged_result=merged, base_result=base, faces=faces,
            max_full_diameter=0.5, max_exemplar_diameter=0.5,
        )

        assert np.array_equal(merged.labels, before_labels)
        assert {k: list(v) for k, v in merged.clusters.items()} == before_clusters


# -----------------------------------------------------------------------------
# Regression: SIGHTING-059 cluster-4 chain merge synthetic reproduction
# -----------------------------------------------------------------------------
class ut_Sighting059Regression:
    def test_chain_merge_cluster_splits_with_default_thresholds(self):
        # Synthesize a 28-face cluster with chained sub-identities.
        # Three sub-groups, each tight internally; the merger stitched them.
        # Max pairwise inside the merged cluster ~ 0.867 (matches user's report).
        rng = np.random.default_rng(42)
        dim = 64

        def cluster_around(center: np.ndarray, n: int, jitter: float = 0.03) -> List[np.ndarray]:
            return [center + rng.standard_normal(dim) * jitter for _ in range(n)]

        # 3 identity centers, far apart
        c_a = np.zeros(dim); c_a[0] = 1.0
        c_b = np.zeros(dim); c_b[1] = 1.0
        c_c = np.zeros(dim); c_c[2] = 1.0

        embs_a = cluster_around(c_a, 12)
        embs_b = cluster_around(c_b, 10)
        embs_c = cluster_around(c_c, 6)
        all_embs = embs_a + embs_b + embs_c

        faces = [_make_face(i, e) for i, e in enumerate(all_embs)]
        base_labels = [10] * 12 + [20] * 10 + [30] * 6
        merged_labels = [99] * 28      # the chain-merge "blob"
        base   = _make_cluster_result(base_labels)
        # Exemplars deliberately span the sub-identities to expose the chain.
        exemplars_merged = {99: [0, 12, 22]}  # one from each sub-group
        merged = ClusterResult(
            labels=np.array(merged_labels, dtype=np.int32),
            clusters={99: list(range(28))},
            cluster_stats={99: {"size": 28}},
            exemplars=exemplars_merged,
            n_clusters=1,
            n_noise=0,
        )

        result = apply_diameter_cap(
            merged_result=merged, base_result=base, faces=faces,
            max_full_diameter=1.2,     # spec default
            max_exemplar_diameter=0.8, # spec default
        )

        decision = result.decisions[0]
        # Either full or exemplar diam should be > 0.8 / 1.2 because the
        # sub-identities are orthogonal in embedding space.
        assert decision.action == "split"
        # After split, must recover the 3 base identities.
        labels = list(result.cluster_result.labels)
        assert sorted(set(labels)) == [10, 20, 30]
        assert result.cluster_result.n_clusters == 3
        # Sizes preserved.
        sizes = {cid: labels.count(cid) for cid in (10, 20, 30)}
        assert sizes == {10: 12, 20: 10, 30: 6}
