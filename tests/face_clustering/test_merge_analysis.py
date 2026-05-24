"""Tests for merge analysis: threshold components, merge_metadata, MergeAnalysisView.

Coverage:
  - T_a/T_b/T_global present in merge_log entries (adaptive mode)
  - merge_metadata.json exported by export_merged_results
  - merge_metadata round-trip: export -> load_pipeline_result
  - backward compatibility: old runs without merge_metadata.json load cleanly
  - MergeAnalysisView gate_rejection_counts computed correctly
  - MergeAnalysisView near_misses contains only 3/4-gate-pass rejections
  - _parse_merge_log handles old-format logs missing new fields
"""
import json
from pathlib import Path

import numpy as np
import pytest

from face_cluster.config import PipelineConfig
from face_cluster.export import export_merged_results
from face_cluster.merge import ConservativeMerger
from face_cluster.types import ClusterResult, FaceRecord, GraphResult
from face_cluster.analysis_views import MergeAnalysisView, MergeDecisionRow
# spec-042 T001: private helpers moved from analysis_views to views.merge_view
# in an earlier refactor; this test file was left out of the update.
from face_cluster.views.merge_view import _parse_merge_log
from face_cluster.pipeline import PipelineResult


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_faces(n: int) -> list:
    faces = []
    for i in range(n):
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        faces.append(FaceRecord(
            face_id=i, image_id=f"img_{i}", bbox=(0, 0, 10, 10),
            embedding_normalized=emb, blur_score=100.0, area=500.0,
            is_core=True, image_path=f"/tmp/img_{i}.jpg",
        ))
    return faces


def _make_cluster_result(n_faces: int, n_clusters: int) -> ClusterResult:
    size = n_faces // n_clusters
    clusters, exemplars = {}, {}
    labels = np.full(n_faces, -1, dtype=np.int32)
    for cid in range(n_clusters):
        members = list(range(cid * size, (cid + 1) * size))
        clusters[cid] = members
        exemplars[cid] = members[:1]
        for fi in members:
            labels[fi] = cid
    return ClusterResult(
        labels=labels, clusters=clusters,
        cluster_stats={cid: {"diameter": 0.1} for cid in clusters},
        exemplars=exemplars, n_clusters=n_clusters, n_noise=0,
    )


def _make_graph_result(n: int) -> GraphResult:
    """All-zero distance matrix so clusters won't merge (too-close check won't trigger)."""
    import networkx as nx
    mat = np.zeros((n, n), dtype=np.float32)
    return GraphResult(
        neighbors=[[] for _ in range(n)],
        neighbor_distances=[[] for _ in range(n)],
        edges=[],
        G=nx.Graph(),
        distance_matrix=mat,
    )


def _make_pipeline_result(faces, cr, merge_log=None, merge_metadata=None) -> PipelineResult:
    return PipelineResult(
        faces=faces,
        cluster_result=cr,
        merged_cluster_result=cr,
        output_dir=Path("/tmp/fake"),
        summary={"n_core": len(faces), "n_noise": 0, "n_clusters": cr.n_clusters},
        merge_log=merge_log or [],
        merge_metadata=merge_metadata,
    )


# ---------------------------------------------------------------------------
# 1. T_a/T_b/T_global in merge_log (adaptive mode)
# ---------------------------------------------------------------------------

class ut_MergeLogThresholdComponents:
    def test_merge_log_contains_threshold_fields(self):
        """merge_log entries must always contain T_a, T_b, T_global keys.

        Adaptive threshold mode was removed. T_a/T_b/T_global are always present
        in the log dict but always None (reserved for future use).
        """
        n = 6
        faces = _make_faces(n)
        # Two tight clusters close together
        emb_a = np.random.randn(512).astype(np.float32)
        emb_a /= np.linalg.norm(emb_a)
        for i in range(3):
            faces[i].embedding_normalized = emb_a + np.random.randn(512).astype(np.float32) * 0.01
            faces[i].embedding_normalized /= np.linalg.norm(faces[i].embedding_normalized)
        emb_b = -emb_a  # opposite direction = far apart
        for i in range(3, 6):
            faces[i].embedding_normalized = emb_b + np.random.randn(512).astype(np.float32) * 0.01
            faces[i].embedding_normalized /= np.linalg.norm(faces[i].embedding_normalized)

        cr = _make_cluster_result(6, 2)
        graph = _make_graph_result(6)

        cfg = PipelineConfig(merge_enabled=True, merge_candidate_threshold=2.0)
        merger = ConservativeMerger(cfg)
        _, log, _ = merger.merge_clusters_with_logging(cr, graph)

        assert len(log) > 0, "Expected at least one merge candidate evaluated"
        for entry in log:
            assert "T_a" in entry
            assert "T_b" in entry
            assert "T_global" in entry

    def test_threshold_components_are_none(self):
        """T_a/T_b/T_global are always None — adaptive threshold mode was removed."""
        cr = _make_cluster_result(4, 2)
        graph = _make_graph_result(4)
        cfg = PipelineConfig(merge_enabled=True, merge_candidate_threshold=2.0)
        merger = ConservativeMerger(cfg)
        _, log, _ = merger.merge_clusters_with_logging(cr, graph)
        for entry in log:
            assert entry.get("T_a") is None
            assert entry.get("T_b") is None
            assert entry.get("T_global") is None


# ---------------------------------------------------------------------------
# 2. merge_metadata.json exported
# ---------------------------------------------------------------------------

class ut_MergeMetadataExport:
    def test_merge_metadata_written_when_provided(self, tmp_path):
        faces = _make_faces(4)
        merged_cr = _make_cluster_result(4, 2)
        metadata = {"cluster_thresholds": {0: 0.25, 1: 0.30}, "global_threshold": 0.28,
                    "n_candidates_proposed": 1, "n_iterations": 1}
        export_merged_results(faces, merged_cr, [], tmp_path,
                              core_indices=None, merge_metadata=metadata)
        assert (tmp_path / "merge_metadata.json").exists()
        loaded = json.loads((tmp_path / "merge_metadata.json").read_text())
        assert loaded["global_threshold"] == pytest.approx(0.28)
        assert loaded["cluster_thresholds"]["0"] == pytest.approx(0.25)

    def test_merge_metadata_not_written_when_none(self, tmp_path):
        faces = _make_faces(4)
        merged_cr = _make_cluster_result(4, 2)
        export_merged_results(faces, merged_cr, [], tmp_path,
                              core_indices=None, merge_metadata=None)
        assert not (tmp_path / "merge_metadata.json").exists()


# ---------------------------------------------------------------------------
# 3. merge_metadata round-trip via loader
# ---------------------------------------------------------------------------

class ut_MergeMetadataRoundTrip:
    def _write_minimal_run(self, tmp_path: Path) -> None:
        import pandas as pd
        faces = _make_faces(4)
        merged_cr = _make_cluster_result(4, 2)
        # Write base artifacts
        rows = [{"face_id": f.face_id, "image_path": f.image_path,
                 "image_id": f.image_id, "crop_path": "",
                 "cluster_id": i // 2, "is_core": True,
                 "blur_score": 100.0, "area": 500.0,
                 "yaw": None, "pitch": None, "roll": None}
                for i, f in enumerate(faces)]
        pd.DataFrame(rows).to_csv(tmp_path / "faces.csv", index=False)
        pd.DataFrame([{"cluster_id": 0, "size": 2, "exemplar_face_ids": "[0]", "diameter": 0.1},
                      {"cluster_id": 1, "size": 2, "exemplar_face_ids": "[2]", "diameter": 0.1}]
                     ).to_csv(tmp_path / "clusters.csv", index=False)
        run_json = {"summary": {"n_core": 4, "n_noise": 0}, "stages": {}, "status": "done"}
        (tmp_path / "pipeline_run.json").write_text(json.dumps(run_json))
        # Write merged artifacts
        export_merged_results(
            faces, merged_cr, [],
            output_dir=tmp_path, core_indices=None,
            merge_metadata={"cluster_thresholds": {0: 0.20, 1: 0.22},
                            "global_threshold": 0.21,
                            "n_candidates_proposed": 1, "n_iterations": 1},
        )

    def test_round_trip_preserves_merge_metadata(self, tmp_path):
        from face_cluster.loader import load_pipeline_result
        self._write_minimal_run(tmp_path)
        result = load_pipeline_result(tmp_path)
        assert result.merge_metadata is not None
        assert "cluster_thresholds" in result.merge_metadata
        assert result.merge_metadata["global_threshold"] == pytest.approx(0.21)

    def test_backward_compat_no_merge_metadata_file(self, tmp_path):
        """Runs without merge_metadata.json must load with merge_metadata=None."""
        from face_cluster.loader import load_pipeline_result
        import pandas as pd
        faces = _make_faces(2)
        rows = [{"face_id": f.face_id, "image_path": f.image_path,
                 "image_id": f.image_id, "crop_path": "",
                 "cluster_id": i, "is_core": True,
                 "blur_score": 100.0, "area": 500.0,
                 "yaw": None, "pitch": None, "roll": None}
                for i, f in enumerate(faces)]
        pd.DataFrame(rows).to_csv(tmp_path / "faces.csv", index=False)
        pd.DataFrame([{"cluster_id": 0, "size": 1, "exemplar_face_ids": "[0]", "diameter": 0.0},
                      {"cluster_id": 1, "size": 1, "exemplar_face_ids": "[1]", "diameter": 0.0}]
                     ).to_csv(tmp_path / "clusters.csv", index=False)
        (tmp_path / "pipeline_run.json").write_text(
            json.dumps({"summary": {}, "stages": {}, "status": "done"})
        )
        result = load_pipeline_result(tmp_path)
        assert result.merge_metadata is None


# ---------------------------------------------------------------------------
# 4. MergeAnalysisView gate counts
# ---------------------------------------------------------------------------

class ut_MergeAnalysisViewGateCounts:
    def _make_rejection(self, pe, ps, pm, pd) -> MergeDecisionRow:
        n = sum([pe, ps, pm, pd])
        return MergeDecisionRow(
            cluster_a=0, cluster_b=1,
            exemplar_dist=0.4, threshold_used=0.3, support=0, action="rejected",
            rejection_reason="test", exemplar_face_ids_a=[], exemplar_face_ids_b=[],
            passes_exemplar=pe, passes_support=ps, passes_margin=pm, passes_diameter=pd,
            n_gates_passed=n,
        )

    def test_gate_rejection_counts(self):
        rejections = [
            self._make_rejection(False, True, True, True),   # exemplar fails
            self._make_rejection(False, True, True, True),   # exemplar fails
            self._make_rejection(True, False, True, True),   # support fails
        ]
        faces = _make_faces(2)
        cr = _make_cluster_result(2, 1)
        result = _make_pipeline_result(faces, cr,
            merge_log=[],  # already parsed
        )
        result.merge_log = None  # bypass _parse_merge_log
        # Construct view directly
        from face_cluster.views.merge_view import _compute_gate_stats
        counts, sole = _compute_gate_stats(rejections)
        assert counts["exemplar"] == 2
        assert counts["support"] == 1
        assert counts["margin"] == 0
        assert counts["diameter"] == 0
        # sole blockers: all three rejections are sole-blocked
        assert sole["exemplar"] == 2
        assert sole["support"] == 1

    def test_sole_blocker_multi_fail_not_counted(self):
        """Sole-blocker should not count when multiple gates fail."""
        from face_cluster.views.merge_view import _compute_gate_stats
        rejections = [
            self._make_rejection(False, False, True, True),  # two gates fail
        ]
        _, sole = _compute_gate_stats(rejections)
        assert sole["exemplar"] == 0
        assert sole["support"] == 0


# ---------------------------------------------------------------------------
# 5. MergeAnalysisView near-misses
# ---------------------------------------------------------------------------

class ut_MergeAnalysisViewNearMisses:
    def _rejection_row(self, n_passed: int) -> dict:
        flags = [True] * n_passed + [False] * (4 - n_passed)
        return {
            "iteration": 1, "cluster_a": 0, "cluster_b": 1,
            "cluster_a_size": 2, "cluster_b_size": 2,
            "exemplar_dist": 0.5, "threshold_used": 0.4,
            "T_a": None, "T_b": None, "T_global": None,
            "support": 1, "required_support": 2,
            "post_diameter": 0.6, "max_allowed_diameter": 0.5,
            "action": "rejected",
            "passes_exemplar": flags[0], "passes_support": flags[1],
            "passes_margin": flags[2], "passes_diameter": flags[3],
            "rejection_reason": "test", "actually_merged": False,
        }

    def test_near_misses_only_three_of_four(self):
        cr = _make_cluster_result(4, 2)
        log = [
            self._rejection_row(2),  # 2/4 — not a near miss
            self._rejection_row(3),  # 3/4 — near miss
            self._rejection_row(3),  # 3/4 — near miss
            self._rejection_row(1),  # 1/4 — not
        ]
        # Deduplicate by key (min/max cluster pair) — use different cluster IDs
        log[0]["cluster_b"] = 2
        log[1]["cluster_b"] = 3
        log[2]["cluster_b"] = 4
        log[3]["cluster_b"] = 5
        faces = _make_faces(2)
        result = _make_pipeline_result(faces, cr, merge_log=log)
        result.merged_cluster_result = cr

        # patch cluster ids to exist in cr to avoid key error in _exemplar_face_ids
        for entry in log:
            entry["cluster_a"] = list(cr.clusters.keys())[0]
            entry["cluster_b"] = list(cr.clusters.keys())[-1]
        view = MergeAnalysisView.compute(result)
        # near_misses must only contain 3/4 rows
        assert all(r.n_gates_passed == 3 for r in view.near_misses)


# ---------------------------------------------------------------------------
# 6. _parse_merge_log backward compat (old-format logs missing new fields)
# ---------------------------------------------------------------------------

class ut_ParseMergeLogBackwardCompat:
    def test_old_log_missing_gate_fields(self):
        """Old logs without passes_* fields must load with safe defaults."""
        cr = _make_cluster_result(4, 2)
        faces = _make_faces(4)
        old_format_log = [
            {
                "cluster_a": 0, "cluster_b": 1,
                "exemplar_dist": 0.35, "threshold_used": 0.30,
                "support": 2, "action": "rejected",
                "rejection_reason": "exemplar_dist too high",
                "actually_merged": False,
                # No passes_* fields, no T_a/T_b/T_global, no sizes
            }
        ]
        merges, rejections = _parse_merge_log(old_format_log, cr, faces)
        assert len(rejections) == 1
        r = rejections[0]
        assert r.passes_exemplar is False
        assert r.passes_support is False
        assert r.T_a is None
        assert r.cluster_a_size == 0
        assert r.n_gates_passed == 0


# ---------------------------------------------------------------------------
# 7. group_merge_candidates() — transitive grouping + cohesion classification
# ---------------------------------------------------------------------------

from face_cluster.merge import group_merge_candidates, CandidateGroup


class ut_GroupMergeCandidates:
    """Unit tests for the group_merge_candidates() algorithm in merge.py."""

    def test_transitive_grouping_forms_components(self):
        """Pairs (0,1), (1,2), (3,4) -> 2 groups: {0,1,2} and {3,4}."""
        pairs = [(0, 1), (1, 2), (3, 4)]
        gates = [4, 4, 4]
        groups = group_merge_candidates(pairs, gates)
        assert len(groups) == 2
        cluster_sets = [frozenset(g.cluster_ids) for g in groups]
        assert frozenset({0, 1, 2}) in cluster_sets
        assert frozenset({3, 4}) in cluster_sets

    def test_single_pair_group(self):
        """A single pair with no shared cluster IDs -> 1 group with 1 pair."""
        groups = group_merge_candidates([(5, 7)], [3])
        assert len(groups) == 1
        assert groups[0].cluster_ids == [5, 7]
        assert len(groups[0].pair_keys) == 1

    def test_empty_input_returns_empty(self):
        groups = group_merge_candidates([], [])
        assert groups == []

    def test_all_pairs_accounted_for(self):
        """Sum of pairs across all groups must equal the number of input pairs."""
        pairs = [(0, 1), (1, 2), (3, 4), (3, 5), (6, 7)]
        gates = [4, 3, 2, 1, 4]
        groups = group_merge_candidates(pairs, gates)
        total = sum(len(g.pair_keys) for g in groups)
        assert total == len(pairs)

    def test_confidence_auto_approve_all_4_gates(self):
        """Group where every pair passes 4/4 gates -> auto_approve."""
        pairs = [(0, 1), (1, 2), (0, 2)]
        gates = [4, 4, 4]
        groups = group_merge_candidates(pairs, gates)
        assert len(groups) == 1
        assert groups[0].confidence == "auto_approve"
        assert groups[0].cohesion == 1.0

    def test_confidence_auto_reject_all_low_gates(self):
        """Group where every pair passes <=2/4 gates -> auto_reject."""
        pairs = [(0, 1), (1, 2)]
        gates = [1, 2]
        groups = group_merge_candidates(pairs, gates)
        assert len(groups) == 1
        assert groups[0].confidence == "auto_reject"

    def test_confidence_review_mixed_gates(self):
        """Mix of 4/4 and 3/4 pairs below cohesion threshold -> review."""
        # 1 pair at 4/4, 2 pairs at 3/4 -> cohesion = 1/3 = 33% < 80%
        pairs = [(0, 1), (1, 2), (0, 2)]
        gates = [4, 3, 3]
        groups = group_merge_candidates(pairs, gates)
        assert len(groups) == 1
        assert groups[0].confidence == "review"

    def test_cohesion_promotion_to_auto_approve(self):
        """5/6 pairs at 4/4, 1 pair at 3/4 -> cohesion=83% >= 80% AND min_gates=3 >= 3 -> auto_approve."""
        # Need a 4-cluster component with 6 pairs
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        gates = [4, 4, 4, 4, 4, 3]  # 5/6 = 83.3% cohesion, min_gates = 3
        groups = group_merge_candidates(pairs, gates)
        assert len(groups) == 1
        assert groups[0].cohesion == pytest.approx(5 / 6)
        assert groups[0].confidence == "auto_approve"

    def test_no_cohesion_promotion_when_min_gates_below_threshold(self):
        """High cohesion but min_gates=2 -> no promotion, stays review."""
        # 4/5 pairs at 4/4 -> cohesion=80%, but one pair at 2/4 -> min_gates=2
        pairs = [(0, 1), (0, 2), (1, 2), (2, 3), (0, 3)]
        gates = [4, 4, 4, 4, 2]  # cohesion = 4/5 = 80%, min_gates = 2
        groups = group_merge_candidates(pairs, gates)
        assert len(groups) == 1
        # min_gates=2 < min_gates_for_promotion=3 -> NOT promoted
        assert groups[0].confidence == "review"

    def test_sort_order_review_first(self):
        """Groups should be sorted: review first, then auto_approve, then auto_reject."""
        # Two separate components
        pairs = [(0, 1), (2, 3), (4, 5)]
        gates = [3, 4, 1]  # review, auto_approve, auto_reject
        groups = group_merge_candidates(pairs, gates)
        assert len(groups) == 3
        assert groups[0].confidence == "review"
        assert groups[1].confidence == "auto_approve"
        assert groups[2].confidence == "auto_reject"

    def test_min_max_gates_computed_correctly(self):
        """min_gates and max_gates reflect the actual range across pairs in the group."""
        pairs = [(0, 1), (1, 2), (0, 2)]
        gates = [2, 3, 4]
        groups = group_merge_candidates(pairs, gates)
        assert len(groups) == 1
        assert groups[0].min_gates == 2
        assert groups[0].max_gates == 4

    def test_pair_keys_are_normalized(self):
        """pair_keys must always be (min_id, max_id) regardless of input order."""
        pairs = [(5, 2)]  # reversed order
        gates = [4]
        groups = group_merge_candidates(pairs, gates)
        assert groups[0].pair_keys == [(2, 5)]


# ---------------------------------------------------------------------------
# 8. compute_ml_merge_view — ML-mode analysis
# ---------------------------------------------------------------------------

def _make_merge_payload(predict_proba_value: float = 0.9, n_features: int = 3):
    """Build a minimal valid model_payload for testing compute_ml_merge_view.

    Uses LogisticRegression fitted on synthetic data so predictions are stable.
    predict_proba_value determines the approximate merge probability.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from face_cluster.features import VERSION as FEAT_VERSION

    # Feature subset — must exist in FeatureComputer.to_dataframe() output
    feature_names = ["min_exemplar_dist", "support_fraction", "size_a"]

    # Perfectly separable training data -> high-confidence predictions
    # Class 1 (merge): very low distance, high support
    # Class 0 (reject): high distance, low support
    rng = np.random.default_rng(42)
    n = 40
    X_merge = np.column_stack([
        rng.uniform(0.0, 0.1, n),   # min_exemplar_dist low
        rng.uniform(0.7, 1.0, n),   # support_fraction high
        rng.integers(2, 5, n),      # size_a
    ])
    X_reject = np.column_stack([
        rng.uniform(0.6, 1.0, n),   # min_exemplar_dist high
        rng.uniform(0.0, 0.2, n),   # support_fraction low
        rng.integers(2, 5, n),
    ])
    X = np.vstack([X_merge, X_reject]).astype(float)
    y = np.array([1] * n + [0] * n)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000, C=10.0, random_state=0)
    model.fit(X_s, y)

    return {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "metadata": {
            "model_type": "logistic_regression",
            "feature_version": FEAT_VERSION,
        },
    }


def _make_close_cluster_result(faces):
    """Two clusters where embeddings are very similar (close distance -> high merge prob)."""
    n = len(faces)
    half = n // 2
    # All faces point in the same direction -> near-zero cosine distance
    base = np.ones(512, dtype=np.float32)
    base /= np.linalg.norm(base)
    for i, face in enumerate(faces):
        noise = np.random.RandomState(i).randn(512).astype(np.float32) * 0.01
        emb = base + noise
        face.embedding_normalized = (emb / np.linalg.norm(emb)).astype(np.float32)

    clusters = {0: list(range(half)), 1: list(range(half, n))}
    exemplars = {0: [0], 1: [half]}
    labels = np.array([0] * half + [1] * (n - half), dtype=np.int32)
    return ClusterResult(
        labels=labels, clusters=clusters, cluster_stats={},
        exemplars=exemplars, n_clusters=2, n_noise=0,
    )


class ut_MLMergeView:
    """Tests for compute_ml_merge_view() in analysis_views.py."""

    def test_compute_returns_merge_analysis_view(self):
        """compute_ml_merge_view returns a MergeAnalysisView with ml_threshold set."""
        from face_cluster.analysis_views import compute_ml_merge_view

        faces = _make_faces(6)
        cr = _make_close_cluster_result(faces)
        result = _make_pipeline_result(faces, cr)
        payload = _make_merge_payload()

        view = compute_ml_merge_view(result, payload, threshold=0.5)

        assert isinstance(view, MergeAnalysisView)
        assert view.ml_threshold == pytest.approx(0.5)
        # All rows must have ml_prob and ml_pred populated
        all_rows = view.merges + view.rejections
        assert len(all_rows) > 0, "Expected at least one candidate pair"
        for row in all_rows:
            assert row.ml_prob is not None
            assert row.ml_pred is not None

    def test_high_prob_pairs_in_merges_list(self):
        """Pairs with prob >= threshold appear in view.merges."""
        from face_cluster.analysis_views import compute_ml_merge_view

        faces = _make_faces(6)
        cr = _make_close_cluster_result(faces)
        result = _make_pipeline_result(faces, cr)
        payload = _make_merge_payload()

        # Use threshold=0.01 so nearly all pairs end up in merges
        view = compute_ml_merge_view(result, payload, threshold=0.01)

        merge_probs = [r.ml_prob for r in view.merges]
        assert all(p >= 0.01 for p in merge_probs), (
            f"merge list contains prob below threshold: {merge_probs}"
        )
        assert all(r.ml_pred == 1 for r in view.merges)

    def test_low_prob_pairs_in_rejections_list(self):
        """Pairs with ml_pred==0 appear in view.rejections."""
        from face_cluster.analysis_views import compute_ml_merge_view

        faces = _make_faces(6)
        cr = _make_close_cluster_result(faces)
        result = _make_pipeline_result(faces, cr)
        payload = _make_merge_payload()

        # Use threshold=0.99 so nearly all pairs end up in rejections
        view = compute_ml_merge_view(result, payload, threshold=0.99)

        assert all(r.ml_pred == 0 for r in view.rejections)

    def test_borderline_pairs_in_near_misses(self):
        """Pairs with 0.4 < ml_prob < 0.6 appear in view.near_misses."""
        from face_cluster.analysis_views import compute_ml_merge_view

        faces = _make_faces(6)
        cr = _make_close_cluster_result(faces)
        result = _make_pipeline_result(faces, cr)
        payload = _make_merge_payload()

        view = compute_ml_merge_view(result, payload, threshold=0.5)

        for row in view.near_misses:
            assert 0.4 < row.ml_prob < 0.6

    def test_feature_version_mismatch_raises(self):
        """ValueError raised if model feature_version != FeatureComputer.VERSION."""
        from face_cluster.analysis_views import compute_ml_merge_view

        faces = _make_faces(4)
        cr = _make_close_cluster_result(faces)
        result = _make_pipeline_result(faces, cr)

        bad_payload = _make_merge_payload()
        bad_payload["metadata"]["feature_version"] = 999  # wrong version

        with pytest.raises(ValueError, match="Feature version mismatch"):
            compute_ml_merge_view(result, bad_payload)

    def test_no_embeddings_raises(self):
        """ValueError raised if no face has embedding_normalized."""
        from face_cluster.analysis_views import compute_ml_merge_view

        faces = _make_faces(4)
        for face in faces:
            face.embedding_normalized = None
        cr = _make_cluster_result(4, 2)
        result = _make_pipeline_result(faces, cr)
        payload = _make_merge_payload()

        with pytest.raises(ValueError, match="No face embeddings"):
            compute_ml_merge_view(result, payload)

    def test_probability_to_gate_mapping(self):
        """_prob_to_gate_count maps probabilities to correct gate counts."""
        from face_cluster.views.merge_view import _prob_to_gate_count

        assert _prob_to_gate_count(0.85) == 4
        assert _prob_to_gate_count(0.65) == 3
        assert _prob_to_gate_count(0.45) == 2
        assert _prob_to_gate_count(0.32) == 1
        assert _prob_to_gate_count(0.25) == 0


# ---------------------------------------------------------------------------
# 9. compute_pair_feature_contributions — feature attribution
# ---------------------------------------------------------------------------

class ut_FeatureContributions:
    """Tests for compute_pair_feature_contributions() in analysis_views.py."""

    def _minimal_lr_payload(self, feature_names):
        """Fit a LR model on synthetic data with given feature_names."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from face_cluster.features import VERSION as FEAT_VERSION

        rng = np.random.default_rng(7)
        n = 50
        X = rng.standard_normal((n, len(feature_names)))
        y = (X[:, 0] > 0).astype(int)  # class depends only on first feature

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=1000, random_state=0)
        model.fit(X_s, y)

        return {
            "model": model,
            "scaler": scaler,
            "feature_names": list(feature_names),
            "metadata": {
                "model_type": "logistic_regression",
                "feature_version": FEAT_VERSION,
            },
        }

    def test_logistic_regression_returns_top3(self):
        """LR contributions: returns <= top_n entries sorted by |contribution|."""
        from face_cluster.analysis_views import compute_pair_feature_contributions

        feature_names = ["feat_a", "feat_b", "feat_c", "feat_d"]
        payload = self._minimal_lr_payload(feature_names)
        features = {"feat_a": 0.9, "feat_b": 0.1, "feat_c": 0.5, "feat_d": 0.3}

        contribs = compute_pair_feature_contributions(features, payload, top_n=3)

        assert len(contribs) == 3
        # Each entry: (name, value, direction)
        for name, value, direction in contribs:
            assert name in feature_names
            assert isinstance(value, float)
            assert direction in ("-> merge", "-> reject")
        # Sorted by abs value descending
        abs_vals = [abs(v) for _, v, _ in contribs]
        assert abs_vals == sorted(abs_vals, reverse=True)

    def test_direction_matches_contribution_sign(self):
        """direction must be '-> merge' for positive contributions, '-> reject' for negative."""
        from face_cluster.analysis_views import compute_pair_feature_contributions

        feature_names = ["f1", "f2"]
        payload = self._minimal_lr_payload(feature_names)
        features = {"f1": 2.0, "f2": -1.0}  # force varied signs via extreme values

        contribs = compute_pair_feature_contributions(features, payload, top_n=2)

        for name, value, direction in contribs:
            if value >= 0:
                assert direction == "-> merge"
            else:
                assert direction == "-> reject"

    def test_top_n_limits_output_length(self):
        """top_n=1 returns exactly 1 entry (or fewer if model has <1 features)."""
        from face_cluster.analysis_views import compute_pair_feature_contributions

        feature_names = ["f1", "f2", "f3"]
        payload = self._minimal_lr_payload(feature_names)
        features = {"f1": 0.5, "f2": 0.3, "f3": 0.8}

        contribs = compute_pair_feature_contributions(features, payload, top_n=1)

        assert len(contribs) == 1
