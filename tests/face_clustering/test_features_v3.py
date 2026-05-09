"""Unit tests for face_cluster/features/ V3 sub-modules and orchestrator."""

import math
import numpy as np
import pytest

from face_cluster.types import FaceRecord, ClusterResult
from face_cluster.features import (
    FeatureComputer,
    MergeFeatureContext,
    ClusterPairFeatures,
)
from face_cluster.features.distance import compute_distance_features
from face_cluster.features.geometry import compute_geometry_features
from face_cluster.features.source_images import compute_source_image_features
from face_cluster.features.quality import compute_quality_features


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _face(face_id: int, image_path: str = "img_a.jpg", blur: float = 100.0,
          area: float = 5000.0, pose=None) -> FaceRecord:
    return FaceRecord(
        face_id=face_id,
        image_id=image_path,
        bbox=(0, 0, 100, 100),
        blur_score=blur,
        area=area,
        image_path=image_path,
        pose=pose,
        is_core=True,
    )


def _dist_matrix_2x2(d: float) -> np.ndarray:
    """2x2 distance matrix where off-diagonal = d."""
    dm = np.zeros((2, 2), dtype=float)
    dm[0, 1] = dm[1, 0] = d
    return dm


def _make_context(nodes_a, nodes_b, dm, faces, exemplars=None):
    """Build a MergeFeatureContext for the given two clusters."""
    n = len(faces)
    clusters = {0: nodes_a, 1: nodes_b}
    exemplars_map = exemplars or {0: nodes_a, 1: nodes_b}
    cr = ClusterResult(
        labels=np.array([0] * len(nodes_a) + [1] * len(nodes_b)),
        clusters=clusters,
        cluster_stats={0: {"diameter": 0.1}, 1: {"diameter": 0.1}},
        exemplars=exemplars_map,
        n_clusters=2,
        n_noise=0,
    )
    return MergeFeatureContext(cluster_result=cr, faces=faces, distance_matrix=dm)


# ---------------------------------------------------------------------------
# Group A: distance features
# ---------------------------------------------------------------------------

class ut_DistanceFeatures:
    def test_values_correct(self):
        nodes_a, nodes_b = [0], [1]
        dm = _dist_matrix_2x2(0.25)
        feats = compute_distance_features(
            nodes_a, nodes_b, nodes_a, nodes_b, dm, support_threshold=0.35
        )
        assert feats["min_exemplar_dist"] == pytest.approx(0.25)
        assert feats["min_cross_dist"] == pytest.approx(0.25)
        assert feats["p50_cross_dist"] == pytest.approx(0.25)
        assert feats["support_fraction"] == pytest.approx(1.0)
        assert feats["n_cross_pairs_below_threshold"] == 1

    def test_no_support_when_far(self):
        nodes_a, nodes_b = [0], [1]
        dm = _dist_matrix_2x2(0.9)
        feats = compute_distance_features(
            nodes_a, nodes_b, nodes_a, nodes_b, dm, support_threshold=0.35
        )
        assert feats["support_fraction"] == pytest.approx(0.0)
        assert feats["n_cross_pairs_below_threshold"] == 0

    def test_iqr_zero_for_single_pair(self):
        nodes_a, nodes_b = [0], [1]
        dm = _dist_matrix_2x2(0.3)
        feats = compute_distance_features(
            nodes_a, nodes_b, nodes_a, nodes_b, dm, support_threshold=0.35
        )
        assert feats["cross_dist_iqr"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Group B+C: geometry features
# ---------------------------------------------------------------------------

class ut_GeometryFeatures:
    def test_single_face_clusters(self):
        dm = _dist_matrix_2x2(0.3)
        feats = compute_geometry_features(
            [0], [1], [0], [1], dm,
            t_global=0.2, merge_exemplar_threshold=0.35
        )
        assert feats["size_a"] == 1
        assert feats["size_b"] == 1
        assert feats["diameter_a"] == pytest.approx(0.0)
        assert feats["post_merge_diameter"] == pytest.approx(0.3)
        assert feats["diameter_expansion"] == pytest.approx(1.0)  # dia_max=0 → 1.0

    def test_dist_to_threshold_ratio(self):
        dm = _dist_matrix_2x2(0.1)
        feats = compute_geometry_features(
            [0], [1], [0], [1], dm,
            t_global=0.2, merge_exemplar_threshold=0.4
        )
        assert feats["dist_to_threshold_ratio"] == pytest.approx(0.1 / 0.4)

    def test_size_ratio_symmetric(self):
        # 3 nodes in A, 1 in B → ratio should be 3.0
        n = 4
        dm = np.zeros((n, n))
        dm[0, 3] = dm[3, 0] = 0.2
        dm[1, 3] = dm[3, 1] = 0.3
        dm[2, 3] = dm[3, 2] = 0.4
        feats = compute_geometry_features(
            [0, 1, 2], [3], [0], [3], dm,
            t_global=0.2, merge_exemplar_threshold=0.35
        )
        assert feats["size_ratio"] == pytest.approx(3.0)
        assert feats["size_sum"] == 4


# ---------------------------------------------------------------------------
# Group D: source image features
# ---------------------------------------------------------------------------

class ut_SourceImageFeatures:
    def test_no_shared_images(self):
        faces = [_face(0, "img_a.jpg"), _face(1, "img_b.jpg")]
        dm = _dist_matrix_2x2(0.3)
        feats = compute_source_image_features([0], [1], faces, dm)
        assert feats["n_images_a"] == 1
        assert feats["n_images_b"] == 1
        assert feats["shared_source_images"] == 0
        assert feats["shared_source_ratio"] == pytest.approx(0.0)
        assert math.isnan(feats["same_image_min_dist"])

    def test_shared_image_detected(self):
        faces = [_face(0, "img_a.jpg"), _face(1, "img_a.jpg")]
        dm = _dist_matrix_2x2(0.25)
        feats = compute_source_image_features([0], [1], faces, dm)
        assert feats["shared_source_images"] == 1
        assert feats["shared_source_ratio"] == pytest.approx(1.0)
        assert feats["same_image_min_dist"] == pytest.approx(0.25)

    def test_missing_image_path_ignored(self):
        f0 = _face(0, "img_a.jpg")
        f1 = FaceRecord(face_id=1, image_id="x", bbox=(0, 0, 1, 1), image_path=None)
        dm = _dist_matrix_2x2(0.3)
        feats = compute_source_image_features([0], [1], [f0, f1], dm)
        assert feats["n_images_b"] == 0


# ---------------------------------------------------------------------------
# Group G: quality features
# ---------------------------------------------------------------------------

class ut_QualityFeatures:
    def test_basic_blur_area(self):
        faces = [_face(0, blur=80.0, area=4000.0), _face(1, blur=60.0, area=2000.0)]
        feats = compute_quality_features([0], [1], faces)
        assert feats["mean_blur_a"] == pytest.approx(80.0)
        assert feats["blur_min_b"] == pytest.approx(60.0)
        assert feats["area_ratio"] == pytest.approx(2.0)

    def test_frontal_fraction_with_pose(self):
        f0 = _face(0, pose=(5.0, 5.0, 0.0))   # frontal
        f1 = _face(1, pose=(50.0, 5.0, 0.0))  # not frontal
        feats = compute_quality_features([0], [1], [f0, f1])
        assert feats["frontal_frac_a"] == pytest.approx(1.0)
        assert feats["frontal_frac_b"] == pytest.approx(0.0)
        assert feats["frontal_frac_min"] == pytest.approx(0.0)

    def test_no_pose_defaults_to_frontal(self):
        faces = [_face(0), _face(1)]
        feats = compute_quality_features([0], [1], faces)
        assert feats["frontal_frac_a"] == pytest.approx(1.0)
        assert feats["pose_diff"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class ut_FeatureComputer:
    def _two_cluster_setup(self):
        """3 faces per cluster, clear separation."""
        n = 6
        dm = np.full((n, n), 0.8)
        np.fill_diagonal(dm, 0.0)
        # Intra-cluster: tight
        for i in range(3):
            for j in range(3):
                dm[i, j] = dm[j, i] = 0.05
                dm[i + 3, j + 3] = dm[j + 3, i + 3] = 0.05
        # Cross-cluster: far
        for i in range(3):
            for j in range(3):
                dm[i, j + 3] = dm[j + 3, i] = 0.8

        faces = [
            _face(i, f"img_{i}.jpg", blur=100.0, area=5000.0)
            for i in range(6)
        ]
        return faces, dm

    def test_compute_all_pairs_finds_candidate(self):
        """When clusters are close, compute_all_pairs finds the pair."""
        n = 4
        dm = np.full((n, n), 0.8)
        np.fill_diagonal(dm, 0.0)
        dm[0, 2] = dm[2, 0] = 0.2
        dm[1, 3] = dm[3, 1] = 0.3
        dm[0, 3] = dm[3, 0] = 0.35
        dm[1, 2] = dm[2, 1] = 0.35

        faces = [_face(i, f"img_{i}.jpg") for i in range(4)]
        cr = ClusterResult(
            labels=np.array([0, 0, 1, 1]),
            clusters={0: [0, 1], 1: [2, 3]},
            cluster_stats={0: {"diameter": 0.1}, 1: {"diameter": 0.1}},
            exemplars={0: [0, 1], 1: [2, 3]},
            n_clusters=2,
            n_noise=0,
        )
        ctx = MergeFeatureContext(cluster_result=cr, faces=faces, distance_matrix=dm)
        fc = FeatureComputer()
        pairs = fc.compute_all_pairs(ctx, candidate_threshold=0.45)
        assert len(pairs) == 1
        feats = next(iter(pairs.values()))
        assert feats.min_exemplar_dist == pytest.approx(0.2)

    def test_compute_all_pairs_skips_far_pairs(self):
        """Clusters with exemplar distance > threshold are excluded."""
        n = 4
        dm = np.full((n, n), 0.9)
        np.fill_diagonal(dm, 0.0)
        faces = [_face(i) for i in range(4)]
        cr = ClusterResult(
            labels=np.array([0, 0, 1, 1]),
            clusters={0: [0, 1], 1: [2, 3]},
            cluster_stats={},
            exemplars={0: [0, 1], 1: [2, 3]},
            n_clusters=2,
            n_noise=0,
        )
        ctx = MergeFeatureContext(cluster_result=cr, faces=faces, distance_matrix=dm)
        fc = FeatureComputer()
        pairs = fc.compute_all_pairs(ctx, candidate_threshold=0.45)
        assert len(pairs) == 0

    def test_to_dataframe_shape(self):
        """to_dataframe produces correct columns and row count."""
        n = 4
        dm = np.zeros((n, n))
        dm[0, 2] = dm[2, 0] = 0.2
        faces = [_face(i, f"img_{i}.jpg") for i in range(4)]
        cr = ClusterResult(
            labels=np.array([0, 0, 1, 1]),
            clusters={0: [0, 1], 1: [2, 3]},
            cluster_stats={},
            exemplars={0: [0, 1], 1: [2, 3]},
            n_clusters=2,
            n_noise=0,
        )
        ctx = MergeFeatureContext(cluster_result=cr, faces=faces, distance_matrix=dm)
        fc = FeatureComputer()
        pairs = fc.compute_all_pairs(ctx, candidate_threshold=0.45)
        df = fc.to_dataframe(pairs)
        assert "cluster_a" in df.columns
        assert "cluster_b" in df.columns
        assert "min_exemplar_dist" in df.columns
        assert len(df) == len(pairs)

    def test_context_asserts_index_space(self):
        """MergeFeatureContext must reject mismatched distance matrix."""
        dm = np.zeros((3, 3))  # 3x3 but only 2 faces
        faces = [_face(0), _face(1)]
        cr = ClusterResult(
            labels=np.array([0, 1]),
            clusters={0: [0], 1: [1]},
            cluster_stats={},
            exemplars={},
            n_clusters=2,
            n_noise=0,
        )
        with pytest.raises(AssertionError):
            MergeFeatureContext(cluster_result=cr, faces=faces, distance_matrix=dm)
