"""Pipeline integration tests on 100-image sample from Google_Germany.

Baseline run (2026-04-03, seed=42):
  n_faces=130, n_core=110, n_holdout=20, n_clusters=7, n_noise=60
  max_cluster_size=11 (22% of core), crop_manifest=130

Test bounds are set around these observed values with ±50% headroom to
accommodate minor algorithm changes while still catching regressions.
"""
import pytest
import pandas as pd
import json
import numpy as np
from pathlib import Path

from face_cluster import FaceClusteringPipeline, PipelineConfig
from tests.conftest import get_test_data_dir

TEST_DATA_100 = get_test_data_dir() / "face_clustering_100"


@pytest.fixture(scope="class")
def pipeline_result(tmp_path_factory):
    """Run pipeline once on 100-image sample; reuse result across all tests."""
    if not TEST_DATA_100.exists():
        pytest.fail(
            f"Test data directory not found: {TEST_DATA_100}\n"
            "Run: python scripts/create_test_sample.py "
            "--source D:/Google_Germany --dest test_data/face_clustering_100 --n 100 --seed 42"
        )
    if not any(TEST_DATA_100.iterdir()):
        pytest.fail(f"Test data directory is empty: {TEST_DATA_100}")

    tmp = tmp_path_factory.mktemp("output_100")
    # production defaults — no relaxation
    return FaceClusteringPipeline().run(
        PipelineConfig.full_run(TEST_DATA_100, tmp / "output")
    )


class ut_FaceClusteringPipeline_100:
    """Integration tests on 100-image real-world sample.

    Checks structural properties of the pipeline output:
    - Detection and quality gating rates
    - Cluster count in sane range (not degenerate)
    - Analysis layer is computable at all three levels
    """

    # -- Output integrity ----------------------------------------------------

    def test_all_output_files_exist(self, pipeline_result):
        out = pipeline_result.output_dir
        for fname in ("faces.csv", "clusters.csv", "export_summary.json", "crop_manifest.json"):
            assert (out / fname).exists(), f"Missing output file: {fname}"

    def test_no_null_image_paths(self, pipeline_result):
        faces_df = pd.read_csv(pipeline_result.output_dir / "faces.csv")
        assert not faces_df.empty, "faces.csv is empty"
        null_count = faces_df["image_path"].isna().sum()
        assert null_count == 0, f"{null_count} faces have null image_path"

    def test_face_ids_globally_unique(self, pipeline_result):
        """Regression for SIGHTING-012: face_id_counter resetting per image."""
        faces_df = pd.read_csv(pipeline_result.output_dir / "faces.csv")
        assert not faces_df.empty
        n_total = len(faces_df)
        n_unique = faces_df["face_id"].nunique()
        assert n_unique == n_total, (
            f"face_id not unique: {n_unique} unique IDs for {n_total} faces"
        )

    def test_crop_manifest_format_is_flat_string(self, pipeline_result):
        """Regression for SIGHTING-013: manifest format must be {id: path_str}, not {id: {crop_path: ...}}.

        Any code that reads crop_manifest.json must use entry as a string, not call .get() on it.
        """
        with open(pipeline_result.output_dir / "crop_manifest.json") as f:
            manifest = json.load(f)
        assert len(manifest) > 0, "crop_manifest.json is empty"
        first_key = next(iter(manifest))
        first_entry = manifest[first_key]
        assert isinstance(first_entry, str), (
            f"crop_manifest entry must be a str path, got {type(first_entry).__name__}: {first_entry!r}"
        )
        # The path must resolve to an existing file relative to output_dir
        crop_path = pipeline_result.output_dir / first_entry
        assert crop_path.exists(), f"Crop file does not exist: {crop_path}"

    def test_crop_count_matches_face_count(self, pipeline_result):
        """Regression for SIGHTING-012: crop_manifest must have one entry per face."""
        with open(pipeline_result.output_dir / "crop_manifest.json") as f:
            manifest = json.load(f)
        faces_df = pd.read_csv(pipeline_result.output_dir / "faces.csv")
        assert len(manifest) == len(faces_df), (
            f"crop_manifest has {len(manifest)} entries but faces.csv has {len(faces_df)} rows"
        )

    # -- Detection and quality gating ----------------------------------------

    def test_detects_faces_in_majority_of_images(self, pipeline_result):
        """At least 50% of 100 images must yield a detectable face."""
        n_faces = pipeline_result.summary["n_faces"]
        assert n_faces >= 50, f"Only {n_faces} faces detected in 100 images — embedder may be broken"

    def test_quality_gate_rejects_some_faces(self, pipeline_result):
        """Regression for SIGHTING-010: holdout path must be exercised."""
        n_holdout = pipeline_result.summary["n_faces"] - pipeline_result.summary["n_core"]
        assert n_holdout > 0, "Quality gate rejected no faces — gating may be broken"

    def test_quality_gate_passes_majority(self, pipeline_result):
        """Quality gate must not reject everything — regression for over-gating (SIGHTING-004)."""
        n_faces = pipeline_result.summary["n_faces"]
        n_core = pipeline_result.summary["n_core"]
        assert n_core > 0, "Quality gate accepted zero faces"
        assert n_core / n_faces >= 0.4, (
            f"Quality gate passed only {100*n_core/n_faces:.0f}% of faces — threshold may be too strict"
        )

    # -- Clustering structure ------------------------------------------------

    def test_cluster_count_in_expected_range(self, pipeline_result):
        """Baseline: 7 clusters. Allow 3-25 to catch regressions without being brittle."""
        n = pipeline_result.cluster_result.n_clusters
        assert 3 <= n <= 25, (
            f"Expected 3-25 clusters for 100-image album sample, got {n}. "
            "Baseline was 7. Check distance_threshold or kNN graph."
        )

    def test_no_dominant_cluster(self, pipeline_result):
        """No single cluster should contain >50% of core faces (degenerate merge)."""
        faces_df = pd.read_csv(pipeline_result.output_dir / "faces.csv")
        core = faces_df[faces_df["cluster_id"] != -1]
        assert len(core) > 0, "No clustered faces at all"
        max_size = core["cluster_id"].value_counts().iloc[0]
        assert max_size < len(core) * 0.5, (
            f"Largest cluster has {max_size}/{len(core)} core faces ({100*max_size/len(core):.0f}%) — over-merging"
        )

    def test_all_clusters_have_exemplars(self, pipeline_result):
        cr = pipeline_result.cluster_result
        for cid in cr.clusters:
            ex = cr.exemplars.get(cid, [])
            assert len(ex) >= 1, f"Cluster {cid} has no exemplars"

    # -- Analysis layer: RunOverview -----------------------------------------

    def test_run_overview_computable(self, pipeline_result):
        from face_cluster.analysis_views import RunOverview
        overview = RunOverview.compute(pipeline_result)
        assert overview.n_clusters == pipeline_result.cluster_result.n_clusters
        assert overview.n_faces_detected == pipeline_result.summary["n_faces"]
        assert len(overview.cluster_rows) == overview.n_clusters

    def test_run_overview_umap_shape(self, pipeline_result):
        from face_cluster.analysis_views import RunOverview
        overview = RunOverview.compute(pipeline_result)
        assert overview.umap_coords is not None, "UMAP computation failed"
        n_core = pipeline_result.summary["n_core"]
        assert overview.umap_coords.shape == (n_core, 2), (
            f"UMAP shape {overview.umap_coords.shape} != ({n_core}, 2)"
        )
        assert overview.umap_labels is not None
        assert len(overview.umap_labels) == n_core

    def test_run_overview_cluster_rows_have_valid_distances(self, pipeline_result):
        from face_cluster.analysis_views import RunOverview
        overview = RunOverview.compute(pipeline_result)
        for row in overview.cluster_rows:
            assert row.diameter >= 0, f"Cluster {row.cluster_id} has negative diameter"
            assert row.avg_intra_dist >= 0
            assert row.nearest_cluster_dist >= 0

    # -- Analysis layer: ClusterView -----------------------------------------

    def test_cluster_view_computable(self, pipeline_result):
        from face_cluster.analysis_views import ClusterView
        cr = pipeline_result.cluster_result
        largest_cid = max(cr.clusters, key=lambda c: len(cr.clusters[c]))
        view = ClusterView.compute(pipeline_result, largest_cid)
        assert view.cluster_id == largest_cid
        assert view.size == len(cr.clusters[largest_cid])
        assert view.diameter >= 0
        assert len(view.exemplar_face_ids) >= 1
        assert len(view.faces) == view.size

    def test_cluster_view_nearest_clusters_non_empty(self, pipeline_result):
        from face_cluster.analysis_views import ClusterView
        cr = pipeline_result.cluster_result
        if len(cr.clusters) < 2:
            pytest.skip("Need at least 2 clusters to test nearest-cluster computation")
        largest_cid = max(cr.clusters, key=lambda c: len(cr.clusters[c]))
        view = ClusterView.compute(pipeline_result, largest_cid)
        assert len(view.nearest_clusters) >= 1, "No nearest clusters computed"
        # Distances must be in valid range
        for nc in view.nearest_clusters:
            assert 0.0 <= nc.min_exemplar_dist <= 2.0

    # -- Analysis layer: FaceView --------------------------------------------

    def test_face_view_computable_for_core_face(self, pipeline_result):
        from face_cluster.analysis_views import FaceView
        faces_df = pd.read_csv(pipeline_result.output_dir / "faces.csv")
        core = faces_df[faces_df["cluster_id"] != -1]
        assert len(core) > 0, "No core faces to test FaceView"
        face_id = int(core.iloc[0]["face_id"])
        view = FaceView.compute(pipeline_result, face_id)
        assert view.face_id == face_id
        assert view.cluster_id >= 0
        assert view.gate_result == "core"
        assert view.gate_rejection_reason is None
        assert view.blur_score > 0

    def test_face_view_computable_for_holdout_face(self, pipeline_result):
        from face_cluster.analysis_views import FaceView
        faces_df = pd.read_csv(pipeline_result.output_dir / "faces.csv")
        holdout = faces_df[faces_df["cluster_id"] == -1]
        assert len(holdout) > 0, "No holdout faces — test_quality_gate_rejects_some_faces should have caught this"
        # Find one that was quality-gated (not just noise) — blur < 50
        quality_gated = holdout[~holdout["face_id"].isin(
            [f.face_id for f in pipeline_result.faces if f.is_core]
        )]
        if quality_gated.empty:
            # All holdout are noise — use any holdout face
            quality_gated = holdout
        face_id = int(quality_gated.iloc[0]["face_id"])
        view = FaceView.compute(pipeline_result, face_id)
        assert view.face_id == face_id
        assert view.gate_result == "holdout"
        assert view.gate_rejection_reason is not None

    def test_face_view_closest_faces_populated(self, pipeline_result):
        from face_cluster.analysis_views import FaceView
        cr = pipeline_result.cluster_result
        # Pick a face from the largest cluster (most likely to have close same-cluster neighbors)
        largest_cid = max(cr.clusters, key=lambda c: len(cr.clusters[c]))
        member_idx = cr.clusters[largest_cid][0]
        face_id = pipeline_result.faces[member_idx].face_id
        view = FaceView.compute(pipeline_result, face_id)
        assert len(view.closest_same_cluster) >= 1, "No same-cluster neighbors found"
        assert len(view.closest_other_clusters) >= 1, "No cross-cluster neighbors found"
        for cf in view.closest_same_cluster + view.closest_other_clusters:
            assert 0.0 <= cf.distance <= 2.0
