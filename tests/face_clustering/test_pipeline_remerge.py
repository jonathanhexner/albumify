"""E2E tests for PipelineConfig.remerge() flow.

Tests the path: full_run -> save_manual_merge_snapshot -> remerge -> valid output.

This exercises the _load_source_remerge source loader, which:
  1. Loads faces + embeddings from a manual_merge snapshot
  2. Converts face-list cluster_result back to graph-local indices
  3. Builds a GraphResult from the embedding distance matrix
  4. Passes these to the merge/exemplar/export stages
"""
import json
from pathlib import Path

import pytest

from face_cluster import FaceClusteringPipeline, PipelineConfig, save_manual_merge_snapshot
from face_cluster.loader import load_pipeline_result
from tests.conftest import get_test_data_dir

TEST_DATA_DIR = get_test_data_dir() / "face_clustering"


@pytest.fixture(scope="module")
def full_run_result(tmp_path_factory):
    """Run a full pipeline once; reuse across remerge tests."""
    if not TEST_DATA_DIR.exists():
        pytest.fail(f"Test data directory not found: {TEST_DATA_DIR}")
    tmp = tmp_path_factory.mktemp("remerge_base")
    return FaceClusteringPipeline().run(
        PipelineConfig.full_run(TEST_DATA_DIR, tmp / "full", merge_enabled=True)
    )


class ut_PipelineRemerge:
    """E2E tests: remerge pipeline from a manual_merge snapshot."""

    @pytest.fixture(scope="class")
    def snapshot_dir(self, full_run_result, tmp_path_factory):
        """Save a manual_merge snapshot from the full run result."""
        tmp = tmp_path_factory.mktemp("snap")
        snap = tmp / "snapshot"

        # Use merged result if available, else base
        merged_cr = full_run_result.merged_cluster_result or full_run_result.cluster_result

        save_manual_merge_snapshot(
            faces=full_run_result.faces,
            merged_cluster_result=merged_cr,
            approved_pairs=[],
            rejected_pairs=[],
            config=PipelineConfig(),
            output_dir=snap,
            parent_output_dir=full_run_result.output_dir,
            parent_run_id="test_full_run",
            merge_round=1,
        )
        return snap

    @pytest.fixture(scope="class")
    def remerge_result(self, snapshot_dir, tmp_path_factory):
        """Run remerge from the snapshot."""
        tmp = tmp_path_factory.mktemp("remerge_out")
        return FaceClusteringPipeline().run(
            PipelineConfig.remerge(snapshot_dir, tmp / "remerged", merge_enabled=True)
        )

    def test_remerge_produces_valid_output(self, remerge_result):
        assert remerge_result is not None
        assert remerge_result.cluster_result is not None
        assert remerge_result.cluster_result.n_clusters >= 1

    def test_remerge_writes_required_files(self, remerge_result):
        out = remerge_result.output_dir
        for fname in ("faces.csv", "clusters.csv", "pipeline_run.json", "crop_manifest.json"):
            assert (out / fname).exists(), f"Missing: {fname}"

    def test_remerge_faces_csv_has_cluster_assignments(self, remerge_result):
        import pandas as pd
        df = pd.read_csv(remerge_result.output_dir / "faces.csv")
        assert not df.empty
        assert "cluster_id" in df.columns
        # At least some faces should be clustered
        clustered = df[df["cluster_id"] != -1]
        assert len(clustered) >= 1, "No clustered faces in remerge output"

    def test_remerge_cluster_count_lte_base(self, full_run_result, remerge_result):
        """Remerge (with merge_enabled=True) can only reduce or keep cluster count."""
        base_n = (full_run_result.merged_cluster_result or
                  full_run_result.cluster_result).n_clusters
        remerge_n = (remerge_result.merged_cluster_result or
                     remerge_result.cluster_result).n_clusters
        assert remerge_n <= base_n, (
            f"Remerge increased cluster count: {base_n} -> {remerge_n}"
        )

    def test_remerge_writes_merged_files(self, remerge_result):
        """With merge_enabled=True, faces_merged.csv must exist."""
        out = remerge_result.output_dir
        assert (out / "faces_merged.csv").exists(), "faces_merged.csv missing from remerge"
        assert (out / "clusters_merged.csv").exists()
        assert (out / "merge_log.json").exists()

    def test_remerge_loader_round_trip(self, remerge_result):
        """load_pipeline_result must reconstruct the remerge result correctly."""
        loaded = load_pipeline_result(remerge_result.output_dir)
        assert loaded.cluster_result.n_clusters == remerge_result.cluster_result.n_clusters
        assert len(loaded.faces) == len(remerge_result.faces)

    def test_remerge_pipeline_run_json(self, remerge_result):
        rec = json.loads((remerge_result.output_dir / "pipeline_run.json").read_text())
        assert rec.get("mode") == "remerge"
        assert "source_run" in rec

    def test_remerge_with_exemplars(self, snapshot_dir, tmp_path_factory):
        """PipelineConfig.remerge(with_exemplars=True) must also work."""
        tmp = tmp_path_factory.mktemp("remerge_ex")
        result = FaceClusteringPipeline().run(
            PipelineConfig.remerge(snapshot_dir, tmp / "remerged_ex",
                                   with_exemplars=True, merge_enabled=True)
        )
        assert result.cluster_result.n_clusters >= 1
