"""Tests that pipeline.py writes action_log rows correctly.

Uses the real test images so the hook exercises the full path including
_init_context, _build_result, and _finalize.
"""
import pytest
from pathlib import Path

from face_cluster import FaceClusteringPipeline, PipelineConfig
from face_cluster.run_history_db import list_actions, init_table
from tests.conftest import get_test_data_dir

TEST_DATA_DIR = get_test_data_dir() / "face_clustering"


@pytest.fixture()
def db(tmp_path):
    db_path = tmp_path / "hist.db"
    init_table(db_path=db_path)
    return db_path


def _monkeypatch_db(monkeypatch, db_path):
    """Redirect run_history_db to a tmp DB for isolation."""
    import face_cluster.run_history_db as rh
    monkeypatch.setattr(rh, "get_db_path", lambda: db_path)


class ut_PipelineHistoryHook:

    def test_successful_run_writes_complete_row(self, tmp_path, monkeypatch, db):
        _monkeypatch_db(monkeypatch, db)
        out = tmp_path / "out"
        result = FaceClusteringPipeline().run(PipelineConfig.full_run(TEST_DATA_DIR, out))

        rows = list_actions(types=["pipeline_run"], db_path=db)
        assert len(rows) == 1, f"Expected 1 row, got {len(rows)}"
        row = rows[0]
        assert row["status"] == "complete"
        assert row["n_faces"] == result.summary["n_faces"]
        assert row["n_clusters"] == result.summary["n_clusters"]
        assert row["output_dir"] is not None
        assert row["log_file"] is not None

    def test_failed_run_writes_failed_row(self, tmp_path, monkeypatch, db):
        _monkeypatch_db(monkeypatch, db)

        # Patch a stage method to raise so the pipeline fails mid-run
        import face_cluster.pipeline as pl
        original_cluster = FaceClusteringPipeline._cluster

        def _boom(self, ctx):
            raise RuntimeError("Injected failure for test")

        monkeypatch.setattr(FaceClusteringPipeline, "_cluster", _boom)

        out = tmp_path / "out_fail"
        with pytest.raises(Exception):
            FaceClusteringPipeline().run(PipelineConfig.full_run(TEST_DATA_DIR, out))

        rows = list_actions(types=["pipeline_run"], db_path=db)
        assert len(rows) == 1
        row = rows[0]
        assert row["status"] == "failed"
        assert row["error"] is not None
        assert "Injected failure" in row["error"] or row["error"] != ""

    def test_recluster_writes_recluster_type(self, tmp_path, monkeypatch, db):
        _monkeypatch_db(monkeypatch, db)
        # First do a full run to get embeddings
        out_full = tmp_path / "full"
        FaceClusteringPipeline().run(PipelineConfig.full_run(TEST_DATA_DIR, out_full))

        # Then recluster
        out_rc = tmp_path / "rc"
        FaceClusteringPipeline().run(PipelineConfig.recluster(out_full, out_rc))

        rows = list_actions(types=["recluster"], db_path=db)
        assert len(rows) == 1
        assert rows[0]["status"] == "complete"
        assert rows[0]["action_type"] == "recluster"
