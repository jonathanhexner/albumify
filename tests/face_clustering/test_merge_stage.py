"""Tests for merge stage, recluster, and merged export/loader contract.

Coverage:
  - ConservativeMerger wired into pipeline (_merge stage)
  - export_merged_results writes correct files
  - loader loads merged artifacts and populates PipelineResult.merged_cluster_result
  - recluster() produces valid output from an existing run directory
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from face_cluster import FaceClusteringPipeline, PipelineConfig
from sim_bench.run_db._schema import SCHEMA_VERSION
from face_cluster.export import export_merged_results
from face_cluster.loader import load_pipeline_result
from face_cluster.types import ClusterResult, FaceRecord


# ---------------------------------------------------------------------------
# Minimal synthetic fixtures
# ---------------------------------------------------------------------------

def _make_faces(n: int) -> list[FaceRecord]:
    """Return n FaceRecord objects with random normalised embeddings."""
    faces = []
    for i in range(n):
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        faces.append(FaceRecord(
            face_id=i,
            image_id=f"img_{i}",
            bbox=(0, 0, 10, 10),
            embedding_normalized=emb,
            blur_score=100.0,
            area=500.0,
            is_core=True,
            image_path=f"/tmp/img_{i}.jpg",
        ))
    return faces


def _make_cluster_result(n_faces: int, n_clusters: int) -> ClusterResult:
    """Return a minimal ClusterResult with n_clusters even-sized clusters."""
    size = n_faces // n_clusters
    clusters = {}
    exemplars = {}
    labels = np.full(n_faces, -1, dtype=np.int32)
    for cid in range(n_clusters):
        members = list(range(cid * size, (cid + 1) * size))
        clusters[cid] = members
        exemplars[cid] = members[:1]
        for fi in members:
            labels[fi] = cid
    return ClusterResult(
        labels=labels,
        clusters=clusters,
        cluster_stats={cid: {"diameter": 0.1} for cid in clusters},
        exemplars=exemplars,
        n_clusters=n_clusters,
        n_noise=0,
    )


# ---------------------------------------------------------------------------
# export_merged_results / loader contract
# ---------------------------------------------------------------------------

class ut_MergedExportLoaderContract:
    """Verify writer-reader contract for merged CSV files."""

    def test_export_writes_required_files(self, tmp_path):
        faces = _make_faces(6)
        merged_cr = _make_cluster_result(6, 2)
        merge_log = [
            {"iteration": 1, "cluster_a": 0, "cluster_b": 1,
             "exemplar_dist": 0.3, "threshold_used": 0.4,
             "support": 5, "required_support": 2,
             "post_diameter": 0.35, "max_allowed_diameter": 0.5,
             "action": "merged", "passes_exemplar": True,
             "passes_support": True, "passes_margin": True,
             "passes_diameter": True, "rejection_reason": None,
             "actually_merged": True,
             "cluster_a_size": 3, "cluster_b_size": 3},
        ]
        export_merged_results(
            faces=faces,
            merged_cluster_result=merged_cr,
            merge_log=merge_log,
            output_dir=tmp_path,
            core_indices=None,  # clusters already in face-list space for this test
        )
        assert (tmp_path / "faces_merged.csv").exists()
        assert (tmp_path / "clusters_merged.csv").exists()
        assert (tmp_path / "merge_log.json").exists()

    def test_faces_merged_schema(self, tmp_path):
        faces = _make_faces(4)
        merged_cr = _make_cluster_result(4, 2)
        export_merged_results(faces, merged_cr, [], tmp_path, core_indices=None)

        df = pd.read_csv(tmp_path / "faces_merged.csv")
        assert set(df.columns) >= {"face_id", "cluster_id"}
        assert df["face_id"].dtype in (int, np.int64, np.int32)
        assert df["cluster_id"].dtype in (int, np.int64, np.int32)
        assert len(df) == 4

    def test_clusters_merged_schema(self, tmp_path):
        faces = _make_faces(4)
        merged_cr = _make_cluster_result(4, 2)
        export_merged_results(faces, merged_cr, [], tmp_path, core_indices=None)

        df = pd.read_csv(tmp_path / "clusters_merged.csv")
        assert set(df.columns) >= {"cluster_id", "size", "exemplar_face_ids"}
        assert len(df) == 2

    def test_merge_log_json_serializable(self, tmp_path):
        """merge_log.json must be valid JSON even with numpy scalar values."""
        faces = _make_faces(2)
        merged_cr = _make_cluster_result(2, 1)
        log_with_numpy = [{"exemplar_dist": np.float64(0.25), "support": np.int32(3),
                           "passes_exemplar": np.bool_(True)}]
        export_merged_results(faces, merged_cr, log_with_numpy, tmp_path, core_indices=None)
        loaded = json.loads((tmp_path / "merge_log.json").read_text())
        assert isinstance(loaded[0]["exemplar_dist"], float)
        assert isinstance(loaded[0]["support"], int)

    def test_loader_reads_merged_artifacts(self, tmp_path):
        """load_pipeline_result() must populate merged_cluster_result when files exist."""
        faces = _make_faces(4)
        merged_cr = _make_cluster_result(4, 2)
        export_merged_results(faces, merged_cr, [], tmp_path, core_indices=None)

        # Also write the base files needed by loader
        faces_rows = [{"face_id": f.face_id, "image_path": f.image_path,
                        "image_id": f.image_id, "crop_path": "",
                        "cluster_id": 0, "is_core": True,
                        "blur_score": f.blur_score, "area": f.area,
                        "yaw": None, "pitch": None, "roll": None}
                      for f in faces]
        pd.DataFrame(faces_rows).to_csv(tmp_path / "faces.csv", index=False)
        pd.DataFrame([{"cluster_id": 0, "size": 4, "exemplar_face_ids": "[0]", "diameter": 0.1}
                      ]).to_csv(tmp_path / "clusters.csv", index=False)

        run_rec = {"run_id": "test", "source_album": "/tmp",
                   "output_dir": str(tmp_path), "started_at": "2026-01-01T00:00:00",
                   "config": {}, "stages": {}, "status": "complete",
                   "summary": {"n_faces": 4, "n_core": 4, "n_clusters": 1, "n_noise": 0}}
        (tmp_path / "pipeline_run.json").write_text(json.dumps(run_rec))

        result = load_pipeline_result(tmp_path)
        assert result.merged_cluster_result is not None
        assert result.merged_cluster_result.n_clusters == 2
        assert result.merge_log is not None


# ---------------------------------------------------------------------------
# E2E pipeline tests using real test data
# ---------------------------------------------------------------------------

TEST_DATA_DIR = Path(__file__).parent.parent.parent / "test_data" / "face_clustering"


@pytest.fixture(scope="module")
def merged_pipeline_result(tmp_path_factory):
    """Run pipeline with merge_enabled=True on test data."""
    if not TEST_DATA_DIR.exists():
        pytest.fail(f"Test data directory not found: {TEST_DATA_DIR}")
    tmp_path = tmp_path_factory.mktemp("merge_e2e")
    return FaceClusteringPipeline().run(
        PipelineConfig.full_run(TEST_DATA_DIR, tmp_path / "output", merge_enabled=True)
    )


class ut_MergeStageE2E:
    """E2E tests: merge stage in full pipeline."""

    def test_pipeline_result_has_merged_fields(self, merged_pipeline_result):
        """When merge_enabled=True, PipelineResult must have merged_cluster_result."""
        result = merged_pipeline_result
        # merged_cluster_result may be None if no merges happened, but the field must exist
        assert hasattr(result, "merged_cluster_result")
        assert hasattr(result, "merge_log")

    def test_merged_files_written(self, merged_pipeline_result):
        out = merged_pipeline_result.output_dir
        assert (out / "faces_merged.csv").exists()
        assert (out / "clusters_merged.csv").exists()
        assert (out / "merge_log.json").exists()

    def test_v4_dual_write_present(self, merged_pipeline_result):
        """spec-030 Phase 1: every real pipeline run produces a v4 subdir
        alongside the legacy artifacts."""
        v4 = merged_pipeline_result.output_dir / "_v4"
        assert v4.exists(), "spec-030 Phase 1 dual-write missing"
        names = sorted(p.name for p in v4.iterdir())
        from sim_bench.run_db._schema import EXPECTED_ARTIFACTS
        assert names == sorted(EXPECTED_ARTIFACTS), (
            f"v4 subdir listdir mismatch: got {names}, expected {sorted(EXPECTED_ARTIFACTS)}"
        )

    def test_v4_full_round_trip_real_images(self, merged_pipeline_result):
        """spec-030 end-to-end on real JPEGs:
            real images -> face detect -> embed -> cluster -> merge
                -> RunExporter (writer) -> disk -> RunStore (reader)
        Every piece of in-memory pipeline state must come back bit-identical
        (or float-close) through the writer/reader pair.

        This is the test that proves Phases 1+2 work end-to-end on real data;
        all other tests use synthetic FaceRecord fixtures.
        """
        import numpy as np
        from sim_bench.run_db.store import RunStore
        from face_cluster.types import MergeDecisionRow

        result = merged_pipeline_result
        store = RunStore(result.output_dir / "_v4")

        # ---- run-level metadata --------------------------------------------------
        meta = store.metadata()
        assert meta.n_faces == len(result.faces)
        assert meta.n_clusters_base == result.cluster_result.n_clusters
        assert meta.n_clusters_final == (
            result.merged_cluster_result.n_clusters
            if result.merged_cluster_result else result.cluster_result.n_clusters
        )
        assert meta.n_merges == sum(1 for e in (result.merge_log or []) if e.get("actually_merged"))
        # spec-054: assert against the current SCHEMA_VERSION constant
        # (was hardcoded `== 4` and silently rotted when spec-040 Phase 4
        # bumped to 5). Future bumps re-validate without test edits;
        # tests/architecture/test_schema_history.py forces SCHEMA_HISTORY
        # to be updated on every bump so the version still has meaning.
        assert meta.schema_version == SCHEMA_VERSION

        # ---- faces table: face_id set + bbox + blur match exactly ----------------
        read_faces = store.faces()
        assert len(read_faces) == len(result.faces)
        in_mem = {f.face_id: f for f in result.faces}
        for rf in read_faces:
            mf = in_mem[rf.face_id]
            assert rf.image_path == mf.image_path
            # bbox: stored as REAL, comes back as float — exact equality acceptable
            # because both sides cast through float() the same way.
            assert tuple(map(float, rf.bbox)) == pytest.approx(tuple(map(float, mf.bbox)))
            assert rf.is_core == mf.is_core
            assert rf.blur_score == pytest.approx(mf.blur_score, rel=1e-6)

        # ---- embeddings: shape, dtype, alignment, bit-identical values -----------
        emb = store.embeddings()
        assert emb.matrix.shape == (len(result.faces), 512)
        assert emb.matrix.dtype == np.float32
        assert emb.face_ids.tolist() == [f.face_id for f in result.faces]
        # Each face's stored embedding must equal what was in memory at write time.
        for i, mf in enumerate(result.faces):
            in_mem_vec = mf.embedding_normalized if mf.embedding_normalized is not None else mf.embedding
            if in_mem_vec is not None:
                np.testing.assert_array_equal(
                    emb.matrix[i],
                    np.asarray(in_mem_vec, dtype=np.float32),
                    err_msg=f"embedding mismatch for face_id {mf.face_id}",
                )

        # ---- merge_log: every field of every row, including booleans + inf -------
        read_log = store.merge_log()
        in_mem_log = result.merge_log or []
        assert len(read_log) == len(in_mem_log)
        # Sort both by (iteration, cluster_a, cluster_b) for stable comparison.
        read_sorted = sorted(read_log, key=lambda r: (r.iteration, r.cluster_a, r.cluster_b))
        mem_sorted = sorted(in_mem_log, key=lambda e: (e["iteration"], e["cluster_a"], e["cluster_b"]))
        for r_row, m_dict in zip(read_sorted, mem_sorted):
            assert isinstance(r_row, MergeDecisionRow)
            for fname in MergeDecisionRow.field_names():
                got = getattr(r_row, fname)
                want = m_dict[fname]
                if isinstance(want, float) and not (want == want):  # NaN
                    assert got != got, f"{fname}: NaN lost"
                elif isinstance(want, float) and want in (float("inf"), float("-inf")):
                    assert got == want, f"{fname}: lost inf"
                else:
                    assert got == want, f"row ({r_row.iteration},{r_row.cluster_a},{r_row.cluster_b}) field {fname!r}: got {got!r} want {want!r}"

        # ---- clusters: base & final cluster_id sets match in-memory state -------
        read_base = store.clusters("base")
        assert read_base.n_clusters == result.cluster_result.n_clusters
        assert set(read_base.clusters.keys()) == set(result.cluster_result.clusters.keys())
        if result.merged_cluster_result is not None:
            read_final = store.clusters("final")
            assert read_final.n_clusters == result.merged_cluster_result.n_clusters
            assert set(read_final.clusters.keys()) == set(result.merged_cluster_result.clusters.keys())

        # ---- crops: every face's crop file exists and resolves --------------------
        for f in result.faces:
            p = store.crop_path(f.face_id)
            assert p.is_file(), f"crop missing for face_id {f.face_id}: {p}"

        # ---- iteration_count: max(merge_log iteration) ---------------------------
        expected_iters = max((int(e["iteration"]) for e in in_mem_log), default=0)
        assert store.iteration_count() == expected_iters

    def test_v4_db_merge_decisions_match_legacy_log(self, merged_pipeline_result):
        """spec-030: v4 DB has every merge_log row preserved with all 28 fields,
        and n_merges in run_metadata matches actually_merged=True count."""
        import sqlite3
        from face_cluster.types import MergeDecisionRow
        v4 = merged_pipeline_result.output_dir / "_v4"
        legacy_log = merged_pipeline_result.merge_log or []

        conn = sqlite3.connect(v4 / "face_clustering.db")
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM merge_decisions").fetchall()
        cols = [r[1] for r in conn.execute("PRAGMA table_info(merge_decisions)").fetchall()]
        meta = dict(conn.execute("SELECT * FROM run_metadata").fetchone())
        conn.close()

        assert tuple(cols) == MergeDecisionRow.field_names()
        assert len(rows) == len(legacy_log)
        n_actually_merged_db = sum(1 for r in rows if r["actually_merged"])
        n_actually_merged_log = sum(1 for e in legacy_log if e.get("actually_merged"))
        assert n_actually_merged_db == n_actually_merged_log == meta["n_merges"]

    def test_every_merge_log_row_has_actually_merged(self, merged_pipeline_result):
        """spec-030 / SIGHTING-058 producer-side contract: every row written by
        ConservativeMerger must carry actually_merged, including the terminal
        iteration where no valid merges existed (regression for early-return
        path that previously skipped the stamping)."""
        log = merged_pipeline_result.merge_log or []
        missing = [(i, e.get("iteration"), e.get("action"))
                   for i, e in enumerate(log)
                   if "actually_merged" not in e]
        assert not missing, (
            f"merge_log rows missing actually_merged: {missing}"
        )

    def test_base_result_unchanged(self, merged_pipeline_result):
        """Base cluster_result must still be present and valid."""
        cr = merged_pipeline_result.cluster_result
        assert cr is not None
        assert cr.n_clusters >= 1

    def test_merged_cluster_count_lte_base(self, merged_pipeline_result):
        """Merging can only reduce cluster count, never increase it."""
        result = merged_pipeline_result
        if result.merged_cluster_result is None:
            pytest.skip("No merges performed on this test data")
        assert result.merged_cluster_result.n_clusters <= result.cluster_result.n_clusters

    def test_loader_round_trip(self, merged_pipeline_result, tmp_path):
        """load_pipeline_result must reconstruct the merged result from disk."""
        loaded = load_pipeline_result(merged_pipeline_result.output_dir)
        assert loaded.merged_cluster_result is not None
        assert loaded.merged_cluster_result.n_clusters == merged_pipeline_result.merged_cluster_result.n_clusters
        assert loaded.merge_log is not None


class ut_ReclusterE2E:
    """E2E tests: recluster() from an existing run directory."""

    @pytest.fixture(scope="class")
    def source_result(self, tmp_path_factory):
        """Run base pipeline once."""
        if not TEST_DATA_DIR.exists():
            pytest.fail(f"Test data directory not found: {TEST_DATA_DIR}")
        tmp = tmp_path_factory.mktemp("recluster_source")
        return FaceClusteringPipeline().run(
            PipelineConfig.full_run(TEST_DATA_DIR, tmp / "source")
        )

    def test_recluster_produces_valid_output(self, source_result, tmp_path):
        result = FaceClusteringPipeline().run(
            PipelineConfig.recluster(source_result.output_dir, tmp_path / "recluster",
                                     K=3, distance_threshold=0.4)
        )
        assert result is not None
        assert result.cluster_result.n_clusters >= 1

    def test_recluster_writes_required_files(self, source_result, tmp_path):
        result = FaceClusteringPipeline().run(
            PipelineConfig.recluster(source_result.output_dir, tmp_path / "recluster2")
        )
        out = result.output_dir
        for fname in ("faces.csv", "clusters.csv", "pipeline_run.json", "crop_manifest.json"):
            assert (out / fname).exists(), f"Missing: {fname}"

    def test_recluster_pipeline_run_has_source_run_field(self, source_result, tmp_path):
        result = FaceClusteringPipeline().run(
            PipelineConfig.recluster(source_result.output_dir, tmp_path / "recluster3")
        )
        rec = json.loads((result.output_dir / "pipeline_run.json").read_text())
        assert rec.get("mode") == "recluster"
        assert "source_run" in rec

    def test_recluster_with_merge(self, source_result, tmp_path):
        result = FaceClusteringPipeline().run(
            PipelineConfig.recluster(source_result.output_dir, tmp_path / "recluster_merge",
                                     merge_enabled=True)
        )
        assert hasattr(result, "merged_cluster_result")
