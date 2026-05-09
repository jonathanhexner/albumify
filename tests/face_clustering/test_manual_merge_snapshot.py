"""Writer-reader contract test for manual_merge_snapshot.py.

Calls save_manual_merge_snapshot() then load_pipeline_result() on the output
and asserts field types and values match -- verifying the writer-reader contract
defined in face_cluster/loader.py.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from face_cluster.config import PipelineConfig
from face_cluster.loader import load_pipeline_result
from face_cluster.manual_merge_snapshot import save_manual_merge_snapshot
from face_cluster.types import ClusterResult, FaceRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_faces(n: int) -> list[FaceRecord]:
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
            is_core=(i < 6),  # first 6 are core, last 2 holdout
            image_path=f"/tmp/img_{i}.jpg",
            pose=(5.0, 3.0, 2.0) if i < 6 else None,
        ))
    return faces


def _make_merged_result(n_faces: int, n_clusters: int, core_count: int) -> ClusterResult:
    """Return a ClusterResult in face-list index space (as PipelineResult stores it)."""
    size = core_count // n_clusters
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
# Writer-reader contract tests
# ---------------------------------------------------------------------------

class ut_ManualMergeSnapshotContract:
    """Verify save_manual_merge_snapshot / load_pipeline_result round-trip."""

    @pytest.fixture()
    def snapshot_dir(self, tmp_path):
        """Write a snapshot and return the directory."""
        parent_dir = tmp_path / "parent"
        parent_dir.mkdir()
        snap_dir = tmp_path / "snap"

        faces = _make_faces(8)  # 6 core, 2 holdout
        merged_cr = _make_merged_result(n_faces=8, n_clusters=2, core_count=6)
        config = PipelineConfig()

        # Write a minimal crop_manifest.json in parent to test manifest copy
        crops_dir = parent_dir / "crops"
        crops_dir.mkdir()
        for f in faces[:6]:  # only core faces have crops
            crop_file = crops_dir / f"face_{f.face_id}.jpg"
            crop_file.write_bytes(b"fake")
        manifest = {str(f.face_id): f"crops/face_{f.face_id}.jpg" for f in faces[:6]}
        (parent_dir / "crop_manifest.json").write_text(json.dumps(manifest))

        save_manual_merge_snapshot(
            faces=faces,
            merged_cluster_result=merged_cr,
            approved_pairs=[(0, 1)],
            rejected_pairs=[(2, 3)],
            config=config,
            output_dir=snap_dir,
            parent_output_dir=parent_dir,
            parent_run_id="test_parent_run",
            merge_round=1,
        )
        return snap_dir

    def test_required_files_written(self, snapshot_dir):
        for fname in ("faces.csv", "clusters.csv", "embeddings.npy",
                      "embedding_face_ids.npy", "crop_manifest.json", "pipeline_run.json"):
            assert (snapshot_dir / fname).exists(), f"Missing: {fname}"

    def test_faces_csv_schema(self, snapshot_dir):
        df = pd.read_csv(snapshot_dir / "faces.csv")
        assert set(df.columns) >= {"face_id", "image_path", "cluster_id", "is_core",
                                   "blur_score", "area"}
        assert len(df) == 8  # all faces (core + holdout)
        assert df["face_id"].dtype in (int, np.int64)
        assert df["cluster_id"].dtype in (int, np.int64)

    def test_faces_csv_cluster_assignments(self, snapshot_dir):
        """Approved pair (0,1) merges cluster 0 and 1 -> all 6 core faces in one cluster."""
        df = pd.read_csv(snapshot_dir / "faces.csv")
        # Approved pair (0,1) merges cluster 0 and cluster 1; canonical = min = 0
        core_cluster_ids = df.loc[df["face_id"].isin([0, 1, 2, 3, 4, 5]), "cluster_id"].tolist()
        assert len(set(core_cluster_ids)) == 1, "All core faces must be in the same merged cluster"
        assert core_cluster_ids[0] == 0, "Canonical cluster ID should be min(0, 1) = 0"
        assert df.loc[df["face_id"].isin([6, 7]), "cluster_id"].tolist() == [-1, -1]

    def test_clusters_csv_schema(self, snapshot_dir):
        df = pd.read_csv(snapshot_dir / "clusters.csv")
        assert set(df.columns) >= {"cluster_id", "size", "exemplar_face_ids"}
        # approved pair (0,1) collapses 2 clusters into 1
        assert len(df) == 1
        assert df["cluster_id"].dtype in (int, np.int64)
        assert df["size"].dtype in (int, np.int64)

    def test_exemplar_face_ids_are_valid(self, snapshot_dir):
        """exemplar_face_ids in clusters.csv must be face_id values, not list indices."""
        import ast
        clusters_df = pd.read_csv(snapshot_dir / "clusters.csv")
        faces_df = pd.read_csv(snapshot_dir / "faces.csv")
        valid_face_ids = set(faces_df["face_id"].tolist())
        for _, row in clusters_df.iterrows():
            ex_ids = ast.literal_eval(str(row["exemplar_face_ids"]))
            assert len(ex_ids) >= 1, f"Cluster {row['cluster_id']} has no exemplars"
            for eid in ex_ids:
                assert eid in valid_face_ids, f"exemplar face_id {eid} not in faces.csv"

    def test_embeddings_shape_and_dtype(self, snapshot_dir):
        emb = np.load(snapshot_dir / "embeddings.npy")
        fids = np.load(snapshot_dir / "embedding_face_ids.npy")
        assert emb.dtype == np.float32
        assert fids.dtype == np.int32
        assert emb.shape == (8, 512)
        assert fids.shape == (8,)

    def test_core_embeddings_nonzero(self, snapshot_dir):
        """Core faces must have non-zero embeddings in embeddings.npy."""
        emb = np.load(snapshot_dir / "embeddings.npy")
        fids = np.load(snapshot_dir / "embedding_face_ids.npy")
        for i, fid in enumerate(fids):
            if fid < 6:  # core faces
                assert np.any(emb[i]), f"Core face {fid} has zero embedding"

    def test_crop_manifest_has_absolute_paths(self, snapshot_dir):
        with open(snapshot_dir / "crop_manifest.json") as fh:
            manifest = json.load(fh)
        assert len(manifest) > 0
        for fid, path_str in manifest.items():
            assert Path(path_str).is_absolute(), (
                f"crop_manifest entry for face {fid} is not absolute: {path_str!r}"
            )
            assert Path(path_str).exists(), (
                f"crop_manifest entry for face {fid} does not exist: {path_str}"
            )

    def test_pipeline_run_json_schema(self, snapshot_dir):
        rec = json.loads((snapshot_dir / "pipeline_run.json").read_text())
        assert rec["source_type"] == "manual_merge"
        assert "run_id" in rec
        assert "parent_run_id" in rec
        assert "approved_pairs" in rec
        assert "rejected_pairs" in rec
        assert rec["merge_round"] == 1
        assert rec["status"] == "complete"
        # approved pair (0,1) collapses 2 -> 1 cluster
        assert rec["summary"]["n_clusters"] == 1
        assert rec["summary"]["n_manual_merges"] == 1

    def test_load_pipeline_result_round_trip(self, snapshot_dir):
        """load_pipeline_result must reconstruct faces and cluster_result from snapshot."""
        loaded = load_pipeline_result(snapshot_dir)

        assert len(loaded.faces) == 8
        # approved pair (0,1) collapses 2 clusters into 1
        assert loaded.cluster_result.n_clusters == 1

        fids = [f.face_id for f in loaded.faces]
        assert len(fids) == len(set(fids)), "face_id not unique after round-trip"

        assert sum(1 for f in loaded.faces if f.is_core) == 6
        assert sum(1 for f in loaded.faces if not f.is_core) == 2

    def test_load_pipeline_result_cluster_assignments(self, snapshot_dir):
        """Approved pairs must be reflected in the round-tripped cluster assignments."""
        loaded = load_pipeline_result(snapshot_dir)

        cr = loaded.cluster_result
        for cid, members in cr.clusters.items():
            assert len(members) > 0
        # All 6 core faces must land in the single merged cluster
        assert cr.n_clusters == 1
        assert len(list(cr.clusters.values())[0]) == 6

    def test_load_embeddings_reconstructed(self, snapshot_dir):
        """Core faces must have embeddings after round-trip."""
        loaded = load_pipeline_result(snapshot_dir)
        core_faces = [f for f in loaded.faces if f.is_core]
        for face in core_faces:
            assert face.embedding_normalized is not None, (
                f"Core face {face.face_id} has no embedding after round-trip"
            )
            assert face.embedding_normalized.shape == (512,)


class ut_ManualMergeProvenance:
    """Spec 012: manual_merge snapshot writes origin and parent_cluster_ids (T026)."""

    def test_merged_cluster_has_manual_merge_origin(self, tmp_path):
        parent_dir = tmp_path / "parent"
        parent_dir.mkdir()
        snap_dir = tmp_path / "snap"
        faces = _make_faces(8)
        merged_cr = _make_merged_result(n_faces=8, n_clusters=2, core_count=6)
        (parent_dir / "crop_manifest.json").write_text("{}")

        save_manual_merge_snapshot(
            faces=faces,
            merged_cluster_result=merged_cr,
            approved_pairs=[(0, 1)],
            rejected_pairs=[],
            config=PipelineConfig(),
            output_dir=snap_dir,
            parent_output_dir=parent_dir,
            parent_run_id="parent_123",
            merge_round=1,
        )

        df = pd.read_csv(snap_dir / "clusters.csv")
        assert "origin" in df.columns
        assert "parent_cluster_ids" in df.columns
        # Merged cluster should have manual_merge origin
        row = df.iloc[0]
        assert row["origin"] == "manual_merge"
        parents = json.loads(str(row["parent_cluster_ids"]))
        assert len(parents) > 0
