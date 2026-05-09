"""Tests for face_cluster.export.export_results."""
import json
import logging
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from face_cluster.types import FaceRecord, ClusterResult, GateResult, QualityVerdict
from face_cluster.config import PipelineConfig
from face_cluster.export import export_results, export_merged_results, save_merge_decisions, load_merge_decisions


def make_face(face_id: int, image_path: str = "test.jpg", is_core: bool = True) -> FaceRecord:
    return FaceRecord(
        face_id=face_id,
        image_id="test.jpg",
        bbox=(0, 0, 100, 100),
        area=10000.0,
        blur_score=100.0,
        image_path=image_path,
        is_core=is_core,
    )


def make_cluster_result(face_indices_per_cluster: dict) -> ClusterResult:
    """Build a minimal ClusterResult from {cluster_id: [face_indices]}."""
    all_labels = []
    clusters = {}
    for cid, indices in face_indices_per_cluster.items():
        clusters[cid] = indices
    n_clusters = len(clusters)
    labels = np.array([0] * sum(len(v) for v in clusters.values()))
    return ClusterResult(
        labels=labels,
        clusters=clusters,
        cluster_stats={cid: {"diameter": 0.2} for cid in clusters},
        exemplars={cid: [] for cid in clusters},
        n_clusters=n_clusters,
        n_noise=0,
    )


def test_faces_csv_has_required_columns(tmp_path):
    """faces.csv must contain face_id, image_path, cluster_id, is_core, blur_score."""
    faces = [make_face(0), make_face(1)]
    cluster_result = make_cluster_result({0: [0, 1]})
    export_results(faces, cluster_result, {}, tmp_path, PipelineConfig(), "test_album")
    df = pd.read_csv(tmp_path / "faces.csv")
    for col in ["face_id", "image_path", "cluster_id", "is_core", "blur_score"]:
        assert col in df.columns, f"Missing required column: {col}"


def test_no_null_image_paths_raises_warning(tmp_path, caplog):
    """Face with image_path=None should log a warning."""
    faces = [make_face(0, image_path=None)]
    cluster_result = make_cluster_result({0: [0]})
    with caplog.at_level(logging.WARNING, logger="face_cluster.export"):
        export_results(faces, cluster_result, {}, tmp_path, PipelineConfig(), "test_album")
    assert any("null image_path" in record.message for record in caplog.records), (
        "Expected warning about null image_path not found in logs"
    )


def test_summary_json_has_source_album(tmp_path):
    """export_summary.json must contain source_album with correct value."""
    faces = [make_face(0)]
    cluster_result = make_cluster_result({0: [0]})
    export_results(faces, cluster_result, {}, tmp_path, PipelineConfig(), "my_test_album")
    with open(tmp_path / "export_summary.json") as f:
        summary = json.load(f)
    assert "source_album" in summary
    assert summary["source_album"] == "my_test_album"


def test_cluster_ids_in_csv_match_cluster_result(tmp_path):
    """cluster_id values in faces.csv should match what's in ClusterResult."""
    faces = [make_face(0), make_face(1), make_face(2)]
    # cluster 0 has face indices 0,1; cluster 1 has face index 2
    cluster_result = make_cluster_result({0: [0, 1], 1: [2]})
    export_results(faces, cluster_result, {}, tmp_path, PipelineConfig(), "test_album")
    df = pd.read_csv(tmp_path / "faces.csv")
    # face at index 0 -> cluster 0, face at index 2 -> cluster 1
    assert df[df["face_id"] == 0]["cluster_id"].iloc[0] == 0
    assert df[df["face_id"] == 2]["cluster_id"].iloc[0] == 1


def test_exemplar_face_ids_are_actual_face_ids_not_graph_indices(tmp_path):
    """SIGHTING-016: exemplar_face_ids in clusters.csv must be real face_ids,
    not graph-local core-set indices.

    Scenario: 5 faces, 3 core (indices 0,2,4 in faces list).
    Graph-local nodes 0,1,2 map to face-list indices 0,2,4 via core_indices.
    Cluster 0 has graph-nodes [0,1], exemplar is graph-node [1].
    Cluster 1 has graph-node [2], exemplar is graph-node [2].

    Expected in clusters.csv:
      Cluster 0: exemplar_face_ids = [20]  (faces[2].face_id, NOT graph-node 1)
      Cluster 1: exemplar_face_ids = [40]  (faces[4].face_id, NOT graph-node 2)
    """
    faces = [
        make_face(0),       # core, index 0
        make_face(10, is_core=False),  # holdout, index 1
        make_face(20),      # core, index 2
        make_face(30, is_core=False),  # holdout, index 3
        make_face(40),      # core, index 4
    ]
    core_indices = [0, 2, 4]  # graph-local 0->faces[0], 1->faces[2], 2->faces[4]

    cluster_result = ClusterResult(
        labels=np.array([0, -1, 0, -1, 1]),
        clusters={0: [0, 1], 1: [2]},    # graph-local indices
        cluster_stats={0: {"diameter": 0.15}, 1: {"diameter": 0.1}},
        exemplars={0: [1], 1: [2]},       # graph-local: exemplar of C0 is node 1, C1 is node 2
        n_clusters=2,
        n_noise=0,
    )

    export_results(faces, cluster_result, {}, tmp_path, PipelineConfig(),
                   "test_album", core_indices=core_indices)

    cdf = pd.read_csv(tmp_path / "clusters.csv")
    fdf = pd.read_csv(tmp_path / "faces.csv")

    for _, row in cdf.iterrows():
        cid = int(row["cluster_id"])
        exemplar_ids = json.loads(row["exemplar_face_ids"])
        member_face_ids = fdf[fdf["cluster_id"] == cid]["face_id"].tolist()
        assert len(exemplar_ids) > 0, f"Cluster {cid} has no exemplars"
        for eid in exemplar_ids:
            assert eid in member_face_ids, (
                f"Cluster {cid}: exemplar face_id {eid} not in members {member_face_ids}. "
                f"Likely writing graph-local indices instead of face_ids."
            )

    # Verify exact expected values
    c0 = cdf[cdf["cluster_id"] == 0].iloc[0]
    c1 = cdf[cdf["cluster_id"] == 1].iloc[0]
    assert json.loads(c0["exemplar_face_ids"]) == [20], "C0 exemplar should be face_id 20"
    assert json.loads(c1["exemplar_face_ids"]) == [40], "C1 exemplar should be face_id 40"


# ---------------------------------------------------------------------------
# ut_MergeDecisions — writer-reader contract for merge_decisions.json
# ---------------------------------------------------------------------------

_SAMPLE_DECISIONS = [
    {
        "cluster_a": 3,
        "cluster_b": 7,
        "decision": "approve",
        "n_gates_passed": 4,
        "exemplar_dist": 0.312,
        "threshold_used": 0.35,
        "support": 4,
        "required_support": 2,
        "margin_gap": 0.08,
        "post_diameter": 0.41,
        "run_id": "test_run",
        "timestamp": "2026-04-12T14:30:00",
    },
    {
        "cluster_a": 1,
        "cluster_b": 5,
        "decision": "reject",
        "n_gates_passed": 2,
        "exemplar_dist": 0.45,
        "threshold_used": 0.35,
        "support": 1,
        "required_support": 2,
        "margin_gap": None,
        "post_diameter": 0.60,
        "run_id": "test_run",
        "timestamp": "2026-04-12T14:30:00",
    },
]


class ut_MergeDecisions:

    def test_save_load_roundtrip(self, tmp_path):
        """save_merge_decisions + load_merge_decisions must round-trip all fields."""
        save_merge_decisions(_SAMPLE_DECISIONS, tmp_path)
        loaded = load_merge_decisions(tmp_path)

        assert loaded is not None
        assert len(loaded) == len(_SAMPLE_DECISIONS)

        for orig, got in zip(_SAMPLE_DECISIONS, loaded):
            assert got["cluster_a"] == orig["cluster_a"]
            assert got["cluster_b"] == orig["cluster_b"]
            assert got["decision"] == orig["decision"]
            assert isinstance(got["n_gates_passed"], int)
            assert isinstance(got["exemplar_dist"], float)
            assert isinstance(got["threshold_used"], float)
            assert got["run_id"] == orig["run_id"]
            assert got["timestamp"] == orig["timestamp"]

    def test_load_missing_returns_none(self, tmp_path):
        """load_merge_decisions returns None if the file does not exist."""
        result = load_merge_decisions(tmp_path)
        assert result is None

    def test_file_is_valid_json_list(self, tmp_path):
        """File must be a JSON array readable by json.load."""
        import json as _json
        save_merge_decisions(_SAMPLE_DECISIONS, tmp_path)
        with open(tmp_path / "merge_decisions.json", encoding="utf-8") as f:
            raw = _json.load(f)
        assert isinstance(raw, list)
        assert len(raw) == 2

    def test_malformed_json_raises_value_error(self, tmp_path):
        """load_merge_decisions must raise ValueError on malformed JSON."""
        (tmp_path / "merge_decisions.json").write_text("not-valid-json", encoding="utf-8")
        with pytest.raises(ValueError, match="Malformed"):
            load_merge_decisions(tmp_path)

    def test_approve_and_reject_decisions_preserved(self, tmp_path):
        """Both 'approve' and 'reject' values survive the round-trip."""
        save_merge_decisions(_SAMPLE_DECISIONS, tmp_path)
        loaded = load_merge_decisions(tmp_path)
        decisions = {(d["cluster_a"], d["cluster_b"]): d["decision"] for d in loaded}
        assert decisions[(3, 7)] == "approve"
        assert decisions[(1, 5)] == "reject"


# ---------------------------------------------------------------------------
# Phase 4: observability columns + provenance (spec 012)
# ---------------------------------------------------------------------------

def _make_face_with_verdict(face_id: int, blur=120.0, is_core=True) -> FaceRecord:
    face = make_face(face_id, is_core=is_core)
    face.blur_score = blur
    face.det_score = 0.92
    face.d10_score = 0.18 if is_core else None
    face.rejection_reason = None if is_core else "blur"
    face.quality_verdict = QualityVerdict(
        gates={"blur": GateResult(value=blur, threshold=50.0, passed=blur >= 50.0)},
        rejection_reason=None if is_core else "blur",
    )
    return face


class ut_ExportFacesExtended:
    def test_quality_columns_written(self, tmp_path):
        faces = [_make_face_with_verdict(0, blur=120.0, is_core=True)]
        cr = make_cluster_result({0: [0]})
        export_results(faces, cr, {}, tmp_path, PipelineConfig(), "album")
        df = pd.read_csv(tmp_path / "faces.csv")
        assert "quality_blur_value" in df.columns
        assert "quality_blur_pass" in df.columns
        assert "quality_rejection_reason" in df.columns
        assert "det_score" in df.columns
        assert "d10_score" in df.columns

    def test_det_score_and_d10_persisted(self, tmp_path):
        faces = [_make_face_with_verdict(0, blur=120.0, is_core=True)]
        cr = make_cluster_result({0: [0]})
        export_results(faces, cr, {}, tmp_path, PipelineConfig(), "album")
        df = pd.read_csv(tmp_path / "faces.csv")
        assert abs(df.iloc[0]["det_score"] - 0.92) < 1e-4
        assert abs(df.iloc[0]["d10_score"] - 0.18) < 1e-4

    def test_rejection_reason_null_for_core(self, tmp_path):
        faces = [_make_face_with_verdict(0, blur=120.0, is_core=True)]
        cr = make_cluster_result({0: [0]})
        export_results(faces, cr, {}, tmp_path, PipelineConfig(), "album")
        df = pd.read_csv(tmp_path / "faces.csv")
        assert pd.isna(df.iloc[0]["quality_rejection_reason"])

    def test_base_clusters_have_origin_base(self, tmp_path):
        faces = [_make_face_with_verdict(0), _make_face_with_verdict(1)]
        cr = make_cluster_result({0: [0, 1]})
        export_results(faces, cr, {}, tmp_path, PipelineConfig(), "album")
        df = pd.read_csv(tmp_path / "clusters.csv")
        assert (df["origin"] == "base").all()
        assert (df["parent_cluster_ids"] == "[]").all()

    def test_roundtrip_via_loader(self, tmp_path):
        from face_cluster.loader import load_pipeline_result
        import json as _json

        faces = [_make_face_with_verdict(0, blur=120.0, is_core=True)]
        cr = make_cluster_result({0: [0]})
        export_results(faces, cr, {}, tmp_path, PipelineConfig(), "album")
        # loader requires pipeline_run.json
        (tmp_path / "pipeline_run.json").write_text(
            _json.dumps({"run_id": "test", "summary": {}, "stages": {}}), encoding="utf-8"
        )
        result = load_pipeline_result(tmp_path)
        loaded_face = result.faces[0]
        assert loaded_face.det_score == pytest.approx(0.92)
        assert loaded_face.quality_verdict is not None
        assert loaded_face.quality_verdict.rejection_reason is None


class ut_ExportClustersProvenance:
    def test_stage_base_created_after_merge(self, tmp_path):
        faces = [make_face(i) for i in range(4)]
        base_cr = make_cluster_result({0: [0, 1], 1: [2, 3]})
        export_results(faces, base_cr, {}, tmp_path, PipelineConfig(), "album")
        (tmp_path / "pipeline_run.json").write_text(
            json.dumps({"run_id": "t", "summary": {}, "stages": {}}), encoding="utf-8"
        )

        # A merge log that merges cluster 1 into cluster 0
        merge_log = [{"cluster_a": 0, "cluster_b": 1, "actually_merged": True,
                      "iteration": 1, "action": "merge"}]
        merged_cr = make_cluster_result({0: [0, 1, 2, 3]})
        export_merged_results(faces, merged_cr, merge_log, tmp_path)

        assert (tmp_path / "clusters_stage_base.csv").exists()
        stage_df = pd.read_csv(tmp_path / "clusters_stage_base.csv")
        assert len(stage_df) == 2  # original two clusters

    def test_merged_cluster_has_auto_merge_origin(self, tmp_path):
        faces = [make_face(i) for i in range(4)]
        base_cr = make_cluster_result({0: [0, 1], 1: [2, 3]})
        export_results(faces, base_cr, {}, tmp_path, PipelineConfig(), "album")
        (tmp_path / "pipeline_run.json").write_text(
            json.dumps({"run_id": "t", "summary": {}, "stages": {}}), encoding="utf-8"
        )

        merge_log = [{"cluster_a": 0, "cluster_b": 1, "actually_merged": True,
                      "iteration": 1, "action": "merge"}]
        merged_cr = make_cluster_result({0: [0, 1, 2, 3]})
        export_merged_results(faces, merged_cr, merge_log, tmp_path)

        df = pd.read_csv(tmp_path / "clusters.csv")
        row = df[df["cluster_id"] == 0].iloc[0]
        assert row["origin"] == "auto_merge"
        parents = json.loads(row["parent_cluster_ids"])
        assert 1 in parents

    def test_no_merge_keeps_base_origin(self, tmp_path):
        faces = [make_face(i) for i in range(2)]
        base_cr = make_cluster_result({0: [0, 1]})
        export_results(faces, base_cr, {}, tmp_path, PipelineConfig(), "album")
        (tmp_path / "pipeline_run.json").write_text(
            json.dumps({"run_id": "t", "summary": {}, "stages": {}}), encoding="utf-8"
        )

        export_merged_results(faces, base_cr, [], tmp_path)
        df = pd.read_csv(tmp_path / "clusters.csv")
        assert (df["origin"] == "base").all()


class ut_LoaderBackwardCompat:
    def test_legacy_faces_csv_loads_without_error(self, tmp_path):
        """A faces.csv without quality columns loads cleanly; new fields are None."""
        import json as _json
        legacy_df = pd.DataFrame([{
            "face_id": 0, "image_path": "a.jpg", "image_id": "a",
            "crop_path": "", "cluster_id": 0, "is_core": True,
            "blur_score": 100.0, "area": 5000.0,
            "yaw": 5.0, "pitch": 2.0, "roll": 1.0,
        }])
        legacy_df.to_csv(tmp_path / "faces.csv", index=False)
        pd.DataFrame([{"cluster_id": 0, "size": 1, "exemplar_face_ids": "[0]", "diameter": 0.1}]).to_csv(
            tmp_path / "clusters.csv", index=False
        )
        (tmp_path / "pipeline_run.json").write_text(
            _json.dumps({"run_id": "legacy", "summary": {}, "stages": {}}), encoding="utf-8"
        )

        from face_cluster.loader import load_pipeline_result
        result = load_pipeline_result(tmp_path)
        face = result.faces[0]
        assert face.det_score is None
        assert face.d10_score is None
        assert face.quality_verdict is None
        assert face.rejection_reason is None


class ut_MergeMetadata:
    def test_merge_metadata_has_threshold_keys(self, tmp_path):
        from face_cluster.merge import ConservativeMerger
        from face_cluster.knn_graph import KNNGraphBuilder
        from face_cluster.clustering import ConnectedComponentsClusterer
        from face_cluster.exemplars import D10ExemplarSelector
        import numpy as np

        # Build a tiny 4-face cluster result
        rng = np.random.default_rng(42)
        emb = rng.standard_normal((4, 512)).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)

        faces = [FaceRecord(face_id=i, image_id=f"img{i}", bbox=(0, 0, 10, 10),
                            embedding_normalized=emb[i]) for i in range(4)]
        cfg = PipelineConfig(K=3, distance_threshold=0.8, merge_enabled=True,
                             blur_min=0.0, min_cluster_size=2)
        builder = KNNGraphBuilder(cfg)
        graph = builder.build_graph(faces, list(range(4)))
        clusterer = ConnectedComponentsClusterer(cfg)
        cr = clusterer.cluster(graph, list(range(4)))
        selector = D10ExemplarSelector(cfg)
        cr, _ = selector.select_exemplars(cr, graph)

        merger = ConservativeMerger(cfg)
        _, _, metadata = merger.merge_clusters_with_logging(cr, graph)

        assert "merge_exemplar_threshold" in metadata
        assert "merge_candidate_threshold" in metadata
