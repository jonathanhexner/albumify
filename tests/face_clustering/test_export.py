"""Tests for face_cluster.export.export_results."""
import json
import logging
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from face_cluster.types import FaceRecord, ClusterResult
from face_cluster.config import PipelineConfig
from face_cluster.export import export_results


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
