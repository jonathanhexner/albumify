"""Contract test: save_merge_features -> load_merge_features round-trip."""

import numpy as np
import pandas as pd
import pytest

from face_cluster.export import save_merge_features, load_merge_features
from face_cluster.features import FeatureComputer, MergeFeatureContext
from face_cluster.types import FaceRecord, ClusterResult


def _make_face(face_id: int, image_path: str = "img.jpg") -> FaceRecord:
    return FaceRecord(
        face_id=face_id,
        image_id=image_path,
        bbox=(0, 0, 100, 100),
        area=5000.0,
        blur_score=100.0,
        image_path=image_path,
        is_core=True,
    )


def _make_context() -> MergeFeatureContext:
    n = 4
    dm = np.zeros((n, n))
    dm[0, 2] = dm[2, 0] = 0.2
    dm[0, 3] = dm[3, 0] = 0.3
    dm[1, 2] = dm[2, 1] = 0.25
    dm[1, 3] = dm[3, 1] = 0.35

    faces = [_make_face(i, f"img_{i}.jpg") for i in range(4)]
    cr = ClusterResult(
        labels=np.array([0, 0, 1, 1]),
        clusters={0: [0, 1], 1: [2, 3]},
        cluster_stats={0: {"diameter": 0.1}, 1: {"diameter": 0.1}},
        exemplars={0: [0, 1], 1: [2, 3]},
        n_clusters=2,
        n_noise=0,
    )
    return MergeFeatureContext(cluster_result=cr, faces=faces, distance_matrix=dm)


def test_save_load_roundtrip(tmp_path):
    """Writer → reader: all expected columns present with correct types."""
    ctx = _make_context()
    fc = FeatureComputer()
    pairs = fc.compute_all_pairs(ctx, candidate_threshold=0.45)
    df = fc.to_dataframe(pairs)

    df["label"] = 1
    df["run_id"] = "test_run"
    df["timestamp"] = "2026-04-14T00:00:00"
    df["feature_version"] = FeatureComputer.VERSION

    save_merge_features(df, tmp_path)
    loaded = load_merge_features(tmp_path)

    assert loaded is not None
    assert len(loaded) == len(df)

    # Required columns always present
    for col in ("cluster_a", "cluster_b", "label", "run_id", "timestamp", "feature_version"):
        assert col in loaded.columns, f"Missing column: {col}"

    # Feature columns have correct types (float or int — not object/str)
    for col in ("min_exemplar_dist", "p50_cross_dist", "support_fraction",
                "size_a", "size_b", "shared_source_images"):
        assert col in loaded.columns, f"Missing feature column: {col}"
        assert loaded[col].dtype.kind in ("f", "i", "u"), (
            f"{col} has unexpected dtype {loaded[col].dtype}"
        )

    # Values are not all null for core features
    assert loaded["min_exemplar_dist"].notna().all()
    assert loaded["size_a"].notna().all()


def test_load_returns_none_if_missing(tmp_path):
    """load_merge_features returns None when file does not exist."""
    assert load_merge_features(tmp_path) is None


def test_feature_version_column(tmp_path):
    """feature_version column matches FeatureComputer.VERSION."""
    ctx = _make_context()
    fc = FeatureComputer()
    pairs = fc.compute_all_pairs(ctx)
    df = fc.to_dataframe(pairs)
    df["label"] = 0
    df["run_id"] = "r"
    df["timestamp"] = "2026-01-01T00:00:00"
    df["feature_version"] = FeatureComputer.VERSION

    save_merge_features(df, tmp_path)
    loaded = load_merge_features(tmp_path)
    assert int(loaded["feature_version"].iloc[0]) == FeatureComputer.VERSION
