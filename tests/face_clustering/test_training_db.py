"""Tests for face_cluster.training_db — SQLite training data storage."""

import json
import pytest
import pandas as pd
from pathlib import Path

from face_cluster.training_db import (
    init_training_table,
    upsert_training_samples,
    load_training_data,
    training_data_summary,
)


def _sample(run_id="run1", cluster_a=0, cluster_b=1, label=1):
    return {
        "run_id":          run_id,
        "album_path":      "/tmp/album",
        "cluster_a":       cluster_a,
        "cluster_b":       cluster_b,
        "label":           label,
        "feature_version": 3,
        "features_json":   json.dumps({"min_exemplar_dist": 0.25, "size_a": 5, "size_b": 3}),
        "output_dir":      "/tmp/output",
        "exemplar_ids":    json.dumps({"a": [0, 1], "b": [2, 3]}),
        "saved_at":        "2026-04-16T10:00:00",
    }


class ut_TrainingDB:

    def test_upsert_and_load_roundtrip(self, tmp_path):
        db = tmp_path / "test.db"
        samples = [_sample("run1", 0, 1, label=1), _sample("run1", 2, 3, label=0)]
        n = upsert_training_samples(samples, db_path=db)

        assert n == 2
        df = load_training_data(db)
        assert len(df) == 2
        assert set(df["run_id"]) == {"run1"}
        # features_json must be expanded into columns
        assert "min_exemplar_dist" in df.columns
        assert "size_a" in df.columns
        assert df["min_exemplar_dist"].iloc[0] == pytest.approx(0.25)

    def test_summary_counts(self, tmp_path):
        db = tmp_path / "test.db"
        upsert_training_samples([
            _sample("run1", 0, 1, label=1),
            _sample("run1", 2, 3, label=0),
            _sample("run2", 0, 1, label=1),
        ], db_path=db)

        s = training_data_summary(db)
        assert s["total_rows"]  == 3
        assert s["n_approved"]  == 2
        assert s["n_rejected"]  == 1
        assert s["n_unlabeled"] == 0
        assert s["n_runs"]      == 2
        assert s["latest_feature_version"] == 3

    def test_upsert_is_idempotent(self, tmp_path):
        db = tmp_path / "test.db"
        s = _sample("run1", 0, 1, label=1)
        upsert_training_samples([s], db_path=db)
        upsert_training_samples([s], db_path=db)  # same primary key — should overwrite

        df = load_training_data(db)
        assert len(df) == 1, "Duplicate upsert must not create a second row"

    def test_upsert_overwrites_label(self, tmp_path):
        """Re-saving a pair with a different label (changed decision) updates the row."""
        db = tmp_path / "test.db"
        upsert_training_samples([_sample("run1", 0, 1, label=1)], db_path=db)
        upsert_training_samples([_sample("run1", 0, 1, label=0)], db_path=db)

        df = load_training_data(db)
        assert len(df) == 1
        assert int(df["label"].iloc[0]) == 0

    def test_load_empty_returns_dataframe(self, tmp_path):
        db = tmp_path / "test.db"
        df = load_training_data(db)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_summary_on_empty_db(self, tmp_path):
        db = tmp_path / "test.db"
        s = training_data_summary(db)

        assert s["total_rows"] == 0

    def test_unlabeled_sample(self, tmp_path):
        db = tmp_path / "test.db"
        s = _sample("run1", 0, 1, label=None)
        s["label"] = None
        upsert_training_samples([s], db_path=db)

        summary = training_data_summary(db)
        assert summary["n_unlabeled"] == 1
        assert summary["n_approved"]  == 0
