"""Tests for face_cluster.ml_trainer — MergeTrainer training / save / load / predict."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from face_cluster.ml_trainer import (
    MergeTrainer,
    TrainConfig,
    TrainResult,
    FEATURE_GROUPS,
    _feature_names_for_groups,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(n: int = 80, seed: int = 42) -> pd.DataFrame:
    """Synthetic DataFrame with enough labeled samples to train."""
    rng = np.random.default_rng(seed)
    n_features = sum(len(v) for v in FEATURE_GROUPS.values())
    feat_names  = _feature_names_for_groups(list(FEATURE_GROUPS.keys()))

    X = rng.uniform(0.0, 1.0, size=(n, len(feat_names)))
    # Labels are correlated with the first feature (min_exemplar_dist)
    labels = (X[:, 0] < 0.45).astype(int)

    df = pd.DataFrame(X, columns=feat_names)
    df["label"]   = labels
    df["run_id"]  = [f"run_{i % 3}" for i in range(n)]
    df["cluster_a"] = list(range(n))
    df["cluster_b"] = [i + n for i in range(n)]
    return df


# ---------------------------------------------------------------------------
# ut_MergeTrainer
# ---------------------------------------------------------------------------

class ut_MergeTrainer:

    def test_train_logistic_regression_basic(self):
        df     = _make_df(80)
        config = TrainConfig(model_type="logistic_regression")
        result = MergeTrainer().train(df, config)

        assert isinstance(result, TrainResult)
        assert result.n_samples == 80
        assert result.n_train > 0
        assert result.n_test  > 0
        assert result.n_train + result.n_test <= 80

    def test_metrics_present(self):
        result = MergeTrainer().train(_make_df(80), TrainConfig())

        for split in ("train", "test"):
            assert split in result.metrics
            for key in ("accuracy", "f1", "precision", "recall"):
                assert key in result.metrics[split]
                assert 0.0 <= result.metrics[split][key] <= 1.0

    def test_confusion_matrix_shape(self):
        result = MergeTrainer().train(_make_df(80), TrainConfig())
        cm = result.confusion_matrix_test
        assert len(cm) == 2
        assert all(len(row) == 2 for row in cm)
        assert sum(sum(row) for row in cm) == result.n_test

    def test_roc_curve_populated(self):
        result = MergeTrainer().train(_make_df(80), TrainConfig())
        roc = result.roc_curve_test
        assert "fpr" in roc and "tpr" in roc and "auc" in roc
        assert len(roc["fpr"]) == len(roc["tpr"])
        assert 0.0 <= roc["auc"] <= 1.0

    def test_feature_importance_keys(self):
        config = TrainConfig(feature_groups=["A"])
        result = MergeTrainer().train(_make_df(80), config)
        expected_feats = set(_feature_names_for_groups(["A"]))
        assert set(result.feature_importance.keys()).issubset(expected_feats)

    def test_feature_group_selection_reduces_features(self):
        cfg_a   = TrainConfig(feature_groups=["A"])
        cfg_all = TrainConfig(feature_groups=["A", "BC", "D", "G"])
        res_a   = MergeTrainer().train(_make_df(80), cfg_a)
        res_all = MergeTrainer().train(_make_df(80), cfg_all)
        assert len(res_a.feature_names) < len(res_all.feature_names)
        assert len(res_a.feature_names) == len(FEATURE_GROUPS["A"])

    def test_too_few_samples_raises(self):
        df = _make_df(20)
        with pytest.raises(ValueError, match="labeled samples"):
            MergeTrainer().train(df, TrainConfig())

    def test_save_load_roundtrip(self, tmp_path):
        result  = MergeTrainer().train(_make_df(80), TrainConfig())
        path    = MergeTrainer.save_model(result, tmp_path)

        assert path.exists()
        assert path.suffix == ".pkl"

        payload = MergeTrainer.load_model(path)
        assert "model"         in payload
        assert "scaler"        in payload
        assert "feature_names" in payload
        assert "metadata"      in payload
        assert payload["metadata"]["model_type"] == "logistic_regression"

    def test_saved_model_feature_names_match(self, tmp_path):
        config  = TrainConfig(feature_groups=["A", "D"])
        result  = MergeTrainer().train(_make_df(80), config)
        path    = MergeTrainer.save_model(result, tmp_path)
        payload = MergeTrainer.load_model(path)

        assert payload["feature_names"] == result.feature_names

    def test_predict_output_shape(self, tmp_path):
        df     = _make_df(80)
        result = MergeTrainer().train(df, TrainConfig())
        path   = MergeTrainer.save_model(result, tmp_path)

        payload  = MergeTrainer.load_model(path)
        new_data = _make_df(10, seed=99)
        out      = MergeTrainer().predict(payload, new_data)

        assert "ml_pred" in out.columns
        assert "ml_prob" in out.columns
        assert len(out) == 10
        assert set(out["ml_pred"].unique()).issubset({0, 1})

    def test_predict_feature_mismatch_raises(self, tmp_path):
        result  = MergeTrainer().train(_make_df(80), TrainConfig())
        path    = MergeTrainer.save_model(result, tmp_path)
        payload = MergeTrainer.load_model(path)

        bad_df = pd.DataFrame({"wrong_col": [0.1, 0.2]})
        with pytest.raises(ValueError, match="Feature mismatch"):
            MergeTrainer().predict(payload, bad_df)

    def test_by_run_split_strategy(self):
        df = _make_df(80)
        # Give samples 4 distinct runs so by_run split has something to work with
        df["run_id"] = [f"run_{i % 4}" for i in range(len(df))]
        config = TrainConfig(split_strategy="by_run")
        result = MergeTrainer().train(df, config)
        assert result.n_train > 0
        assert result.n_test  > 0

    def test_mlp_model_type(self):
        config = TrainConfig(
            model_type="mlp",
            hyperparams={"hidden_layer_sizes": (16,), "max_iter": 100},
        )
        result = MergeTrainer().train(_make_df(80), config)
        assert result.config.model_type == "mlp"
        assert result.metrics["test"]["accuracy"] >= 0.0

    def test_created_at_is_iso8601(self):
        result = MergeTrainer().train(_make_df(80), TrainConfig())
        # Must parse without error
        from datetime import datetime
        datetime.fromisoformat(result.created_at)
