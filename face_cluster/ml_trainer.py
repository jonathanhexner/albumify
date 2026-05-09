"""ML merge classifier — train, evaluate, save, and predict.

Wraps sklearn (and optionally xgboost) classifiers.
Keeps face_cluster independent of the main sim_bench ORM stack.

Usage:
    from face_cluster.ml_trainer import MergeTrainer, TrainConfig
    from face_cluster.training_db import load_training_data

    df      = load_training_data()
    config  = TrainConfig(model_type="logistic_regression", feature_groups=["A", "BC"])
    trainer = MergeTrainer()
    result  = trainer.train(df, config)
    path    = MergeTrainer.save_model(result, Path.home() / ".sim_bench/models")
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from face_cluster import run_history_db

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_MIN_SAMPLES = 30


# ---------------------------------------------------------------------------
# Feature group registry — mirrors ClusterPairFeatures field names
# ---------------------------------------------------------------------------

FEATURE_GROUPS: Dict[str, List[str]] = {
    "A": [
        "min_exemplar_dist", "exemplar_dist_mean", "exemplar_dist_std",
        "min_cross_dist", "p10_cross_dist", "p25_cross_dist", "p50_cross_dist",
        "p75_cross_dist", "p90_cross_dist", "cross_dist_iqr",
        "support_fraction", "n_cross_pairs_below_threshold",
    ],
    "BC": [
        "size_a", "size_b", "size_min", "size_ratio", "size_sum",
        "diameter_a", "diameter_b", "diameter_max", "diameter_ratio",
        "post_merge_diameter", "diameter_expansion",
        "mean_intra_dist_a", "mean_intra_dist_b",
        "exemplar_count_a", "exemplar_count_b",
        "t_a", "t_b", "t_local", "t_global",
        "dist_to_threshold_ratio",
    ],
    "D": [
        "n_images_a", "n_images_b", "shared_source_images",
        "shared_source_ratio", "same_image_min_dist",
    ],
    "G": [
        "mean_blur_a", "mean_blur_b", "blur_min_a", "blur_min_b",
        "frontal_frac_a", "frontal_frac_b", "frontal_frac_min",
        "pose_diff", "yaw_std_a", "yaw_std_b",
        "mean_area_a", "mean_area_b", "area_ratio",
    ],
}


# ---------------------------------------------------------------------------
# Configuration + result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    model_type: str = "logistic_regression"    # logistic_regression | xgboost | mlp
    test_fraction: float = 0.15
    val_fraction: float = 0.15
    split_strategy: str = "random_stratified"  # random_stratified | by_run
    cv_folds: int = 5
    feature_groups: List[str] = field(default_factory=lambda: ["A", "BC", "D", "G"])
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    random_state: int = 42


@dataclass
class TrainResult:
    model: Any                             # fitted sklearn/xgboost estimator
    scaler: StandardScaler
    feature_names: List[str]
    metrics: Dict[str, Dict[str, float]]   # {"train": {...}, "test": {...}}
    confusion_matrix_test: List[List[int]]
    roc_curve_test: Dict[str, Any]         # {fpr: [...], tpr: [...], auc: float}
    feature_importance: Dict[str, float]
    config: TrainConfig
    created_at: str
    n_samples: int
    n_train: int
    n_test: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _feature_names_for_groups(groups: List[str]) -> List[str]:
    names: List[str] = []
    for g in groups:
        names.extend(FEATURE_GROUPS.get(g, []))
    return names


def _make_estimator(config: TrainConfig) -> Any:
    hp = config.hyperparams
    if config.model_type == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            class_weight=hp.get("class_weight", "balanced"),
            C=hp.get("C", 1.0),
            random_state=config.random_state,
        )
    elif config.model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed. Run: pip install xgboost"
            ) from exc
        return XGBClassifier(
            eval_metric="logloss",
            use_label_encoder=False,
            n_estimators=hp.get("n_estimators", 100),
            max_depth=hp.get("max_depth", 3),
            learning_rate=hp.get("learning_rate", 0.1),
            subsample=hp.get("subsample", 0.8),
            random_state=config.random_state,
        )
    elif config.model_type == "mlp":
        return MLPClassifier(
            max_iter=hp.get("max_iter", 500),
            hidden_layer_sizes=hp.get("hidden_layer_sizes", (32, 16)),
            activation=hp.get("activation", "relu"),
            learning_rate_init=hp.get("learning_rate", 0.001),
            random_state=config.random_state,
        )
    else:
        raise ValueError(f"Unknown model_type: {config.model_type!r}")


def _extract_importance(
    model: Any, feature_names: List[str], model_type: str
) -> Dict[str, float]:
    try:
        if model_type == "logistic_regression":
            return dict(zip(feature_names, model.coef_[0].tolist()))
        elif model_type == "xgboost":
            return dict(zip(feature_names, model.feature_importances_.tolist()))
        elif model_type == "mlp":
            w = model.coefs_[0]
            return dict(zip(feature_names, np.abs(w).sum(axis=1).tolist()))
    except Exception:
        pass
    return {}


def _eval_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
) -> Dict[str, float]:
    out: Dict[str, float] = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            out["auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass
    return out


def _split_by_run(
    df: pd.DataFrame,
    test_frac: float,
    val_frac: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assign entire runs to train/val/test to prevent album-level data leakage."""
    run_ids = df["run_id"].unique()
    n_total = len(run_ids)
    n_test  = max(1, round(n_total * test_frac))
    n_val   = max(1, round(n_total * val_frac))

    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(run_ids).tolist()

    test_runs  = shuffled[:n_test]
    val_runs   = shuffled[n_test : n_test + n_val]
    train_runs = shuffled[n_test + n_val :]

    return (
        df[df["run_id"].isin(train_runs)],
        df[df["run_id"].isin(val_runs)],
        df[df["run_id"].isin(test_runs)],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class MergeTrainer:
    """Train, evaluate, save, and predict with merge classifiers.

    All methods are stateless; use save_model / load_model for persistence.
    """

    FEATURE_GROUPS = FEATURE_GROUPS

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame, config: TrainConfig) -> TrainResult:
        """Full pipeline: validate → split → scale → fit → evaluate.

        Args:
            df:     DataFrame from load_training_data() — must have a 'label'
                    column and the feature columns named in config.feature_groups.
            config: Training configuration.

        Returns:
            Populated TrainResult.

        Raises:
            ValueError: fewer than _MIN_SAMPLES labeled rows, or feature mismatch.
        """
        _action_id = run_history_db.start_action("ml_train", payload={
            "model_type":      config.model_type,
            "feature_groups":  config.feature_groups,
            "n_labeled":       int(df["label"].notna().sum()) if "label" in df.columns else 0,
            "hyperparams":     config.hyperparams,
        })
        labeled = df[df["label"].notna()].copy()
        if len(labeled) < _MIN_SAMPLES:
            raise ValueError(
                f"Need at least {_MIN_SAMPLES} labeled samples to train; "
                f"found {len(labeled)}."
            )

        feature_names = [
            f for f in _feature_names_for_groups(config.feature_groups)
            if f in labeled.columns
        ]
        if not feature_names:
            raise ValueError(
                "No feature columns found in DataFrame for selected feature groups. "
                f"Groups requested: {config.feature_groups}"
            )

        # --- Split ---
        if config.split_strategy == "by_run" and "run_id" in labeled.columns:
            df_train, _df_val, df_test = _split_by_run(
                labeled, config.test_fraction, config.val_fraction, config.random_state
            )
            X_train = df_train[feature_names].fillna(0.0).values.astype(float)
            y_train = df_train["label"].astype(int).values
            X_test  = df_test[feature_names].fillna(0.0).values.astype(float)
            y_test  = df_test["label"].astype(int).values
        else:
            X = labeled[feature_names].fillna(0.0).values.astype(float)
            y = labeled["label"].astype(int).values
            test_size = config.test_fraction + config.val_fraction
            stratify  = y if len(np.unique(y)) > 1 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                stratify=stratify,
                random_state=config.random_state,
            )

        if len(y_train) == 0 or len(y_test) == 0:
            raise ValueError(
                "Train/test split produced an empty partition. "
                "Try more data or a different split strategy."
            )

        # --- Scale ---
        scaler    = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        # --- Fit ---
        estimator = _make_estimator(config)
        estimator.fit(X_train_s, y_train)

        # --- Predict ---
        y_pred_train = estimator.predict(X_train_s)
        y_pred_test  = estimator.predict(X_test_s)

        y_prob_train: Optional[np.ndarray] = None
        y_prob_test:  Optional[np.ndarray] = None
        if hasattr(estimator, "predict_proba"):
            try:
                y_prob_train = estimator.predict_proba(X_train_s)[:, 1]
                y_prob_test  = estimator.predict_proba(X_test_s)[:, 1]
            except Exception:
                pass

        # --- Metrics ---
        train_metrics = _eval_metrics(y_train, y_pred_train, y_prob_train)
        test_metrics  = _eval_metrics(y_test,  y_pred_test,  y_prob_test)

        # --- ROC curve ---
        roc: Dict[str, Any] = {"fpr": [], "tpr": [], "auc": test_metrics.get("auc", 0.0)}
        if y_prob_test is not None and len(np.unique(y_test)) == 2:
            try:
                fpr, tpr, _ = roc_curve(y_test, y_prob_test)
                roc = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "auc": float(roc_auc_score(y_test, y_prob_test)),
                }
            except Exception:
                pass

        # --- Confusion matrix ---
        cm = confusion_matrix(y_test, y_pred_test).tolist()

        # --- Feature importance ---
        importance = _extract_importance(estimator, feature_names, config.model_type)

        logger.info(
            "Trained %s: acc=%.3f  f1=%.3f  auc=%.3f  (test: n=%d)",
            config.model_type,
            test_metrics["accuracy"],
            test_metrics.get("f1", 0.0),
            test_metrics.get("auc", 0.0),
            len(y_test),
        )

        result = TrainResult(
            model=estimator,
            scaler=scaler,
            feature_names=feature_names,
            metrics={"train": train_metrics, "test": test_metrics},
            confusion_matrix_test=cm,
            roc_curve_test=roc,
            feature_importance=importance,
            config=config,
            created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            n_samples=len(labeled),
            n_train=len(y_train),
            n_test=len(y_test),
        )
        run_history_db.complete_action(_action_id, payload_update={
            "n_train":   len(y_train),
            "n_test":    len(y_test),
            "accuracy":  test_metrics.get("accuracy"),
            "f1":        test_metrics.get("f1"),
            "auc":       test_metrics.get("auc"),
        })
        return result

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, model_payload: Dict, df: pd.DataFrame) -> pd.DataFrame:
        """Apply a loaded model bundle to candidate pairs.

        Args:
            model_payload: dict returned by MergeTrainer.load_model().
            df:            DataFrame with at least the feature columns the model expects.

        Returns:
            Copy of df with added 'ml_pred' (int 0/1) and 'ml_prob' (float) columns.

        Raises:
            ValueError: if required feature columns are missing from df.
        """
        model      = model_payload["model"]
        scaler     = model_payload["scaler"]
        feat_names = model_payload["feature_names"]

        missing = [f for f in feat_names if f not in df.columns]
        if missing:
            raise ValueError(
                f"Feature mismatch — model expects {len(feat_names)} features; "
                f"{len(missing)} missing from DataFrame: {missing[:5]}"
                + ("..." if len(missing) > 5 else "")
            )

        X   = df[feat_names].fillna(0.0).values.astype(float)
        X_s = scaler.transform(X)

        out = df.copy()
        out["ml_pred"] = model.predict(X_s).astype(int)

        if hasattr(model, "predict_proba"):
            try:
                out["ml_prob"] = model.predict_proba(X_s)[:, 1].astype(float)
            except Exception:
                out["ml_prob"] = float("nan")
        else:
            out["ml_prob"] = float("nan")

        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_model(result: TrainResult, model_dir: Path) -> Path:
        """Pickle the model bundle to model_dir. Returns the saved file path."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Timestamp-based filename: merge_lr_20260417_143000.pkl
        ts    = result.created_at.replace(":", "").replace("-", "").replace("T", "_")
        short = result.config.model_type[:3]
        fname = f"merge_{short}_{ts}.pkl"
        path  = model_dir / fname

        payload = {
            "model":         result.model,
            "scaler":        result.scaler,
            "feature_names": result.feature_names,
            "metadata": {
                "model_type":      result.config.model_type,
                "feature_version": 3,
                "n_samples":       result.n_samples,
                "n_train":         result.n_train,
                "n_test":          result.n_test,
                "feature_groups":  result.config.feature_groups,
                "split_strategy":  result.config.split_strategy,
                "hyperparams":     result.config.hyperparams,
                "metrics":         result.metrics,
                "created_at":      result.created_at,
            },
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)

        logger.info("Model saved: %s", path)
        return path

    @staticmethod
    def load_model(model_path: Path) -> Dict:
        """Load a saved model bundle.

        Returns:
            dict with keys: model, scaler, feature_names, metadata.
        """
        with open(model_path, "rb") as fh:
            return pickle.load(fh)
