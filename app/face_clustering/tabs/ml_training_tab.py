"""Tab 10: ML Training — train and evaluate merge classifiers."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from face_cluster import run_history_db
from face_cluster.ml_trainer import MergeTrainer, TrainConfig, FEATURE_GROUPS as _FEATURE_GROUPS
from face_cluster.training_db import (
    load_training_data, training_data_summary, save_model_record, load_model_records,
)

from state import _AsyncState
from run_panels import _render_log_expander


def _run_training(df: pd.DataFrame, config: TrainConfig):
    trainer = MergeTrainer()
    return trainer.train(df, config)


def render_ml_training_tab():
    import plotly.express as px
    import plotly.graph_objects as go

    st.header("ML Training")
    st.caption("Train, evaluate, and save merge classifiers from labeled data.")

    summary       = training_data_summary()
    labeled_count = (summary.get("n_approved") or 0) + (summary.get("n_rejected") or 0)
    if labeled_count < 30:
        st.warning(f"Need at least **30** labeled samples to train. Currently have **{labeled_count}**.")
        return

    df         = load_training_data()
    labeled_df = df[df["label"].notna()].copy() if not df.empty else pd.DataFrame()
    all_runs   = sorted(labeled_df["run_id"].unique().tolist()) if not labeled_df.empty else []

    st.subheader("1. Dataset")
    selected_runs = st.multiselect("Runs to include", options=all_runs, default=all_runs, key="ml_runs")
    if not selected_runs:
        st.warning("Select at least one run.")
        return
    subset = labeled_df[labeled_df["run_id"].isin(selected_runs)]
    n_sel  = len(subset)
    n_app  = int((subset["label"] == 1).sum())
    n_rej  = int((subset["label"] == 0).sum())
    st.caption(f"Selected: **{n_sel}** samples ({n_app} approve / {n_rej} reject)")
    split_strategy = st.radio(
        "Split strategy", ["random_stratified", "by_run"],
        format_func=lambda s: "Random stratified" if s == "random_stratified" else "By run (no album leakage)",
        horizontal=True, key="ml_split_strategy",
    )
    sc1, sc2 = st.columns(2)
    test_pct  = sc1.slider("Test %", 5, 40, 15, 5, key="ml_test_pct") / 100
    val_pct   = sc2.slider("Val %",  5, 30, 15, 5, key="ml_val_pct")  / 100
    train_pct = max(0.0, 1.0 - test_pct - val_pct)
    n_train   = round(n_sel * train_pct)
    n_test    = round(n_sel * test_pct)
    n_val     = round(n_sel * val_pct)
    st.caption(f"Approx split: Train={n_train}  Val={n_val}  Test={n_test}")
    if n_train < 15:
        st.error("Train partition is too small (< 15). Reduce test/val percentages.")
        return

    st.divider()
    st.subheader("2. Model")
    model_type = st.selectbox(
        "Model type", ["logistic_regression", "xgboost", "mlp"],
        format_func=lambda s: {"logistic_regression": "Logistic Regression",
                               "xgboost": "XGBoost", "mlp": "MLP (Neural Net)"}[s],
        key="ml_model_type",
    )
    hyperparams: dict = {}
    if model_type == "logistic_regression":
        with st.expander("Logistic Regression parameters", expanded=True):
            hc1, hc2 = st.columns(2)
            hyperparams["C"]            = hc1.number_input("C (regularization)", 0.001, 100.0, 1.0, key="ml_hp_C")
            hyperparams["class_weight"] = hc2.selectbox("Class weight", ["balanced", None], key="ml_hp_cw")
    elif model_type == "xgboost":
        try:
            import xgboost  # noqa: F401
            with st.expander("XGBoost parameters", expanded=True):
                hc1, hc2, hc3, hc4 = st.columns(4)
                hyperparams["n_estimators"]  = hc1.number_input("n_estimators",  10, 500, 100, key="ml_hp_ne")
                hyperparams["max_depth"]     = hc2.number_input("max_depth",      1,  10,   3, key="ml_hp_md")
                hyperparams["learning_rate"] = hc3.number_input("learning_rate", 0.001, 1.0, 0.1, key="ml_hp_lr")
                hyperparams["subsample"]     = hc4.number_input("subsample",     0.1, 1.0,  0.8, key="ml_hp_ss")
        except ImportError:
            st.error("xgboost is not installed. Run: `pip install xgboost`")
            return
    elif model_type == "mlp":
        with st.expander("MLP parameters", expanded=True):
            hc1, hc2, hc3 = st.columns(3)
            layers_str = hc1.text_input("Hidden layers (comma-sep)", "32,16", key="ml_hp_hl")
            activation = hc2.selectbox("Activation", ["relu", "tanh", "logistic"], key="ml_hp_act")
            lr_val     = hc3.number_input("Learning rate", 0.0001, 0.1, 0.001, format="%.4f", key="ml_hp_mlr")
            hyperparams["hidden_layer_sizes"] = tuple(int(x.strip()) for x in layers_str.split(","))
            hyperparams["activation"]    = activation
            hyperparams["learning_rate"] = lr_val

    st.divider()
    st.subheader("3. Features")
    group_labels = {
        "A":  f"Group A: Distance ({len(_FEATURE_GROUPS['A'])} features)",
        "BC": f"Groups B+C: Geometry ({len(_FEATURE_GROUPS['BC'])} features)",
        "D":  f"Group D: Source images ({len(_FEATURE_GROUPS['D'])} features)",
        "G":  f"Group G: Quality/Pose ({len(_FEATURE_GROUPS['G'])} features)",
    }
    selected_groups = []
    gc = st.columns(len(group_labels))
    for i, (grp, label) in enumerate(group_labels.items()):
        if gc[i].checkbox(label, value=True, key=f"ml_grp_{grp}"):
            selected_groups.append(grp)
    if not selected_groups:
        st.warning("Select at least one feature group.")
        return
    total_feats = sum(len(_FEATURE_GROUPS[g]) for g in selected_groups)
    st.caption(f"Selected: **{total_feats}** features")

    st.divider()
    worker: Optional[_AsyncState] = st.session_state.ml_train_worker
    if st.button("Train Model", type="primary", key="ml_train_btn"):
        config = TrainConfig(
            model_type=model_type, test_fraction=test_pct, val_fraction=val_pct,
            split_strategy=split_strategy, feature_groups=selected_groups, hyperparams=hyperparams,
        )
        w = _AsyncState()
        w.start(_run_training, subset.copy(), config)
        st.session_state.ml_train_worker = w
        st.session_state.ml_train_result = None
        st.rerun()
    if worker is not None:
        if worker.is_running:
            st.info("Training in progress...")
            time.sleep(0.4)
            st.rerun()
        elif worker.has_error:
            st.error(f"Training failed: {worker.error}")
            _render_log_expander(worker.drain_logs(), "Training log")
        elif worker.is_done and st.session_state.ml_train_result is None:
            st.session_state.ml_train_result = worker.result
    result = st.session_state.ml_train_result
    if result is None:
        return

    st.divider()
    st.subheader("4. Results")
    tm  = result.metrics.get("test", {})
    trm = result.metrics.get("train", {})
    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Accuracy",  f"{tm.get('accuracy', 0):.3f}", delta=f"train {trm.get('accuracy', 0):.3f}")
    rc2.metric("F1",        f"{tm.get('f1', 0):.3f}",       delta=f"train {trm.get('f1', 0):.3f}")
    rc3.metric("AUC-ROC",   f"{tm.get('auc', 0):.3f}")
    rc4.metric("Precision", f"{tm.get('precision', 0):.3f}")
    st.caption(f"Data: {result.n_samples} total  |  Train: {result.n_train}  |  Test: {result.n_test}  |  Trained: {result.created_at}")
    cm_col, roc_col = st.columns(2)
    with cm_col:
        st.write("**Confusion Matrix (test)**")
        cm = result.confusion_matrix_test
        if cm:
            labels  = ["Reject (0)", "Approve (1)"]
            fig_cm  = go.Figure(data=go.Heatmap(
                z=cm, x=labels, y=labels, colorscale="Blues", showscale=False,
                text=[[str(v) for v in row] for row in cm], texttemplate="%{text}",
            ))
            fig_cm.update_layout(xaxis_title="Predicted", yaxis_title="Actual",
                                 height=280, margin=dict(t=10, b=40, l=80, r=10))
            st.plotly_chart(fig_cm, use_container_width=True)
    with roc_col:
        st.write("**ROC Curve (test)**")
        roc = result.roc_curve_test
        if roc.get("fpr"):
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines",
                                         name=f"AUC={roc['auc']:.3f}", line=dict(color="#3498db", width=2)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                         line=dict(dash="dash", color="gray"), showlegend=False))
            fig_roc.update_layout(xaxis_title="FPR", yaxis_title="TPR",
                                  height=280, margin=dict(t=10, b=40, l=60, r=10),
                                  legend=dict(x=0.6, y=0.05))
            st.plotly_chart(fig_roc, use_container_width=True)
    if result.feature_importance:
        st.write("**Feature Importance**")
        imp     = sorted(result.feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        imp_df  = pd.DataFrame(imp[:20], columns=["feature", "importance"])
        fig_imp = px.bar(imp_df, x="importance", y="feature", orientation="h",
                         height=max(250, len(imp_df) * 20))
        fig_imp.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(t=10))
        st.plotly_chart(fig_imp, use_container_width=True)

    st.divider()
    st.subheader("5. Save Model")
    model_name = st.text_input(
        "Model name",
        value=f"merge_{result.config.model_type[:3]}_{result.created_at.replace(':', '').replace('-', '').replace('T', '_')}",
        key="ml_model_name",
    )
    if st.button("Save Model", key="ml_save_btn"):
        model_dir = Path.home() / ".sim_bench" / "models"
        trainer   = MergeTrainer()
        path      = trainer.save_model(result, model_dir)
        save_model_record({
            "name":            model_name,
            "model_type":      result.config.model_type,
            "model_path":      str(path),
            "feature_version": 3,
            "n_samples":       result.n_samples,
            "n_train":         result.n_train,
            "n_test":          result.n_test,
            "accuracy":        result.metrics.get("test", {}).get("accuracy"),
            "f1":              result.metrics.get("test", {}).get("f1"),
            "auc_roc":         result.metrics.get("test", {}).get("auc"),
            "created_at":      result.created_at,
            "metadata_json":   json.dumps(result.metrics),
        })
        st.success(f"Model saved: `{path.name}`")

    st.divider()
    st.subheader("6. Saved Models")
    models_df = load_model_records()
    if models_df.empty:
        st.info("No saved models yet.")
    else:
        display_cols = ["name", "model_type", "accuracy", "f1", "auc_roc", "n_samples", "created_at"]
        avail = [c for c in display_cols if c in models_df.columns]
        st.dataframe(models_df[avail], hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("7. Apply to Current Run")
    current_result = st.session_state.pipeline_result
    if current_result is None:
        st.info("Load a run first (History tab) to apply a model.")
        return
    if models_df.empty:
        st.info("No saved models. Save a model above first.")
        return
    model_options  = models_df["name"].tolist() if "name" in models_df.columns else []
    sel_model_name = st.selectbox("Model", model_options, key="ml_apply_model")
    if st.button("Open in Merge Analysis", key="ml_open_in_merge_btn", type="primary"):
        st.session_state["merge_mode_selector_radio"] = "ML Model"
        st.session_state["ml_selected_model"] = sel_model_name
        st.info("Switch to the **Merge Analysis** tab. The model is pre-selected — click **Apply threshold**.")
    with st.expander("Preview predictions (top 5 pairs)", expanded=False):
        pass
    st.divider()
    st.caption("Or run predictions here for a quick comparison view:")
    if st.button("Predict", key="ml_predict_btn") and sel_model_name:
        model_row  = models_df[models_df["name"] == sel_model_name].iloc[0]
        model_path = Path(model_row["model_path"])
        if not model_path.exists():
            st.error(f"Model file not found: {model_path}")
            return
        trainer = MergeTrainer()
        payload = trainer.load_model(model_path)
        _aid    = run_history_db.start_action("model_load", payload={
            "model_name": sel_model_name, "model_path": str(model_path),
            "run_id": current_result.summary.get("run_id"),
            "output_dir": str(current_result.output_dir),
        })
        run_history_db.complete_action(_aid)
        pair_features = st.session_state.merge_pair_features
        if not pair_features:
            st.warning("No candidate pair features in memory. Load a run and open the Merge Analysis tab first.")
            return
        fc      = __import__("face_cluster.features", fromlist=["FeatureComputer"]).FeatureComputer()
        pred_df = fc.to_dataframe(pair_features)
        pred_df = trainer.predict(payload, pred_df)
        decisions = st.session_state.merge_approval_decisions
        pred_df["heuristic"]   = pred_df.apply(
            lambda r: decisions.get((int(r["cluster_a"]), int(r["cluster_b"])), "n/a"), axis=1
        )
        pred_df["ml_decision"] = pred_df["ml_pred"].map({1: "merge", 0: "reject"})
        pred_df["agree"]       = pred_df.apply(
            lambda r: "OK" if (
                (r["ml_decision"] == "merge"  and r["heuristic"] == "approve") or
                (r["ml_decision"] == "reject" and r["heuristic"] == "reject")
            ) else ("?" if r["heuristic"] == "n/a" else "DISAGREE"), axis=1
        )
        show_cols  = ["cluster_a", "cluster_b", "ml_decision", "ml_prob", "heuristic", "agree"]
        avail_pred = [c for c in show_cols if c in pred_df.columns]
        st.dataframe(pred_df[avail_pred].sort_values("ml_prob", ascending=False),
                     hide_index=True, use_container_width=True)
        n_agree    = (pred_df["agree"] == "OK").sum()
        n_disagree = (pred_df["agree"] == "DISAGREE").sum()
        st.caption(f"ML and heuristic agree on {n_agree}/{len(pred_df)} pairs  |  "
                   f"{n_disagree} disagreement(s) — review those manually")
