"""Private ML merge prediction panel for the merge analysis tab."""
from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from face_cluster.analysis_views import MergeAnalysisView, compute_ml_merge_view
from face_cluster.ml_trainer import MergeTrainer
from face_cluster.pipeline import PipelineResult
from face_cluster.training_db import load_model_records

from state import _AsyncState
from _merge_helpers import _compute_merge_features
from _merge_decisions_panel import _render_merge_analysis


def _apply_ml_prefill(view: "MergeAnalysisView", threshold: float) -> None:
    decisions = dict(st.session_state.merge_approval_decisions)
    sources   = dict(st.session_state.merge_decision_sources)
    for row in view.merges + view.rejections:
        if row.ml_prob is None:
            continue
        key = (min(row.cluster_a, row.cluster_b), max(row.cluster_a, row.cluster_b))
        if sources.get(key) == "human":
            continue
        if row.ml_prob >= threshold:
            decisions[key] = "approve"
            sources[key]   = "ml"
        elif row.ml_prob < (1.0 - threshold):
            decisions[key] = "reject"
            sources[key]   = "ml"
        else:
            decisions.pop(key, None)
            sources.pop(key, None)
    st.session_state.merge_approval_decisions = decisions
    st.session_state.merge_decision_sources   = sources


def _render_ml_overview_panel(view, model_name, threshold, model_metadata):
    with st.expander("ML Prediction Overview", expanded=True):
        meta_col1, meta_col2 = st.columns([3, 2])
        with meta_col1:
            st.caption(
                f"**Model**: {model_name}  |  **Threshold**: {threshold:.2f}  |  "
                f"**AUC**: {model_metadata.get('auc_roc') or '?'}  |  "
                f"**F1**: {model_metadata.get('f1') or '?'}  |  "
                f"**N train**: {model_metadata.get('n_samples') or '?'}"
            )
        all_rows   = view.merges + view.rejections
        probs      = [r.ml_prob for r in all_rows if r.ml_prob is not None]
        n_total    = len(probs)
        n_merge    = sum(1 for p in probs if p >= threshold)
        n_reject   = n_total - n_merge
        n_borderline = sum(1 for p in probs if 0.4 < p < 0.6)
        with meta_col2:
            ov_c1, ov_c2, ov_c3 = st.columns(3)
            ov_c1.metric("Merge",      n_merge)
            ov_c2.metric("Reject",     n_reject)
            ov_c3.metric("Borderline", n_borderline)
        if probs:
            import plotly.graph_objects as _go
            bins   = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
            counts = [sum(1 for p in probs if bins[i] <= p < bins[i + 1]) for i in range(len(bins) - 1)]
            labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]
            fig    = _go.Figure(_go.Bar(
                x=labels, y=counts,
                marker_color=["#e74c3c" if bins[i] < 0.5 else "#2ecc71" for i in range(len(bins) - 1)],
            ))
            fig.update_layout(height=180, margin=dict(t=10, b=40, l=40, r=10),
                              xaxis_title="ML probability", yaxis_title="pairs")
            st.plotly_chart(fig, use_container_width=True)
        if len(probs) >= 3:
            import statistics as _stats
            variance = _stats.variance(probs)
            if variance < 0.1:
                st.warning(f"Model shows low confidence separation (variance {variance:.3f} < 0.1).")


def _render_merge_analysis_ml_mode(result: PipelineResult) -> None:
    models_df = load_model_records()
    if models_df.empty:
        st.warning("No saved ML models. Train and save a model in the **ML Training** tab first.")
        return
    model_options = models_df["name"].tolist() if "name" in models_df.columns else []
    ml_col1, ml_col2, ml_col3 = st.columns([2, 2, 1])
    with ml_col1:
        sel_model_name = st.selectbox("Model", model_options, key="ml_selected_model")
    with ml_col2:
        threshold = st.slider("Merge threshold", 0.1, 0.9, 0.5, 0.05, key="ml_threshold")
    with ml_col3:
        wider = st.checkbox("Wider candidates", key="ml_wider_candidates",
                            help="Raises candidate distance cutoff to 0.65")
    candidate_threshold = 0.65 if wider else 0.45
    if st.button("Apply threshold", key="ml_apply_btn", type="primary"):
        if sel_model_name:
            model_row  = models_df[models_df["name"] == sel_model_name].iloc[0]
            model_path = Path(model_row["model_path"])
            if not model_path.exists():
                st.error(f"Model file not found: {model_path}")
                return
            payload    = MergeTrainer.load_model(model_path)
            _threshold = threshold
            _cand      = candidate_threshold
            st.session_state.ml_model_payload = payload

            def _run_ml_predict():
                return compute_ml_merge_view(result, payload, _threshold, _cand)

            w = _AsyncState()
            st.session_state.ml_predict_worker = w
            st.session_state.ml_prefilled      = False
            w.start(_run_ml_predict)
            st.rerun()
        return
    ml_worker: _AsyncState = st.session_state.ml_predict_worker
    if ml_worker is None:
        st.info("Select a model and click **Apply threshold** to see ML predictions.")
        return
    if ml_worker.is_running:
        st.info("Computing ML predictions...")
        time.sleep(0.3)
        st.rerun()
        return
    if ml_worker.has_error:
        st.error(f"ML prediction failed: {ml_worker.error}")
        return
    view: MergeAnalysisView = ml_worker.result
    if not st.session_state.ml_prefilled:
        _apply_ml_prefill(view, threshold)
        st.session_state.ml_prefilled   = True
        st.session_state.ml_merge_view  = view
        st.session_state.ml_pair_features = view.pair_features or {}
    if st.session_state.merge_pair_features is None:
        st.session_state.merge_pair_features = _compute_merge_features(result)
    model_metadata = models_df[models_df["name"] == sel_model_name].iloc[0].to_dict() if sel_model_name else {}
    _render_ml_overview_panel(view, sel_model_name or "", threshold, model_metadata)
    _render_merge_analysis(view, result)
