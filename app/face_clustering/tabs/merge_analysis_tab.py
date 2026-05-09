"""Tab 7: Merge Analysis — heuristic and ML merge decision review."""
from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from face_cluster import FaceClusteringPipeline, PipelineConfig, save_manual_merge_snapshot
from face_cluster.analysis_views import MergeAnalysisView, MergeComparisonView, compute_ml_merge_view, compute_pair_feature_contributions
from face_cluster.export import save_merge_decisions, save_merge_features, load_merge_features
from face_cluster.features import FeatureComputer, MergeFeatureContext
from face_cluster.loader import load_pipeline_result
from face_cluster.ml_trainer import MergeTrainer
from face_cluster.pipeline import PipelineResult
from face_cluster.run_naming import RunDirSpec, allocate_run_dir
from face_cluster.training_db import upsert_training_samples, load_model_records

from state import _AsyncState, _invalidate_run_caches, _prefill_approval_decisions
from session_helpers import (
    _render_pending_labels_indicator, _save_pending_labels_to_session,
    _collect_all_merge_labels, _materialize_merge_step,
)
from nav_helpers import _breadcrumb, _no_result
from run_panels import _list_available_runs
from cache_helpers import _crop_for_face
from constants import (
    _GALLERY_FILTERS, _GALLERY_SORTS, _GALLERY_PAGE_SIZE,
    _GROUP_FILTERS, _GROUP_PAGE_SIZE, _CONF_COLOR, _CONF_LABEL,
)

from _merge_helpers import (
    _get_distance_matrix, _get_merge_candidate_threshold, _is_nan,
    _compute_merge_features, _render_pair_crops,
    _exemplar_face_ids_for_cluster, _cluster_size,
)
from _merge_decisions_panel import (
    _render_criteria_reference, _render_merge_run_provenance, _render_merge_summary,
    _render_gate_bottleneck, _render_threshold_distribution, _render_absorbed_clusters,
    _save_merge_features_if_available, _save_approval_decisions, _render_approval_controls,
    _render_grouped_merge_gallery, _render_gate_badges, _render_flat_merge_gallery,
    _render_merge_gallery, _render_merge_analysis,
)
from _merge_ml_panel import (
    _apply_ml_prefill, _render_ml_overview_panel, _render_merge_analysis_ml_mode,
)


# ---------------------------------------------------------------------------
# Main tab render
# ---------------------------------------------------------------------------

def render_merge_comparison_tab():
    """Backward-compatible alias."""
    render_merge_analysis_tab()


def render_merge_analysis_tab():
    st.header("Merge Analysis")
    _breadcrumb()
    remerge_worker: _AsyncState = st.session_state.remerge_worker
    if remerge_worker is not None:
        if remerge_worker.is_running:
            st.info(f"Computing remerge... ({remerge_worker.elapsed_s():.0f}s)")
            time.sleep(0.4)
            st.rerun()
            return
        elif remerge_worker.has_error:
            st.error(f"Remerge failed: {remerge_worker.error}")
            st.session_state.remerge_worker = None
        else:
            st.session_state.pipeline_result       = remerge_worker.result
            st.session_state.remerge_worker        = None
            st.session_state.merge_approval_decisions = {}
            st.session_state.merge_approval_result = None
            st.session_state.pending_candidates    = []
            st.session_state.pending_decisions     = {}
            st.session_state.merge_analysis_worker = None
            st.session_state.merge_pair_features   = None
            _invalidate_run_caches()
            st.rerun()
            return
    result_now = st.session_state.pipeline_result
    runs       = _list_available_runs(complete_only=True)
    merge_runs = [r for r in runs if (Path(r["_dir"]) / "merge_log.json").exists()]
    if not merge_runs and result_now is None:
        _no_result()
        return
    if result_now is not None:
        _render_merge_run_provenance(result_now)
    if merge_runs:
        current_dir = str(result_now.output_dir) if result_now else None
        other_runs  = [r for r in merge_runs if r["_dir"] != current_dir]
        if other_runs:
            with st.expander("Load a different run", expanded=False):
                run_labels  = {
                    r["_dir"]: f"{r['output_folder']}  |  {r['album']}  |  {r['faces'] or '?'} faces  |  {r['started']}"
                    for r in merge_runs
                }
                run_dirs    = list(run_labels.keys())
                default_idx = run_dirs.index(current_dir) if current_dir in run_dirs else 0
                selected_dir = st.selectbox(
                    "Select run", run_dirs, index=default_idx,
                    format_func=lambda d: run_labels[d],
                    key="merge_analysis_run_selector",
                )
                if current_dir is not None and current_dir == selected_dir:
                    st.caption("Already loaded.")
                elif st.button("Load run", key="merge_analysis_load_btn"):
                    loaded = load_pipeline_result(Path(selected_dir))
                    st.session_state.pipeline_result = loaded
                    _invalidate_run_caches()
                    st.rerun()
                    return
    result = st.session_state.pipeline_result
    if result is None:
        _no_result()
        return
    if not st.session_state.merge_approval_decisions:
        _prefill_approval_decisions(result)
    if result.merged_cluster_result is None and not (result.merge_log):
        st.info("No merge data available. Enable **merge_enabled** and re-run or recluster.")
        return
    merge_mode_label = st.radio(
        "Merge source", options=["Heuristic (4-gate)", "ML Model"],
        horizontal=True, key="merge_mode_selector_radio",
    )
    is_ml_mode = "ML" in merge_mode_label
    st.session_state.merge_mode = "ml_model" if is_ml_mode else "heuristic"
    if is_ml_mode:
        _render_merge_analysis_ml_mode(result)
        return
    worker: _AsyncState = st.session_state.merge_analysis_worker
    if worker is None:
        w = _AsyncState()
        st.session_state.merge_analysis_worker = w
        w.start(MergeAnalysisView.compute, result)
        st.rerun()
        return
    if worker.is_running:
        st.info("Computing merge analysis...")
        time.sleep(0.3)
        st.rerun()
        return
    if worker.has_error:
        st.error(f"Merge analysis failed: {worker.error}")
        return
    view: MergeAnalysisView = worker.result
    if st.session_state.merge_pair_features is None:
        st.session_state.merge_pair_features = _compute_merge_features(result)
    _render_merge_analysis(view, result)
