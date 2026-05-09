"""Tab 6: Clusters (Merged) — merged cluster overview."""
from __future__ import annotations

import time

import streamlit as st

from face_cluster.analysis_views import RunOverview
from face_cluster.pipeline import PipelineResult

from state import _AsyncState
from nav_helpers import _breadcrumb, _no_result
from gallery_panels import _render_cluster_gallery, _make_merged_result


def render_merged_clusters_tab():
    st.header("Clusters (Merged)")
    _breadcrumb()
    result = st.session_state.pipeline_result
    if result is None:
        _no_result()
        return
    manual_cr = st.session_state.merge_approval_result
    if manual_cr is not None:
        st.info(
            f"Showing **manually approved** merge result: **{manual_cr.n_clusters}** clusters "
            f"(heuristic had {result.merged_cluster_result.n_clusters if result.merged_cluster_result else 'n/a'}). "
            "Go to Merge Analysis tab to change or reset."
        )
        merged_result = PipelineResult(
            faces=result.faces,
            cluster_result=manual_cr,
            output_dir=result.output_dir,
            summary={**result.summary, "n_clusters": manual_cr.n_clusters, "n_noise": manual_cr.n_noise},
        )
    elif result.merged_cluster_result is None:
        st.info(
            "No merged result available. "
            "Enable **merge_enabled** in Pipeline Config and re-run or recluster."
        )
        return
    else:
        merged_result = _make_merged_result(result)

    worker: _AsyncState = st.session_state.merged_overview_worker
    if worker is None:
        w = _AsyncState()
        st.session_state.merged_overview_worker = w
        w.start(RunOverview.compute, merged_result)
        st.rerun()
        return
    if worker.is_running:
        st.info("Computing merged overview (UMAP may take ~10s)...")
        time.sleep(0.5)
        st.rerun()
        return
    if worker.has_error:
        st.error(f"Merged overview failed: {worker.error}")
        return
    overview = worker.result
    c1, c2, c3 = st.columns(3)
    c1.metric("Clusters (merged)", overview.n_clusters,
              delta=overview.n_clusters - result.cluster_result.n_clusters)
    c2.metric("Noise faces", overview.n_noise)
    c3.metric("Clustered faces", overview.n_core - overview.n_noise)
    st.subheader("Merged Clusters")
    _render_cluster_gallery(overview, merged_result, tab_key="merged_gallery")
