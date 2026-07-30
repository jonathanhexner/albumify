"""spec-045 Phase 6 — Force Merge expander.

Consumes the typed Service surface — preview returns ForceMergePreview,
apply returns ForceMergeResult. UI emits 3 PASS/FAIL gate badges.
"""
from __future__ import annotations

from typing import List

import streamlit as st

from app.face_clustering_v2.components.metric_strip import render_metric_strip
from face_cluster.views.cluster_analysis import ClusterAnalysisService
from face_cluster.views.metric_specs import FORCE_MERGE_STRIP


def render_force_merge(
    service: ClusterAnalysisService, *, cluster_ids: List[int], current_cluster: int
) -> None:
    """Render the Force Merge expander.

    Side effects on apply: writes ``st.session_state['current_run_dir']`` to
    the new snapshot dir + reruns.
    """
    if len(cluster_ids) < 2:
        return
    with st.expander("Force Merge", expanded=False):
        col_a, col_b = st.columns(2)
        a = col_a.selectbox(
            "Cluster A",
            options=cluster_ids,
            index=cluster_ids.index(current_cluster),
            key="fm_cluster_a",
        )
        b_options = [c for c in cluster_ids if c != a]
        b = col_b.selectbox("Cluster B", options=b_options, key="fm_cluster_b")

        if st.button("Preview Merge", key="fm_preview"):
            try:
                preview = service.preview_force_merge(cluster_a=a, cluster_b=b)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Preview failed: {exc}")
                return
            st.session_state["fm_last_preview"] = preview

        preview = st.session_state.get("fm_last_preview")
        if preview is not None and (preview.cluster_a, preview.cluster_b) == (a, b):
            _render_preview_block(preview)
            if st.button(f"Confirm: merge C{a} + C{b}", type="primary", key="fm_confirm"):
                try:
                    merge_round = int(st.session_state.get("fm_merge_round", 0)) + 1
                    result = service.apply_force_merge(cluster_a=a, cluster_b=b, merge_round=merge_round)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Apply failed: {exc}")
                    return
                st.session_state["fm_merge_round"] = merge_round
                st.session_state["current_run_dir"] = str(result.snapshot_dir)
                st.success(f"Wrote snapshot {result.snapshot_dir.name} (round {result.merge_round}).")
                st.rerun()


def _render_preview_block(preview) -> None:
    """3-gate PASS/FAIL badges + summary line."""
    render_metric_strip(preview, FORCE_MERGE_STRIP, n_cols=3)
    tag = "candidate" if preview.is_candidate else "not a candidate"
    st.caption(
        f"{preview.n_gates_passed}/3 gates pass — {tag}. "
        f"sizes A={preview.cluster_a_size} B={preview.cluster_b_size}."
    )
