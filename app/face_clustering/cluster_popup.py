"""Cluster detail popup -- click a cluster row to open a full detail modal.

Usage from gallery_panels.py:
    On row select in cluster table, set st.session_state.cluster_popup_id = cid

The popup is opened by maybe_show_cluster_popup() called from main.py.
"""
from __future__ import annotations

import urllib.parse
from pathlib import Path
from typing import Optional

import streamlit as st

from face_cluster.analysis_views import ClusterView
from face_cluster.pipeline import PipelineResult

from cache_helpers import _crop_for_face


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

@st.dialog("Cluster Detail", width="large")
def _cluster_detail_dialog(result: PipelineResult, cluster_id: int) -> None:
    output_dir = Path(result.output_dir)

    # Compute ClusterView -- cached in session_state.cluster_popup_cache
    cache: dict = st.session_state.get("cluster_popup_cache", {})
    cache_key = (cluster_id, str(output_dir))
    if cache_key not in cache:
        with st.spinner(f"Loading cluster {cluster_id}..."):
            try:
                view = ClusterView.compute(result, cluster_id)
            except Exception as exc:
                st.error(f"Could not load cluster {cluster_id}: {exc}")
                if st.button("Close", key="cpopup_err_close"):
                    st.session_state.cluster_popup_id = None
                    st.rerun()
                return
        cache[cache_key] = view
        st.session_state.cluster_popup_cache = cache
    else:
        view = cache[cache_key]

    # ---- Metrics row ----------------------------------------------------
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Faces", view.size)
    c2.metric("Diameter", f"{view.diameter:.3f}")
    c3.metric("Avg intra-dist", f"{view.avg_intra_dist:.3f}")
    c4.metric("Exemplars", len(view.exemplar_face_ids))
    c5.metric("Outliers", len(view.outlier_face_ids))

    if view.split_signal:
        st.warning("Split signal -- bimodal distribution. Cluster may contain two people.")

    # ---- All exemplar crops ---------------------------------------------
    st.subheader("Exemplars")
    _GRID = 6
    ex_ids = view.exemplar_face_ids
    for row_start in range(0, len(ex_ids), _GRID):
        row_fids = ex_ids[row_start:row_start + _GRID]
        cols = st.columns(len(row_fids))
        for i, fid in enumerate(row_fids):
            with cols[i]:
                img = _crop_for_face(fid, output_dir)
                if img:
                    st.image(img)
                st.caption(f"face_{fid:04d}")
                if st.button("Detail", key=f"cp_ex_{cluster_id}_{row_start}_{i}",
                             use_container_width=True):
                    st.session_state.cluster_popup_id = None
                    st.session_state.face_popup_id = fid
                    st.rerun()

    # ---- Nearest clusters -----------------------------------------------
    if view.nearest_clusters:
        st.subheader("Nearest Clusters")
        near = view.nearest_clusters[:3]
        cols = st.columns(len(near))
        for i, nc in enumerate(near):
            with cols[i]:
                # Show top exemplar of neighbor cluster
                nc_ex_indices = result.cluster_result.exemplars.get(nc.cluster_id, [])
                if nc_ex_indices:
                    nc_fid = result.faces[nc_ex_indices[0]].face_id
                    nc_img = _crop_for_face(nc_fid, output_dir)
                    if nc_img:
                        st.image(nc_img, width=80)
                st.caption(
                    f"C{nc.cluster_id} ({nc.size} faces)\n"
                    f"dist={nc.min_exemplar_dist:.3f}"
                )

    st.divider()

    # ---- Navigation buttons ---------------------------------------------
    btn_nav, btn_new, btn_close = st.columns(3)
    with btn_nav:
        if st.button("Go to Cluster Analysis", key=f"cpopup_nav_{cluster_id}",
                     use_container_width=True):
            st.session_state.selected_cluster = cluster_id
            st.session_state.cluster_worker = None
            st.session_state.cluster_debug_worker = None
            st.session_state.cluster_popup_id = None
            st.rerun()
    with btn_new:
        params = urllib.parse.urlencode({
            "load_run": str(output_dir),
            "cluster": cluster_id,
        })
        st.markdown(
            f'<a href="?{params}" target="_blank" '
            f'style="display:inline-block;width:100%;text-align:center;'
            f'padding:0.4em 0;border:1px solid #ccc;border-radius:4px;'
            f'text-decoration:none;color:inherit;">'
            f'Open in new window</a>',
            unsafe_allow_html=True,
        )
    with btn_close:
        if st.button("Close", key=f"cpopup_close_{cluster_id}",
                     use_container_width=True):
            st.session_state.cluster_popup_id = None
            st.rerun()


# ---------------------------------------------------------------------------
# Entry point -- called from main.py on every render cycle
# ---------------------------------------------------------------------------

def maybe_show_cluster_popup(result: Optional[PipelineResult]) -> None:
    """Open the cluster detail dialog if cluster_popup_id is set.

    Skips if face_popup_id is also set (face popup takes priority).
    Call this from main.py after maybe_show_face_popup.
    """
    if st.session_state.get("face_popup_id") is not None:
        return
    cluster_id = st.session_state.get("cluster_popup_id")
    if cluster_id is not None and result is not None:
        _cluster_detail_dialog(result, cluster_id)
