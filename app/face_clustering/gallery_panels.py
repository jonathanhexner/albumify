"""Cluster gallery rendering helpers."""
from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from PIL import Image

from face_cluster.analysis_views import ClusterView
from face_cluster.pipeline import PipelineResult

from state import _AsyncState
from cache_helpers import _crop_for_face
from face_popup import face_detail_btn


def _pil_to_data_url(img: Image.Image, size: int = 48) -> str:
    """Convert a PIL Image to a small base64 data URL for use in ImageColumn."""
    thumb = img.copy()
    thumb.thumbnail((size, size), Image.Resampling.LANCZOS)
    if thumb.mode != "RGB":
        thumb = thumb.convert("RGB")
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _render_cluster_view(view, result: PipelineResult):
    """Render core cluster metrics, exemplars, and face grid for any ClusterView."""
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Faces",          view.size)
    c2.metric("Diameter",       f"{view.diameter:.3f}")
    c3.metric("Avg intra-dist", f"{view.avg_intra_dist:.3f}")
    c4.metric("Exemplars",      len(view.exemplar_face_ids))
    c5.metric("Outliers",       len(view.outlier_face_ids))
    if view.split_signal:
        st.warning("Split signal -- bimodal distribution. Cluster may contain two people.")
    st.subheader("Exemplars")
    if view.exemplar_face_ids:
        n_total_ex = len(view.exemplar_face_ids)
        show_all   = st.checkbox(
            f"Show all {n_total_ex} exemplars", key=f"show_all_ex_cv_{id(view)}"
        ) if n_total_ex > 5 else False
        n_show = n_total_ex if show_all else min(n_total_ex, 5)
        for row_start in range(0, n_show, 5):
            row_fids = view.exemplar_face_ids[row_start:row_start + 5]
            cols     = st.columns(len(row_fids))
            for col_i, (col, fid) in enumerate(zip(cols, row_fids)):
                with col:
                    img = _crop_for_face(fid, result.output_dir)
                    if img:
                        st.image(img, caption=f"face_{fid:04d}")
                    face_detail_btn(fid, key=f"gp_ex_{id(view)}_{row_start}_{col_i}")
    st.subheader("All Faces")
    _GRID = 8
    sorted_faces = sorted(
        view.faces,
        key=lambda fr: (0 if fr.face_id in view.exemplar_face_ids else 1,
                        fr.dist_to_exemplar if fr.dist_to_exemplar is not None else 9.0),
    )
    for row_start in range(0, len(sorted_faces), _GRID):
        row_faces = sorted_faces[row_start:row_start + _GRID]
        cols      = st.columns(_GRID)
        for col_i, (col, fr) in enumerate(zip(cols, row_faces)):
            with col:
                img = _crop_for_face(fr.face_id, result.output_dir)
                st.image(img) if img else st.markdown("_(no crop)_")
                tag      = "EX " if fr.face_id in view.exemplar_face_ids else ("! " if fr.is_outlier else "")
                dist_str = f"{fr.dist_to_exemplar:.3f}" if fr.dist_to_exemplar is not None else "-"
                st.caption(f"{tag}face_{fr.face_id:04d}\nd={dist_str}")
                face_detail_btn(fr.face_id, key=f"gp_cv_{id(view)}_{row_start}_{col_i}")


def _render_cluster_detail(
    result: PipelineResult, cluster_id: int, worker_key: str, debug_worker_key: Optional[str]
):
    """Shared cluster drill-down renderer."""
    worker: _AsyncState = st.session_state.get(worker_key)
    if worker is None or (worker.is_done and worker.result
                          and worker.result.cluster_id != cluster_id):
        w = _AsyncState()
        st.session_state[worker_key] = w
        w.start(ClusterView.compute, result, cluster_id)
        st.rerun()
        return
    if worker.is_running:
        st.info(f"Loading cluster {cluster_id}...")
        time.sleep(0.3)
        st.rerun()
        return
    if worker.has_error:
        st.error(f"Cluster view failed: {worker.error}")
        return
    _render_cluster_view(worker.result, result)


def _render_cluster_gallery(
    overview: "RunOverview",
    result: "PipelineResult",
    tab_key: str,
) -> None:
    """Cluster table with exemplar thumbnails and popup on row selection."""
    if not overview.cluster_rows:
        st.info("No clusters found.")
        return

    exemplars_map = result.cluster_result.exemplars
    faces = result.faces

    rows = overview.cluster_rows
    _N_THUMB = 3  # number of exemplar thumbnails per row
    table_data = []
    for r in rows:
        ex_indices = exemplars_map.get(r.cluster_id, [])
        # If no exemplars, fall back to first cluster members
        if not ex_indices:
            member_indices = result.cluster_result.clusters.get(r.cluster_id, [])
            ex_indices = member_indices[:_N_THUMB]
        # Build thumbnail URLs for up to _N_THUMB exemplars
        thumbs = [None] * _N_THUMB
        for i, idx in enumerate(ex_indices[:_N_THUMB]):
            fid = faces[idx].face_id
            img = _crop_for_face(fid, result.output_dir)
            if img:
                thumbs[i] = _pil_to_data_url(img, size=80)
        row_data = {}
        for i in range(_N_THUMB):
            row_data[f"Ex{i+1}"] = thumbs[i]
        row_data.update({
            "Cluster":     r.cluster_id,
            "Faces":       r.size,
            "Diameter":    round(r.diameter, 3),
            "Avg dist":    round(r.avg_intra_dist, 3),
            "Exemplars":   r.n_exemplars,
            "Nearest C":   r.nearest_cluster_id,
            "Nearest dist": round(r.nearest_cluster_dist, 3),
            "Merge cand?": "yes" if r.merge_candidate else "",
        })
        table_data.append(row_data)
    df = pd.DataFrame(table_data)

    st.caption(f"{len(df)} clusters -- click a row for details")
    event = st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        key=f"{tab_key}_table",
        column_config={
            "Ex1":          st.column_config.ImageColumn("Ex1",         width="small"),
            "Ex2":          st.column_config.ImageColumn("Ex2",         width="small"),
            "Ex3":          st.column_config.ImageColumn("Ex3",         width="small"),
            "Cluster":      st.column_config.NumberColumn("Cluster",    format="%d",   width="small"),
            "Faces":        st.column_config.NumberColumn("Faces",      format="%d",   width="small"),
            "Diameter":     st.column_config.NumberColumn("Diameter",   format="%.3f", width="small"),
            "Avg dist":     st.column_config.NumberColumn("Avg dist",   format="%.3f", width="small"),
            "Exemplars":    st.column_config.NumberColumn("Exemplars",  format="%d",   width="small"),
            "Nearest C":    st.column_config.NumberColumn("Nearest C",  format="%d",   width="small"),
            "Nearest dist": st.column_config.NumberColumn("Nearest dist", format="%.3f", width="small"),
            "Merge cand?":  st.column_config.TextColumn("Merge?",                      width="small"),
        },
    )

    sel_rows = event.selection.get("rows", []) if hasattr(event, "selection") else []
    if not sel_rows:
        return

    selected_cid = int(df.iloc[sel_rows[0]]["Cluster"])
    if st.session_state.get("cluster_popup_id") != selected_cid:
        st.session_state.cluster_popup_id = selected_cid
        st.rerun()


def _make_merged_result(result: PipelineResult) -> Optional[PipelineResult]:
    """Return a PipelineResult with merged_cluster_result as the active clustering."""
    if result.merged_cluster_result is None:
        return None
    merged_summary = {
        **result.summary,
        "n_clusters": result.merged_cluster_result.n_clusters,
        "n_noise":    result.merged_cluster_result.n_noise,
    }
    return PipelineResult(
        faces=result.faces,
        cluster_result=result.merged_cluster_result,
        output_dir=result.output_dir,
        summary=merged_summary,
    )
