"""Tab 8: Face Analysis — per-face attributes and nearest neighbours."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from face_cluster.analysis_views import FaceView

from state import _AsyncState
from nav_helpers import _breadcrumb, _no_result
from cache_helpers import _crop_for_face, _load_faces_df
from gallery_panels import _pil_to_data_url
from quality_panels import _render_quality_report
from face_popup import face_detail_btn


def _build_face_display_df(faces_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Build a display dataframe for the face table with thumbnails."""
    cols = ["face_id", "is_core", "quality_rejection_reason",
            "blur_score", "area", "det_score",
            "yaw", "pitch", "roll",
            "cluster_id", "image_path"]
    available = [c for c in cols if c in faces_df.columns]
    df = faces_df[available].copy()

    # Thumbnail column
    def _thumb(face_id):
        img = _crop_for_face(int(face_id), output_dir)
        return _pil_to_data_url(img, size=60) if img else None
    df["Thumb"] = df["face_id"].apply(_thumb)

    df["Gate"] = df["is_core"].map(lambda v: "CORE" if v else "HOLDOUT")
    df["Image"] = df["image_path"].apply(
        lambda p: Path(str(p)).name if pd.notna(p) and p else ""
    )
    df["Cluster"] = df["cluster_id"].apply(
        lambda c: f"C{int(c)}" if pd.notna(c) and int(c) >= 0 else "noise"
    )

    rename = {
        "face_id":                   "ID",
        "quality_rejection_reason":  "Rejected by",
        "blur_score":                "Blur",
        "area":                      "Area",
        "det_score":                 "Det score",
        "yaw":                       "Yaw",
        "pitch":                     "Pitch",
        "roll":                      "Roll",
    }
    df = df.rename(columns=rename)

    display_cols = ["Thumb", "ID", "Gate", "Rejected by", "Blur", "Area",
                    "Det score", "Yaw", "Pitch", "Roll", "Cluster", "Image"]
    return df[[c for c in display_cols if c in df.columns]]


def render_face_analysis_tab():
    st.header("Face Analysis")
    _breadcrumb()
    result = st.session_state.pipeline_result
    if result is None:
        _no_result()
        return
    faces_df = _load_faces_df(result.output_dir)
    if faces_df.empty:
        st.warning("No faces loaded.")
        return

    # ---- Face selection table -----------------------------------------------
    disp_df = _build_face_display_df(faces_df, Path(result.output_dir))
    face_ids_list = list(faces_df["face_id"].astype(int))

    # Determine pre-select index (set by popup "Open in Face Analysis" or prior selection)
    default_face = st.session_state.selected_face
    if default_face in face_ids_list:
        default_row = face_ids_list.index(default_face)
    else:
        default_row = 0

    st.caption(f"{len(disp_df)} faces — click a row to inspect")
    event = st.dataframe(
        disp_df,
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        key="fa_face_table",
        column_config={
            "Thumb":     st.column_config.ImageColumn("",           width="small"),
            "ID":        st.column_config.NumberColumn("ID",        format="%d", width="small"),
            "Gate":      st.column_config.TextColumn("Gate",        width="small"),
            "Rejected by": st.column_config.TextColumn("Rejected by", width="medium"),
            "Blur":      st.column_config.NumberColumn("Blur",      format="%.1f", width="small"),
            "Area":      st.column_config.NumberColumn("Area",      format="%d",   width="small"),
            "Det score": st.column_config.NumberColumn("Det score", format="%.2f", width="small"),
            "Yaw":       st.column_config.NumberColumn("Yaw",       format="%.1f", width="small"),
            "Pitch":     st.column_config.NumberColumn("Pitch",     format="%.1f", width="small"),
            "Roll":      st.column_config.NumberColumn("Roll",      format="%.1f", width="small"),
            "Cluster":   st.column_config.TextColumn("Cluster",     width="small"),
            "Image":     st.column_config.TextColumn("Image",       width="large"),
        },
    )

    sel_rows = event.selection.get("rows", []) if hasattr(event, "selection") else []
    if sel_rows:
        selected_face = int(disp_df.iloc[sel_rows[0]]["ID"])
    else:
        selected_face = face_ids_list[default_row]

    if selected_face != st.session_state.selected_face:
        st.session_state.selected_face = selected_face
        st.session_state.face_worker   = None

    # ---- Async detail compute -----------------------------------------------
    worker: _AsyncState = st.session_state.face_worker
    if worker is None:
        w = _AsyncState()
        st.session_state.face_worker = w
        w.start(FaceView.compute, result, selected_face)
        st.rerun()
        return
    if worker.is_running:
        st.info(f"Analysing face_{selected_face:04d}...")
        time.sleep(0.4)
        st.rerun()
        return
    if worker.has_error:
        st.error(f"Face analysis failed: {worker.error}")
        return

    # ---- Detail view --------------------------------------------------------
    view = worker.result
    st.divider()
    col_img, col_attrs = st.columns([1, 3])
    with col_img:
        img = _crop_for_face(view.face_id, result.output_dir)
        st.image(img, caption=f"face_{view.face_id:04d}") if img else st.caption("(no crop)")
    with col_attrs:
        gate_color = "green" if view.gate_result == "core" else "red"
        st.markdown(f"**Gate**: :{gate_color}[{view.gate_result.upper()}]")
        if view.gate_rejection_reason:
            st.caption(f"Rejection: {view.gate_rejection_reason}")
        st.markdown(f"**Cluster**: {view.cluster_id if view.cluster_id >= 0 else 'noise/holdout'}")
        st.markdown(f"**Source**: `{Path(view.image_path).name}`  rank #{view.rank_in_image}")
        st.markdown(f"**Blur**: {view.blur_score:.1f}  |  **Area**: {int(view.area):,} px²")
        if view.pose:
            yaw, pitch, roll = view.pose
            st.markdown(f"**Pose**: yaw {yaw:.1f}  pitch {pitch:.1f}  roll {roll:.1f}")
    st.subheader("Closest — Same Cluster")
    if view.closest_same_cluster:
        cols = st.columns(min(len(view.closest_same_cluster), 5))
        for i, cf in enumerate(view.closest_same_cluster):
            with cols[i]:
                img = _crop_for_face(cf.face_id, result.output_dir)
                if img:
                    st.image(img)
                st.caption(f"face_{cf.face_id:04d}\nd={cf.distance:.3f}")
                face_detail_btn(cf.face_id, key=f"fa_same_{selected_face}_{i}")
    else:
        st.info("No same-cluster neighbors.")
    st.subheader("Closest — Other Clusters")
    if view.closest_other_clusters:
        cols = st.columns(min(len(view.closest_other_clusters), 5))
        for i, cf in enumerate(view.closest_other_clusters):
            with cols[i]:
                img = _crop_for_face(cf.face_id, result.output_dir)
                if img:
                    st.image(img)
                lbl = f"C{cf.cluster_id}" if cf.cluster_id >= 0 else "noise"
                st.caption(f"face_{cf.face_id:04d}\n{lbl}  d={cf.distance:.3f}")
                face_detail_btn(cf.face_id, key=f"fa_other_{selected_face}_{i}")
    else:
        st.info("No cross-cluster neighbors.")
    face_row = faces_df[faces_df["face_id"] == selected_face]
    if not face_row.empty:
        _render_quality_report(face_row.iloc[0])
    if view.coimage_faces:
        st.subheader(f"Other Faces in {Path(view.image_path).name}")
        cols = st.columns(min(len(view.coimage_faces), 6))
        for i, fr in enumerate(view.coimage_faces):
            with cols[i]:
                img = _crop_for_face(fr.face_id, result.output_dir)
                if img:
                    st.image(img)
                lbl = f"C{fr.cluster_id}" if fr.cluster_id >= 0 else "holdout"
                st.caption(f"face_{fr.face_id:04d}\n{lbl}")
                face_detail_btn(fr.face_id, key=f"fa_coimg_{selected_face}_{i}")
