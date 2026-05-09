"""Face detail popup — click any face in the app to open a full detail modal.

Usage from any tab:
    from face_popup import face_detail_btn
    face_detail_btn(face_id, key="some_unique_suffix")

The popup is opened by maybe_show_face_popup() called unconditionally from main.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import streamlit as st

from face_cluster.analysis_views import FaceView
from face_cluster.pipeline import PipelineResult

from cache_helpers import _crop_for_face, _load_faces_df
from quality_panels import _render_quality_report


_COMMENT_MAX = 1024


# ---------------------------------------------------------------------------
# Public trigger helper — import this in any tab that renders face crops
# ---------------------------------------------------------------------------

def face_detail_btn(face_id: int, key: str) -> None:
    """Render a small Detail button under a face crop.

    When clicked, sets face_popup_id in session_state and reruns.
    key must be unique within the page — use a context+index suffix, e.g.
        face_detail_btn(fid, key=f"base_gallery_C{cid}_{i}")
    """
    if st.button("Detail", key=f"popup_btn_{key}", use_container_width=True):
        st.session_state.face_popup_id = face_id
        st.rerun()


# ---------------------------------------------------------------------------
# Comment persistence
# ---------------------------------------------------------------------------

def _load_comments(output_dir: Path) -> dict:
    p = output_dir / "face_comments.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_comment(output_dir: Path, face_id: int, text: str) -> bool:
    """Write comment for face_id into face_comments.json. Returns True on success."""
    if len(text) > _COMMENT_MAX:
        st.error(f"Comment exceeds {_COMMENT_MAX} characters.")
        return False
    comments = _load_comments(output_dir)
    comments[str(face_id)] = text
    try:
        p = output_dir / "face_comments.json"
        p.write_text(json.dumps(comments, indent=2, ensure_ascii=False), encoding="utf-8")
        return True
    except Exception as exc:
        st.warning(f"Could not save comment: {exc}")
        return False


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

@st.dialog("Face Detail", width="large")
def _face_detail_dialog(result: PipelineResult, face_id: int) -> None:
    output_dir = Path(result.output_dir)

    # Compute FaceView — cached in session_state.face_popup_cache
    cache: dict = st.session_state.get("face_popup_cache", {})
    cache_key = (face_id, str(output_dir))
    if cache_key not in cache:
        with st.spinner(f"Analysing face_{face_id:04d}..."):
            try:
                view = FaceView.compute(result, face_id)
            except Exception as exc:
                st.error(f"Could not load face {face_id}: {exc}")
                if st.button("Close", key="popup_err_close"):
                    st.session_state.face_popup_id = None
                    st.rerun()
                return
        cache[cache_key] = view
        st.session_state.face_popup_cache = cache
    else:
        view = cache[cache_key]

    # ---- Header: crop + core attributes --------------------------------
    img_main = _crop_for_face(face_id, output_dir)
    col_img, col_attrs = st.columns([1, 3])
    with col_img:
        if img_main:
            st.image(img_main, caption=f"face_{face_id:04d}")
        else:
            st.caption("(no crop)")
    with col_attrs:
        gate_color = "green" if view.gate_result == "core" else "red"
        st.markdown(f"**Gate**: :{gate_color}[{view.gate_result.upper()}]")
        if view.gate_rejection_reason:
            st.caption(f"Rejection: {view.gate_rejection_reason}")
        cluster_label = str(view.cluster_id) if view.cluster_id >= 0 else "noise/holdout"
        st.markdown(f"**Cluster**: {cluster_label}")
        st.markdown(f"**Source**: `{Path(view.image_path).name}`  rank #{view.rank_in_image}")
        st.markdown(f"**Blur**: {view.blur_score:.1f}  |  **Area**: {int(view.area):,} px2")
        if view.pose:
            yaw, pitch, roll = view.pose
            st.markdown(f"**Pose**: yaw {yaw:.1f}  pitch {pitch:.1f}  roll {roll:.1f}")

    # ---- Quality report (per-gate pass/fail + det_score + d10) ---------
    faces_df = _load_faces_df(output_dir)
    face_row = faces_df[faces_df["face_id"] == face_id]
    if not face_row.empty:
        _render_quality_report(face_row.iloc[0])

    # ---- Closest — Same Cluster ----------------------------------------
    if view.closest_same_cluster:
        st.subheader("Closest - Same Cluster")
        n_same = min(len(view.closest_same_cluster), 5)
        cols = st.columns(n_same)
        for i, cf in enumerate(view.closest_same_cluster[:n_same]):
            with cols[i]:
                img2 = _crop_for_face(cf.face_id, output_dir)
                if img2:
                    st.image(img2)
                st.caption(f"face_{cf.face_id:04d}\nd={cf.distance:.3f}")
                face_detail_btn(cf.face_id, key=f"ps_{face_id}_{i}")
    else:
        st.caption("No same-cluster neighbours (holdout or noise face).")

    # ---- Closest — Other Clusters --------------------------------------
    if view.closest_other_clusters:
        st.subheader("Closest - Other Clusters")
        n_other = min(len(view.closest_other_clusters), 5)
        cols = st.columns(n_other)
        for i, cf in enumerate(view.closest_other_clusters[:n_other]):
            with cols[i]:
                img2 = _crop_for_face(cf.face_id, output_dir)
                if img2:
                    st.image(img2)
                lbl = f"C{cf.cluster_id}" if cf.cluster_id >= 0 else "noise"
                st.caption(f"face_{cf.face_id:04d}\n{lbl}  d={cf.distance:.3f}")
                face_detail_btn(cf.face_id, key=f"po_{face_id}_{i}")

    # ---- Co-image faces -----------------------------------------------
    if view.coimage_faces:
        st.subheader(f"Other Faces in {Path(view.image_path).name}")
        n_co = min(len(view.coimage_faces), 6)
        cols = st.columns(n_co)
        for i, fr in enumerate(view.coimage_faces[:n_co]):
            with cols[i]:
                img2 = _crop_for_face(fr.face_id, output_dir)
                if img2:
                    st.image(img2)
                lbl = f"C{fr.cluster_id}" if fr.cluster_id >= 0 else "holdout"
                st.caption(f"face_{fr.face_id:04d}\n{lbl}")
                face_detail_btn(fr.face_id, key=f"pc_{face_id}_{i}")

    st.divider()

    # ---- Comment -------------------------------------------------------
    st.subheader("Comment")
    comments = _load_comments(output_dir)
    existing = comments.get(str(face_id), "")
    new_comment = st.text_area(
        "Note (e.g. 'falsely filtered', 'bad detection')",
        value=existing,
        max_chars=_COMMENT_MAX,
        key=f"popup_comment_{face_id}",
        label_visibility="collapsed",
        placeholder="Add a note about this face...",
    )

    btn_save, btn_nav, btn_close = st.columns(3)
    with btn_save:
        if st.button("Save comment", key=f"popup_save_{face_id}", use_container_width=True):
            if _save_comment(output_dir, face_id, new_comment):
                st.success("Saved.")
    with btn_nav:
        if st.button("Open in Face Analysis", key=f"popup_nav_{face_id}",
                     use_container_width=True):
            st.session_state.selected_face = face_id
            st.session_state.face_worker   = None
            st.session_state.face_popup_id = None
            st.rerun()
    with btn_close:
        if st.button("Close", key=f"popup_close_{face_id}", use_container_width=True):
            st.session_state.face_popup_id = None
            st.rerun()


# ---------------------------------------------------------------------------
# Entry point — called from main.py on every render cycle
# ---------------------------------------------------------------------------

def maybe_show_face_popup(result: Optional[PipelineResult]) -> None:
    """Open the face detail dialog if face_popup_id is set in session_state.

    Call this unconditionally from main.py after the tab block.
    """
    face_id = st.session_state.get("face_popup_id")
    if face_id is not None and result is not None:
        _face_detail_dialog(result, face_id)
