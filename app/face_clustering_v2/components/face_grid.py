"""spec-045 Phase 6 — face thumbnail grid (8 cols).

Used by Cluster Analysis (all faces) and reused by future Gallery /
Face Analysis tabs. Takes a concrete :class:`ClusterView` (SIGHTING-079
sync rewrite).
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from face_cluster.run_layout import crop_path
from face_cluster.views.cluster_view import ClusterView

GRID_COLS = 8


def render_face_grid(view: ClusterView, *, run_dir: Path) -> None:
    """Render an 8-column grid of face thumbnails, exemplars first.

    Args:
        view:    the computed ClusterView for the current cluster.
        run_dir: parent run dir; crops live at ``run_dir / crops / face_{id:04d}.jpg``.
    """
    faces = view.faces
    if not faces:
        st.caption("No faces in this cluster.")
        return

    for i in range(0, len(faces), GRID_COLS):
        cols = st.columns(GRID_COLS)
        for j, face in enumerate(faces[i : i + GRID_COLS]):
            with cols[j]:
                # Crop path convention owned by face_cluster.run_layout.crop_path
                # (spec-066 D3 — was hardcoded here and in face_analysis_tab).
                crop = crop_path(run_dir, face.face_id)
                if crop.is_file():
                    # Defensive: bad / corrupted crop must NOT crash the whole
                    # page (PIL raises UnidentifiedImageError for empty / non-image
                    # files; that error bubbles up to st.exception and red-boxes
                    # the whole render). Skip silently — caption still shows.
                    try:
                        st.image(str(crop), width=110)
                    except Exception:
                        pass
                role_tag = {"exemplar": "EX", "core": "", "attached": "·"}.get(face.role, "")
                outlier_tag = "!" if face.is_outlier else ""
                area_tag = f" A={face.area_ratio:.1%}" if face.area_ratio is not None else ""
                st.caption(f"`face_{face.face_id:04d}` {role_tag}{outlier_tag} d={face.dist_to_exemplar:.3f}{area_tag}")
                # spec-080: select the face AND switch to the Face Analysis view.
                if st.button("Open", key=f"open_face_{face.face_id}"):
                    st.session_state["selected_face_id"] = int(face.face_id)
                    from app.face_clustering_v2._nav import navigate_to
                    navigate_to("Face Analysis")
