"""spec-045 Phase 6 — face thumbnail grid (8 cols).

Used by Cluster Analysis (all faces) and reused by future Gallery /
Face Analysis tabs. Takes a concrete :class:`ClusterView` (SIGHTING-079
sync rewrite).
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

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

    crops_dir = run_dir / "crops"
    for i in range(0, len(faces), GRID_COLS):
        cols = st.columns(GRID_COLS)
        for j, face in enumerate(faces[i : i + GRID_COLS]):
            with cols[j]:
                # STOPGAP (2026-05-29): the v5 writer names crops
                # ``face_{id:04d}_aligned.jpg`` but FaceRow doesn't carry the
                # crop path yet (the proper fix — surfacing FaceRecord.crop_path
                # onto FaceRow — collides with the in-flight spec-056/057/058
                # refactor on FaceRecord/RunStore/Pandera). Hardcoding the
                # suffix here is brittle; revisit after specs 056-058 land.
                # TODO(spec-061 audit): replace with face.crop_path once the
                # refactor settles and the field can be safely added.
                crop = crops_dir / f"face_{face.face_id:04d}_aligned.jpg"
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
                # spec-064 cross-tab nav: writes selected_face_id; user then
                # clicks the Face Analysis tab manually (Streamlit has no
                # programmatic tab-switch API).
                if st.button("Open", key=f"open_face_{face.face_id}"):
                    st.session_state["selected_face_id"] = int(face.face_id)
                    st.rerun()
