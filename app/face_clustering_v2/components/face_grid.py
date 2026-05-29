"""spec-045 Phase 6 — face thumbnail grid (8 cols).

Used by Cluster Analysis (all faces) and reused by future Gallery /
Face Analysis tabs. Stateless given an :class:`AsyncHandle[ClusterView]`.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from face_cluster.views._async import AsyncHandle
from face_cluster.views.cluster_view import ClusterView

GRID_COLS = 8


def render_face_grid(handle: AsyncHandle[ClusterView], *, run_dir: Path) -> None:
    """Render an 8-column grid of face thumbnails, exemplars first.

    Args:
        handle:  the in-flight ClusterView compute.
        run_dir: parent run dir; crops live at ``run_dir / crops / face_{id:04d}.jpg``.
    """
    if handle.poll() != "done" or handle.result is None:
        return
    view: ClusterView = handle.result
    faces = view.faces
    if not faces:
        st.caption("No faces in this cluster.")
        return

    crops_dir = run_dir / "crops"
    for i in range(0, len(faces), GRID_COLS):
        cols = st.columns(GRID_COLS)
        for j, face in enumerate(faces[i : i + GRID_COLS]):
            with cols[j]:
                crop = crops_dir / f"face_{face.face_id:04d}.jpg"
                if crop.exists():
                    st.image(str(crop), width=110)
                role_tag = {"exemplar": "EX", "core": "", "attached": "·"}.get(face.role, "")
                outlier_tag = "!" if face.is_outlier else ""
                st.caption(f"`face_{face.face_id:04d}` {role_tag}{outlier_tag} d={face.dist_to_exemplar:.3f}")
