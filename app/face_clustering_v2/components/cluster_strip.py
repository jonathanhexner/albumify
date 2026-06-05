"""spec-066 — one-cluster thumbnail strip for the Gallery tab.

Render-only. Takes pre-resolved face_ids (the tab calls
``ClusterAnalysisService.exemplar_face_ids``) and draws: cluster header +
"Open in Cluster Analysis" + the exemplar thumbnails wrapped into rows of
``STRIP_COLS``. Faces that failed the quality gate get a warning badge
(G5-flag, read-only). Per-face drill-in lives in Face Metrics (clickable
rows) and Cluster Analysis — Streamlit can't make a bare image clickable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import streamlit as st

from app.face_clustering_v2._telemetry import component_render
from face_cluster.run_layout import crop_path

STRIP_COLS = 8


def render_cluster_strip(
    *,
    cluster_id: int,
    size: int,
    face_ids: Sequence[int],
    run_dir: Path,
    low_quality: set,
) -> None:
    """Render one cluster's strip.

    Args:
        cluster_id: the cluster to display.
        size: total member count (for the header; may exceed len(face_ids)).
        face_ids: pre-resolved ids to show, exemplars first.
        run_dir: parent run dir; crops resolved via ``crop_path``.
        low_quality: face_ids that failed the quality gate (G5-flag badge).

    Side effects: "Open in Cluster Analysis" -> one-shot ``_goto_cluster``
    (+ ``selected_cluster``) + rerun, consumed by cluster_picker.
    """
    header = st.columns([4, 2])
    header[0].markdown(f"**Cluster {cluster_id}** · {size} faces")
    if header[1].button("Open in Cluster Analysis", key=f"gal_open_cluster_{cluster_id}"):
        # One-shot nav request consumed by cluster_picker (writes the picker's
        # widget key — selected_cluster alone can't move a keyed selectbox).
        st.session_state["_goto_cluster"] = int(cluster_id)
        st.session_state["selected_cluster"] = int(cluster_id)
        st.rerun()

    if not face_ids:
        st.caption("No faces to show for this cluster.")
        return

    # Wrap thumbnails into rows of STRIP_COLS so a big cluster's faces stack
    # onto multiple rows instead of squashing into one (user feedback 2026-06-05).
    n_rendered = 0
    for i in range(0, len(face_ids), STRIP_COLS):
        cols = st.columns(STRIP_COLS)
        for col, fid in zip(cols, face_ids[i:i + STRIP_COLS]):
            with col:
                crop = crop_path(run_dir, fid)
                if crop.is_file():
                    try:
                        st.image(str(crop), width=96)
                        n_rendered += 1
                    except Exception:
                        pass
                flag = " :warning:" if fid in low_quality else ""
                st.caption(f"`{fid:04d}`{flag}")

    component_render(
        "gallery", "cluster_strip",
        cluster_id=cluster_id, n_thumbs=n_rendered, n_flagged=len(low_quality),
    )
