"""spec-064 — face detail panel (scores + nearest + verdicts).

Renders the non-image portion of the Face Analysis tab: header, 5-metric
strip, gate rejection banner, nearest faces list. Takes a concrete
:class:`FaceView` — Streamlit-bound only on the render side.
"""
from __future__ import annotations

import streamlit as st

from face_cluster.views.face_view import FaceView


def render_face_detail_panel(view: FaceView) -> None:
    """Render header + 5-metric strip + nearest faces + verdicts."""
    st.subheader(
        f"face_{view.face_id:04d}  ·  cluster {view.cluster_id}  ·  {view.gate_result}"
    )
    yaw, pitch, roll = view.pose if view.pose else (float("nan"),) * 3
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Blur", f"{view.blur_score:.1f}")
    c2.metric("Yaw", f"{yaw:.1f}")
    c3.metric("Pitch", f"{pitch:.1f}")
    c4.metric("Roll", f"{roll:.1f}")
    c5.metric("Area", f"{view.area:.0f}")
    if view.gate_rejection_reason:
        st.warning(f"Rejected: {view.gate_rejection_reason}")
    if view.closest_same_cluster or view.closest_other_clusters:
        with st.expander("Nearest faces", expanded=True):
            for cf in (view.closest_same_cluster + view.closest_other_clusters)[:10]:
                st.text(f"face_{cf.face_id:04d}  C{cf.cluster_id}  d={cf.distance:.3f}")
