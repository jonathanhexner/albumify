"""spec-064 — face detail panel (scores + nearest + verdicts).

Renders the non-image portion of the Face Analysis tab: header, 5-metric
strip, gate rejection banner, nearest faces list. Takes a concrete
:class:`FaceView` — Streamlit-bound only on the render side.
"""
from __future__ import annotations

import streamlit as st

from app.face_clustering_v2.components.metric_strip import render_metric_strip
from face_cluster.views.face_metrics import FACE_METRIC_COLUMNS
from face_cluster.views.face_view import FaceView


def render_face_detail_panel(view: FaceView) -> None:
    """Render header + metric strip (spec-072 registry) + nearest + verdicts."""
    st.subheader(
        f"face_{view.face_id:04d}  ·  cluster {view.cluster_id}  ·  {view.gate_result}"
    )
    # spec-072: one declarative metric strip from FACE_METRIC_COLUMNS — same
    # registry the Face Metrics table uses. Metrics FaceView lacks render "—".
    render_metric_strip(view, FACE_METRIC_COLUMNS)
    if view.gate_rejection_reason:
        st.warning(f"Rejected: {view.gate_rejection_reason}")
    if view.closest_same_cluster or view.closest_other_clusters:
        with st.expander("Nearest faces", expanded=True):
            for cf in (view.closest_same_cluster + view.closest_other_clusters)[:10]:
                st.text(f"face_{cf.face_id:04d}  C{cf.cluster_id}  d={cf.distance:.3f}")
