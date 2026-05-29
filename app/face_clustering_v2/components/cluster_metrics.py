"""spec-045 Phase 6 — cluster-metrics strip + split-signal banner.

Renders the 5-metric headline + an optional "split signal" banner when the
clusterer detected a bimodal intra-cluster distance distribution.

Reads only the AsyncHandle. No DB, no FS.
"""
from __future__ import annotations

import streamlit as st

from face_cluster.views._async import AsyncHandle
from face_cluster.views.cluster_view import ClusterView


def render_cluster_metrics(handle: AsyncHandle[ClusterView]) -> None:
    """Render the metric strip for the cluster the handle is computing.

    States:
        - ``running`` / ``pending``: spinner placeholder.
        - ``failed``: ``st.error`` with the exception message.
        - ``done``: 5-metric strip + provenance + (optional) split-signal banner.
    """
    state = handle.poll()
    if state in ("pending", "running"):
        st.caption("Analysing cluster…")
        return
    if state == "failed":
        st.error(f"Cluster compute failed: {handle.error}")
        return
    if state == "cancelled" or handle.result is None:
        return
    view: ClusterView = handle.result

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Faces", view.size)
    c2.metric("Diameter", f"{view.diameter:.3f}")
    c3.metric("Avg intra-dist", f"{view.avg_intra_dist:.3f}")
    c4.metric("Exemplars", len(view.exemplar_face_ids))
    c5.metric("Outliers", len(view.outlier_face_ids))

    if view.split_signal:
        st.warning(
            "**Split signal:** bimodal intra-cluster distance distribution "
            "(gap > 0.12 and > 3× mean gap). This cluster may be two people."
        )
