"""spec-045 Phase 6 — cluster-metrics strip + split-signal banner.

Renders the 5-metric headline + an optional "split signal" banner when the
clusterer detected a bimodal intra-cluster distance distribution.

Takes a concrete :class:`ClusterView` (sync compute). SIGHTING-079 fix:
the AsyncHandle variant never advanced past "Analysing cluster…" because
Streamlit doesn't poll background threads. Sync + ``st.spinner`` in the
tab body fits Streamlit's lifecycle.
"""
from __future__ import annotations

import streamlit as st

from face_cluster.views.cluster_view import ClusterView


def render_cluster_metrics(view: ClusterView) -> None:
    """Render the 5-metric strip + (optional) split-signal banner."""
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
