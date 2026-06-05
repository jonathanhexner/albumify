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

from app.face_clustering_v2.components.metric_strip import render_metric_strip
from face_cluster.views.cluster_view import ClusterView
from face_cluster.views.metric_specs import CLUSTER_METRIC_STRIP


def render_cluster_metrics(view: ClusterView) -> None:
    """Render the 5-metric strip + (optional) split-signal banner."""
    render_metric_strip(view, CLUSTER_METRIC_STRIP)

    if view.split_signal:
        st.warning(
            "**Split signal:** bimodal intra-cluster distance distribution "
            "(gap > 0.12 and > 3× mean gap). This cluster may be two people."
        )
