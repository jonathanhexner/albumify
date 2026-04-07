"""Threshold statistics display component."""

import streamlit as st

from app.face_clustering_debug.models.schemas import ClusterInfo


def render_threshold_info(cluster: ClusterInfo) -> None:
    """Display threshold stats for a cluster.

    Args:
        cluster: Cluster whose threshold stats to display.
    """
    cols = st.columns(4)
    cols[0].metric("Threshold", f"{cluster.threshold:.3f}")
    cols[1].metric("Raw", f"{cluster.raw_threshold:.3f}")
    cols[2].metric("IQR", f"{cluster.iqr:.3f}")
    cols[3].metric("Q1 / Q3", f"{cluster.q1:.2f} / {cluster.q3:.2f}")
