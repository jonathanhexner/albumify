"""spec-045 Phase 6 — cluster picker component.

Renders a selectbox of ``"Cluster {id} ({size} faces)"`` labels. Persists
the chosen id under ``st.session_state['selected_cluster']``.
"""
from __future__ import annotations

from typing import List, Optional

import streamlit as st

from face_cluster.views._base import ClusterRow


def render_cluster_picker(rows: List[ClusterRow]) -> Optional[int]:
    """Return the chosen cluster_id (or None if the run has no clusters).

    Side effects: writes ``st.session_state['selected_cluster']`` on change.
    """
    if not rows:
        st.info("This run has no clusters to analyse.")
        return None
    options = [r.cluster_id for r in rows]
    labels = {r.cluster_id: f"Cluster {r.cluster_id} ({r.size} faces)" for r in rows}
    prior = st.session_state.get("selected_cluster")
    default_idx = options.index(prior) if prior in options else 0
    chosen = st.selectbox(
        "Cluster",
        options=options,
        index=default_idx,
        format_func=lambda cid: labels[cid],
        key="cluster_analysis_picker",
    )
    st.session_state["selected_cluster"] = chosen
    return chosen
