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
    wkey = "cluster_analysis_picker"

    # Cross-tab nav (spec-066): another tab — Gallery's "Open in Cluster
    # Analysis", nearest-clusters' "Go to" — requests a cluster via a ONE-SHOT
    # ``_goto_cluster`` flag. We must write the *widget key* here, because
    # Streamlit ignores ``index=`` once a keyed selectbox has a stored value.
    # ``pop`` makes it one-shot so it never overrides the user's own in-tab
    # selection on later reruns (``selected_cluster`` lags by a render and
    # can't be used to tell apart "external request" from "user just picked").
    goto = st.session_state.pop("_goto_cluster", None)
    if goto in options:
        st.session_state[wkey] = goto
    elif wkey not in st.session_state:
        prior = st.session_state.get("selected_cluster")
        st.session_state[wkey] = prior if prior in options else options[0]

    chosen = st.selectbox(
        "Cluster",
        options=options,
        format_func=lambda cid: labels[cid],
        key=wkey,
    )
    st.session_state["selected_cluster"] = chosen
    return chosen
