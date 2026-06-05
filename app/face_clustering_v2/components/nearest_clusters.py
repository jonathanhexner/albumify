"""spec-045 Phase 6 — nearest-clusters list with Go-To button.

Takes a concrete :class:`ClusterView` (SIGHTING-079 sync rewrite). Reads
the currently-selected cluster from session state for the Go-To target.
"""
from __future__ import annotations

import streamlit as st

from face_cluster.views.cluster_view import ClusterView


def render_nearest_clusters(view: ClusterView) -> None:
    """Render the top-10 nearest clusters strip from the cluster view.

    Side effects: when the user clicks "Go to C{id}", writes
    ``st.session_state['selected_cluster']`` to that id and reruns.
    """
    if not view.nearest_clusters:
        return
    with st.expander(f"Nearest clusters ({len(view.nearest_clusters)})", expanded=False):
        for row in view.nearest_clusters[:10]:
            cols = st.columns([1, 1, 1, 1, 1, 1])
            cols[0].text(f"C{row.cluster_id}")
            cols[1].text(f"size={row.size}")
            cols[2].text(f"d_min={row.min_exemplar_dist:.3f}")
            cols[3].text(f"p10={row.p10_cross_dist:.3f}")
            cols[4].text("MERGE" if row.merge_candidate else "—")
            if cols[5].button("Go to", key=f"nc_goto_{row.cluster_id}"):
                # One-shot nav request (see cluster_picker): writing the widget
                # key is the only thing that actually moves a keyed selectbox.
                st.session_state["_goto_cluster"] = row.cluster_id
                st.session_state["selected_cluster"] = row.cluster_id
                st.rerun()
