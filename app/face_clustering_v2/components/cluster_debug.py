"""spec-045 Phase 6 — graph-debug section for one cluster.

Renders 4-metric strip + (optional) chain/sparse warnings + bridge faces +
heatmap + edges expander. Stateless given an AsyncHandle[ClusterDebugView].
"""
from __future__ import annotations

import streamlit as st

from face_cluster.views._async import AsyncHandle
from face_cluster.views.cluster_debug_view import ClusterDebugView


def render_cluster_debug(handle: AsyncHandle[ClusterDebugView]) -> None:
    """Render graph diagnostics for the cluster the handle is computing."""
    state = handle.poll()
    if state in ("pending", "running"):
        st.caption("Computing graph diagnostics…")
        return
    if state == "failed":
        st.error(f"Graph compute failed: {handle.error}")
        return
    if state == "cancelled" or handle.result is None:
        return
    dbg: ClusterDebugView = handle.result

    with st.expander("Graph debug", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Edges", f"{dbg.n_edges} / {dbg.max_possible_edges}")
        c2.metric("Density", f"{dbg.edge_density:.1%}")
        c3.metric("Chain score", f"{dbg.chain_score:.2f}",
                  help="diameter / (2 × median dist). > 1.5 suggests chain.")
        c4.metric("Bridge faces", len(dbg.bridge_face_ids))

        if dbg.chain_score > 1.5:
            st.warning(
                f"**Chain structure** (score {dbg.chain_score:.2f}). "
                f"Diameter {dbg.diameter:.3f} ≫ median {dbg.median_dist:.3f}."
            )
        elif dbg.edge_density < 0.15 and dbg.n_faces > 4:
            st.warning(f"**Sparse graph** ({dbg.edge_density:.1%} density).")

        if dbg.bridge_face_ids:
            st.markdown(
                f"**Bridge faces**: `{', '.join(f'face_{fid:04d}' for fid in dbg.bridge_face_ids)}`"
            )

        if dbg.distance_matrix is not None and len(dbg.distance_matrix) > 1:
            import plotly.express as px
            labels = [f"f{fid}" for fid in dbg.face_ids_order]
            fig = px.imshow(
                dbg.distance_matrix, x=labels, y=labels,
                color_continuous_scale="RdYlGn_r",
                zmin=0.0, zmax=min(0.6, float(dbg.distance_matrix.max()) + 0.05),
                labels=dict(color="cosine dist"), aspect="equal",
            )
            fig.update_layout(height=max(300, 18 * dbg.n_faces + 100))
            st.plotly_chart(fig, use_container_width=True)
