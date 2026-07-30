"""spec-071 — side-by-side face crops of the two clusters in a merge pair.

Lets the operator answer "were these actually the same person?" — the visual
half of V1's Merge Analysis. Reads nothing; takes the typed ``PairFaces``.
"""
from __future__ import annotations

import streamlit as st

from face_cluster.views.merged_clusters import PairFaces


def render_cluster_pair_crops(pair: PairFaces) -> None:
    """Two columns of thumbnails: cluster_a on the left, cluster_b on the right."""
    col_a, col_b = st.columns(2)
    with col_a:
        st.caption(f"cluster_a = {pair.cluster_a} · {len(pair.a_face_ids)} faces")
        if pair.a_crops:
            st.image(pair.a_crops, width=64)
        else:
            st.caption("(no crops resolvable for this cluster)")
    with col_b:
        st.caption(f"cluster_b = {pair.cluster_b} · {len(pair.b_face_ids)} faces")
        if pair.b_crops:
            st.image(pair.b_crops, width=64)
        else:
            st.caption("(no crops resolvable for this cluster)")
