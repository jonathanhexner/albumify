"""Distance lookup page — compute distance between any two faces."""

import numpy as np
import streamlit as st
from scipy.spatial.distance import cosine

from app.face_clustering_debug.components.face_detail import render_face_detail
from app.face_clustering_debug.services.protocols import DataLoaderProtocol


def render_distance_lookup_page(loader: DataLoaderProtocol, method: str) -> None:
    st.header("📏 Distance Lookup")
    st.caption("Select any two faces to compute their Euclidean and cosine distance, "
               "and see whether that distance falls within each cluster's threshold T.")

    result = loader.load_clustering_result(method)
    if result is None:
        st.error(f"No results for method **{method}**")
        return

    n = len(result.faces)
    col_a, col_b = st.columns(2)
    idx_a = col_a.number_input("Face A", 0, n - 1, 0, help="Face index")
    idx_b = col_b.number_input("Face B", 0, n - 1, min(1, n - 1), help="Face index")

    emb_a = result.embeddings[int(idx_a)]
    emb_b = result.embeddings[int(idx_b)]
    euclidean = float(np.linalg.norm(emb_a - emb_b))
    cos_dist = float(cosine(emb_a, emb_b))

    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("📐 Euclidean distance", f"{euclidean:.4f}")
    m2.metric("📐 Cosine distance", f"{cos_dist:.4f}")

    _render_threshold_comparison(result, int(idx_a), int(idx_b), cos_dist)

    st.divider()
    face_a = result.faces[int(idx_a)]
    face_b = result.faces[int(idx_b)]

    with col_a:
        st.markdown(f"**Face {idx_a}** — cluster {int(result.labels[int(idx_a)])}")
        crop = loader.get_face_crop(int(idx_a))
        if crop:
            render_face_detail(face_a, crop)
    with col_b:
        st.markdown(f"**Face {idx_b}** — cluster {int(result.labels[int(idx_b)])}")
        crop = loader.get_face_crop(int(idx_b))
        if crop:
            render_face_detail(face_b, crop)


def _render_threshold_comparison(result, idx_a: int, idx_b: int, cos_dist: float) -> None:
    thresholds = {c.cluster_id: c.threshold for c in result.clusters}
    for idx in (idx_a, idx_b):
        label = int(result.labels[idx])
        if label >= 0 and label in thresholds:
            t = thresholds[label]
            within = cos_dist <= t
            icon = "✅ within" if within else "❌ outside"
            st.caption(f"Face {idx} (cluster {label}, T={t:.3f}): cosine {cos_dist:.4f} is **{icon}** threshold")
