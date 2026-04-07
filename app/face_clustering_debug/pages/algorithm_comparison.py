"""Algorithm comparison page — side-by-side metrics for two methods."""

import pandas as pd
import streamlit as st

from app.face_clustering_debug.components.face_grid import render_face_grid
from app.face_clustering_debug.services.protocols import DataLoaderProtocol


def render_algorithm_comparison_page(loader: DataLoaderProtocol) -> None:
    st.header("🆚 Algorithm Comparison")
    st.caption("Compare cluster assignments and metrics across two methods side-by-side.")

    methods = loader.get_available_methods()
    if len(methods) < 2:
        st.warning("⚠️ At least two methods are required to compare.")
        return

    col_l, col_r = st.columns(2)
    method_a = col_l.selectbox("Method A", methods, index=0, key="cmp_a")
    method_b = col_r.selectbox("Method B", methods, index=min(1, len(methods) - 1), key="cmp_b")

    result_a = loader.load_clustering_result(method_a)
    result_b = loader.load_clustering_result(method_b)

    if result_a is None or result_b is None:
        st.error("Could not load one or both methods.")
        return

    _render_metrics_side_by_side(col_l, col_r, result_a, result_b, method_a, method_b)
    st.divider()
    _render_diff_summary(result_a, result_b)
    st.divider()
    _render_cluster_size_comparison(result_a, result_b, method_a, method_b)


def _render_metrics_side_by_side(col_l, col_r, result_a, result_b, label_a, label_b) -> None:
    col_l.subheader(f"📊 {label_a}")
    col_l.metric("Clusters", result_a.n_clusters)
    col_l.metric("Noise", result_a.n_noise)

    delta_c = result_b.n_clusters - result_a.n_clusters
    delta_n = result_b.n_noise - result_a.n_noise
    col_r.subheader(f"📊 {label_b}")
    col_r.metric("Clusters", result_b.n_clusters, delta=delta_c)
    col_r.metric("Noise", result_b.n_noise, delta=delta_n, delta_color="inverse")


def _render_diff_summary(result_a, result_b) -> None:
    st.subheader("📋 Assignment Differences")
    if len(result_a.labels) != len(result_b.labels):
        st.warning("Label arrays differ in length — cannot compare assignments.")
        return

    changed = [i for i in range(len(result_a.labels)) if result_a.labels[i] != result_b.labels[i]]
    pct = 100 * len(changed) / max(len(result_a.labels), 1)
    st.metric("Faces with changed cluster", f"{len(changed)}  ({pct:.1f}%)")
    if changed:
        st.caption(f"Indices: {changed[:60]}{'…' if len(changed) > 60 else ''}")


def _render_cluster_size_comparison(result_a, result_b, label_a: str, label_b: str) -> None:
    st.subheader("📐 Cluster Size Distribution")
    sizes_a = sorted([len(c.face_indices) for c in result_a.clusters], reverse=True)
    sizes_b = sorted([len(c.face_indices) for c in result_b.clusters], reverse=True)
    max_len = max(len(sizes_a), len(sizes_b))
    rows = {
        label_a: sizes_a + [None] * (max_len - len(sizes_a)),
        label_b: sizes_b + [None] * (max_len - len(sizes_b)),
    }
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
