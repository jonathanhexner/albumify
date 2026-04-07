"""Embedding Analysis page - UMAP visualizations and distance diagnostics."""

import io
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from app.face_clustering_debug.services.protocols import DataLoaderProtocol


def compute_umap_embedding(embeddings: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    """Compute 2D UMAP embedding."""
    try:
        import umap
    except ImportError:
        st.error("UMAP not installed. Run: pip install umap-learn")
        return None

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42)
    return reducer.fit_transform(embeddings)


def render_embedding_analysis_page(loader: DataLoaderProtocol) -> None:
    st.header("Embedding Analysis")
    st.caption("UMAP visualizations and distance diagnostics for understanding embedding quality.")

    # Load data
    try:
        embeddings = loader.load_embeddings()
        faces = loader.load_faces()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    st.info(f"Loaded {len(embeddings)} embeddings ({embeddings.shape[1]} dimensions)")

    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "Global UMAP",
        "Per-Cluster UMAP",
        "Distance Comparison",
        "Condensed Tree"
    ])

    with tab1:
        render_global_umap(loader, embeddings, faces)

    with tab2:
        render_per_cluster_umap(loader, embeddings, faces)

    with tab3:
        render_distance_comparison(embeddings, faces)

    with tab4:
        render_condensed_tree(embeddings)


def render_global_umap(loader: DataLoaderProtocol, embeddings: np.ndarray, faces) -> None:
    """Render global UMAP colored by cluster."""
    st.subheader("Global UMAP - All Faces")

    col1, col2 = st.columns([1, 3])

    with col1:
        n_neighbors = st.slider("n_neighbors", 5, 50, 15, key="global_nn")
        min_dist = st.slider("min_dist", 0.0, 1.0, 0.1, key="global_md")

        # Get available methods
        methods = loader.get_available_methods()
        if not methods:
            st.warning("No clustering results available")
            return

        method = st.selectbox("Clustering method", methods, key="global_method")

        color_by = st.selectbox("Color by", [
            "Cluster",
            "Confidence",
            "Frontal Score",
        ], key="global_color")

    if st.button("Compute UMAP", key="compute_global"):
        with st.spinner("Computing UMAP projection..."):
            umap_coords = compute_umap_embedding(embeddings, n_neighbors, min_dist)

        if umap_coords is None:
            return

        # Get clustering result
        result = loader.load_clustering_result(method)
        if result is None:
            st.error("Failed to load clustering result")
            return

        # Build dataframe
        df = pd.DataFrame({
            "x": umap_coords[:, 0],
            "y": umap_coords[:, 1],
            "face_idx": list(range(len(embeddings))),
            "cluster": result.labels,
        })

        # Add face metadata
        faces_by_idx = {f.index: f for f in faces}
        df["confidence"] = [faces_by_idx.get(i, type('obj', (object,), {'confidence': 0})).confidence for i in df["face_idx"]]
        df["frontal_score"] = [getattr(faces_by_idx.get(i), 'frontal_score', 0) or 0 for i in df["face_idx"]]

        # Color mapping
        if color_by == "Cluster":
            df["color"] = df["cluster"].astype(str)
            # Mark noise as special
            df.loc[df["cluster"] == -1, "color"] = "Noise"
        elif color_by == "Confidence":
            df["color"] = df["confidence"]
        else:
            df["color"] = df["frontal_score"]

        with col2:
            st.scatter_chart(
                df,
                x="x",
                y="y",
                color="color" if color_by == "Cluster" else None,
                size=20,
            )

            # Stats
            n_clusters = len(set(result.labels) - {-1})
            n_noise = sum(1 for l in result.labels if l == -1)
            st.caption(f"Clusters: {n_clusters} | Noise: {n_noise}")


def render_per_cluster_umap(loader: DataLoaderProtocol, embeddings: np.ndarray, faces) -> None:
    """Render UMAP for individual clusters."""
    st.subheader("Per-Cluster UMAP")
    st.caption("Visualize faces within a single cluster to identify sub-groups or outliers.")

    methods = loader.get_available_methods()
    if not methods:
        st.warning("No clustering results available")
        return

    col1, col2 = st.columns([1, 3])

    with col1:
        method = st.selectbox("Method", methods, key="cluster_method")
        result = loader.load_clustering_result(method)

        if result is None:
            st.error("Failed to load result")
            return

        cluster_ids = sorted(set(result.labels) - {-1})
        if not cluster_ids:
            st.warning("No clusters found")
            return

        cluster_id = st.selectbox("Cluster", cluster_ids, key="cluster_id")

        color_by = st.selectbox("Color by", [
            "Confidence",
            "Frontal Score",
            "Face Index",
        ], key="cluster_color")

        n_neighbors = st.slider("n_neighbors", 3, 30, 10, key="cluster_nn")

    # Get faces in this cluster
    cluster_mask = result.labels == cluster_id
    cluster_indices = np.where(cluster_mask)[0]

    if len(cluster_indices) < 4:
        st.warning(f"Cluster {cluster_id} has only {len(cluster_indices)} faces - too few for UMAP")
        return

    if st.button("Compute Cluster UMAP", key="compute_cluster"):
        cluster_embeddings = embeddings[cluster_mask]

        with st.spinner("Computing UMAP..."):
            # Adjust n_neighbors if needed
            actual_nn = min(n_neighbors, len(cluster_embeddings) - 1)
            umap_coords = compute_umap_embedding(cluster_embeddings, actual_nn, 0.1)

        if umap_coords is None:
            return

        faces_by_idx = {f.index: f for f in faces}

        df = pd.DataFrame({
            "x": umap_coords[:, 0],
            "y": umap_coords[:, 1],
            "face_idx": cluster_indices,
        })

        df["confidence"] = [faces_by_idx.get(i, type('obj', (object,), {'confidence': 0})).confidence for i in df["face_idx"]]
        df["frontal_score"] = [getattr(faces_by_idx.get(i), 'frontal_score', 0) or 0 for i in df["face_idx"]]

        if color_by == "Confidence":
            df["color"] = df["confidence"]
        elif color_by == "Frontal Score":
            df["color"] = df["frontal_score"]
        else:
            df["color"] = df["face_idx"]

        with col2:
            st.scatter_chart(df, x="x", y="y", size=30)

            # Show face thumbnails
            st.write("**Faces in cluster:**")
            cols = st.columns(min(10, len(cluster_indices)))
            for i, idx in enumerate(cluster_indices[:10]):
                with cols[i % 10]:
                    crop = loader.get_face_crop(idx)
                    if crop:
                        st.image(crop, caption=f"#{idx}", width=60)


def render_distance_comparison(embeddings: np.ndarray, faces) -> None:
    """Compare distances between user-specified groups."""
    st.subheader("Distance Distribution Comparison")
    st.caption("Compare embedding distances between known same-person and different-person pairs.")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Group 1 (Same Person)**")
        group1_input = st.text_input(
            "Face indices (comma-separated)",
            placeholder="e.g., 0,81,79,82",
            key="group1"
        )

    with col2:
        st.write("**Group 2 (Different Person)**")
        group2_input = st.text_input(
            "Face indices (comma-separated)",
            placeholder="e.g., 2,46,183",
            key="group2"
        )

    if not group1_input or not group2_input:
        st.info("Enter face indices for both groups to compare distances.")
        return

    try:
        group1 = [int(x.strip()) for x in group1_input.split(",") if x.strip()]
        group2 = [int(x.strip()) for x in group2_input.split(",") if x.strip()]
    except ValueError:
        st.error("Invalid input - use comma-separated integers")
        return

    # Validate indices
    max_idx = len(embeddings) - 1
    invalid = [i for i in group1 + group2 if i > max_idx]
    if invalid:
        st.error(f"Invalid face indices (max is {max_idx}): {invalid}")
        return

    if st.button("Analyze Distances", key="analyze_dist"):
        from scipy.spatial.distance import cosine

        # Compute intra-group distances
        intra1 = []
        for i, idx1 in enumerate(group1):
            for idx2 in group1[i+1:]:
                intra1.append(cosine(embeddings[idx1], embeddings[idx2]))

        intra2 = []
        for i, idx1 in enumerate(group2):
            for idx2 in group2[i+1:]:
                intra2.append(cosine(embeddings[idx1], embeddings[idx2]))

        # Compute inter-group distances
        inter = []
        for idx1 in group1:
            for idx2 in group2:
                inter.append(cosine(embeddings[idx1], embeddings[idx2]))

        # Display results
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Group 1 Median", f"{np.median(intra1):.3f}" if intra1 else "N/A")
            st.metric("Group 1 Max", f"{max(intra1):.3f}" if intra1 else "N/A")

        with col2:
            st.metric("Group 2 Median", f"{np.median(intra2):.3f}" if intra2 else "N/A")
            st.metric("Group 2 Max", f"{max(intra2):.3f}" if intra2 else "N/A")

        with col3:
            st.metric("Inter-Group Min", f"{min(inter):.3f}" if inter else "N/A")
            st.metric("Inter-Group Median", f"{np.median(inter):.3f}" if inter else "N/A")

        # Analysis
        if intra1 and intra2 and inter:
            max_intra = max(max(intra1), max(intra2))
            min_inter = min(inter)

            if max_intra > min_inter:
                overlap = max_intra - min_inter
                st.error(f"**OVERLAP DETECTED**: {overlap:.3f}")
                st.write("Some same-person pairs have higher distance than different-person pairs.")
                st.write("This indicates embedding quality issues - clustering parameters alone cannot fix this.")
            else:
                gap = min_inter - max_intra
                st.success(f"**GOOD SEPARATION**: Gap of {gap:.3f}")
                st.write(f"Threshold between {max_intra:.3f} and {min_inter:.3f} should work.")

        # Histogram data
        hist_data = pd.DataFrame({
            "Distance": intra1 + intra2 + inter,
            "Type": (["Group 1 (intra)"] * len(intra1) +
                    ["Group 2 (intra)"] * len(intra2) +
                    ["Inter-group"] * len(inter))
        })

        st.bar_chart(hist_data.groupby(["Type", pd.cut(hist_data["Distance"], bins=15)]).size().unstack(fill_value=0))


def render_condensed_tree(embeddings: np.ndarray) -> None:
    """Render HDBSCAN condensed tree visualization."""
    st.subheader("HDBSCAN Condensed Tree")
    st.caption("Visualize the cluster hierarchy to understand merge decisions.")

    try:
        import hdbscan
    except ImportError:
        st.error("HDBSCAN not installed")
        return

    col1, col2 = st.columns([1, 3])

    with col1:
        min_cluster_size = st.slider("min_cluster_size", 2, 20, 5, key="tree_mcs")
        min_samples = st.slider("min_samples", 1, 10, 2, key="tree_ms")

    if st.button("Generate Tree", key="gen_tree"):
        with st.spinner("Running HDBSCAN..."):
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='cosine',
                gen_min_span_tree=True,
            )
            clusterer.fit(embeddings)

        with col2:
            # Try to plot condensed tree
            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(12, 8))
                clusterer.condensed_tree_.plot(
                    select_clusters=True,
                    selection_palette=['#4CAF50', '#2196F3', '#f44336', '#FF9800', '#9C27B0'],
                    ax=ax
                )
                ax.set_title("HDBSCAN Condensed Tree")
                ax.set_facecolor('#1a1a2e')
                fig.patch.set_facecolor('#1a1a2e')

                st.pyplot(fig)
                plt.close()

                # Stats
                n_clusters = len(set(clusterer.labels_) - {-1})
                n_noise = sum(1 for l in clusterer.labels_ if l == -1)
                st.caption(f"Found {n_clusters} clusters, {n_noise} noise points")

            except Exception as e:
                st.error(f"Failed to plot tree: {e}")
                st.write("Cluster labels:", set(clusterer.labels_))
