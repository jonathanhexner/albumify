"""
Debug page for Hybrid Closest Face clustering algorithm.
Shows detailed merge decisions, thresholds, and cross-cluster d3 analysis.
"""

import streamlit as st
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from scipy.spatial.distance import cdist
from PIL import Image


def find_nearest_faces_with_clusters(
    cluster_indices: List[int],
    all_embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 10
) -> List[Tuple[int, float, int]]:
    """Find k nearest faces outside cluster with their cluster labels.
    
    Returns:
        List of (face_idx, distance, cluster_label) tuples
    """
    cluster_label = labels[cluster_indices[0]]
    cluster_embeddings = all_embeddings[cluster_indices]
    external_mask = labels != cluster_label
    
    if not np.any(external_mask):
        return []
    
    external_indices = np.where(external_mask)[0]
    external_embeddings = all_embeddings[external_indices]
    external_labels = labels[external_indices]
    
    # Compute distances to all external faces
    distances = cdist(cluster_embeddings, external_embeddings, metric='euclidean')
    min_distances = np.min(distances, axis=0)
    
    # Get top k nearest
    nearest_k_indices = np.argsort(min_distances)[:k]
    
    results = []
    for idx in nearest_k_indices:
        global_idx = external_indices[idx]
        distance = min_distances[idx]
        cluster_label = external_labels[idx]
        results.append((int(global_idx), float(distance), int(cluster_label)))
    
    return results


def find_closest_cluster(
    cluster_indices: List[int],
    all_embeddings: np.ndarray,
    labels: np.ndarray
) -> Tuple[int, float]:
    """Find the closest other cluster by min distance.
    
    Returns:
        (closest_cluster_label, min_distance)
    """
    cluster_label = labels[cluster_indices[0]]
    cluster_embeddings = all_embeddings[cluster_indices]
    
    unique_labels = [l for l in np.unique(labels) if l != cluster_label and l >= 0]
    if not unique_labels:
        return (-1, float('inf'))
    
    min_distance = float('inf')
    closest_label = -1
    
    for other_label in unique_labels:
        other_indices = np.where(labels == other_label)[0]
        other_embeddings = all_embeddings[other_indices]
        
        distances = cdist(cluster_embeddings, other_embeddings, metric='euclidean')
        curr_min = float(np.min(distances))
        
        if curr_min < min_distance:
            min_distance = curr_min
            closest_label = other_label
    
    return (int(closest_label), min_distance)


def compute_cross_cluster_d3(
    cluster_a_indices: List[int],
    cluster_b_indices: List[int],
    all_embeddings: np.ndarray,
    k: int = 3
) -> Dict[str, Any]:
    """Compute cross-cluster d3 statistics for two clusters."""
    embeddings_a = all_embeddings[cluster_a_indices]
    embeddings_b = all_embeddings[cluster_b_indices]
    
    # For each face in A, compute d3 using B's faces
    d3_a_to_b = []
    for face_a in embeddings_a:
        dists = cdist([face_a], embeddings_b, metric='euclidean')[0]
        k_actual = min(k, len(cluster_b_indices))
        d3 = np.sort(dists)[:k_actual][-1] if k_actual > 0 else float('inf')
        d3_a_to_b.append(d3)
    
    # For each face in B, compute d3 using A's faces
    d3_b_to_a = []
    for face_b in embeddings_b:
        dists = cdist([face_b], embeddings_a, metric='euclidean')[0]
        k_actual = min(k, len(cluster_a_indices))
        d3 = np.sort(dists)[:k_actual][-1] if k_actual > 0 else float('inf')
        d3_b_to_a.append(d3)
    
    return {
        'd3_a_to_b': d3_a_to_b,
        'd3_b_to_a': d3_b_to_a,
        'n_faces_a': len(cluster_a_indices),
        'n_faces_b': len(cluster_b_indices)
    }


def render_cluster_detail(
    cluster_label: int,
    cluster_indices: List[int],
    results_dir: Path,
    all_embeddings: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    knn_k: int
):
    """Render detailed view of a cluster with nearest external faces."""
    st.subheader(f"Cluster {cluster_label} ({len(cluster_indices)} faces)")
    st.caption(f"Threshold: {threshold:.3f}")
    
    # Show cluster faces
    st.markdown("**Cluster Faces:**")
    cols = st.columns(min(10, len(cluster_indices)))
    for i, face_idx in enumerate(cluster_indices[:10]):
        with cols[i]:
            crop_path = results_dir / 'face_crops' / f'face_{face_idx:04d}.jpg'
            if crop_path.exists():
                img = Image.open(crop_path)
                st.image(img, caption=f"Face {face_idx}", use_container_width=True)
    
    if len(cluster_indices) > 10:
        st.caption(f"... and {len(cluster_indices) - 10} more faces")
    
    # Find closest cluster
    closest_cluster_label, closest_distance = find_closest_cluster(
        cluster_indices, all_embeddings, labels
    )
    
    st.markdown(f"**Closest Cluster:** Cluster {closest_cluster_label} (distance: {closest_distance:.3f})")
    
    # Find 10 nearest external faces
    nearest_faces = find_nearest_faces_with_clusters(
        cluster_indices, all_embeddings, labels, k=10
    )
    
    if nearest_faces:
        st.markdown("**10 Nearest Faces Outside Cluster:**")
        cols = st.columns(10)
        for i, (face_idx, distance, face_cluster) in enumerate(nearest_faces):
            with cols[i]:
                crop_path = results_dir / 'face_crops' / f'face_{face_idx:04d}.jpg'
                if crop_path.exists():
                    img = Image.open(crop_path)
                    color = "🔴" if distance <= threshold else "🟢"
                    st.image(img, use_container_width=True)
                    st.caption(f"{color} Face {face_idx}")
                    st.caption(f"d={distance:.3f}")
                    st.caption(f"Cluster {face_cluster}")


def render_merge_test(
    results_dir: Path,
    labels: np.ndarray,
    all_embeddings: np.ndarray,
    knn_k: int,
    iqr_multiplier: float,
    threshold_floor: float,
    threshold_ceiling: float,
    early_exit_mult: float
):
    """Interactive merge testing between two clusters."""
    st.subheader("🧪 Test Merge Between Clusters")
    
    unique_clusters = sorted([l for l in np.unique(labels) if l >= 0])
    
    col1, col2 = st.columns(2)
    with col1:
        cluster_a = st.selectbox("Cluster A", unique_clusters, key="merge_test_a")
    with col2:
        cluster_b = st.selectbox("Cluster B", [c for c in unique_clusters if c != cluster_a], key="merge_test_b")
    
    if st.button("Test Merge Criteria"):
        # Get cluster data
        indices_a = np.where(labels == cluster_a)[0].tolist()
        indices_b = np.where(labels == cluster_b)[0].tolist()
        embeddings_a = all_embeddings[indices_a]
        embeddings_b = all_embeddings[indices_b]
        
        # Compute thresholds
        def compute_d3_threshold(embeddings, knn_k, iqr_mult, floor, ceiling):
            distances = cdist(embeddings, embeddings, metric='euclidean')
            k = min(knn_k, len(embeddings) - 1)
            d3_values = []
            for i in range(len(embeddings)):
                sorted_dists = np.sort(distances[i])[1:k+1]
                d3_values.append(sorted_dists[-1] if len(sorted_dists) > 0 else 0)
            
            median_d3 = np.median(d3_values)
            q1, q3 = np.percentile(d3_values, [25, 75])
            iqr = q3 - q1
            raw_threshold = median_d3 + iqr_mult * iqr
            threshold = max(min(raw_threshold, ceiling), floor)
            
            return threshold, median_d3, iqr, d3_values
        
        threshold_a, median_a, iqr_a, d3_a = compute_d3_threshold(
            embeddings_a, knn_k, iqr_multiplier, threshold_floor, threshold_ceiling
        )
        threshold_b, median_b, iqr_b, d3_b = compute_d3_threshold(
            embeddings_b, knn_k, iqr_multiplier, threshold_floor, threshold_ceiling
        )
        
        st.markdown("### Cluster Thresholds")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Cluster {cluster_a}:**")
            st.write(f"- Threshold: **{threshold_a:.3f}**")
            st.write(f"- Median d3: {median_a:.3f}")
            st.write(f"- IQR: {iqr_a:.3f}")
        with col2:
            st.markdown(f"**Cluster {cluster_b}:**")
            st.write(f"- Threshold: **{threshold_b:.3f}**")
            st.write(f"- Median d3: {median_b:.3f}")
            st.write(f"- IQR: {iqr_b:.3f}")
        
        merge_threshold = min(threshold_a, threshold_b)
        st.markdown(f"**Merge Threshold:** `min(T_A, T_B) = {merge_threshold:.3f}`")
        
        # Early exit check
        st.markdown("### Stage 1: Early Exit Check")
        
        # Compute exemplar distances
        exemplar_indices_a = np.argsort(d3_a)[:min(10, len(d3_a))]
        exemplar_indices_b = np.argsort(d3_b)[:min(10, len(d3_b))]
        exemplar_embeddings_a = embeddings_a[exemplar_indices_a]
        exemplar_embeddings_b = embeddings_b[exemplar_indices_b]
        
        exemplar_dists = cdist(exemplar_embeddings_a, exemplar_embeddings_b, metric='euclidean')
        min_exemplar_dist = float(np.min(exemplar_dists))
        early_exit_threshold = early_exit_mult * max(threshold_a, threshold_b)
        
        st.write(f"- Min exemplar distance: **{min_exemplar_dist:.3f}**")
        st.write(f"- Early exit threshold: **{early_exit_threshold:.3f}** ({early_exit_mult}×max(T_A, T_B))")
        
        if min_exemplar_dist > early_exit_threshold:
            st.error(f"❌ **EARLY EXIT**: Min exemplar distance ({min_exemplar_dist:.3f}) > threshold ({early_exit_threshold:.3f})")
            st.info("Clusters rejected by early exit filter. No further checks performed.")
            return
        
        st.success(f"✅ **PASS**: Proceed to cross-cluster d3 check")
        
        # Stage 2: Cross-cluster d3
        st.markdown("### Stage 2: Cross-Cluster d3 Check")
        
        cross_d3 = compute_cross_cluster_d3(indices_a, indices_b, all_embeddings, knn_k)
        d3_a_to_b = cross_d3['d3_a_to_b']
        d3_b_to_a = cross_d3['d3_b_to_a']
        
        # Count faces that fit
        fits_a_to_b = sum(1 for d3 in d3_a_to_b if d3 <= merge_threshold)
        fits_b_to_a = sum(1 for d3 in d3_b_to_a if d3 <= merge_threshold)
        total_fits = fits_a_to_b + fits_b_to_a
        
        st.write(f"- Faces from A fitting into B: **{fits_a_to_b}** / {len(d3_a_to_b)}")
        st.write(f"- Faces from B fitting into A: **{fits_b_to_a}** / {len(d3_b_to_a)}")
        st.write(f"- **Total fits: {total_fits}**")
        
        min_faces_required = 2  # merge_min_faces parameter
        
        if total_fits >= min_faces_required:
            st.success(f"✅ **MERGE**: {total_fits} faces fit (>= {min_faces_required} required)")
        else:
            st.error(f"❌ **NO MERGE**: {total_fits} faces fit (< {min_faces_required} required)")
        
        # Show distribution
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        
        ax[0].hist(d3_a_to_b, bins=20, alpha=0.7, label='A→B')
        ax[0].axvline(merge_threshold, color='r', linestyle='--', label=f'Threshold={merge_threshold:.3f}')
        ax[0].set_xlabel('Cross-cluster d3')
        ax[0].set_ylabel('Count')
        ax[0].set_title(f'Cluster {cluster_a} faces → Cluster {cluster_b}')
        ax[0].legend()
        
        ax[1].hist(d3_b_to_a, bins=20, alpha=0.7, label='B→A', color='orange')
        ax[1].axvline(merge_threshold, color='r', linestyle='--', label=f'Threshold={merge_threshold:.3f}')
        ax[1].set_xlabel('Cross-cluster d3')
        ax[1].set_ylabel('Count')
        ax[1].set_title(f'Cluster {cluster_b} faces → Cluster {cluster_a}')
        ax[1].legend()
        
        st.pyplot(fig)


def render_debug_hybrid_closest(
    results_dir: Path,
    metadata: List[Dict[str, Any]],
    hybrid_closest_results: Dict[str, Any],
    all_embeddings: np.ndarray
):
    """Render debug page for Hybrid Closest Face clustering."""
    st.header("🔧 Debug: Hybrid Closest Face")
    
    stats = hybrid_closest_results.get('stats', {})
    labels = np.array(hybrid_closest_results.get('labels', []))
    params = stats.get('params', {})
    
    # Configuration controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ Algorithm Parameters")
    st.sidebar.caption("Adjust to test different configurations")
    
    knn_k = st.sidebar.number_input("knn_k", value=params.get('knn_k', 3), min_value=1, max_value=10)
    iqr_multiplier = st.sidebar.number_input(
        "iqr_multiplier", 
        value=params.get('iqr_multiplier', 2.5), 
        min_value=1.0, 
        max_value=5.0, 
        step=0.1,
        help="Higher = more permissive thresholds"
    )
    threshold_floor = st.sidebar.number_input(
        "threshold_floor", 
        value=params.get('threshold_floor', 0.30), 
        min_value=0.0, 
        max_value=1.0, 
        step=0.05
    )
    threshold_ceiling = st.sidebar.number_input(
        "threshold_ceiling", 
        value=params.get('threshold_ceiling', 1.50), 
        min_value=0.5, 
        max_value=2.0, 
        step=0.1
    )
    early_exit_multiplier = st.sidebar.number_input(
        "early_exit_multiplier", 
        value=params.get('early_exit_multiplier', 2.0), 
        min_value=1.0, 
        max_value=5.0, 
        step=0.1,
        help="Early exit if min_exemplar_dist > this × max(T_A, T_B)"
    )
    
    # Algorithm info
    st.markdown("""
    **Algorithm:** Cross-cluster d3 matching
    
    - **Threshold:** T = median(d3) + iqr_multiplier×IQR(d3) for all faces in cluster
    - **Merge:** If ≥2 faces have cross-cluster d3 < min(T_A, T_B)
    - **Early Exit:** Skip merge check if min(exemplar distances) > early_exit_multiplier × max(T_A, T_B)
    """)
    
    # Merge test section
    render_merge_test(
        results_dir, labels, all_embeddings, 
        knn_k, iqr_multiplier, threshold_floor, threshold_ceiling, early_exit_multiplier
    )
    
    st.markdown("---")
    
    # Cluster explorer
    st.subheader("📊 Cluster Explorer")
    
    unique_clusters = sorted([l for l in np.unique(labels) if l >= 0])
    selected_cluster = st.selectbox("Select Cluster", unique_clusters, key="cluster_explorer")
    
    cluster_indices = np.where(labels == selected_cluster)[0].tolist()
    embeddings_cluster = all_embeddings[cluster_indices]
    
    # Compute threshold for this cluster
    distances = cdist(embeddings_cluster, embeddings_cluster, metric='euclidean')
    k = min(knn_k, len(embeddings_cluster) - 1)
    d3_values = []
    for i in range(len(embeddings_cluster)):
        sorted_dists = np.sort(distances[i])[1:k+1]
        d3_values.append(sorted_dists[-1] if len(sorted_dists) > 0 else 0)
    
    median_d3 = np.median(d3_values)
    q1, q3 = np.percentile(d3_values, [25, 75])
    iqr = q3 - q1
    raw_threshold = median_d3 + iqr_multiplier * iqr
    threshold = max(min(raw_threshold, threshold_ceiling), threshold_floor)
    
    render_cluster_detail(
        selected_cluster, cluster_indices, results_dir, 
        all_embeddings, labels, threshold, knn_k
    )
