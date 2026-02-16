"""
Face Clustering Comparison Streamlit App

Visual comparison of clustering methods (HDBSCAN vs Hybrid HDBSCAN+kNN).
Includes detailed cluster exploration with metrics and distance analysis.

Usage:
    streamlit run app/face_clustering_comparison.py
"""

import streamlit as st
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from PIL import Image, ImageDraw
from collections import defaultdict
from scipy.spatial.distance import cdist
import pandas as pd

st.set_page_config(
    page_title="Face Clustering Comparison",
    page_icon="👥",
    layout="wide"
)


@st.cache_data
def load_benchmark_results(results_file: Path) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


@st.cache_data
def load_embeddings(embeddings_file: Path) -> Optional[np.ndarray]:
    """Load face embeddings from numpy file."""
    if embeddings_file.exists():
        return np.load(embeddings_file)
    return None


def get_available_benchmarks(results_dir: Path) -> List[Path]:
    """Find all benchmark result files."""
    if not results_dir.exists():
        return []
    return sorted(results_dir.glob('benchmark_*.json'), reverse=True)


def get_face_crop_path(results_dir: Path, face_index: int) -> Path:
    """Get path to face crop image."""
    return results_dir / 'face_crops' / f'face_{face_index:04d}.jpg'


def draw_landmarks_on_face(face_img: Image.Image, landmarks: List[List[float]]) -> Image.Image:
    """Draw landmarks on face image."""
    if not landmarks or len(landmarks) != 5:
        return face_img
    
    img_with_landmarks = face_img.copy()
    draw = ImageDraw.Draw(img_with_landmarks)
    
    # Scale landmarks to image size
    w, h = face_img.size
    scaled_landmarks = [(x * w, y * h) for x, y in landmarks]
    
    # Draw landmarks as red circles
    radius = max(2, int(min(w, h) * 0.03))
    for x, y in scaled_landmarks:
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill='red', outline='white')
    
    return img_with_landmarks


def compute_distance_matrix(face_indices: List[int], all_embeddings: np.ndarray) -> np.ndarray:
    """Compute distance matrix for faces in a cluster."""
    cluster_embeddings = all_embeddings[face_indices]
    return cdist(cluster_embeddings, cluster_embeddings, metric='cosine')


def find_nearest_external_faces(cluster_indices: List[int], all_embeddings: np.ndarray, 
                                labels: np.ndarray, k: int = 5) -> List[tuple]:
    """Find k nearest faces outside the cluster."""
    cluster_label = labels[cluster_indices[0]]
    cluster_embeddings = all_embeddings[cluster_indices]
    external_mask = labels != cluster_label
    
    if not np.any(external_mask):
        return []
    
    external_indices = np.where(external_mask)[0]
    external_embeddings = all_embeddings[external_indices]
    
    # Compute distances to all external faces
    distances = cdist(cluster_embeddings, external_embeddings, metric='cosine')
    min_distances = np.min(distances, axis=0)
    
    # Get top k nearest
    nearest_k_indices = np.argsort(min_distances)[:k]
    results = [(int(external_indices[i]), float(min_distances[i])) for i in nearest_k_indices]
    
    return results


def render_cluster_stats_table(method_name: str, cluster_stats: List[Dict], labels: List[int]):
    """Render cluster statistics table."""
    st.subheader(f"📊 {method_name} Cluster Statistics")
    
    if not cluster_stats:
        st.info("No clusters found")
        return
    
    # Create DataFrame
    df_data = []
    for stats in cluster_stats:
        df_data.append({
            'Cluster': stats['cluster_id'],
            'Size': stats['size'],
            'Intra-Min': f"{stats['intra_min']:.3f}",
            'Intra-Max': f"{stats['intra_max']:.3f}",
            'Intra-Mean': f"{stats['intra_mean']:.3f}",
            'Intra-Std': f"{stats['intra_std']:.3f}",
            'Nearest External': f"{stats['nearest_external_dist']:.3f}" if stats['nearest_external_dist'] else 'N/A'
        })
    
    # Count noise
    noise_count = sum(1 for label in labels if label == -1)
    
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(df)} clusters. {noise_count} faces marked as noise.")


def render_cluster_gallery(method_name: str, labels: List[int], metadata: List[Dict], 
                           results_dir: Path, max_faces_per_cluster: int = 20):
    """Render cluster gallery with face thumbnails."""
    st.subheader(f"🖼️ {method_name} Cluster Gallery")
    
    # Group faces by cluster
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        if label != -1:  # Skip noise
            clusters[label].append(idx)
    
    # Sort clusters by size (descending)
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Filter controls
    col1, col2 = st.columns([1, 3])
    with col1:
        min_size = st.number_input("Min cluster size", min_value=1, value=2, key=f"{method_name}_min_size")
    with col2:
        show_clusters = st.slider("Number of clusters to show", 1, len(sorted_clusters), 
                                 min(20, len(sorted_clusters)), key=f"{method_name}_show_n")
    
    # Display clusters
    displayed = 0
    for cluster_id, face_indices in sorted_clusters:
        if displayed >= show_clusters:
            break
        if len(face_indices) < min_size:
            continue
        
        with st.expander(f"**Cluster {cluster_id}** ({len(face_indices)} faces)", expanded=(displayed < 3)):
            # Display faces in rows
            cols = st.columns(min(10, len(face_indices)))
            for i, face_idx in enumerate(face_indices[:max_faces_per_cluster]):
                crop_path = get_face_crop_path(results_dir, face_idx)
                if crop_path.exists():
                    with cols[i % len(cols)]:
                        st.image(str(crop_path), caption=f"#{face_idx}", use_container_width=True)
            
            if len(face_indices) > max_faces_per_cluster:
                st.caption(f"... and {len(face_indices) - max_faces_per_cluster} more faces")
        
        displayed += 1


def render_cluster_explorer(results_dir: Path, metadata: List[Dict], 
                            labels: List[int], method_name: str, all_embeddings: np.ndarray):
    """Detailed cluster exploration page."""
    st.header(f"🔍 Cluster Explorer - {method_name}")
    
    # Cluster selection
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        if label != -1:
            clusters[label].append(idx)
    
    sorted_cluster_ids = sorted(clusters.keys(), key=lambda x: len(clusters[x]), reverse=True)
    
    if not sorted_cluster_ids:
        st.warning("No clusters to explore")
        return
    
    selected_cluster = st.selectbox(
        "Select cluster to explore",
        sorted_cluster_ids,
        format_func=lambda x: f"Cluster {x} ({len(clusters[x])} faces)",
        key=f"{method_name}_explorer_select"
    )
    
    face_indices = clusters[selected_cluster]
    
    st.markdown(f"### Cluster {selected_cluster} - {len(face_indices)} faces")
    
    # Section 1: Face crops with landmarks
    st.markdown("#### Faces with Landmarks")
    cols = st.columns(min(8, len(face_indices)))
    for i, face_idx in enumerate(face_indices[:24]):  # Limit to 24 faces
        crop_path = get_face_crop_path(results_dir, face_idx)
        if crop_path.exists():
            face_meta = metadata[face_idx]
            face_img = Image.open(crop_path)
            landmarks = face_meta.get('landmarks', [])
            if landmarks:
                face_img = draw_landmarks_on_face(face_img, landmarks)
            with cols[i % len(cols)]:
                st.image(face_img, caption=f"#{face_idx}", use_container_width=True)
    
    # Section 2: Face metrics table
    st.markdown("#### Face Quality Metrics")
    metrics_data = []
    for face_idx in face_indices:
        face_meta = metadata[face_idx]
        metrics_data.append({
            'Face': face_idx,
            'Confidence': f"{face_meta.get('confidence', 0):.2f}",
            'Frontal Score': f"{face_meta.get('frontal_score', 0):.2f}",
            'Roll': f"{face_meta.get('roll_angle', 0):.1f}°",
            'Pitch': f"{face_meta.get('pitch_angle', 0):.1f}°",
            'Yaw': f"{face_meta.get('yaw_angle', 0):.1f}°",
            'Eye/Width': f"{face_meta.get('eye_bbox_ratio', 0):.3f}",
            'Asymmetry': f"{face_meta.get('asymmetry_ratio', 0):.2f}"
        })
    df = pd.DataFrame(metrics_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Section 3: Distance matrix
    st.markdown("#### Intra-Cluster Distance Matrix")
    if len(face_indices) > 1:
        dist_matrix = compute_distance_matrix(face_indices, all_embeddings)
        fig = create_distance_matrix_plot(dist_matrix, face_indices)
        st.pyplot(fig)
    else:
        st.info("Only 1 face in cluster - no distance matrix")
    
    # Section 4: Nearest external faces
    st.markdown("#### 5 Nearest Faces Outside Cluster")
    nearest_external = find_nearest_external_faces(face_indices, all_embeddings, 
                                                   np.array(labels), k=5)
    if nearest_external:
        cols = st.columns(5)
        for i, (ext_idx, dist) in enumerate(nearest_external):
            crop_path = get_face_crop_path(results_dir, ext_idx)
            if crop_path.exists():
                with cols[i]:
                    st.image(str(crop_path), caption=f"#{ext_idx}\nDist: {dist:.3f}", 
                            use_container_width=True)
    else:
        st.info("No external faces found")


def create_distance_matrix_plot(dist_matrix: np.ndarray, face_indices: List[int]):
    """Create distance matrix heatmap."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(dist_matrix, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(face_indices)))
    ax.set_yticks(range(len(face_indices)))
    ax.set_xticklabels([f"#{i}" for i in face_indices], rotation=90, fontsize=8)
    ax.set_yticklabels([f"#{i}" for i in face_indices], fontsize=8)
    ax.set_title("Cosine Distance Matrix")
    plt.colorbar(im, ax=ax, label='Distance')
    plt.tight_layout()
    return fig


def create_inter_cluster_heatmap(
    cluster_ids: List[int],
    cluster_exemplars: Dict[int, List[int]],
    all_embeddings: np.ndarray,
    cluster_thresholds: Dict[int, float]
):
    """Create heatmap of minimum distances between cluster exemplars."""
    import matplotlib.pyplot as plt

    n = len(cluster_ids)
    min_dist_matrix = np.zeros((n, n))

    for i, c1 in enumerate(cluster_ids):
        for j, c2 in enumerate(cluster_ids):
            if i == j:
                min_dist_matrix[i, j] = 0
            elif i < j:
                exemplars_a = cluster_exemplars.get(c1, [])
                exemplars_b = cluster_exemplars.get(c2, [])
                if exemplars_a and exemplars_b:
                    emb_a = all_embeddings[exemplars_a]
                    emb_b = all_embeddings[exemplars_b]
                    dists = cdist(emb_a, emb_b, metric='euclidean')
                    min_dist = float(np.min(dists))
                    min_dist_matrix[i, j] = min_dist
                    min_dist_matrix[j, i] = min_dist

    fig, ax = plt.subplots(figsize=(12, 10))

    # Create threshold-relative coloring
    im = ax.imshow(min_dist_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0.3, vmax=0.9)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"C{c}" for c in cluster_ids], rotation=90, fontsize=8)
    ax.set_yticklabels([f"C{c}" for c in cluster_ids], fontsize=8)
    ax.set_title("Inter-Cluster Minimum Distance (Exemplar-to-Exemplar)")

    # Add text annotations
    for i in range(n):
        for j in range(n):
            if i != j:
                text = ax.text(j, i, f"{min_dist_matrix[i, j]:.2f}",
                              ha="center", va="center", color="black", fontsize=6)

    plt.colorbar(im, ax=ax, label='Min Distance')
    plt.tight_layout()
    return fig, min_dist_matrix


def render_algorithm_explanation():
    """Render expandable algorithm explanation."""
    with st.expander("📖 How the Hybrid HDBSCAN+kNN Algorithm Works", expanded=False):
        st.markdown("""
### Overview

The Hybrid HDBSCAN+kNN algorithm combines density-based clustering (HDBSCAN) with
local cohesion analysis to improve face clustering accuracy. It works in two stages:

---

### Stage 1: Initial Clustering (HDBSCAN)

HDBSCAN identifies dense regions of faces in embedding space. Faces that don't
belong to any dense region are marked as **noise** (label = -1).

---

### Stage 2: Iterative Refinement

For each cluster, we compute:

1. **d3 values**: For each face, the distance to its 3rd nearest neighbor within the cluster.
   - Faces with small d3 are "core" faces (well-connected)
   - Faces with large d3 are "peripheral" (loosely connected)

2. **Threshold T** (Tukey Fence): `T = Q3(d3) + 1.5 × IQR(d3)`
   - Q3 = 75th percentile of d3 values
   - IQR = Q3 - Q1 (interquartile range)
   - Clamped to [floor, ceiling] (default: [0.50, 0.90])
   - **Meaning**: T defines how far a face can be from cluster members and still "belong"

3. **Exemplars**: Top 10 faces with smallest d3 (most core-like faces)
   - Exemplars represent the cluster's "identity"
   - Used for merge and attach decisions

---

### Merge Decision

Two clusters A and B are merged if:
- **Threshold**: Use `min(T_A, T_B)` (more conservative)
- **Pairs**: At least `merge_min_pairs` (default: 3) exemplar pairs have distance ≤ threshold
- **Diversity**: At least `merge_min_distinct` (default: 2) distinct exemplars from EACH cluster are involved

**Why these rules?** Requiring multiple pairs from distinct exemplars prevents merging
based on a single outlier match. The stricter cluster's threshold is used to avoid
accidentally merging tight clusters with loose ones.

---

### Attach Decision

A noise point is attached to a cluster if:
- At least `attach_min_exemplars` (default: 2) exemplars are within the cluster's threshold T
- If multiple clusters qualify, choose the one with the most matches (then closest distance)

---

### Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `threshold_floor` | 0.50 | Minimum T for any cluster (prevents over-merging) |
| `threshold_ceiling` | 0.90 | Maximum T for any cluster (prevents under-merging) |
| `merge_min_pairs` | 3 | Required exemplar pairs within threshold to merge |
| `merge_min_distinct` | 2 | Required distinct exemplars per cluster for merge |
| `attach_min_exemplars` | 2 | Required exemplar matches to attach noise point |

---

### Debugging Tips

- **Clusters not merging?** Check the Merge Decisions tab to see which criterion failed
- **Wrong face attached?** Check Attachment Decisions to see which clusters qualified
- **Threshold too tight/loose?** Use Parameter Tuning to experiment with floor/ceiling
        """)


def render_debug_hybrid_knn(
    results_dir: Path,
    metadata: List[Dict],
    hybrid_knn_results: Dict[str, Any],
    all_embeddings: np.ndarray
):
    """Render the debug page for Hybrid kNN clustering."""
    st.header("🔧 Debug: Hybrid kNN")

    # Show algorithm explanation at the top
    render_algorithm_explanation()

    stats = hybrid_knn_results.get('stats', {})
    debug_data = stats.get('debug', {})
    labels = hybrid_knn_results.get('labels', [])

    if not debug_data:
        st.warning("No debug data available. Re-run benchmark with latest code to generate debug data.")
        st.info("Run: `python scripts/benchmark_face_clustering.py --album-path <path>`")
        return

    cluster_thresholds = debug_data.get('cluster_thresholds', {})
    cluster_exemplars = debug_data.get('cluster_exemplars', {})
    cluster_d3_stats = debug_data.get('cluster_d3_stats', {})
    merge_decisions = debug_data.get('merge_decisions', [])
    attach_decisions = debug_data.get('attach_decisions', [])

    # Convert string keys to int (JSON serialization issue)
    cluster_thresholds = {int(k): v for k, v in cluster_thresholds.items()}
    cluster_exemplars = {int(k): v for k, v in cluster_exemplars.items()}
    cluster_d3_stats = {int(k): v for k, v in cluster_d3_stats.items()}

    # Group faces by cluster
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        if label != -1:
            clusters[label].append(idx)

    sorted_cluster_ids = sorted(clusters.keys(), key=lambda x: len(clusters[x]), reverse=True)

    # Create tabs for different sections
    tabs = st.tabs([
        "📊 Cluster Overview",
        "🌐 Inter-Cluster Distances",
        "🔗 Merge Decisions",
        "📎 Attachment Decisions",
        "⚙️ Parameter Tuning",
        "📏 Face Distance Lookup"
    ])

    # Section 1: Enhanced Cluster Overview
    with tabs[0]:
        render_debug_cluster_overview(
            results_dir, clusters, cluster_thresholds, cluster_exemplars,
            cluster_d3_stats, sorted_cluster_ids
        )

    # Section 2: Inter-Cluster Distance Heatmap
    with tabs[1]:
        render_debug_inter_cluster_distances(
            sorted_cluster_ids, cluster_exemplars, all_embeddings,
            cluster_thresholds, merge_decisions
        )

    # Section 3: Merge Decision Explorer
    with tabs[2]:
        render_debug_merge_decisions(
            results_dir, merge_decisions, cluster_exemplars, all_embeddings,
            cluster_thresholds, stats.get('params', {})
        )

    # Section 4: Attachment Decision Explorer
    with tabs[3]:
        render_debug_attach_decisions(
            results_dir, attach_decisions, cluster_thresholds
        )

    # Section 5: Parameter Tuning
    with tabs[4]:
        render_debug_parameter_tuning(
            all_embeddings, labels, stats.get('params', {})
        )

    # Section 6: Face Distance Lookup
    with tabs[5]:
        render_debug_face_distance_lookup(
            results_dir, all_embeddings, labels, cluster_thresholds, len(metadata)
        )


def render_debug_cluster_overview(
    results_dir: Path,
    clusters: Dict[int, List[int]],
    cluster_thresholds: Dict[int, float],
    cluster_exemplars: Dict[int, List[int]],
    cluster_d3_stats: Dict[int, Dict],
    sorted_cluster_ids: List[int]
):
    """Section 1: Enhanced cluster overview with all faces (no truncation)."""
    st.subheader("📊 Cluster Overview (Full)")

    st.caption("""
    **What this shows:** Each cluster's computed threshold (T), the d3 statistics used to calculate it,
    and all faces in the cluster. Exemplars (faces with smallest d3 = most "core-like") are marked with ⭐.
    The threshold T determines how far a face can be from cluster members and still be considered part of the cluster.
    """)

    # Cluster summary table
    table_data = []
    for cid in sorted_cluster_ids:
        d3_stats = cluster_d3_stats.get(cid, {})
        table_data.append({
            'Cluster': cid,
            'Size': len(clusters[cid]),
            'Threshold': f"{cluster_thresholds.get(cid, 0):.3f}",
            'Q1': f"{d3_stats.get('q1', 0):.3f}",
            'Q3': f"{d3_stats.get('q3', 0):.3f}",
            'IQR': f"{d3_stats.get('iqr', 0):.3f}",
            'Raw T': f"{d3_stats.get('raw_threshold', 0):.3f}",
            'Exemplars': len(cluster_exemplars.get(cid, []))
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Cluster selection for full view
    selected_cluster = st.selectbox(
        "Select cluster for full view",
        sorted_cluster_ids,
        format_func=lambda x: f"Cluster {x} ({len(clusters[x])} faces, T={cluster_thresholds.get(x, 0):.3f})"
    )

    if selected_cluster is not None:
        face_indices = clusters[selected_cluster]
        exemplar_set = set(cluster_exemplars.get(selected_cluster, []))

        st.markdown(f"### Cluster {selected_cluster} - All {len(face_indices)} Faces")
        st.caption(f"Threshold: {cluster_thresholds.get(selected_cluster, 0):.3f} | "
                   f"Exemplars: {len(exemplar_set)} (shown with green border)")

        # Show ALL faces in scrollable grid (no truncation)
        cols_per_row = 12
        for row_start in range(0, len(face_indices), cols_per_row):
            row_indices = face_indices[row_start:row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for i, face_idx in enumerate(row_indices):
                crop_path = get_face_crop_path(results_dir, face_idx)
                if crop_path.exists():
                    is_exemplar = face_idx in exemplar_set
                    caption = f"#{face_idx}" + (" ⭐" if is_exemplar else "")
                    with cols[i]:
                        st.image(str(crop_path), caption=caption, use_container_width=True)


def render_debug_inter_cluster_distances(
    sorted_cluster_ids: List[int],
    cluster_exemplars: Dict[int, List[int]],
    all_embeddings: np.ndarray,
    cluster_thresholds: Dict[int, float],
    merge_decisions: List[Dict]
):
    """Section 2: Inter-cluster distance heatmap."""
    st.subheader("🌐 Inter-Cluster Distance Matrix")

    st.caption("""
    **What this shows:** The minimum embedding distance between each pair of clusters (using exemplar faces only).
    Green = close clusters that might merge. Red = distant clusters.
    The table below compares each pair's distance against their merge threshold (min of both T values).
    If min_distance ≤ merge_threshold, the pair *could* merge (but also needs enough exemplar pairs - see Merge Decisions tab).
    """)

    if len(sorted_cluster_ids) < 2:
        st.info("Need at least 2 clusters for distance matrix")
        return

    # Create and display heatmap
    fig, dist_matrix = create_inter_cluster_heatmap(
        sorted_cluster_ids, cluster_exemplars, all_embeddings, cluster_thresholds
    )
    st.pyplot(fig)

    # Distance table with merge threshold comparison
    st.markdown("#### Distance vs Merge Threshold")
    table_data = []
    for i, c1 in enumerate(sorted_cluster_ids):
        for j, c2 in enumerate(sorted_cluster_ids):
            if i < j:
                min_dist = dist_matrix[i, j]
                t1 = cluster_thresholds.get(c1, 0.5)
                t2 = cluster_thresholds.get(c2, 0.5)
                merge_t = min(t1, t2)
                could_merge = min_dist <= merge_t

                table_data.append({
                    'Cluster A': c1,
                    'Cluster B': c2,
                    'Min Dist': f"{min_dist:.3f}",
                    'T_A': f"{t1:.3f}",
                    'T_B': f"{t2:.3f}",
                    'Merge T': f"{merge_t:.3f}",
                    'Within T?': '✓' if could_merge else '✗'
                })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_debug_merge_decisions(
    results_dir: Path,
    merge_decisions: List[Dict],
    cluster_exemplars: Dict[int, List[int]],
    all_embeddings: np.ndarray,
    cluster_thresholds: Dict[int, float],
    params: Dict[str, Any]
):
    """Section 3: Merge decision explorer."""
    st.subheader("🔗 Merge Decision Explorer")

    st.caption("""
    **What this shows:** Every cluster pair that was evaluated for merging, and why it did or didn't merge.

    **Merge criteria** (ALL must be satisfied):
    1. At least `merge_min_pairs` exemplar pairs must have distance ≤ min(T_A, T_B)
    2. At least `merge_min_distinct` distinct exemplars from cluster A must be involved
    3. At least `merge_min_distinct` distinct exemplars from cluster B must be involved

    **Reason codes:**
    - `merged`: All criteria met, clusters were merged
    - `not_enough_pairs`: Too few exemplar pairs within threshold
    - `not_enough_distinct_a`: Not enough distinct exemplars from cluster A
    - `not_enough_distinct_b`: Not enough distinct exemplars from cluster B
    """)

    if not merge_decisions:
        st.info("No merge decisions recorded")
        return

    # Summary stats
    merged = [d for d in merge_decisions if d['merged']]
    not_merged = [d for d in merge_decisions if not d['merged']]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Pairs Evaluated", len(merge_decisions))
    with col2:
        st.metric("Merged", len(merged))
    with col3:
        st.metric("Not Merged", len(not_merged))

    # Filter options
    show_only = st.radio(
        "Show decisions:",
        ["All", "Merged Only", "Not Merged Only"],
        horizontal=True
    )

    if show_only == "Merged Only":
        decisions_to_show = merged
    elif show_only == "Not Merged Only":
        decisions_to_show = not_merged
    else:
        decisions_to_show = merge_decisions

    # Sort by min_distance
    decisions_to_show = sorted(decisions_to_show, key=lambda x: x['min_distance'])

    # Display as table
    table_data = []
    for d in decisions_to_show:
        table_data.append({
            'Cluster A': d['cluster_a'],
            'Cluster B': d['cluster_b'],
            'Threshold': f"{d['threshold']:.3f}",
            'Min Dist': f"{d['min_distance']:.3f}",
            'Pairs ≤ T': d['n_pairs_within'],
            'Req Pairs': params.get('merge_min_pairs', 3),
            'Distinct A': d['exemplars_a_involved'],
            'Distinct B': d['exemplars_b_involved'],
            'Req Distinct': params.get('merge_min_distinct', 2),
            'Merged?': '✓' if d['merged'] else '✗',
            'Reason': d['reason']
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Detailed view of selected pair
    st.markdown("#### Detailed Pair Analysis")
    if decisions_to_show:
        pair_options = [
            f"C{d['cluster_a']} ↔ C{d['cluster_b']} ({d['reason']})"
            for d in decisions_to_show
        ]
        selected_idx = st.selectbox("Select pair to inspect", range(len(pair_options)),
                                   format_func=lambda i: pair_options[i])

        decision = decisions_to_show[selected_idx]

        col1, col2 = st.columns(2)

        # Show exemplars from each cluster
        with col1:
            st.markdown(f"**Cluster {decision['cluster_a']} Exemplars**")
            exemplars_a = cluster_exemplars.get(decision['cluster_a'], [])
            cols = st.columns(min(5, len(exemplars_a)))
            for i, face_idx in enumerate(exemplars_a[:5]):
                crop_path = get_face_crop_path(results_dir, face_idx)
                if crop_path.exists():
                    with cols[i]:
                        st.image(str(crop_path), caption=f"#{face_idx}", use_container_width=True)

        with col2:
            st.markdown(f"**Cluster {decision['cluster_b']} Exemplars**")
            exemplars_b = cluster_exemplars.get(decision['cluster_b'], [])
            cols = st.columns(min(5, len(exemplars_b)))
            for i, face_idx in enumerate(exemplars_b[:5]):
                crop_path = get_face_crop_path(results_dir, face_idx)
                if crop_path.exists():
                    with cols[i]:
                        st.image(str(crop_path), caption=f"#{face_idx}", use_container_width=True)

        # Show cross-distance matrix if available
        cross_dists = decision.get('cross_distances')
        if cross_dists:
            st.markdown("**Cross-Exemplar Distance Matrix**")
            cross_dists = np.array(cross_dists)
            threshold = decision['threshold']

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(cross_dists, cmap='RdYlGn_r', vmin=0.3, vmax=0.9)

            # Annotate with values, highlight those within threshold
            for i in range(cross_dists.shape[0]):
                for j in range(cross_dists.shape[1]):
                    val = cross_dists[i, j]
                    color = 'green' if val <= threshold else 'black'
                    weight = 'bold' if val <= threshold else 'normal'
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                           color=color, fontweight=weight, fontsize=9)

            ax.set_xlabel(f"Cluster {decision['cluster_b']} Exemplars")
            ax.set_ylabel(f"Cluster {decision['cluster_a']} Exemplars")
            ax.set_title(f"Threshold: {threshold:.3f} | Pairs ≤ T: {decision['n_pairs_within']}")
            plt.colorbar(im, ax=ax, label='Distance')
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown(f"**Why {'merged' if decision['merged'] else 'not merged'}:** {decision['reason']}")


def render_debug_attach_decisions(
    results_dir: Path,
    attach_decisions: List[Dict],
    cluster_thresholds: Dict[int, float]
):
    """Section 4: Attachment decision explorer."""
    st.subheader("📎 Attachment Decision Explorer")

    st.caption("""
    **What this shows:** How noise points (faces not assigned by HDBSCAN) were evaluated for attachment to existing clusters.

    **Attachment criteria:**
    - The noise face must be within threshold T of at least `attach_min_exemplars` exemplars from a cluster
    - If multiple clusters qualify, the one with the most matches wins (ties broken by closest distance)
    - For small clusters (< attach_min_exemplars faces), only 1 match is required

    Select a face to see which clusters it was considered for and why it attached (or didn't).
    """)

    if not attach_decisions:
        st.info("No attachment decisions recorded")
        return

    # Summary
    attached = [d for d in attach_decisions if d['attached_to'] is not None]
    not_attached = [d for d in attach_decisions if d['attached_to'] is None]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Noise Points Evaluated", len(attach_decisions))
    with col2:
        st.metric("Attached", len(attached))
    with col3:
        st.metric("Remained Noise", len(not_attached))

    # Filter
    show_only = st.radio(
        "Show decisions:",
        ["All", "Attached Only", "Not Attached Only"],
        horizontal=True,
        key="attach_filter"
    )

    if show_only == "Attached Only":
        decisions_to_show = attached
    elif show_only == "Not Attached Only":
        decisions_to_show = not_attached
    else:
        decisions_to_show = attach_decisions

    if not decisions_to_show:
        st.info("No decisions to show with current filter")
        return

    # Select a face to inspect
    face_options = [f"Face #{d['face_idx']} → {d['attached_to'] if d['attached_to'] is not None else 'NOISE'}"
                   for d in decisions_to_show]
    selected_idx = st.selectbox("Select face to inspect", range(len(face_options)),
                               format_func=lambda i: face_options[i])

    decision = decisions_to_show[selected_idx]

    # Show the face
    col1, col2 = st.columns([1, 3])
    with col1:
        crop_path = get_face_crop_path(results_dir, decision['face_idx'])
        if crop_path.exists():
            st.image(str(crop_path), caption=f"Face #{decision['face_idx']}", use_container_width=True)

        if decision['attached_to'] is not None:
            st.success(f"Attached to Cluster {decision['attached_to']}")
        else:
            st.error("Not attached (remained noise)")

    # Show candidates
    with col2:
        st.markdown("**Candidates (all clusters considered):**")
        candidates = decision.get('candidates', [])

        if candidates:
            table_data = []
            for c in candidates:
                table_data.append({
                    'Cluster': c['cluster'],
                    'Threshold': f"{c['threshold']:.3f}",
                    'Matches': c['matches'],
                    'Required': c['required'],
                    'Min Dist': f"{c['min_dist']:.3f}",
                    'Qualifies?': '✓' if c['qualifies'] else '✗'
                })

            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No candidate information available")


def render_debug_parameter_tuning(
    all_embeddings: np.ndarray,
    current_labels: List[int],
    current_params: Dict[str, Any]
):
    """Section 5: Parameter tuning with re-run capability."""
    st.subheader("⚙️ Parameter Tuning")

    st.caption("""
    **What this does:** Lets you experiment with different parameter values and immediately see the effect on clustering.

    **Tips:**
    - **Too many clusters?** Lower `threshold_floor` or `merge_min_pairs` to encourage more merging
    - **Clusters too large (mixing identities)?** Raise `threshold_floor` or `merge_min_pairs`
    - **Noise points not attaching?** Lower `attach_min_exemplars` to 1
    - **Wrong faces attaching?** Raise `attach_min_exemplars` to 3

    Changes here are temporary - they don't modify the saved benchmark results.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Threshold Parameters**")
        threshold_floor = st.slider(
            "Threshold Floor",
            min_value=0.3, max_value=0.7,
            value=float(current_params.get('threshold_floor', 0.5)),
            step=0.05,
            help="Minimum threshold for any cluster"
        )
        threshold_ceiling = st.slider(
            "Threshold Ceiling",
            min_value=0.5, max_value=1.2,
            value=float(current_params.get('threshold_ceiling', 0.9)),
            step=0.05,
            help="Maximum threshold for any cluster"
        )

    with col2:
        st.markdown("**Merge/Attach Parameters**")
        merge_min_pairs = st.slider(
            "Merge Min Pairs",
            min_value=1, max_value=5,
            value=int(current_params.get('merge_min_pairs', 3)),
            help="Minimum cross-exemplar pairs within threshold to merge"
        )
        merge_min_distinct = st.slider(
            "Merge Min Distinct",
            min_value=1, max_value=3,
            value=int(current_params.get('merge_min_distinct', 2)),
            help="Minimum distinct exemplars from each cluster involved"
        )
        attach_min_exemplars = st.slider(
            "Attach Min Exemplars",
            min_value=1, max_value=3,
            value=int(current_params.get('attach_min_exemplars', 2)),
            help="Minimum exemplars within threshold to attach noise point"
        )

    if st.button("🔄 Re-run Clustering with New Parameters", type="primary"):
        with st.spinner("Running clustering..."):
            from sim_bench.clustering.hybrid_hdbscan_knn import HybridHDBSCANKNN

            new_config = {
                'method': 'hybrid_hdbscan_knn',
                'params': {
                    'min_cluster_size': current_params.get('min_cluster_size', 2),
                    'min_samples': current_params.get('min_samples', 2),
                    'cluster_selection_epsilon': current_params.get('cluster_selection_epsilon', 0.3),
                    'knn_k': current_params.get('knn_k', 3),
                    'threshold_floor': threshold_floor,
                    'threshold_ceiling': threshold_ceiling,
                    'max_exemplars': current_params.get('max_exemplars', 10),
                    'attach_min_exemplars': attach_min_exemplars,
                    'merge_min_pairs': merge_min_pairs,
                    'merge_min_distinct': merge_min_distinct,
                    'max_iterations': current_params.get('max_iterations', 10),
                }
            }

            clusterer = HybridHDBSCANKNN(new_config)
            new_labels, new_stats = clusterer.cluster(all_embeddings, collect_debug_data=True)

        # Compare results
        st.markdown("### Results Comparison")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Original**")
            original_clusters = len(set(l for l in current_labels if l >= 0))
            original_noise = sum(1 for l in current_labels if l == -1)
            st.metric("Clusters", original_clusters)
            st.metric("Noise", original_noise)

        with col2:
            st.markdown("**New Parameters**")
            st.metric("Clusters", new_stats['n_clusters'],
                     delta=new_stats['n_clusters'] - original_clusters)
            st.metric("Noise", new_stats['n_noise'],
                     delta=new_stats['n_noise'] - original_noise)
            st.metric("Merges", new_stats['total_merges'])
            st.metric("Attached", new_stats['total_attached'])


def render_debug_face_distance_lookup(
    results_dir: Path,
    all_embeddings: np.ndarray,
    labels: List[int],
    cluster_thresholds: Dict[int, float],
    n_faces: int
):
    """Section 6: Face distance lookup tool."""
    st.subheader("📏 Face Distance Lookup")

    st.caption("""
    **What this does:** Shows the embedding distance between any two faces and whether that distance
    is within the relevant cluster thresholds.

    **Use cases:**
    - "Why aren't these two faces in the same cluster?" → Check if distance > threshold
    - "Why did this face attach to cluster X instead of Y?" → Compare distances to each cluster's threshold
    - "Are these two faces the same person?" → Low distance (< 0.5) suggests same identity
    """)

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        face_a = st.number_input("Face A Index", min_value=0, max_value=n_faces-1, value=0)
        crop_path_a = get_face_crop_path(results_dir, face_a)
        if crop_path_a.exists():
            st.image(str(crop_path_a), caption=f"Face #{face_a}", use_container_width=True)
        label_a = labels[face_a] if face_a < len(labels) else -1
        st.caption(f"Cluster: {label_a}")

    with col2:
        face_b = st.number_input("Face B Index", min_value=0, max_value=n_faces-1, value=1)
        crop_path_b = get_face_crop_path(results_dir, face_b)
        if crop_path_b.exists():
            st.image(str(crop_path_b), caption=f"Face #{face_b}", use_container_width=True)
        label_b = labels[face_b] if face_b < len(labels) else -1
        st.caption(f"Cluster: {label_b}")

    with col3:
        if face_a < len(all_embeddings) and face_b < len(all_embeddings):
            emb_a = all_embeddings[face_a]
            emb_b = all_embeddings[face_b]

            # Euclidean distance (what the algorithm uses)
            euclidean_dist = float(np.linalg.norm(emb_a - emb_b))
            # Cosine distance for reference
            cosine_dist = float(1 - np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b)))

            st.markdown("### Distance Metrics")
            st.metric("Euclidean Distance", f"{euclidean_dist:.4f}")
            st.metric("Cosine Distance", f"{cosine_dist:.4f}")

            # Threshold comparison
            st.markdown("### Threshold Comparison")

            if label_a >= 0:
                t_a = cluster_thresholds.get(label_a, 0.5)
                within_a = euclidean_dist <= t_a
                st.write(f"Cluster {label_a} threshold: {t_a:.3f} → {'✓ Within' if within_a else '✗ Outside'}")

            if label_b >= 0:
                t_b = cluster_thresholds.get(label_b, 0.5)
                within_b = euclidean_dist <= t_b
                st.write(f"Cluster {label_b} threshold: {t_b:.3f} → {'✓ Within' if within_b else '✗ Outside'}")

            if label_a >= 0 and label_b >= 0 and label_a != label_b:
                merge_t = min(cluster_thresholds.get(label_a, 0.5), cluster_thresholds.get(label_b, 0.5))
                would_contribute = euclidean_dist <= merge_t
                st.write(f"Merge threshold (min): {merge_t:.3f} → {'✓ Would contribute to merge' if would_contribute else '✗ Too far'}")
        else:
            st.warning("Invalid face indices")


def main():
    st.title("👥 Face Clustering Comparison")
    
    # Sidebar: File selection
    results_dir = Path('results/face_clustering_benchmark')
    available_benchmarks = get_available_benchmarks(results_dir)
    
    if not available_benchmarks:
        st.error(f"No benchmark results found in {results_dir}")
        st.info("Run: `python scripts/benchmark_face_clustering.py --album-path <path>`")
        return
    
    selected_file = st.sidebar.selectbox(
        "Select benchmark result",
        available_benchmarks,
        format_func=lambda x: x.name
    )
    
    results = load_benchmark_results(selected_file)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Album:** {Path(results['album_path']).name}")
    st.sidebar.markdown(f"**Total Faces:** {results['total_faces']}")
    st.sidebar.markdown(f"**Timestamp:** {results['timestamp']}")
    
    # Page selection
    page = st.sidebar.radio(
        "Select View",
        ["📊 Overview & Comparison", "🔍 HDBSCAN Explorer", "🔍 Hybrid kNN Explorer",
         "🔍 Hybrid Closest Explorer", "🔧 Debug: Hybrid kNN"]
    )
    
    metadata = results['face_metadata']
    methods = results['methods']
    
    # Load embeddings if available
    embeddings_filename = results.get('embeddings_file')
    all_embeddings = None
    if embeddings_filename:
        embeddings_path = results_dir / embeddings_filename
        all_embeddings = load_embeddings(embeddings_path)
        if all_embeddings is not None:
            st.sidebar.success(f"✓ Embeddings loaded ({all_embeddings.shape[0]} faces)")
        else:
            st.sidebar.warning(f"⚠️ Embeddings file not found: {embeddings_filename}")
    else:
        st.sidebar.warning("⚠️ No embeddings in this benchmark (older format)")
    
    # Route to page
    if page == "📊 Overview & Comparison":
        st.header("Overview & Comparison")
        
        # Metrics comparison
        ncols = 3 if 'hybrid_closest' in methods else 2
        cols = st.columns(ncols)
        
        with cols[0]:
            render_cluster_stats_table("HDBSCAN", 
                                      methods['hdbscan'].get('cluster_stats', []),
                                      methods['hdbscan']['labels'])
        with cols[1]:
            render_cluster_stats_table("Hybrid kNN",
                                      methods['hybrid_knn'].get('cluster_stats', []),
                                      methods['hybrid_knn']['labels'])
        if ncols == 3:
            with cols[2]:
                render_cluster_stats_table("Hybrid Closest",
                                          methods['hybrid_closest'].get('cluster_stats', []),
                                          methods['hybrid_closest']['labels'])
        
        st.markdown("---")
        
        # Cluster galleries
        cols = st.columns(ncols)
        with cols[0]:
            render_cluster_gallery("HDBSCAN", methods['hdbscan']['labels'], 
                                  metadata, results_dir)
        with cols[1]:
            render_cluster_gallery("Hybrid kNN", methods['hybrid_knn']['labels'],
                                  metadata, results_dir)
        if ncols == 3:
            with cols[2]:
                render_cluster_gallery("Hybrid Closest", methods['hybrid_closest']['labels'],
                                      metadata, results_dir)
    
    elif page == "🔍 HDBSCAN Explorer":
        if all_embeddings is None:
            st.error("⚠️ Embeddings not available. Re-run benchmark to include embeddings.")
        else:
            render_cluster_explorer(results_dir, metadata, methods['hdbscan']['labels'],
                                   "HDBSCAN", all_embeddings)
    
    elif page == "🔍 Hybrid kNN Explorer":
        if all_embeddings is None:
            st.error("⚠️ Embeddings not available. Re-run benchmark to include embeddings.")
        else:
            render_cluster_explorer(results_dir, metadata, methods['hybrid_knn']['labels'],
                                   "Hybrid HDBSCAN+kNN", all_embeddings)
    
    elif page == "🔍 Hybrid Closest Explorer":
        if all_embeddings is None:
            st.error("⚠️ Embeddings not available. Re-run benchmark to include embeddings.")
        elif 'hybrid_closest' not in methods:
            st.error("⚠️ Hybrid Closest results not in this benchmark. Re-run with updated config.")
        else:
            render_cluster_explorer(results_dir, metadata, methods['hybrid_closest']['labels'],
                                   "Hybrid Closest-Face", all_embeddings)

    elif page == "🔧 Debug: Hybrid kNN":
        if all_embeddings is None:
            st.error("⚠️ Embeddings not available. Re-run benchmark to include embeddings.")
        elif 'hybrid_knn' not in methods:
            st.error("⚠️ Hybrid kNN results not in this benchmark.")
        else:
            render_debug_hybrid_knn(results_dir, metadata, methods['hybrid_knn'], all_embeddings)


if __name__ == "__main__":
    main()
