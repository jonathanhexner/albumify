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
        ["📊 Overview & Comparison", "🔍 HDBSCAN Explorer", "🔍 Hybrid kNN Explorer", "🔍 Hybrid Closest Explorer"]
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


if __name__ == "__main__":
    main()
