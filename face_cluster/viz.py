"""Visualization helpers for face clustering notebook."""

import logging
from typing import List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

from face_cluster.types import FaceRecord, GraphResult, ClusterResult

logger = logging.getLogger(__name__)


def plot_distance_matrix(
    distance_matrix: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "Distance Matrix"
):
    """Plot distance matrix heatmap.

    Args:
        distance_matrix: n x n distance matrix
        labels: Optional cluster labels for ordering
        title: Plot title
    """
    if distance_matrix.size == 0:
        logger.warning("Empty distance matrix")
        return

    n = distance_matrix.shape[0]

    # Reorder by cluster labels if provided
    if labels is not None:
        order = np.argsort(labels)
        distance_matrix = distance_matrix[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(distance_matrix, cmap='RdYlGn_r', vmin=0, vmax=1, aspect='auto')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Face Index')
    ax.set_ylabel('Face Index')
    plt.colorbar(im, ax=ax, label='Cosine Distance')

    if labels is not None:
        # Add cluster boundaries
        cluster_ids = np.unique(labels[labels >= 0])
        boundaries = []
        for cid in cluster_ids:
            cluster_indices = np.where(labels == cid)[0]
            if len(cluster_indices) > 0:
                boundaries.append(cluster_indices[-1] + 0.5)

        for boundary in boundaries[:-1]:  # Skip last boundary
            ax.axhline(boundary, color='white', linewidth=2, alpha=0.7)
            ax.axvline(boundary, color='white', linewidth=2, alpha=0.7)

    plt.tight_layout()
    plt.show()


def plot_knn_table(
    neighbors: List[List[int]],
    neighbor_distances: List[List[float]],
    max_samples: int = 10
):
    """Print table of kNN neighbors and distances.

    Args:
        neighbors: List of neighbor indices per sample
        neighbor_distances: List of neighbor distances per sample
        max_samples: Maximum number of samples to show
    """
    logger.info(f"\nTop-K Nearest Neighbors (showing first {max_samples} samples):\n")
    logger.info(f"{'Sample':<8} {'Neighbors':<40} {'Distances':<40}")
    logger.info("-" * 90)

    for i in range(min(max_samples, len(neighbors))):
        neighbor_str = ", ".join(f"{n:3d}" for n in neighbors[i][:5])
        distance_str = ", ".join(f"{d:.3f}" for d in neighbor_distances[i][:5])
        logger.debug(f"{i:<8} {neighbor_str:<40} {distance_str:<40}")


def plot_graph(
    graph_result: GraphResult,
    labels: np.ndarray,
    title: str = "Mutual kNN Graph"
):
    """Plot graph with nodes colored by cluster.

    Args:
        graph_result: Graph result with NetworkX graph
        labels: Cluster labels for coloring
        title: Plot title
    """
    G = graph_result.G

    if G.number_of_nodes() == 0:
        logger.warning("Empty graph")
        return

    fig, ax = plt.subplots(figsize=(14, 10))

    # Layout
    try:
        pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)
    except:
        pos = nx.random_layout(G, seed=42)

    # Color by cluster
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    for label in unique_labels:
        mask = labels == label
        node_list = [i for i in range(len(labels)) if mask[i]]

        if label == -1:
            # Noise - gray, smaller
            nx.draw_networkx_nodes(
                G, pos, nodelist=node_list,
                node_color='gray', node_size=100, alpha=0.5, ax=ax
            )
        else:
            color = colors[label % len(colors)]
            nx.draw_networkx_nodes(
                G, pos, nodelist=node_list,
                node_color=[color], node_size=200, alpha=0.8, ax=ax,
                label=f'C{label} ({int(mask.sum())})'
            )

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, ax=ax)

    ax.set_title(title, fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def print_cluster_summary(cluster_result: ClusterResult):
    """Print summary table of clusters.

    Args:
        cluster_result: Cluster result with statistics
    """
    logger.info(f"\nClustering Summary:")
    logger.info(f"  Clusters: {cluster_result.n_clusters}")
    logger.info(f"  Noise: {cluster_result.n_noise}")
    logger.info("")

    if cluster_result.n_clusters == 0:
        logger.warning("No clusters found")
        return

    # Sort clusters by size
    sorted_clusters = sorted(
        cluster_result.clusters.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    logger.info(f"{'Cluster':<8} {'Size':<6} {'Diameter':<10} {'Median':<10} {'P95':<10} {'Exemplars':<10}")
    logger.info("-" * 70)

    for cluster_id, nodes in sorted_clusters:
        stats = cluster_result.cluster_stats[cluster_id]
        size = stats.get('size', len(nodes))
        diameter = stats.get('diameter', 0.0)
        median = stats.get('median_dist', 0.0)
        p95 = stats.get('p95_dist', 0.0)
        n_exemplars = len(cluster_result.exemplars.get(cluster_id, []))

        logger.debug(
            f"{cluster_id:<8} {size:<6} {diameter:<10.3f} {median:<10.3f} "
            f"{p95:<10.3f} {n_exemplars:<10}"
        )


def show_face_grid(
    faces: List[FaceRecord],
    indices: List[int],
    title: str = "Faces",
    max_faces: int = 50,
    show_metadata: bool = True
):
    """Show grid of face images with metadata.

    Args:
        faces: List of all face records
        indices: Indices of faces to show
        title: Grid title
        max_faces: Maximum number of faces to show
        show_metadata: Whether to show face metadata (ID, blur, pose)
    """
    if len(indices) == 0:
        logger.warning(f"No faces to show for: {title}")
        return

    indices = indices[:max_faces]
    n = len(indices)
    cols = min(10, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 2))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(f"{title} ({n} faces)", fontsize=14)

    for idx, face_idx in enumerate(indices):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        face = faces[face_idx]

        if face.aligned_face is not None:
            ax.imshow(face.aligned_face)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center')

        if show_metadata:
            # Build metadata string
            meta_lines = [f"ID: {face.face_id}"]

            if face.blur_score > 0:
                meta_lines.append(f"Blur: {face.blur_score:.0f}")

            if face.pose is not None:
                yaw, pitch, roll = face.pose
                meta_lines.append(f"Y:{yaw:.0f} P:{pitch:.0f} R:{roll:.0f}")

            meta_lines.append(f"Area: {face.area:.0f}")

            title_text = '\n'.join(meta_lines)
            ax.set_title(title_text, fontsize=8)

        ax.axis('off')

    # Hide unused subplots
    for idx in range(n, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()


def show_cluster_faces(
    faces: List[FaceRecord],
    cluster_result: ClusterResult,
    max_clusters: int = 10,
    max_faces_per_cluster: int = 20,
    highlight_exemplars: bool = True
):
    """Show faces for each cluster.

    Args:
        faces: List of all face records
        cluster_result: Cluster result with assignments
        max_clusters: Maximum number of clusters to show
        max_faces_per_cluster: Maximum faces to show per cluster
        highlight_exemplars: Whether to highlight exemplar faces
    """
    # Sort clusters by size
    sorted_clusters = sorted(
        cluster_result.clusters.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    for cluster_id, nodes in sorted_clusters[:max_clusters]:
        stats = cluster_result.cluster_stats[cluster_id]
        exemplars = cluster_result.exemplars.get(cluster_id, [])

        title = (
            f"Cluster {cluster_id} (size={stats['size']}, "
            f"diameter={stats.get('diameter', 0):.3f}, exemplars={len(exemplars)})"
        )

        # Show exemplars first, then other faces
        if highlight_exemplars and len(exemplars) > 0:
            other_faces = [n for n in nodes if n not in exemplars]
            ordered_nodes = exemplars + other_faces
        else:
            ordered_nodes = nodes

        show_face_grid(
            faces,
            ordered_nodes[:max_faces_per_cluster],
            title=title,
            show_metadata=True
        )


def plot_d10_histogram(
    d10_values: np.ndarray,
    threshold: float,
    cluster_id: int
):
    """Plot histogram of d10 values for a cluster.

    Args:
        d10_values: Array of d10 values
        threshold: Exemplar threshold
        cluster_id: Cluster ID for title
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(d10_values, bins=20, alpha=0.7, edgecolor='black', color='steelblue')
    ax.axvline(
        threshold, color='red', linestyle='--', linewidth=2,
        label=f'Threshold={threshold:.3f}'
    )
    ax.axvline(
        np.median(d10_values), color='green', linestyle='--', linewidth=2,
        label=f'Median={np.median(d10_values):.3f}'
    )

    ax.set_xlabel(f'd{len(d10_values)} (distance to kth neighbor)')
    ax.set_ylabel('Count')
    ax.set_title(f'Cluster {cluster_id}: d10 Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
