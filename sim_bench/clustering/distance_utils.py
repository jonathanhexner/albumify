"""
Distance computation utilities for clustering algorithms.

All functions assume L2-normalized input vectors and use cosine distance:
    cosine_distance = 1 - cosine_similarity

Values are clipped to [0, 2] for numeric safety.
"""

import numpy as np


def cosine_distance_matrix(X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
    """Compute cosine distance matrix between X and Y (or X and X if Y is None).

    Cosine distance = 1 - cosine_similarity, clipped to [0, 2] for numeric safety.
    Assumes X and Y are already L2-normalized.

    Args:
        X: Feature matrix [n, d], L2-normalized
        Y: Feature matrix [m, d], L2-normalized. If None, computes X vs X.

    Returns:
        Distance matrix [n, m] with values in [0, 2], dtype float64
    """
    is_self = Y is None
    if is_self:
        Y = X
    # Use float64 for numerical stability and HDBSCAN compatibility
    similarity = np.asarray(X, dtype=np.float64) @ np.asarray(Y, dtype=np.float64).T
    distance = 1.0 - similarity
    distance = np.clip(distance, 0.0, 2.0)
    # Ensure exact zeros on diagonal for self-distance matrices
    if is_self:
        np.fill_diagonal(distance, 0.0)
    return distance


def cosine_distance_pairwise(X: np.ndarray) -> np.ndarray:
    """Compute condensed pairwise cosine distances (like scipy.spatial.distance.pdist).

    Args:
        X: Feature matrix [n, d], L2-normalized

    Returns:
        Condensed distance array of length n*(n-1)/2, dtype float64
    """
    n = len(X)
    if n < 2:
        return np.array([], dtype=np.float64)

    # Compute full distance matrix
    dist_matrix = cosine_distance_matrix(X)

    # Extract upper triangle (excluding diagonal) as condensed form
    indices = np.triu_indices(n, k=1)
    return dist_matrix[indices]


def cosine_distance_to_set(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute cosine distance from a single vector to a set of vectors.

    Args:
        x: Single feature vector [d] or [1, d]
        Y: Feature matrix [m, d], L2-normalized

    Returns:
        Distance array [m] with values in [0, 2], dtype float64
    """
    x = np.atleast_2d(x)
    return cosine_distance_matrix(x, Y).ravel()


# ---------------------------------------------------------------------------
# Cluster debug utilities
# These functions work with raw distance_matrix + cluster_members for
# frontend/debug tools to compute derived metrics on demand.
# ---------------------------------------------------------------------------

from typing import Dict, List, Tuple, Optional


def closest_distance_to_cluster(
    sample_idx: int,
    cluster_id: int,
    distance_matrix: np.ndarray,
    cluster_members: Dict[int, List[int]]
) -> float:
    """
    Get the minimum distance from a sample to any member of a cluster.

    Args:
        sample_idx: Index of the sample
        cluster_id: ID of the target cluster
        distance_matrix: Pairwise distance matrix (n_samples, n_samples)
        cluster_members: Dict mapping cluster_id -> list of sample indices

    Returns:
        Minimum distance to any cluster member, or np.inf if cluster is empty
    """
    members = cluster_members.get(cluster_id, [])
    if not members:
        return np.inf

    # Exclude self if sample is in the cluster
    other_members = [m for m in members if m != sample_idx]
    if not other_members:
        return np.inf

    distances = distance_matrix[sample_idx, other_members]
    return float(np.min(distances))


def all_cluster_distances(
    sample_idx: int,
    distance_matrix: np.ndarray,
    cluster_members: Dict[int, List[int]],
    exclude_noise: bool = True
) -> Dict[int, float]:
    """
    Get minimum distance from a sample to each cluster.

    Args:
        sample_idx: Index of the sample
        distance_matrix: Pairwise distance matrix
        cluster_members: Dict mapping cluster_id -> list of sample indices
        exclude_noise: If True, skip cluster_id = -1

    Returns:
        Dict mapping cluster_id -> minimum distance to that cluster
    """
    result = {}
    for cluster_id, members in cluster_members.items():
        if exclude_noise and cluster_id == -1:
            continue
        if not members:
            continue

        # Exclude self
        other_members = [m for m in members if m != sample_idx]
        if not other_members:
            result[cluster_id] = np.inf
        else:
            result[cluster_id] = float(np.min(distance_matrix[sample_idx, other_members]))

    return result


def support_count(
    sample_idx: int,
    cluster_id: int,
    distance_matrix: np.ndarray,
    cluster_members: Dict[int, List[int]],
    radius: float
) -> int:
    """
    Count neighbors in a cluster within a given radius.

    Args:
        sample_idx: Index of the sample
        cluster_id: ID of the cluster to check
        distance_matrix: Pairwise distance matrix
        cluster_members: Dict mapping cluster_id -> list of sample indices
        radius: Maximum distance to count as a neighbor

    Returns:
        Number of cluster members within radius (excluding self)
    """
    members = cluster_members.get(cluster_id, [])
    if not members:
        return 0

    # Exclude self
    other_members = [m for m in members if m != sample_idx]
    if not other_members:
        return 0

    distances = distance_matrix[sample_idx, other_members]
    return int(np.sum(distances <= radius))


def separation_margin(
    sample_idx: int,
    current_cluster: int,
    distance_matrix: np.ndarray,
    cluster_members: Dict[int, List[int]]
) -> Tuple[float, Optional[int]]:
    """
    Compute the separation margin: how much closer is the sample to its
    current cluster compared to the next-best cluster.

    Args:
        sample_idx: Index of the sample
        current_cluster: ID of the sample's current cluster
        distance_matrix: Pairwise distance matrix
        cluster_members: Dict mapping cluster_id -> list of sample indices

    Returns:
        Tuple of (margin, next_best_cluster_id)
        - margin: dist_to_next_best - dist_to_current (positive = good separation)
        - next_best_cluster_id: ID of the next closest cluster (None if only one cluster)
    """
    all_dists = all_cluster_distances(sample_idx, distance_matrix, cluster_members)

    if current_cluster not in all_dists:
        return 0.0, None

    dist_to_current = all_dists[current_cluster]

    # Find next best cluster
    other_clusters = {c: d for c, d in all_dists.items() if c != current_cluster}
    if not other_clusters:
        return np.inf, None

    next_best = min(other_clusters.items(), key=lambda x: x[1])
    next_best_cluster, dist_to_next = next_best

    margin = dist_to_next - dist_to_current
    return float(margin), next_best_cluster


def compute_cluster_members_from_labels(labels: np.ndarray) -> Dict[int, List[int]]:
    """
    Convert labels array to cluster_members dict.

    Args:
        labels: Array of cluster labels

    Returns:
        Dict mapping cluster_id -> list of sample indices
    """
    cluster_members: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        label = int(label)
        if label not in cluster_members:
            cluster_members[label] = []
        cluster_members[label].append(idx)
    return cluster_members
