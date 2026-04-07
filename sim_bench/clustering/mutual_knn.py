"""
Mutual K-Nearest Neighbors clustering using connected components.

Algorithm:
1. L2-normalize each embedding row.
2. Compute cosine similarity matrix: S = E @ E.T
3. For each embedding i, find its top-k most similar neighbors (excluding itself).
4. Construct undirected graph using mutual-KNN with similarity threshold:
   - Add edge between i and j only if:
     (j is in top-k neighbors of i) AND (i is in top-k neighbors of j) AND (S[i,j] >= threshold)
5. Run connected components on the graph.
6. Each connected component is a cluster.
"""

import numpy as np
from typing import Dict, Any, Tuple
import logging

from sim_bench.clustering.base import ClusteringMethod

logger = logging.getLogger(__name__)


class MutualKNNClusterer(ClusteringMethod):
    """Mutual KNN clustering using connected components."""

    doc_explanation = """
Mutual KNN builds a graph where edges connect mutually close faces.
Edge (i,j) exists if j is in i's top-k neighbors AND i is in j's top-k neighbors
AND similarity(i,j) >= threshold. Clusters are connected components.

Decision: Two faces cluster together if they're mutual k-nearest neighbors
with similarity above threshold. No noise concept - all faces get a cluster.

Simple and interpretable but sensitive to k and threshold choices.
"""

    decision_parameters = {
        "k": {
            "description": "Number of nearest neighbors to consider",
            "default": 10,
            "decision_role": "Edge requires mutual top-k relationship"
        },
        "similarity_threshold": {
            "description": "Minimum cosine similarity for edge creation",
            "default": 0.70,
            "decision_role": "Edge requires similarity >= this (cosine, so 0.7 = close)"
        },
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.k = self.params.get('k', 10)
        self.similarity_threshold = self.params.get('similarity_threshold', 0.70)

    def cluster(
        self,
        features: np.ndarray,
        collect_debug_data: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Cluster features using mutual KNN connected components.

        Args:
            features: Feature matrix [n_samples, n_features]
            collect_debug_data: If True, collect debug data

        Returns:
            labels: Cluster labels (0-indexed, no noise concept)
            stats: Dictionary with clustering statistics
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        n_samples = features.shape[0]

        if n_samples == 0:
            return np.array([], dtype=np.int32), self._empty_stats()

        if n_samples == 1:
            return np.array([0], dtype=np.int32), self._single_sample_stats()

        # Step 1: L2-normalize embeddings
        normalized = self.normalize_features(features)

        # Step 2: Compute cosine similarity matrix
        similarity = normalized @ normalized.T

        # Step 3: Find top-k neighbors for each sample (excluding self)
        # Set diagonal to -inf so self is never selected as neighbor
        np.fill_diagonal(similarity, -np.inf)

        # Get indices of top-k neighbors for each row
        # argsort returns ascending, so we negate and take first k
        k_actual = min(self.k, n_samples - 1)
        top_k_indices = np.argsort(-similarity, axis=1)[:, :k_actual]

        # Build set of (i, j) pairs where j is in top-k of i
        top_k_sets = [set(top_k_indices[i]) for i in range(n_samples)]

        # Step 4: Build mutual KNN graph
        # Edge (i, j) exists if:
        #   - j in top_k(i)
        #   - i in top_k(j)
        #   - similarity[i, j] >= threshold

        # Restore diagonal for threshold checking (not needed, but cleaner)
        np.fill_diagonal(similarity, 1.0)

        edges_row = []
        edges_col = []

        for i in range(n_samples):
            for j in top_k_sets[i]:
                if j > i:  # Only process each pair once
                    if i in top_k_sets[j] and similarity[i, j] >= self.similarity_threshold:
                        edges_row.append(i)
                        edges_col.append(j)
                        edges_row.append(j)
                        edges_col.append(i)

        n_edges = len(edges_row) // 2  # Each edge added twice for symmetry

        logger.info(
            f"Mutual KNN: {n_samples} samples, k={k_actual}, "
            f"threshold={self.similarity_threshold}, edges={n_edges}"
        )

        # Step 5: Build sparse adjacency matrix and find connected components
        if n_edges > 0:
            data = np.ones(len(edges_row), dtype=np.int8)
            adjacency = csr_matrix(
                (data, (edges_row, edges_col)),
                shape=(n_samples, n_samples)
            )
            n_components, labels = connected_components(
                adjacency, directed=False, return_labels=True
            )
        else:
            # No edges: each sample is its own cluster
            n_components = n_samples
            labels = np.arange(n_samples, dtype=np.int32)

        # Step 6: Compute statistics
        stats = self._compute_stats(labels, n_components, n_edges, k_actual)

        return labels, stats

    def _compute_stats(
        self,
        labels: np.ndarray,
        n_components: int,
        n_edges: int,
        k_actual: int
    ) -> Dict[str, Any]:
        """Compute clustering statistics."""
        cluster_sizes = {}
        for label in range(n_components):
            size = int(np.sum(labels == label))
            if size > 0:
                cluster_sizes[int(label)] = size

        # Count singletons (clusters of size 1)
        n_singletons = sum(1 for size in cluster_sizes.values() if size == 1)

        stats = {
            'algorithm': 'mutual_knn',
            'n_clusters': n_components,
            'n_singletons': n_singletons,
            'n_edges': n_edges,
            'cluster_sizes': cluster_sizes,
            'params': {
                'k': k_actual,
                'similarity_threshold': self.similarity_threshold
            }
        }

        return stats

    def _empty_stats(self) -> Dict[str, Any]:
        """Return stats for empty input."""
        return {
            'algorithm': 'mutual_knn',
            'n_clusters': 0,
            'n_singletons': 0,
            'n_edges': 0,
            'cluster_sizes': {},
            'params': {
                'k': self.k,
                'similarity_threshold': self.similarity_threshold
            }
        }

    def _single_sample_stats(self) -> Dict[str, Any]:
        """Return stats for single sample input."""
        return {
            'algorithm': 'mutual_knn',
            'n_clusters': 1,
            'n_singletons': 1,
            'n_edges': 0,
            'cluster_sizes': {0: 1},
            'params': {
                'k': self.k,
                'similarity_threshold': self.similarity_threshold
            }
        }
