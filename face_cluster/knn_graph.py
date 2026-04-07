"""Mutual kNN graph construction with distance threshold."""

import logging
from typing import List
import numpy as np
import networkx as nx

from face_cluster.types import FaceRecord, GraphResult
from face_cluster.config import PipelineConfig

logger = logging.getLogger(__name__)


class KNNGraphBuilder:
    """Build mutual kNN graph for face clustering.

    Algorithm:
    1. Compute pairwise cosine distances
    2. Find top-K nearest neighbors for each face
    3. Create edges where:
       - j in knn(i) AND i in knn(j) (mutual)
       - AND distance[i,j] <= threshold
    """

    def __init__(self, config: PipelineConfig):
        """Initialize graph builder.

        Args:
            config: Pipeline configuration with K and distance_threshold
        """
        self.config = config

    def build_distance_matrix(
        self,
        faces: List[FaceRecord],
        indices: List[int]
    ) -> np.ndarray:
        """Compute pairwise cosine distance matrix for subset of faces.

        Args:
            faces: All face records
            indices: Indices of faces to include in distance matrix

        Returns:
            Distance matrix (n x n)
        """
        if len(indices) == 0:
            return np.array([])

        # Extract normalized embeddings
        embeddings = np.array([faces[i].embedding_normalized for i in indices])

        # Compute cosine similarity: S = E @ E.T
        similarity = embeddings @ embeddings.T

        # Convert to distance: D = 1 - S, clamp to [0, 2]
        distance = np.clip(1.0 - similarity, 0.0, 2.0)

        # Ensure diagonal is exactly 0
        np.fill_diagonal(distance, 0.0)

        return distance

    def find_knn(
        self,
        distance_matrix: np.ndarray,
        k: int
    ) -> tuple:
        """Find k-nearest neighbors for each sample.

        Args:
            distance_matrix: n x n distance matrix
            k: Number of neighbors to find

        Returns:
            (neighbors, distances) where:
                neighbors[i] = list of k neighbor indices for sample i
                distances[i] = list of k distances for sample i
        """
        n = distance_matrix.shape[0]
        if n == 0:
            return [], []

        k_actual = min(k, n - 1)

        neighbors = []
        neighbor_distances = []

        for i in range(n):
            # Get distances for this sample (excluding self)
            dists = distance_matrix[i].copy()
            dists[i] = np.inf  # Exclude self

            # Find k nearest
            nearest_indices = np.argsort(dists)[:k_actual]
            nearest_dists = dists[nearest_indices]

            neighbors.append(nearest_indices.tolist())
            neighbor_distances.append(nearest_dists.tolist())

        return neighbors, neighbor_distances

    def build_mutual_knn_graph(
        self,
        distance_matrix: np.ndarray,
        k: int,
        threshold: float
    ) -> GraphResult:
        """Build mutual kNN graph with distance threshold.

        An edge (i, j) is created if:
        - j in knn(i) AND i in knn(j) (mutual kNN)
        - distance[i, j] <= threshold

        Args:
            distance_matrix: n x n distance matrix
            k: Number of nearest neighbors
            threshold: Maximum distance for edge creation

        Returns:
            GraphResult with neighbors, edges, and NetworkX graph
        """
        n = distance_matrix.shape[0]

        if n == 0:
            return GraphResult(
                neighbors=[],
                neighbor_distances=[],
                edges=[],
                G=nx.Graph(),
                distance_matrix=distance_matrix
            )

        # Find k-nearest neighbors
        neighbors, neighbor_distances = self.find_knn(distance_matrix, k)

        # Build set representation for fast lookup
        neighbor_sets = [set(neighbors[i]) for i in range(n)]

        # Build graph
        G = nx.Graph()
        G.add_nodes_from(range(n))

        edges = []

        for i in range(n):
            for j_idx, j in enumerate(neighbors[i]):
                # Only process each pair once
                if j > i:
                    # Check mutual kNN
                    if i in neighbor_sets[j]:
                        dist = distance_matrix[i, j]
                        # Check threshold
                        if dist <= threshold:
                            G.add_edge(i, j, distance=dist, weight=1.0 - dist)
                            edges.append((i, j, dist))

        logger.info(
            f"Built mutual kNN graph: {n} nodes, {len(edges)} edges "
            f"(k={k}, threshold={threshold:.3f})"
        )

        return GraphResult(
            neighbors=neighbors,
            neighbor_distances=neighbor_distances,
            edges=edges,
            G=G,
            distance_matrix=distance_matrix
        )

    def build_graph(
        self,
        faces: List[FaceRecord],
        core_indices: List[int]
    ) -> GraphResult:
        """Build mutual kNN graph for core faces.

        Args:
            faces: All face records
            core_indices: Indices of core faces to cluster

        Returns:
            GraphResult with graph structure
        """
        # Build distance matrix
        distance_matrix = self.build_distance_matrix(faces, core_indices)

        if len(core_indices) == 0:
            return GraphResult(
                neighbors=[],
                neighbor_distances=[],
                edges=[],
                G=nx.Graph(),
                distance_matrix=distance_matrix
            )

        # Build mutual kNN graph
        result = self.build_mutual_knn_graph(
            distance_matrix,
            self.config.K,
            self.config.distance_threshold
        )

        return result
