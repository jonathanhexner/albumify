"""D10-based exemplar selection for clusters."""

import logging
from typing import List, Dict
import numpy as np

from face_cluster.types import ClusterResult, GraphResult
from face_cluster.config import PipelineConfig

logger = logging.getLogger(__name__)


class D10ExemplarSelector:
    """Select exemplars using d10 density metric.

    For each cluster:
    1. Compute d10(i) = distance to kth nearest neighbor within cluster
    2. Select candidates with d10 <= threshold
    3. Greedily select up to N exemplars using suppression radius
    """

    def __init__(self, config: PipelineConfig):
        """Initialize exemplar selector.

        Args:
            config: Pipeline configuration with d10 parameters
        """
        self.config = config

    def select_exemplars(
        self,
        cluster_result: ClusterResult,
        graph_result: GraphResult
    ) -> ClusterResult:
        """Select exemplars for each cluster using d10 metric.

        Args:
            cluster_result: Cluster result from connected components
            graph_result: Graph result with distance matrix

        Returns:
            Updated ClusterResult with exemplars filled in
        """
        exemplars = {}

        for cluster_id, nodes in cluster_result.clusters.items():
            cluster_exemplars = self._select_cluster_exemplars(
                nodes,
                graph_result.distance_matrix
            )
            exemplars[cluster_id] = cluster_exemplars

            logger.info(
                f"Cluster {cluster_id}: {len(cluster_exemplars)} exemplars "
                f"from {len(nodes)} faces"
            )

        cluster_result.exemplars = exemplars
        return cluster_result

    def _select_cluster_exemplars(
        self,
        cluster_nodes: List[int],
        distance_matrix: np.ndarray
    ) -> List[int]:
        """Select exemplars for a single cluster.

        Args:
            cluster_nodes: List of node indices in cluster
            distance_matrix: Full distance matrix

        Returns:
            List of exemplar node indices
        """
        size = len(cluster_nodes)

        if size == 0:
            return []

        if size == 1:
            return cluster_nodes

        # Extract pairwise distances within cluster
        indices = np.array(cluster_nodes)
        cluster_dists = distance_matrix[np.ix_(indices, indices)]

        # Compute d10 for each node
        k = min(self.config.d10_k, size - 1)
        d10_values = np.array([
            np.sort(cluster_dists[i])[1:k + 1][-1]  # Skip self (index 0)
            for i in range(size)
        ])

        # Select candidates with d10 <= threshold
        candidate_mask = d10_values <= self.config.exemplars_d10_threshold
        candidate_indices = np.where(candidate_mask)[0]

        if len(candidate_indices) == 0:
            # No candidates, fall back to best d10 value
            logger.warning(
                f"No exemplar candidates with d10 <= {self.config.exemplars_d10_threshold:.3f}, "
                f"using best face (d10={d10_values.min():.3f})"
            )
            best_idx = int(np.argmin(d10_values))
            return [cluster_nodes[best_idx]]

        # Sort candidates by d10 (ascending - smaller is better)
        candidate_indices_sorted = candidate_indices[np.argsort(d10_values[candidate_indices])]

        # Greedy selection with suppression radius
        selected = []
        selected_positions = []

        for i in candidate_indices_sorted:
            # Check if this candidate is too close to already selected exemplars
            too_close = False
            for j in selected:
                dist = cluster_dists[i, j]
                if dist < self.config.exemplar_suppression_radius:
                    too_close = True
                    break

            if not too_close:
                selected.append(i)
                selected_positions.append(cluster_nodes[i])

                # Stop if we have enough exemplars
                if len(selected) >= self.config.N_exemplars_max:
                    break

        # If no exemplars selected (all too close), just take the best one
        if len(selected) == 0:
            best_idx = int(candidate_indices_sorted[0])
            selected_positions = [cluster_nodes[best_idx]]

        return selected_positions

    def get_d10_values(
        self,
        cluster_nodes: List[int],
        distance_matrix: np.ndarray
    ) -> np.ndarray:
        """Compute d10 values for cluster nodes.

        Useful for debugging and visualization.

        Args:
            cluster_nodes: List of node indices in cluster
            distance_matrix: Full distance matrix

        Returns:
            Array of d10 values (one per node)
        """
        size = len(cluster_nodes)

        if size <= 1:
            return np.zeros(size)

        indices = np.array(cluster_nodes)
        cluster_dists = distance_matrix[np.ix_(indices, indices)]

        k = min(self.config.d10_k, size - 1)
        d10_values = np.array([
            np.sort(cluster_dists[i])[1:k + 1][-1]
            for i in range(size)
        ])

        return d10_values
