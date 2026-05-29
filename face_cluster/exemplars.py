"""D10-based exemplar selection for clusters."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from face_cluster.types import ClusterResult, GraphResult
from face_cluster.config import PipelineConfig

logger = logging.getLogger(__name__)


# spec-053: typed boundary for D10ExemplarSelector.calc().

@dataclass(frozen=True, slots=True)
class ExemplarInputs:
    """Per-call data for D10ExemplarSelector.calc()."""
    cluster_result: ClusterResult
    graph_result: GraphResult


@dataclass(frozen=True, slots=True)
class ExemplarResult:
    """Output of D10ExemplarSelector.calc().

    ``cluster_result`` is the SAME ClusterResult passed in, mutated
    to populate the ``exemplars`` field.
    """
    cluster_result: ClusterResult
    node_d10_map: Dict[int, float]


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

    def calc(self, inputs: ExemplarInputs) -> ExemplarResult:
        """Single pipeline entry point (spec-053). Thin facade over
        ``select_exemplars()``."""
        cr, d10 = self.select_exemplars(inputs.cluster_result, inputs.graph_result)
        return ExemplarResult(cluster_result=cr, node_d10_map=d10)

    def select_exemplars(
        self,
        cluster_result: ClusterResult,
        graph_result: GraphResult,
    ) -> Tuple[ClusterResult, Dict[int, float]]:
        """Select exemplars for each cluster using d10 metric.

        Returns:
            (updated ClusterResult, node_d10_map) where node_d10_map maps
            graph-local node index -> d10 value for every core node.
        """
        exemplars: Dict[int, List[int]] = {}
        node_d10_map: Dict[int, float] = {}

        for cluster_id, nodes in cluster_result.clusters.items():
            cluster_exemplars, d10_vals = self._select_cluster_exemplars(
                nodes, graph_result.distance_matrix
            )
            exemplars[cluster_id] = cluster_exemplars
            for node, d10 in zip(nodes, d10_vals):
                node_d10_map[node] = float(d10)

            logger.info(
                f"Cluster {cluster_id}: {len(cluster_exemplars)} exemplars "
                f"from {len(nodes)} faces"
            )

        cluster_result.exemplars = exemplars
        return cluster_result, node_d10_map

    def _select_cluster_exemplars(
        self,
        cluster_nodes: List[int],
        distance_matrix: np.ndarray,
    ) -> Tuple[List[int], np.ndarray]:
        """Select exemplars for a single cluster.

        Returns:
            (exemplar_node_indices, d10_values_per_node)
        """
        size = len(cluster_nodes)

        if size == 0:
            return [], np.array([])

        if size == 1:
            return cluster_nodes, np.array([0.0])

        indices = np.array(cluster_nodes)
        cluster_dists = distance_matrix[np.ix_(indices, indices)]

        k = min(self.config.d10_k, size - 1)
        d10_values = np.array([
            np.sort(cluster_dists[i])[1:k + 1][-1]
            for i in range(size)
        ])

        candidate_mask = d10_values <= self.config.exemplars_d10_threshold
        candidate_indices = np.where(candidate_mask)[0]

        if len(candidate_indices) == 0:
            logger.warning(
                f"No exemplar candidates with d10 <= {self.config.exemplars_d10_threshold:.3f}, "
                f"using best face (d10={d10_values.min():.3f})"
            )
            best_idx = int(np.argmin(d10_values))
            return [cluster_nodes[best_idx]], d10_values

        candidate_indices_sorted = candidate_indices[np.argsort(d10_values[candidate_indices])]

        selected = []
        selected_positions = []

        for i in candidate_indices_sorted:
            too_close = any(
                cluster_dists[i, j] < self.config.exemplar_suppression_radius
                for j in selected
            )
            if not too_close:
                selected.append(i)
                selected_positions.append(cluster_nodes[i])
                if len(selected) >= self.config.N_exemplars_max:
                    break

        if len(selected) == 0:
            best_idx = int(candidate_indices_sorted[0])
            selected_positions = [cluster_nodes[best_idx]]

        return selected_positions, d10_values

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
