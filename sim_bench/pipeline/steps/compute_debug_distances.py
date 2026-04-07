"""Compute Debug Distances step - Pre-compute neighbor distances for debug UI.

Computes for each face:
- 5 closest neighbors within its cluster
- 5 furthest neighbors within its cluster
- 5 closest neighbors from other clusters

Also computes exemplar distance matrices for cluster-to-cluster visualization.
"""

import logging
import time
from typing import Dict, Any, List, Tuple

import numpy as np

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step

logger = logging.getLogger(__name__)


@register_step
class ComputeDebugDistancesStep(BaseStep):
    """Compute debug neighbor distances for UI visualization.

    Pre-computes closest/furthest within-cluster and closest cross-cluster neighbors
    for efficient UI rendering.
    """

    def __init__(self):
        self._metadata = StepMetadata(
            name="compute_debug_distances",
            display_name="Compute Debug Distances",
            description="Pre-compute neighbor distances for debug UI (closest/furthest within, closest outside)",
            category="people",
            requires={"initial_clusters", "knn_graph_result", "face_records"},
            produces={"debug_neighbors"},
            depends_on=["select_exemplars"],
            config_schema={
                "type": "object",
                "properties": {
                    "n_closest_within": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of closest within-cluster neighbors to store"
                    },
                    "n_furthest_within": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of furthest within-cluster neighbors to store"
                    },
                    "n_closest_cross": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of closest cross-cluster neighbors to store"
                    }
                }
            }
        )

    def _compute_within_cluster_neighbors(
        self,
        cluster_id: int,
        cluster_nodes: List[int],
        distance_matrix: np.ndarray,
        n_closest: int,
        n_furthest: int
    ) -> Tuple[Dict[int, List[Tuple[int, float]]], Dict[int, List[Tuple[int, float]]]]:
        """Compute within-cluster neighbors for all faces in cluster.

        Args:
            cluster_id: Cluster ID
            cluster_nodes: List of node indices in cluster
            distance_matrix: Full distance matrix
            n_closest: Number of closest neighbors to store
            n_furthest: Number of furthest neighbors to store

        Returns:
            (closest_dict, furthest_dict) where each maps node_idx -> [(neighbor_idx, distance), ...]
        """
        within_closest = {}
        within_furthest = {}

        if len(cluster_nodes) < 2:
            # Single-face cluster - no neighbors
            return within_closest, within_furthest

        # Extract cluster distance submatrix
        indices = np.array(cluster_nodes)
        cluster_dists = distance_matrix[np.ix_(indices, indices)]

        for i, node_i in enumerate(cluster_nodes):
            # Get distances to all other nodes in cluster
            dists = cluster_dists[i].copy()
            dists[i] = np.inf  # Exclude self

            # Closest neighbors
            n_closest_actual = min(n_closest, len(cluster_nodes) - 1)
            closest_indices = np.argsort(dists)[:n_closest_actual]
            within_closest[node_i] = [
                (cluster_nodes[j], float(dists[j]))
                for j in closest_indices
            ]

            # Furthest neighbors
            n_furthest_actual = min(n_furthest, len(cluster_nodes) - 1)
            dists_finite = dists[dists < np.inf]
            if len(dists_finite) > 0:
                furthest_indices = np.argsort(dists)[-n_furthest_actual:][::-1]
                within_furthest[node_i] = [
                    (cluster_nodes[j], float(dists[j]))
                    for j in furthest_indices
                ]
            else:
                within_furthest[node_i] = []

        return within_closest, within_furthest

    def _compute_cross_cluster_neighbors(
        self,
        cluster_nodes: List[int],
        all_other_nodes: List[int],
        cluster_id_map: Dict[int, int],
        distance_matrix: np.ndarray,
        n_closest: int
    ) -> Dict[int, List[Tuple[int, int, float]]]:
        """Compute cross-cluster neighbors for all faces in cluster.

        Args:
            cluster_nodes: List of node indices in this cluster
            all_other_nodes: List of all node indices NOT in this cluster
            cluster_id_map: Maps node_idx -> cluster_id
            distance_matrix: Full distance matrix
            n_closest: Number of closest cross-cluster neighbors to store

        Returns:
            Dict mapping node_idx -> [(neighbor_idx, neighbor_cluster_id, distance), ...]
        """
        cross_closest = {}

        if not all_other_nodes:
            # No other clusters - no cross-cluster neighbors
            return cross_closest

        for node_i in cluster_nodes:
            # Get distances to all nodes in other clusters
            dists = distance_matrix[node_i, all_other_nodes]

            # Find closest cross-cluster neighbors
            n_closest_actual = min(n_closest, len(all_other_nodes))
            closest_indices = np.argsort(dists)[:n_closest_actual]

            cross_closest[node_i] = [
                (all_other_nodes[j], cluster_id_map[all_other_nodes[j]], float(dists[j]))
                for j in closest_indices
            ]

        return cross_closest

    def _compute_exemplar_distances(
        self,
        cluster_result,
        distance_matrix: np.ndarray
    ) -> Dict[Tuple[int, int], float]:
        """Compute minimum exemplar distances between all cluster pairs.

        Args:
            cluster_result: ClusterResult with exemplars
            distance_matrix: Full distance matrix

        Returns:
            Dict mapping (cluster_i, cluster_j) -> min exemplar distance
        """
        exemplar_distances = {}
        cluster_ids = sorted(cluster_result.clusters.keys())

        for i, cluster_i in enumerate(cluster_ids):
            for cluster_j in cluster_ids[i+1:]:
                exemplars_i = cluster_result.exemplars.get(cluster_i, [])
                exemplars_j = cluster_result.exemplars.get(cluster_j, [])

                if not exemplars_i or not exemplars_j:
                    continue

                # Compute all pairwise distances between exemplars
                min_dist = np.inf
                for ex_i in exemplars_i:
                    for ex_j in exemplars_j:
                        dist = distance_matrix[ex_i, ex_j]
                        if dist < min_dist:
                            min_dist = dist

                exemplar_distances[(cluster_i, cluster_j)] = float(min_dist)
                exemplar_distances[(cluster_j, cluster_i)] = float(min_dist)

        return exemplar_distances

    def process(self, context: PipelineContext, config: dict) -> None:
        """Compute debug neighbor distances.

        Args:
            context: Pipeline context with initial_clusters, knn_graph_result
            config: Step configuration
        """
        start_time = time.time()

        cluster_result = context.initial_clusters
        graph_result = context.knn_graph_result

        if cluster_result is None or graph_result is None:
            logger.warning("No cluster result or graph result")
            context.debug_neighbors = None
            context.report_progress("compute_debug_distances", 1.0, "No clusters")
            return

        if cluster_result.n_clusters == 0:
            logger.warning("No clusters to compute distances for")
            context.debug_neighbors = {
                'within_closest': {},
                'within_furthest': {},
                'cross_cluster': {},
                'exemplar_distances': {}
            }
            context.report_progress("compute_debug_distances", 1.0, "No clusters")
            return

        logger.info(f"Computing debug distances for {cluster_result.n_clusters} clusters")

        n_closest = config.get('n_closest_within', 5)
        n_furthest = config.get('n_furthest_within', 5)
        n_cross = config.get('n_closest_cross', 5)

        distance_matrix = graph_result.distance_matrix
        core_indices = context.core_indices

        # Build node -> cluster_id mapping
        cluster_id_map = {}
        for cluster_id, nodes in cluster_result.clusters.items():
            for node in nodes:
                cluster_id_map[node] = cluster_id

        # Initialize result dictionaries
        within_closest_all = {}
        within_furthest_all = {}
        cross_cluster_all = {}

        # Process each cluster
        context.report_progress("compute_debug_distances", 0.3, "Computing within-cluster neighbors")

        for cluster_id, cluster_nodes in cluster_result.clusters.items():
            # Within-cluster neighbors
            within_closest, within_furthest = self._compute_within_cluster_neighbors(
                cluster_id, cluster_nodes, distance_matrix, n_closest, n_furthest
            )
            within_closest_all.update(within_closest)
            within_furthest_all.update(within_furthest)

            # Cross-cluster neighbors
            all_other_nodes = [
                node for node in core_indices
                if cluster_id_map.get(node, -1) != cluster_id
            ]
            cross_cluster = self._compute_cross_cluster_neighbors(
                cluster_nodes, all_other_nodes, cluster_id_map, distance_matrix, n_cross
            )
            cross_cluster_all.update(cross_cluster)

        # Compute exemplar distances
        context.report_progress("compute_debug_distances", 0.7, "Computing exemplar distances")
        exemplar_distances = self._compute_exemplar_distances(cluster_result, distance_matrix)

        # Validation: All core faces should have neighbors (unless single-face cluster)
        for node in core_indices:
            if cluster_id_map.get(node, -1) == -1:
                continue  # Noise face

            cluster_id = cluster_id_map[node]
            cluster_size = len(cluster_result.clusters[cluster_id])

            if cluster_size > 1:
                assert node in within_closest_all, \
                    f"Face {node} (cluster {cluster_id}) missing within_closest neighbors"

        # Store results in context
        context.debug_neighbors = {
            'within_closest': within_closest_all,
            'within_furthest': within_furthest_all,
            'cross_cluster': cross_cluster_all,
            'exemplar_distances': exemplar_distances
        }

        duration = time.time() - start_time

        # Log with timing
        logger.info("=" * 60)
        logger.info("COMPUTE_DEBUG_DISTANCES: Stage completed")
        logger.info("=" * 60)
        logger.info(f"Duration: {duration:.2f}s")
        logger.info(f"Faces with within-cluster neighbors: {len(within_closest_all)}")
        logger.info(f"Faces with cross-cluster neighbors: {len(cross_cluster_all)}")
        logger.info(f"Cluster pairs with exemplar distances: {len(exemplar_distances) // 2}")
        logger.info("=" * 60)

        context.report_progress(
            "compute_debug_distances", 1.0,
            f"Computed debug distances for {len(within_closest_all)} faces"
        )
