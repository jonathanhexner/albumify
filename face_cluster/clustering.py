"""Connected components clustering and optional cluster splitting."""

import logging
from typing import List, Dict
import numpy as np
import networkx as nx

from face_cluster.types import FaceRecord, GraphResult, ClusterResult
from face_cluster.config import PipelineConfig

logger = logging.getLogger(__name__)


class ConnectedComponentsClusterer:
    """Cluster faces using connected components on mutual kNN graph.

    Optional splitting safeguard:
    - For clusters with diameter > threshold, attempt to split
    - Build internal mutual kNN graph with tighter threshold
    - If splits into multiple components, replace with sub-clusters
    """

    def __init__(self, config: PipelineConfig):
        """Initialize clusterer.

        Args:
            config: Pipeline configuration
        """
        self.config = config

    def cluster(
        self,
        graph_result: GraphResult,
        core_indices: List[int]
    ) -> ClusterResult:
        """Cluster using connected components.

        Args:
            graph_result: Mutual kNN graph result
            core_indices: Original core face indices

        Returns:
            ClusterResult with cluster assignments
        """
        G = graph_result.G
        n = G.number_of_nodes()

        if n == 0:
            return ClusterResult(
                labels=np.array([], dtype=np.int32),
                clusters={},
                cluster_stats={},
                exemplars={},
                n_clusters=0,
                n_noise=0
            )

        # Find connected components
        components = list(nx.connected_components(G))

        # Assign labels
        labels = np.full(n, -1, dtype=np.int32)
        cluster_id = 0

        clusters = {}
        cluster_stats = {}

        for component in components:
            component_list = list(component)
            size = len(component_list)

            # Check minimum cluster size
            if size < self.config.min_cluster_size:
                # Mark as noise
                for node in component_list:
                    labels[node] = -1
            else:
                # Assign cluster ID
                for node in component_list:
                    labels[node] = cluster_id

                clusters[cluster_id] = component_list

                # Compute cluster statistics
                stats = self._compute_cluster_stats(
                    component_list,
                    graph_result.distance_matrix
                )
                cluster_stats[cluster_id] = stats

                cluster_id += 1

        n_clusters = cluster_id
        n_noise = int(np.sum(labels == -1))

        logger.info(
            f"Clustering: {n_clusters} clusters, {n_noise} noise points "
            f"(min_cluster_size={self.config.min_cluster_size})"
        )

        result = ClusterResult(
            labels=labels,
            clusters=clusters,
            cluster_stats=cluster_stats,
            exemplars={},
            n_clusters=n_clusters,
            n_noise=n_noise
        )

        # Optional: split wide clusters
        if self.config.split_enabled:
            result = self._split_wide_clusters(result, graph_result, core_indices)

        return result

    def _compute_cluster_stats(
        self,
        cluster_nodes: List[int],
        distance_matrix: np.ndarray
    ) -> Dict[str, float]:
        """Compute statistics for a cluster.

        Args:
            cluster_nodes: List of node indices in cluster
            distance_matrix: Full distance matrix

        Returns:
            Dictionary with statistics (size, diameter, median_dist, etc.)
        """
        size = len(cluster_nodes)

        if size < 2:
            return {
                'size': size,
                'diameter': 0.0,
                'median_dist': 0.0,
                'mean_dist': 0.0,
                'p95_dist': 0.0,
            }

        # Extract pairwise distances within cluster
        indices = np.array(cluster_nodes)
        cluster_dists = distance_matrix[np.ix_(indices, indices)]

        # Get upper triangle (exclude diagonal)
        upper_tri = cluster_dists[np.triu_indices_from(cluster_dists, k=1)]

        diameter = float(upper_tri.max())
        median_dist = float(np.median(upper_tri))
        mean_dist = float(upper_tri.mean())
        p95_dist = float(np.percentile(upper_tri, 95))

        return {
            'size': size,
            'diameter': diameter,
            'median_dist': median_dist,
            'mean_dist': mean_dist,
            'p95_dist': p95_dist,
        }

    def _split_wide_clusters(
        self,
        result: ClusterResult,
        graph_result: GraphResult,
        core_indices: List[int]
    ) -> ClusterResult:
        """Split clusters with diameter > threshold.

        For each wide cluster:
        1. Build internal mutual kNN graph with tighter threshold
        2. Run connected components
        3. If splits into >=2 components of size >= min_cluster_size,
           replace original cluster with sub-clusters

        Args:
            result: Current cluster result
            graph_result: Original graph result
            core_indices: Original core face indices

        Returns:
            Updated ClusterResult with split clusters
        """
        from face_cluster.knn_graph import KNNGraphBuilder

        new_labels = result.labels.copy()
        new_clusters = {}
        new_cluster_stats = {}
        next_cluster_id = result.n_clusters

        # Check each cluster for splitting
        for cluster_id, nodes in result.clusters.items():
            stats = result.cluster_stats[cluster_id]
            diameter = stats['diameter']

            if diameter > self.config.split_diameter_threshold:
                logger.info(
                    f"Cluster {cluster_id}: diameter {diameter:.3f} > "
                    f"{self.config.split_diameter_threshold:.3f}, attempting split"
                )

                # Extract subgraph distances
                indices = np.array(nodes)
                cluster_dists = graph_result.distance_matrix[np.ix_(indices, indices)]

                # Build internal mutual kNN graph with tighter threshold
                builder = KNNGraphBuilder(self.config)
                internal_graph = builder.build_mutual_knn_graph(
                    cluster_dists,
                    self.config.split_K,
                    self.config.split_distance_threshold
                )

                # Find connected components
                internal_components = list(nx.connected_components(internal_graph.G))

                # Check if split into multiple valid clusters
                valid_components = [
                    list(comp) for comp in internal_components
                    if len(comp) >= self.config.min_cluster_size
                ]

                if len(valid_components) >= 2:
                    logger.info(
                        f"Split cluster {cluster_id} into {len(valid_components)} sub-clusters"
                    )

                    # Remove original cluster assignments
                    for node in nodes:
                        new_labels[node] = -1

                    # Assign new cluster IDs
                    for component in valid_components:
                        # Map back to original node indices
                        original_nodes = [nodes[i] for i in component]

                        for node in original_nodes:
                            new_labels[node] = next_cluster_id

                        new_clusters[next_cluster_id] = original_nodes

                        # Recompute stats
                        stats = self._compute_cluster_stats(
                            original_nodes,
                            graph_result.distance_matrix
                        )
                        new_cluster_stats[next_cluster_id] = stats

                        next_cluster_id += 1

                    # Mark remaining nodes (too small components) as noise
                    noise_components = [
                        list(comp) for comp in internal_components
                        if len(comp) < self.config.min_cluster_size
                    ]
                    for component in noise_components:
                        for i in component:
                            node = nodes[i]
                            new_labels[node] = -1

                else:
                    # Keep original cluster
                    new_clusters[cluster_id] = nodes
                    new_cluster_stats[cluster_id] = stats
            else:
                # Keep original cluster
                new_clusters[cluster_id] = nodes
                new_cluster_stats[cluster_id] = stats

        n_clusters = len(new_clusters)
        n_noise = int(np.sum(new_labels == -1))

        logger.info(
            f"After splitting: {n_clusters} clusters, {n_noise} noise points"
        )

        return ClusterResult(
            labels=new_labels,
            clusters=new_clusters,
            cluster_stats=new_cluster_stats,
            exemplars={},
            n_clusters=n_clusters,
            n_noise=n_noise
        )
