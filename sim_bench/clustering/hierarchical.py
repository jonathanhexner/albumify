"""
Hierarchical clustering for multi-level photo organization.

Enables clustering at multiple levels:
- Level 1: Coarse grouping (e.g., scene type, event)
- Level 2: Fine grouping within level 1 clusters (e.g., specific people, landmarks)

Example use cases:
- Level 1: Cluster by event (beach, museum, party)
- Level 2: Cluster by people within each event

- Level 1: Cluster by location (Paris, Rome, London)
- Level 2: Cluster by landmark within each location
"""

from typing import Dict, Any, Tuple, List
import numpy as np
from pathlib import Path
import logging

from sim_bench.clustering.base import ClusteringMethod
from sim_bench.clustering import load_clustering_method

logger = logging.getLogger(__name__)


class HierarchicalClusterer(ClusteringMethod):
    """
    Two-level hierarchical clustering.

    First clusters at coarse level, then refines within each cluster.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize hierarchical clusterer.

        Args:
            config: Configuration dictionary with keys:
                - algorithm: 'hierarchical'
                - params:
                    - level1_config: Config for first-level clustering
                    - level2_config: Config for second-level clustering
                    - level1_features: Feature type for level 1 (e.g., 'dinov2')
                    - level2_features: Feature type for level 2 (e.g., 'face_embeddings')
                - output: Output configuration
        """
        super().__init__(config)

        self.level1_config = self.params.get('level1_config', {})
        self.level2_config = self.params.get('level2_config', {})

        # Feature types for each level
        self.level1_features = self.params.get('level1_features', 'dinov2')
        self.level2_features = self.params.get('level2_features', 'face_embeddings')

        logger.info(
            f"Initialized HierarchicalClusterer: "
            f"L1={self.level1_config.get('algorithm', 'dbscan')}/{self.level1_features}, "
            f"L2={self.level2_config.get('algorithm', 'dbscan')}/{self.level2_features}"
        )

    def cluster(
        self,
        features_level1: np.ndarray,
        features_level2: np.ndarray = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform hierarchical clustering.

        Args:
            features_level1: Features for first-level clustering [n_samples, dim1]
            features_level2: Features for second-level clustering [n_samples, dim2]
                           If None, uses same features as level 1

        Returns:
            labels: Hierarchical cluster labels (level1_id * 1000 + level2_id)
            stats: Clustering statistics
        """
        n_samples = len(features_level1)

        if features_level2 is None:
            features_level2 = features_level1

        logger.info(f"Hierarchical clustering: {n_samples} samples")

        # Level 1 clustering
        logger.info("Level 1 clustering...")
        level1_clusterer = load_clustering_method(self.level1_config)
        labels_level1, stats_level1 = level1_clusterer.cluster(features_level1)

        # Level 2 clustering within each level 1 cluster
        logger.info("Level 2 clustering...")
        labels_hierarchical = np.full(n_samples, -1, dtype=int)

        level2_stats_list = []
        next_global_id = 0

        # Get unique level 1 clusters (excluding noise -1)
        unique_level1 = np.unique(labels_level1)
        unique_level1 = unique_level1[unique_level1 >= 0]

        for l1_cluster_id in unique_level1:
            # Get indices of samples in this level 1 cluster
            l1_indices = np.where(labels_level1 == l1_cluster_id)[0]

            if len(l1_indices) < 2:
                # Too few samples for clustering, assign directly
                labels_hierarchical[l1_indices] = next_global_id
                next_global_id += 1
                continue

            # Extract features for this subset
            l1_features_subset = features_level2[l1_indices]

            # Cluster at level 2
            level2_clusterer = load_clustering_method(self.level2_config)
            labels_level2, stats_level2 = level2_clusterer.cluster(l1_features_subset)

            level2_stats_list.append(stats_level2)

            # Assign global hierarchical IDs
            unique_level2 = np.unique(labels_level2)

            for l2_cluster_id in unique_level2:
                l2_indices_local = np.where(labels_level2 == l2_cluster_id)[0]
                l2_indices_global = l1_indices[l2_indices_local]

                if l2_cluster_id == -1:
                    # Noise at level 2 stays as noise
                    labels_hierarchical[l2_indices_global] = -1
                else:
                    # Assign unique global ID
                    labels_hierarchical[l2_indices_global] = next_global_id
                    next_global_id += 1

        # Compute statistics
        unique_hierarchical = np.unique(labels_hierarchical)
        unique_hierarchical = unique_hierarchical[unique_hierarchical >= 0]

        n_clusters_hierarchical = len(unique_hierarchical)
        n_noise = np.sum(labels_hierarchical == -1)

        cluster_sizes = {}
        for cluster_id in unique_hierarchical:
            size = np.sum(labels_hierarchical == cluster_id)
            cluster_sizes[int(cluster_id)] = int(size)

        stats = {
            'algorithm': 'hierarchical',
            'n_clusters': n_clusters_hierarchical,
            'n_noise': int(n_noise),
            'noise_ratio': n_noise / n_samples if n_samples > 0 else 0,
            'cluster_sizes': cluster_sizes,
            'level1_stats': stats_level1,
            'level2_stats': level2_stats_list,
            'n_clusters_level1': stats_level1['n_clusters'],
            'avg_clusters_per_level1': (
                n_clusters_hierarchical / stats_level1['n_clusters']
                if stats_level1['n_clusters'] > 0 else 0
            )
        }

        logger.info(
            f"Hierarchical clustering complete: "
            f"L1={stats_level1['n_clusters']} clusters, "
            f"L2={n_clusters_hierarchical} total clusters "
            f"(avg {stats['avg_clusters_per_level1']:.1f} per L1 cluster)"
        )

        return labels_hierarchical, stats
