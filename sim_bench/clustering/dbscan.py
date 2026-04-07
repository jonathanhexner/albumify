"""
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) implementation.
"""

import numpy as np
from typing import Dict, Any, Tuple
from sim_bench.clustering.base import ClusteringMethod


class DBSCANClusterer(ClusteringMethod):
    """DBSCAN clustering implementation."""

    doc_explanation = """
DBSCAN finds dense regions separated by sparse regions.
A point is a core point if it has >= min_samples neighbors within distance eps.
Clusters form by connecting core points within eps of each other.

Decision: Point becomes noise (-1) if it's not a core point and not within
eps of any core point. Border points (within eps of core but not core themselves)
join the nearest core point's cluster.

Sensitive to eps: too small = many noise, too large = single cluster.
"""

    decision_parameters = {
        "eps": {
            "description": "Maximum distance between neighbors",
            "default": 0.3,
            "decision_role": "Core points need min_samples within this distance"
        },
        "min_samples": {
            "description": "Minimum neighbors to be a core point",
            "default": 4,
            "decision_role": "Point is core if neighbors within eps >= this"
        },
        "metric": {
            "description": "Distance metric",
            "default": "cosine",
            "decision_role": "How distance is computed (cosine recommended for embeddings)"
        },
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Extract parameters with defaults
        self.metric = self.params.get('metric', 'cosine')
        self.eps = self.params.get('eps', 0.3)
        self.min_samples = self.params.get('min_samples', 4)
    
    def cluster(
        self,
        features: np.ndarray,
        collect_debug_data: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Cluster features using DBSCAN.

        Args:
            features: Feature matrix [n_samples, n_features]
            collect_debug_data: If True, collect debug data (not used by DBSCAN)

        Returns:
            labels: Cluster labels (-1 for noise)
            stats: Dictionary with clustering statistics
        """
        from sklearn.cluster import DBSCAN
        
        # sklearn's DBSCAN supports cosine metric directly
        # No need for manual normalization or conversion
        clusterer = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric
        )
        labels = clusterer.fit_predict(features)
        
        # Compute statistics
        stats = self._compute_stats(labels)
        
        return labels, stats
    
    def _compute_stats(self, labels: np.ndarray) -> Dict[str, Any]:
        """Compute clustering statistics."""
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(labels).count(-1)
        
        cluster_sizes = {}
        for label in unique_labels:
            if label != -1:
                cluster_sizes[int(label)] = int(np.sum(labels == label))
        
        return {
            'algorithm': 'dbscan',
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_ratio': n_noise / len(labels) if len(labels) > 0 else 0.0,
            'cluster_sizes': cluster_sizes,
            'params': {
                'metric': self.metric,
                'eps': self.eps,
                'min_samples': self.min_samples
            }
        }

