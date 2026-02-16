"""
HDBSCAN (Hierarchical Density-Based Spatial Clustering) implementation.
Automatically determines the number of clusters.
"""

import numpy as np
from typing import Dict, Any, Tuple
from sim_bench.clustering.base import ClusteringMethod


class HDBSCANClusterer(ClusteringMethod):
    """HDBSCAN clustering implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Extract parameters with defaults
        self.metric = self.params.get('metric', 'cosine')
        self.min_cluster_size = self.params.get('min_cluster_size', 5)
        self.min_samples = self.params.get('min_samples', None)
        self.cluster_selection_epsilon = self.params.get('cluster_selection_epsilon', 0.0)
        self.cluster_selection_method = self.params.get('cluster_selection_method', 'eom')
    
    def cluster(
        self,
        features: np.ndarray,
        collect_debug_data: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Cluster features using HDBSCAN.

        Args:
            features: Feature matrix [n_samples, n_features]
            collect_debug_data: If True, collect debug data (not used by HDBSCAN)

        Returns:
            labels: Cluster labels (-1 for noise)
            stats: Dictionary with clustering statistics
        """
        try:
            import hdbscan
        except ImportError:
            raise ImportError(
                "HDBSCAN is not installed. Install it with: pip install hdbscan"
            )
        
        # Normalize for cosine distance
        processed_features = features
        actual_metric = self.metric
        
        if self.metric == 'cosine':
            processed_features = self.normalize_features(features)
            # Use euclidean on normalized features (equivalent to cosine)
            actual_metric = 'euclidean'
        
        # Cluster - build kwargs dynamically to handle None values
        kwargs = {
            'min_cluster_size': self.min_cluster_size,
            'metric': actual_metric,
            'cluster_selection_epsilon': self.cluster_selection_epsilon,
            'cluster_selection_method': self.cluster_selection_method
        }
        
        # Only add min_samples if specified (otherwise HDBSCAN uses min_cluster_size)
        if self.min_samples is not None:
            kwargs['min_samples'] = self.min_samples
        
        clusterer = hdbscan.HDBSCAN(**kwargs)
        labels = clusterer.fit_predict(processed_features)
        
        # Compute statistics
        stats = self._compute_stats(labels, clusterer)
        
        return labels, stats
    
    def _compute_stats(self, labels: np.ndarray, clusterer) -> Dict[str, Any]:
        """Compute clustering statistics."""
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(labels).count(-1)
        
        cluster_sizes = {}
        for label in unique_labels:
            if label != -1:
                cluster_sizes[int(label)] = int(np.sum(labels == label))
        
        stats = {
            'algorithm': 'hdbscan',
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_ratio': n_noise / len(labels) if len(labels) > 0 else 0.0,
            'cluster_sizes': cluster_sizes,
            'params': {
                'metric': self.metric,
                'min_cluster_size': self.min_cluster_size,
                'min_samples': self.min_samples,
                'cluster_selection_epsilon': self.cluster_selection_epsilon,
                'cluster_selection_method': self.cluster_selection_method
            }
        }
        
        # Add cluster persistence (strength) if available
        if hasattr(clusterer, 'cluster_persistence_'):
            stats['cluster_persistence'] = {
                int(k): float(v) 
                for k, v in enumerate(clusterer.cluster_persistence_) 
                if k != -1
            }
        
        return stats

