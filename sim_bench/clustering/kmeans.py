"""
KMeans clustering implementation.
"""

import numpy as np
from typing import Dict, Any, Tuple
from sim_bench.clustering.base import ClusteringMethod


class KMeansClusterer(ClusteringMethod):
    """KMeans clustering implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Extract parameters with defaults
        self.n_clusters = self.params.get('n_clusters', 10)
        self.n_init = self.params.get('n_init', 10)
        self.random_state = self.params.get('random_state', 42)
    
    def cluster(self, features: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Cluster features using KMeans.
        
        Args:
            features: Feature matrix [n_samples, n_features]
            
        Returns:
            labels: Cluster labels
            stats: Dictionary with clustering statistics
        """
        from sklearn.cluster import KMeans
        
        # Normalize features for better clustering
        processed_features = self.normalize_features(features)
        
        # Cluster
        clusterer = KMeans(
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            random_state=self.random_state
        )
        labels = clusterer.fit_predict(processed_features)
        
        # Compute statistics
        stats = self._compute_stats(labels, clusterer.inertia_)
        
        return labels, stats
    
    def _compute_stats(self, labels: np.ndarray, inertia: float) -> Dict[str, Any]:
        """Compute clustering statistics."""
        unique_labels = set(labels)
        
        cluster_sizes = {}
        for label in unique_labels:
            cluster_sizes[int(label)] = int(np.sum(labels == label))
        
        return {
            'algorithm': 'kmeans',
            'n_clusters': len(unique_labels),
            'cluster_sizes': cluster_sizes,
            'inertia': float(inertia),
            'params': {
                'n_clusters': self.n_clusters,
                'n_init': self.n_init,
                'random_state': self.random_state
            }
        }










