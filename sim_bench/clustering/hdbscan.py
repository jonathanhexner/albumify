"""
HDBSCAN (Hierarchical Density-Based Spatial Clustering) implementation.
Automatically determines the number of clusters.

Distance metric: Cosine distance = 1 - cosine_similarity, clipped to [0, 2].
"""

import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sim_bench.clustering.base import ClusteringMethod
from sim_bench.clustering.distance_utils import cosine_distance_matrix

logger = logging.getLogger(__name__)


class HDBSCANClusterer(ClusteringMethod):
    """HDBSCAN clustering implementation with optional PCA preprocessing."""

    doc_explanation = """
HDBSCAN finds clusters based on density without requiring k (number of clusters).
It builds a hierarchy of clusters and extracts the most stable ones.

Decision: A point becomes noise if it lacks sufficient nearby neighbors (density).
Clusters form where points have high mutual reachability (both points consider
each other close). Cluster boundaries are determined by density drops.

Key thresholds: min_cluster_size (minimum points per cluster),
cluster_selection_epsilon (merge clusters closer than this distance).
"""

    decision_parameters = {
        "min_cluster_size": {
            "description": "Minimum number of faces to form a valid cluster",
            "default": 5,
            "decision_role": "Clusters with fewer faces become noise (-1)"
        },
        "min_samples": {
            "description": "Core point density requirement",
            "default": None,
            "decision_role": "Points need this many neighbors within reach to be core; if None uses min_cluster_size"
        },
        "cluster_selection_epsilon": {
            "description": "Distance threshold for merging clusters",
            "default": 0.0,
            "decision_role": "Clusters with linkage distance < epsilon are merged into one"
        },
        "cluster_selection_method": {
            "description": "How to select flat clusters from hierarchy",
            "default": "eom",
            "decision_role": "'eom' (Excess of Mass) maximizes stability; 'leaf' returns finest clusters"
        },
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # PCA dimensionality reduction (None = disabled, 256 = default when enabled)
        self.pca_dim = self.params.get('pca_dim', None)

        # Extract parameters with defaults
        self.metric = self.params.get('metric', 'cosine')
        self.min_cluster_size = self.params.get('min_cluster_size', 5)
        self.min_samples = self.params.get('min_samples', None)
        self.cluster_selection_epsilon = self.params.get('cluster_selection_epsilon', 0.0)
        self.cluster_selection_method = self.params.get('cluster_selection_method', 'eom')
    
    def _apply_pca(self, features: np.ndarray) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Apply PCA dimensionality reduction if configured.

        Returns:
            reduced_features: PCA-transformed features (or original if PCA disabled)
            pca_stats: Dict with variance_explained, or None if PCA disabled
        """
        if self.pca_dim is None:
            return features, None

        from sklearn.decomposition import PCA

        n_samples, n_features = features.shape
        pca_dim = min(self.pca_dim, n_samples, n_features)

        if pca_dim >= n_features:
            logger.debug(f"PCA skipped: pca_dim={pca_dim} >= n_features={n_features}")
            return features, None

        logger.info(f"Applying PCA: {n_features}D -> {pca_dim}D for {n_samples} samples")
        pca = PCA(n_components=pca_dim)
        reduced = pca.fit_transform(features)
        variance_explained = float(np.sum(pca.explained_variance_ratio_))
        logger.info(f"PCA variance explained: {variance_explained:.2%}")

        return reduced, {'pca_dim': pca_dim, 'variance_explained': variance_explained}

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

        # Apply PCA if configured
        features, pca_stats = self._apply_pca(features)

        # Handle distance metric
        if self.metric == 'cosine':
            # Normalize and compute precomputed cosine distance matrix
            normalized_features = self.normalize_features(features)
            dist_matrix = cosine_distance_matrix(normalized_features)
            actual_metric = 'precomputed'
            cluster_input = dist_matrix
        else:
            actual_metric = self.metric
            cluster_input = features

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
        labels = clusterer.fit_predict(cluster_input)

        # Compute statistics
        stats = self._compute_stats(labels, clusterer, pca_stats)

        # Store last run info for UI display
        self.last_run_info = {
            'n_clusters': stats['n_clusters'],
            'n_noise': stats['n_noise'],
            'min_cluster_size_used': self.min_cluster_size,
            'cluster_selection_epsilon_used': self.cluster_selection_epsilon,
            'cluster_sizes': stats['cluster_sizes'],
        }

        return labels, stats
    
    def _compute_stats(
        self, labels: np.ndarray, clusterer, pca_stats: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
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
                'pca_dim': self.pca_dim,
                'metric': self.metric,
                'min_cluster_size': self.min_cluster_size,
                'min_samples': self.min_samples,
                'cluster_selection_epsilon': self.cluster_selection_epsilon,
                'cluster_selection_method': self.cluster_selection_method
            }
        }

        if pca_stats:
            stats['pca'] = pca_stats
        
        # Add cluster persistence (strength) if available
        if hasattr(clusterer, 'cluster_persistence_'):
            stats['cluster_persistence'] = {
                int(k): float(v) 
                for k, v in enumerate(clusterer.cluster_persistence_) 
                if k != -1
            }
        
        return stats

