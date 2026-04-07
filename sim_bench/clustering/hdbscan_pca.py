"""
HDBSCAN clustering with PCA dimensionality reduction.
Reduces embedding dimensions before clustering for better performance.

Distance metric: Cosine distance = 1 - cosine_similarity, clipped to [0, 2].
"""

import numpy as np
from typing import Dict, Any, Tuple
import logging

from sim_bench.clustering.base import ClusteringMethod
from sim_bench.clustering.distance_utils import cosine_distance_matrix

logger = logging.getLogger(__name__)


class HDBSCANPCAClusterer(ClusteringMethod):
    """HDBSCAN clustering with PCA preprocessing."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # PCA parameters
        self.pca_components = self.params.get('pca_components', 128)

        # HDBSCAN parameters
        self.metric = self.params.get('metric', 'cosine')
        self.min_cluster_size = self.params.get('min_cluster_size', 2)
        self.min_samples = self.params.get('min_samples', None)
        self.cluster_selection_epsilon = self.params.get('cluster_selection_epsilon', 0.0)
        self.cluster_selection_method = self.params.get('cluster_selection_method', 'eom')

    def cluster(
        self,
        features: np.ndarray,
        collect_debug_data: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Cluster features using HDBSCAN with PCA preprocessing.

        Args:
            features: Feature matrix [n_samples, n_features]
            collect_debug_data: If True, collect debug data

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

        try:
            from sklearn.decomposition import PCA
        except ImportError:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )

        n_samples, n_features = features.shape

        # Determine PCA components (can't exceed min of samples or features)
        pca_dim = min(self.pca_components, n_samples, n_features)

        logger.info(
            f"HDBSCAN+PCA: reducing {n_features}D -> {pca_dim}D for {n_samples} samples"
        )

        # Apply PCA
        pca = PCA(n_components=pca_dim)
        reduced_features = pca.fit_transform(features)
        variance_explained = float(np.sum(pca.explained_variance_ratio_))

        logger.info(f"PCA variance explained: {variance_explained:.2%}")

        # Handle distance metric
        if self.metric == 'cosine':
            # Normalize and compute precomputed cosine distance matrix
            normalized_features = self.normalize_features(reduced_features)
            dist_matrix = cosine_distance_matrix(normalized_features)
            actual_metric = 'precomputed'
            cluster_input = dist_matrix
        else:
            actual_metric = self.metric
            cluster_input = reduced_features

        # Build HDBSCAN kwargs
        kwargs = {
            'min_cluster_size': self.min_cluster_size,
            'metric': actual_metric,
            'cluster_selection_epsilon': self.cluster_selection_epsilon,
            'cluster_selection_method': self.cluster_selection_method
        }

        if self.min_samples is not None:
            kwargs['min_samples'] = self.min_samples

        # Cluster
        clusterer = hdbscan.HDBSCAN(**kwargs)
        labels = clusterer.fit_predict(cluster_input)

        # Compute statistics
        stats = self._compute_stats(labels, clusterer, pca_dim, variance_explained)

        return labels, stats

    def _compute_stats(
        self,
        labels: np.ndarray,
        clusterer,
        pca_dim: int,
        variance_explained: float
    ) -> Dict[str, Any]:
        """Compute clustering statistics."""
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = int(np.sum(labels == -1))

        cluster_sizes = {}
        for label in unique_labels:
            if label != -1:
                cluster_sizes[int(label)] = int(np.sum(labels == label))

        stats = {
            'algorithm': 'hdbscan_pca',
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_ratio': n_noise / len(labels) if len(labels) > 0 else 0.0,
            'cluster_sizes': cluster_sizes,
            'params': {
                'pca_components': pca_dim,
                'pca_variance_explained': variance_explained,
                'metric': self.metric,
                'min_cluster_size': self.min_cluster_size,
                'min_samples': self.min_samples,
                'cluster_selection_epsilon': self.cluster_selection_epsilon,
                'cluster_selection_method': self.cluster_selection_method
            }
        }

        if hasattr(clusterer, 'cluster_persistence_'):
            stats['cluster_persistence'] = {
                int(k): float(v)
                for k, v in enumerate(clusterer.cluster_persistence_)
                if k != -1
            }

        return stats
