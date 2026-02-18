"""Tests for Mutual KNN clustering."""

import numpy as np
import pytest

from sim_bench.clustering.mutual_knn import MutualKNNClusterer
from sim_bench.clustering.hdbscan_pca import HDBSCANPCAClusterer
from sim_bench.clustering.base import load_clustering_method


class TestMutualKNNClusterer:
    """Test Mutual KNN clustering implementation."""

    def test_basic_clustering(self):
        """Test basic clustering with clear clusters."""
        # Create 3 clear clusters of 5 points each
        np.random.seed(42)
        cluster1 = np.random.randn(5, 512) + np.array([3.0] + [0.0] * 511)
        cluster2 = np.random.randn(5, 512) + np.array([-3.0] + [0.0] * 511)
        cluster3 = np.random.randn(5, 512) + np.array([0.0, 3.0] + [0.0] * 510)

        embeddings = np.vstack([cluster1, cluster2, cluster3])

        config = {
            'algorithm': 'mutual_knn',
            'params': {
                'k': 5,
                'similarity_threshold': 0.5,
            }
        }

        clusterer = MutualKNNClusterer(config)
        labels, stats = clusterer.cluster(embeddings)

        assert len(labels) == 15
        assert stats['algorithm'] == 'mutual_knn'
        assert stats['n_clusters'] >= 1  # Should find some clusters

    def test_empty_input(self):
        """Test with empty input."""
        config = {'algorithm': 'mutual_knn', 'params': {'k': 10}}
        clusterer = MutualKNNClusterer(config)

        labels, stats = clusterer.cluster(np.array([]).reshape(0, 512))

        assert len(labels) == 0
        assert stats['n_clusters'] == 0

    def test_single_sample(self):
        """Test with single sample."""
        config = {'algorithm': 'mutual_knn', 'params': {'k': 10}}
        clusterer = MutualKNNClusterer(config)

        labels, stats = clusterer.cluster(np.random.randn(1, 512))

        assert len(labels) == 1
        assert labels[0] == 0
        assert stats['n_clusters'] == 1

    def test_high_threshold_creates_singletons(self):
        """Test that high threshold creates mostly singletons."""
        np.random.seed(42)
        embeddings = np.random.randn(10, 512)

        config = {
            'algorithm': 'mutual_knn',
            'params': {
                'k': 3,
                'similarity_threshold': 0.99,  # Very high threshold
            }
        }

        clusterer = MutualKNNClusterer(config)
        labels, stats = clusterer.cluster(embeddings)

        # With very high threshold, most points should be singletons
        assert stats['n_singletons'] >= 5

    def test_factory_loading(self):
        """Test loading via factory function."""
        config = {
            'algorithm': 'mutual_knn',
            'params': {
                'k': 10,
                'similarity_threshold': 0.70,
            }
        }

        clusterer = load_clustering_method(config)
        assert isinstance(clusterer, MutualKNNClusterer)


class TestHDBSCANPCAClusterer:
    """Test HDBSCAN+PCA clustering implementation."""

    def test_basic_clustering(self):
        """Test basic clustering with PCA."""
        np.random.seed(42)
        embeddings = np.random.randn(20, 512)

        config = {
            'algorithm': 'hdbscan_pca',
            'params': {
                'pca_components': 64,
                'min_cluster_size': 2,
            }
        }

        clusterer = HDBSCANPCAClusterer(config)
        labels, stats = clusterer.cluster(embeddings)

        assert len(labels) == 20
        assert stats['algorithm'] == 'hdbscan_pca'
        assert stats['params']['pca_components'] == 64
        assert 'pca_variance_explained' in stats['params']

    def test_pca_dimension_capping(self):
        """Test that PCA dimensions are capped to min(samples, features)."""
        np.random.seed(42)
        embeddings = np.random.randn(10, 512)  # Only 10 samples

        config = {
            'algorithm': 'hdbscan_pca',
            'params': {
                'pca_components': 256,  # Request more than samples
                'min_cluster_size': 2,
            }
        }

        clusterer = HDBSCANPCAClusterer(config)
        labels, stats = clusterer.cluster(embeddings)

        # PCA components should be capped to 10 (n_samples)
        assert stats['params']['pca_components'] == 10

    def test_factory_loading(self):
        """Test loading via factory function."""
        config = {
            'algorithm': 'hdbscan_pca',
            'params': {
                'pca_components': 128,
            }
        }

        clusterer = load_clustering_method(config)
        assert isinstance(clusterer, HDBSCANPCAClusterer)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
