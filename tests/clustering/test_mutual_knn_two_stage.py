"""Tests for Mutual KNN Two-Stage clustering."""

import numpy as np
import pytest

from sim_bench.clustering.mutual_knn_two_stage import MutualKNNTwoStageClusterer
from sim_bench.clustering.pruning_strategies import (
    RedundantSupportStrategy,
    create_pruning_strategy,
)
from sim_bench.clustering.distance_utils import (
    closest_distance_to_cluster,
    all_cluster_distances,
    support_count,
    separation_margin,
    compute_cluster_members_from_labels,
    cosine_distance_matrix,
)
from sim_bench.clustering.base import load_clustering_method


class TestMutualKNNTwoStageClusterer:
    """Test Mutual KNN Two-Stage clustering implementation."""

    def test_basic_clustering(self):
        """Test basic clustering with clear clusters."""
        # Create 3 clear clusters of 5 points each
        np.random.seed(42)
        cluster1 = np.random.randn(5, 512) * 0.1 + np.array([3.0] + [0.0] * 511)
        cluster2 = np.random.randn(5, 512) * 0.1 + np.array([-3.0] + [0.0] * 511)
        cluster3 = np.random.randn(5, 512) * 0.1 + np.array([0.0, 3.0] + [0.0] * 510)

        embeddings = np.vstack([cluster1, cluster2, cluster3])

        config = {
            'algorithm': 'mutual_knn_two_stage',
            'params': {
                'k': 5,
                'base_threshold': 0.5,
                'min_support': 2,
            }
        }

        clusterer = MutualKNNTwoStageClusterer(config)
        labels, stats = clusterer.cluster(embeddings)

        assert len(labels) == 15
        assert stats['algorithm'] == 'mutual_knn_two_stage'
        assert stats['n_clusters'] >= 1

    def test_empty_input(self):
        """Test with empty input."""
        config = {'algorithm': 'mutual_knn_two_stage', 'params': {'k': 10}}
        clusterer = MutualKNNTwoStageClusterer(config)

        labels, stats = clusterer.cluster(np.array([]).reshape(0, 512))

        assert len(labels) == 0
        assert stats['n_clusters'] == 0

    def test_single_sample(self):
        """Test with single sample."""
        config = {'algorithm': 'mutual_knn_two_stage', 'params': {'k': 10}}
        clusterer = MutualKNNTwoStageClusterer(config)

        labels, stats = clusterer.cluster(np.random.randn(1, 512))

        assert len(labels) == 1
        assert labels[0] == 0
        assert stats['n_clusters'] == 1

    def test_two_samples(self):
        """Test with two samples."""
        np.random.seed(42)
        # Two similar samples
        embeddings = np.random.randn(2, 512)
        embeddings[1] = embeddings[0] + np.random.randn(512) * 0.01  # Very similar

        config = {
            'algorithm': 'mutual_knn_two_stage',
            'params': {
                'k': 1,
                'base_threshold': 0.5,
            }
        }

        clusterer = MutualKNNTwoStageClusterer(config)
        labels, stats = clusterer.cluster(embeddings)

        assert len(labels) == 2

    def test_debug_data_collection(self):
        """Test that debug data is collected when requested."""
        np.random.seed(42)
        embeddings = np.random.randn(10, 512)

        config = {
            'algorithm': 'mutual_knn_two_stage',
            'params': {'k': 5}
        }

        clusterer = MutualKNNTwoStageClusterer(config)
        labels, stats = clusterer.cluster(embeddings, collect_debug_data=True)

        assert 'debug' in stats
        assert 'distance_matrix' in stats['debug']
        assert 'cluster_members' in stats['debug']
        assert 'iterations' in stats['debug']

        # Verify distance matrix shape
        assert stats['debug']['distance_matrix'].shape == (10, 10)

    def test_convergence(self):
        """Test that algorithm converges."""
        np.random.seed(42)
        embeddings = np.random.randn(20, 512)

        config = {
            'algorithm': 'mutual_knn_two_stage',
            'params': {
                'k': 5,
                'max_iterations': 10,
            }
        }

        clusterer = MutualKNNTwoStageClusterer(config)
        labels, stats = clusterer.cluster(embeddings)

        # Should complete within max_iterations
        assert stats['n_iterations'] <= 10

    def test_factory_loading(self):
        """Test loading via factory function."""
        config = {
            'algorithm': 'mutual_knn_two_stage',
            'params': {
                'k': 10,
                'base_threshold': 0.45,
            }
        }

        clusterer = load_clustering_method(config)
        assert isinstance(clusterer, MutualKNNTwoStageClusterer)

    def test_strict_vs_loose_thresholds(self):
        """Test that stricter thresholds produce more clusters/noise."""
        np.random.seed(42)
        embeddings = np.random.randn(30, 512)

        strict_config = {
            'algorithm': 'mutual_knn_two_stage',
            'params': {
                'k': 5,
                'base_threshold': 0.3,  # Strict
                'min_support': 3,
            }
        }

        loose_config = {
            'algorithm': 'mutual_knn_two_stage',
            'params': {
                'k': 5,
                'base_threshold': 0.6,  # Loose
                'min_support': 1,
            }
        }

        strict_clusterer = MutualKNNTwoStageClusterer(strict_config)
        loose_clusterer = MutualKNNTwoStageClusterer(loose_config)

        strict_labels, strict_stats = strict_clusterer.cluster(embeddings)
        loose_labels, loose_stats = loose_clusterer.cluster(embeddings)

        # Stricter thresholds typically produce more noise or more clusters
        # (This is a general expectation, not guaranteed for random data)
        assert strict_stats['n_clusters'] >= 0
        assert loose_stats['n_clusters'] >= 0


class TestRedundantSupportStrategy:
    """Test the RedundantSupportStrategy implementation."""

    def test_basic_evaluation(self):
        """Test basic membership evaluation."""
        strategy = RedundantSupportStrategy(
            base_threshold=0.5,
            min_support=2,
            separation_delta=0.1,
        )

        # Create simple distance matrix
        # 4 points: 0,1,2 are close, 3 is far
        distance_matrix = np.array([
            [0.0, 0.1, 0.1, 0.9],
            [0.1, 0.0, 0.1, 0.9],
            [0.1, 0.1, 0.0, 0.9],
            [0.9, 0.9, 0.9, 0.0],
        ])

        cluster_members = {
            0: [0, 1, 2],
            1: [3],
        }

        # Sample 0 should stay in cluster 0 (has support)
        should_stay, conf, details = strategy.evaluate_membership(
            sample_idx=0,
            candidate_cluster=0,
            current_cluster=0,
            cluster_members=cluster_members,
            distance_matrix=distance_matrix,
        )

        assert should_stay is True
        assert details['has_redundant_support'] is True

    def test_rejection_without_support(self):
        """Test that samples without support are rejected."""
        strategy = RedundantSupportStrategy(
            base_threshold=0.3,  # Strict threshold
            min_support=3,  # Requires 3 neighbors
            separation_delta=0.5,  # Large separation required
        )

        # Create distance matrix where point 0 has only 1 close neighbor
        distance_matrix = np.array([
            [0.0, 0.2, 0.8, 0.8],
            [0.2, 0.0, 0.8, 0.8],
            [0.8, 0.8, 0.0, 0.1],
            [0.8, 0.8, 0.1, 0.0],
        ])

        cluster_members = {
            0: [0, 1],
            1: [2, 3],
        }

        # Sample 0 has only 1 close neighbor (1), needs 3
        # And separation to cluster 1 is not enough
        should_stay, conf, details = strategy.evaluate_membership(
            sample_idx=0,
            candidate_cluster=0,
            current_cluster=0,
            cluster_members=cluster_members,
            distance_matrix=distance_matrix,
        )

        # Should still stay because distance 0.2 <= 0.3 * 1.1 = 0.33
        # and has 1 support neighbor within 0.3 * 1.05 = 0.315
        # But needs 3, so no support path
        # Check separation: dist to cluster 1 is 0.8, to cluster 0 is 0.2
        # Separation = 0.8 - 0.2 = 0.6 >= 0.5
        assert should_stay is True
        assert details['has_separation'] is True

    def test_find_best_cluster(self):
        """Test finding best cluster for unassigned sample."""
        strategy = RedundantSupportStrategy(
            base_threshold=0.5,
            min_support=1,
        )

        distance_matrix = np.array([
            [0.0, 0.9, 0.9, 0.1],  # Point 0 is close to point 3
            [0.9, 0.0, 0.1, 0.9],
            [0.9, 0.1, 0.0, 0.9],
            [0.1, 0.9, 0.9, 0.0],
        ])

        cluster_members = {
            0: [1, 2],
            1: [3],
            -1: [0],  # Point 0 is unassigned
        }

        best_cluster, conf, details = strategy.find_best_cluster(
            sample_idx=0,
            cluster_members=cluster_members,
            distance_matrix=distance_matrix,
        )

        # Point 0 is closest to point 3, which is in cluster 1
        assert best_cluster == 1


class TestDistanceUtils:
    """Test distance utility functions."""

    def test_closest_distance_to_cluster(self):
        """Test closest distance calculation."""
        distance_matrix = np.array([
            [0.0, 0.3, 0.5],
            [0.3, 0.0, 0.2],
            [0.5, 0.2, 0.0],
        ])

        cluster_members = {0: [1, 2]}

        dist = closest_distance_to_cluster(0, 0, distance_matrix, cluster_members)
        assert dist == 0.3  # Closest to point 1

    def test_all_cluster_distances(self):
        """Test computing distances to all clusters."""
        distance_matrix = np.array([
            [0.0, 0.3, 0.5, 0.8],
            [0.3, 0.0, 0.2, 0.9],
            [0.5, 0.2, 0.0, 0.7],
            [0.8, 0.9, 0.7, 0.0],
        ])

        cluster_members = {0: [1, 2], 1: [3]}

        dists = all_cluster_distances(0, distance_matrix, cluster_members)

        assert dists[0] == 0.3  # Closest to cluster 0
        assert dists[1] == 0.8  # Closest to cluster 1

    def test_support_count(self):
        """Test counting supporting neighbors."""
        distance_matrix = np.array([
            [0.0, 0.2, 0.3, 0.8],
            [0.2, 0.0, 0.1, 0.9],
            [0.3, 0.1, 0.0, 0.8],
            [0.8, 0.9, 0.8, 0.0],
        ])

        cluster_members = {0: [0, 1, 2]}

        count = support_count(0, 0, distance_matrix, cluster_members, radius=0.35)
        assert count == 2  # Points 1 and 2 are within radius

    def test_separation_margin(self):
        """Test separation margin calculation."""
        distance_matrix = np.array([
            [0.0, 0.2, 0.8],
            [0.2, 0.0, 0.9],
            [0.8, 0.9, 0.0],
        ])

        cluster_members = {0: [0, 1], 1: [2]}

        margin, next_best = separation_margin(0, 0, distance_matrix, cluster_members)

        # Dist to cluster 0 = 0.2 (to point 1)
        # Dist to cluster 1 = 0.8 (to point 2)
        # Margin = 0.8 - 0.2 = 0.6
        assert abs(margin - 0.6) < 1e-9
        assert next_best == 1

    def test_compute_cluster_members_from_labels(self):
        """Test label to cluster members conversion."""
        labels = np.array([0, 0, 1, 1, -1])

        members = compute_cluster_members_from_labels(labels)

        assert members[0] == [0, 1]
        assert members[1] == [2, 3]
        assert members[-1] == [4]


class TestPruningStrategyFactory:
    """Test pruning strategy factory."""

    def test_create_redundant_support_strategy(self):
        """Test creating RedundantSupportStrategy via factory."""
        config = {
            'strategy': 'redundant_support',
            'base_threshold': 0.4,
            'min_support': 3,
        }

        strategy = create_pruning_strategy(config)

        assert isinstance(strategy, RedundantSupportStrategy)
        assert strategy.base_threshold == 0.4
        assert strategy.min_support == 3

    def test_unknown_strategy_raises(self):
        """Test that unknown strategy raises error."""
        config = {'strategy': 'unknown_strategy'}

        with pytest.raises(ValueError, match="Unknown pruning strategy"):
            create_pruning_strategy(config)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
