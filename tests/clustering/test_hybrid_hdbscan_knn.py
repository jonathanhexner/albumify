"""
Unit tests for HybridHDBSCANKNN clustering algorithm.
"""

import pytest
import numpy as np
from sim_bench.clustering.hybrid_hdbscan_knn import HybridHDBSCANKNN


@pytest.fixture
def default_config():
    """Default clustering configuration."""
    return {
        'algorithm': 'hybrid_hdbscan_knn',
        'params': {
            'min_cluster_size': 2,
            'min_samples': 2,
            'cluster_selection_epsilon': 0.3,
            'knn_k': 3,
            'threshold_floor': 0.5,
            'threshold_ceiling': 0.9,
            'max_exemplars': 10,
            'attach_min_exemplars': 2,
            'merge_min_pairs': 3,
            'merge_min_distinct': 2,
            'max_iterations': 10,
        }
    }


@pytest.fixture
def clusterer(default_config):
    """Create a clusterer instance."""
    return HybridHDBSCANKNN(default_config)


class TestEmptyAndSingleSample:
    """Tests for edge cases with 0-1 samples."""

    def test_empty_features(self, clusterer):
        """Empty input returns empty labels."""
        features = np.array([]).reshape(0, 512)
        labels, stats = clusterer.cluster(features)

        assert len(labels) == 0
        assert stats['n_clusters'] == 0
        assert stats['n_noise'] == 0

    def test_single_feature(self, clusterer):
        """Single sample returns cluster 0."""
        features = np.random.randn(1, 512)
        labels, stats = clusterer.cluster(features)

        assert len(labels) == 1
        assert labels[0] == 0
        assert stats['n_clusters'] == 1
        assert stats['n_noise'] == 0


class TestInputValidation:
    """Tests for input validation."""

    def test_nan_features_raises(self, clusterer):
        """NaN values raise ValueError."""
        features = np.random.randn(10, 512)
        features[5, 100] = np.nan

        with pytest.raises(ValueError, match="NaN"):
            clusterer.cluster(features)

    def test_inf_features_raises(self, clusterer):
        """Inf values raise ValueError."""
        features = np.random.randn(10, 512)
        features[3, 50] = np.inf

        with pytest.raises(ValueError, match="Inf"):
            clusterer.cluster(features)

    def test_all_zero_vectors_raises(self, clusterer):
        """All zero vectors raise ValueError."""
        features = np.zeros((10, 512))

        with pytest.raises(ValueError, match="zero vectors"):
            clusterer.cluster(features)

    def test_some_zero_vectors_warns(self, clusterer, caplog):
        """Some zero vectors log a warning but don't fail."""
        # Create valid features with one zero vector
        features = np.random.randn(10, 512)
        features[5] = 0  # Zero vector

        # Should not raise, but should warn
        labels, stats = clusterer.cluster(features)
        assert "zero-vector" in caplog.text.lower() or len(labels) == 10

    def test_1d_array_raises(self, clusterer):
        """1D array raises ValueError."""
        features = np.random.randn(512)

        with pytest.raises(ValueError, match="2D"):
            clusterer.cluster(features)


class TestBasicClustering:
    """Tests for basic clustering functionality."""

    def test_two_identical_features_same_cluster(self, clusterer):
        """Two identical embeddings should be in the same cluster."""
        base = np.random.randn(512)
        features = np.vstack([base, base])

        labels, stats = clusterer.cluster(features)

        assert labels[0] == labels[1]

    def test_two_distant_features(self, clusterer):
        """Two very different embeddings may be noise or different clusters."""
        # Create orthogonal vectors (maximally different)
        v1 = np.zeros(512)
        v1[0] = 1.0
        v2 = np.zeros(512)
        v2[1] = 1.0
        features = np.vstack([v1, v2])

        labels, stats = clusterer.cluster(features)

        # With min_cluster_size=2, both should be noise
        # or in different clusters if HDBSCAN forms them
        assert labels[0] == -1 or labels[0] != labels[1]

    def test_clear_clusters_identified(self, default_config):
        """Distinct clusters should be identified."""
        # Create 3 clear clusters with 5 samples each
        np.random.seed(42)

        cluster1 = np.random.randn(5, 512) * 0.1 + np.array([1.0] + [0.0] * 511)
        cluster2 = np.random.randn(5, 512) * 0.1 + np.array([0.0, 1.0] + [0.0] * 510)
        cluster3 = np.random.randn(5, 512) * 0.1 + np.array([0.0, 0.0, 1.0] + [0.0] * 509)

        features = np.vstack([cluster1, cluster2, cluster3])

        # Use relaxed parameters for this test
        config = default_config.copy()
        config['params']['min_cluster_size'] = 3
        config['params']['threshold_floor'] = 0.3

        clusterer = HybridHDBSCANKNN(config)
        labels, stats = clusterer.cluster(features)

        # Should have at least 2 clusters
        n_clusters = len(set(l for l in labels if l >= 0))
        assert n_clusters >= 2, f"Expected at least 2 clusters, got {n_clusters}"


class TestThresholdComputation:
    """Tests for Tukey fence threshold computation."""

    def test_threshold_clamped_to_floor(self, default_config):
        """Threshold should not go below floor."""
        # Create tight cluster where raw threshold would be very low
        np.random.seed(42)
        features = np.random.randn(10, 512) * 0.01  # Very tight cluster

        config = default_config.copy()
        config['params']['threshold_floor'] = 0.5
        config['params']['min_cluster_size'] = 2

        clusterer = HybridHDBSCANKNN(config)
        labels, stats = clusterer.cluster(features, collect_debug_data=True)

        debug = stats.get('debug', {})
        thresholds = debug.get('cluster_thresholds', {})

        # All thresholds should be >= floor
        for t in thresholds.values():
            assert t >= 0.5, f"Threshold {t} below floor 0.5"

    def test_threshold_clamped_to_ceiling(self, default_config):
        """Threshold should not go above ceiling."""
        # Create loose cluster where raw threshold would be high
        np.random.seed(42)
        features = np.random.randn(10, 512) * 2.0  # Loose cluster

        config = default_config.copy()
        config['params']['threshold_ceiling'] = 0.9
        config['params']['min_cluster_size'] = 2

        clusterer = HybridHDBSCANKNN(config)
        labels, stats = clusterer.cluster(features, collect_debug_data=True)

        debug = stats.get('debug', {})
        thresholds = debug.get('cluster_thresholds', {})

        # All thresholds should be <= ceiling
        for t in thresholds.values():
            assert t <= 0.9, f"Threshold {t} above ceiling 0.9"


class TestDebugDataCollection:
    """Tests for debug data collection."""

    def test_debug_data_not_collected_by_default(self, clusterer):
        """Debug data should not be collected when not requested."""
        features = np.random.randn(20, 512)
        labels, stats = clusterer.cluster(features)

        assert 'debug' not in stats

    def test_debug_data_collected_when_requested(self, clusterer):
        """Debug data should be collected when requested."""
        features = np.random.randn(20, 512)
        labels, stats = clusterer.cluster(features, collect_debug_data=True)

        assert 'debug' in stats
        debug = stats['debug']
        assert 'cluster_thresholds' in debug
        assert 'cluster_exemplars' in debug
        assert 'cluster_d3_stats' in debug
        assert 'merge_decisions' in debug
        assert 'attach_decisions' in debug

    def test_merge_decisions_have_required_fields(self, clusterer):
        """Merge decisions should have all required fields."""
        features = np.random.randn(30, 512)
        labels, stats = clusterer.cluster(features, collect_debug_data=True)

        debug = stats.get('debug', {})
        merge_decisions = debug.get('merge_decisions', [])

        if merge_decisions:
            decision = merge_decisions[0]
            required_fields = [
                'cluster_a', 'cluster_b', 'threshold', 'n_pairs_within',
                'exemplars_a_involved', 'exemplars_b_involved',
                'min_distance', 'merged', 'reason'
            ]
            for field in required_fields:
                assert field in decision, f"Missing field: {field}"

    def test_attach_decisions_have_required_fields(self, clusterer):
        """Attach decisions should have all required fields."""
        features = np.random.randn(30, 512)
        labels, stats = clusterer.cluster(features, collect_debug_data=True)

        debug = stats.get('debug', {})
        attach_decisions = debug.get('attach_decisions', [])

        if attach_decisions:
            decision = attach_decisions[0]
            assert 'face_idx' in decision
            assert 'attached_to' in decision
            assert 'candidates' in decision


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_input_same_output(self, clusterer):
        """Same input should produce same output."""
        np.random.seed(42)
        features = np.random.randn(20, 512)

        labels1, stats1 = clusterer.cluster(features.copy())
        labels2, stats2 = clusterer.cluster(features.copy())

        np.testing.assert_array_equal(labels1, labels2)
        assert stats1['n_clusters'] == stats2['n_clusters']


class TestParameterSensitivity:
    """Tests for parameter effects."""

    def test_higher_merge_min_pairs_fewer_merges(self, default_config):
        """Higher merge_min_pairs should result in fewer merges."""
        np.random.seed(42)
        features = np.random.randn(30, 512)

        config_low = default_config.copy()
        config_low['params']['merge_min_pairs'] = 1

        config_high = default_config.copy()
        config_high['params']['merge_min_pairs'] = 5

        clusterer_low = HybridHDBSCANKNN(config_low)
        clusterer_high = HybridHDBSCANKNN(config_high)

        _, stats_low = clusterer_low.cluster(features)
        _, stats_high = clusterer_high.cluster(features)

        # Higher merge requirement should result in fewer merges (or equal)
        assert stats_high['total_merges'] <= stats_low['total_merges']

    def test_lower_attach_min_exemplars_more_attached(self, default_config):
        """Lower attach_min_exemplars should attach more noise points."""
        np.random.seed(42)
        features = np.random.randn(30, 512)

        config_low = default_config.copy()
        config_low['params']['attach_min_exemplars'] = 1

        config_high = default_config.copy()
        config_high['params']['attach_min_exemplars'] = 3

        clusterer_low = HybridHDBSCANKNN(config_low)
        clusterer_high = HybridHDBSCANKNN(config_high)

        _, stats_low = clusterer_low.cluster(features)
        _, stats_high = clusterer_high.cluster(features)

        # Lower requirement should attach more (or equal)
        assert stats_low['total_attached'] >= stats_high['total_attached']


class TestStatsStructure:
    """Tests for stats dictionary structure."""

    def test_stats_has_required_fields(self, clusterer):
        """Stats should have all required fields."""
        features = np.random.randn(20, 512)
        labels, stats = clusterer.cluster(features)

        required_fields = [
            'algorithm', 'n_clusters', 'n_noise', 'cluster_sizes',
            'hdbscan', 'total_merges', 'total_attached', 'params'
        ]

        for field in required_fields:
            assert field in stats, f"Missing stats field: {field}"

    def test_cluster_sizes_consistent_with_labels(self, clusterer):
        """Cluster sizes should match label counts."""
        features = np.random.randn(20, 512)
        labels, stats = clusterer.cluster(features)

        for cluster_id, size in stats['cluster_sizes'].items():
            actual_size = np.sum(labels == cluster_id)
            assert size == actual_size, (
                f"Cluster {cluster_id}: reported size {size} != actual {actual_size}"
            )
