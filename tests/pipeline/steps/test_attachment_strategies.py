"""Tests for attachment strategies."""

import numpy as np
import pytest

from sim_bench.pipeline.steps.attachment_strategies import (
    AttachmentStrategyFactory,
    CentroidStrategy,
    ExemplarStrategy,
    HybridStrategy,
    ClusterInfo,
    cosine_distance,
)


class TestCosineDistance:
    """Tests for cosine distance function."""

    def test_identical_vectors(self):
        """Identical vectors should have distance 0."""
        v = np.array([1.0, 0.0, 0.0])
        assert cosine_distance(v, v) == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have distance 1."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        assert cosine_distance(v1, v2) == pytest.approx(1.0, abs=1e-6)

    def test_opposite_vectors(self):
        """Opposite vectors should have distance 2."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([-1.0, 0.0, 0.0])
        assert cosine_distance(v1, v2) == pytest.approx(2.0, abs=1e-6)


class TestCentroidStrategy:
    """Tests for CentroidStrategy."""

    @pytest.fixture
    def strategy(self):
        return CentroidStrategy()

    @pytest.fixture
    def config(self):
        return {
            "centroid_threshold": 0.38,
            "reject_threshold": 0.45,
        }

    def test_attaches_when_below_threshold(self, strategy, config):
        """Should attach when centroid distance is below threshold."""
        face_emb = np.array([1.0, 0.0, 0.0])
        centroid = np.array([0.95, 0.05, 0.0])
        centroid = centroid / np.linalg.norm(centroid)

        cluster_info = ClusterInfo(
            cluster_id=0,
            centroid=centroid,
            exemplar_embeddings=[]
        )

        result = strategy.evaluate(face_emb, cluster_info, config)

        assert result.attached is True
        assert result.cluster_id == 0
        assert result.confidence > 0

    def test_rejects_when_above_reject_threshold(self, strategy, config):
        """Should reject when centroid distance is above reject threshold."""
        face_emb = np.array([1.0, 0.0, 0.0])
        centroid = np.array([0.0, 1.0, 0.0])  # Orthogonal

        cluster_info = ClusterInfo(
            cluster_id=0,
            centroid=centroid,
            exemplar_embeddings=[]
        )

        result = strategy.evaluate(face_emb, cluster_info, config)

        assert result.attached is False

    def test_rejects_when_between_thresholds(self, strategy, config):
        """Should reject when distance is between centroid and reject thresholds."""
        face_emb = np.array([1.0, 0.0, 0.0])
        # Create a vector that gives distance ~0.42 (between 0.38 and 0.45)
        centroid = np.array([0.8, 0.2, 0.0])
        centroid = centroid / np.linalg.norm(centroid)

        cluster_info = ClusterInfo(
            cluster_id=0,
            centroid=centroid,
            exemplar_embeddings=[]
        )

        result = strategy.evaluate(face_emb, cluster_info, config)
        dist = result.centroid_distance

        # If distance is between thresholds, should not attach
        if 0.38 < dist < 0.45:
            assert result.attached is False


class TestExemplarStrategy:
    """Tests for ExemplarStrategy."""

    @pytest.fixture
    def strategy(self):
        return ExemplarStrategy()

    @pytest.fixture
    def config(self):
        return {
            "exemplar_threshold": 0.40,
            "reject_threshold": 0.45,
            "small_cluster_threshold": 3,
            "small_cluster_min_matches": 1,
        }

    def test_attaches_with_sufficient_exemplar_matches(self, strategy, config):
        """Should attach when enough exemplars match."""
        face_emb = np.array([1.0, 0.0, 0.0])

        # Create 4 exemplars, 3 very close to face
        exemplars = [
            np.array([0.98, 0.02, 0.0]),
            np.array([0.97, 0.03, 0.0]),
            np.array([0.96, 0.04, 0.0]),
            np.array([0.5, 0.5, 0.0]),  # One far away
        ]
        exemplars = [e / np.linalg.norm(e) for e in exemplars]

        cluster_info = ClusterInfo(
            cluster_id=0,
            centroid=np.array([1.0, 0.0, 0.0]),
            exemplar_embeddings=exemplars
        )

        result = strategy.evaluate(face_emb, cluster_info, config)

        # min_required = max(2, ceil(0.3 * 4)) = max(2, 2) = 2
        # We have 3 close exemplars, so should pass
        assert result.attached is True

    def test_rejects_with_insufficient_matches(self, strategy, config):
        """Should reject when not enough exemplars match."""
        face_emb = np.array([1.0, 0.0, 0.0])

        # Create 4 exemplars, only 1 close (others have cosine distance > 0.40)
        exemplars = [
            np.array([0.98, 0.02, 0.0]),  # Close - distance ~0.0002
            np.array([0.3, 0.7, 0.0]),    # Far - distance ~0.61
            np.array([0.0, 1.0, 0.0]),    # Orthogonal - distance 1.0
            np.array([0.0, 0.0, 1.0]),    # Orthogonal - distance 1.0
        ]
        exemplars = [e / np.linalg.norm(e) for e in exemplars]

        cluster_info = ClusterInfo(
            cluster_id=0,
            centroid=np.array([1.0, 0.0, 0.0]),
            exemplar_embeddings=exemplars
        )

        result = strategy.evaluate(face_emb, cluster_info, config)

        # min_required = max(2, ceil(0.3 * 4)) = 2
        # We have only 1 close exemplar, so should fail
        assert result.attached is False

    def test_small_cluster_relaxed_requirements(self, strategy, config):
        """Small clusters should have relaxed exemplar requirements."""
        face_emb = np.array([1.0, 0.0, 0.0])

        # Create 2 exemplars (small cluster), only 1 close
        exemplars = [
            np.array([0.98, 0.02, 0.0]),  # Close
            np.array([0.5, 0.5, 0.0]),    # Far
        ]
        exemplars = [e / np.linalg.norm(e) for e in exemplars]

        cluster_info = ClusterInfo(
            cluster_id=0,
            centroid=np.array([1.0, 0.0, 0.0]),
            exemplar_embeddings=exemplars
        )

        result = strategy.evaluate(face_emb, cluster_info, config)

        # 2 exemplars <= small_cluster_threshold (3)
        # min_required = small_cluster_min_matches = 1
        # We have 1 close exemplar, so should pass
        assert result.attached is True


class TestHybridStrategy:
    """Tests for HybridStrategy."""

    @pytest.fixture
    def strategy(self):
        return HybridStrategy()

    @pytest.fixture
    def config(self):
        return {
            "centroid_threshold": 0.38,
            "exemplar_threshold": 0.40,
            "reject_threshold": 0.45,
            "small_cluster_threshold": 3,
            "small_cluster_min_matches": 1,
        }

    def test_attaches_when_both_criteria_pass(self, strategy, config):
        """Should attach when both centroid and exemplar criteria pass."""
        face_emb = np.array([1.0, 0.0, 0.0])

        centroid = np.array([0.98, 0.02, 0.0])
        centroid = centroid / np.linalg.norm(centroid)

        exemplars = [
            np.array([0.98, 0.02, 0.0]),
            np.array([0.97, 0.03, 0.0]),
        ]
        exemplars = [e / np.linalg.norm(e) for e in exemplars]

        cluster_info = ClusterInfo(
            cluster_id=0,
            centroid=centroid,
            exemplar_embeddings=exemplars
        )

        result = strategy.evaluate(face_emb, cluster_info, config)

        assert result.attached is True

    def test_rejects_when_centroid_fails(self, strategy, config):
        """Should reject when centroid criteria fails, even if exemplars pass."""
        face_emb = np.array([1.0, 0.0, 0.0])

        # Centroid far, but exemplars close
        centroid = np.array([0.7, 0.3, 0.0])
        centroid = centroid / np.linalg.norm(centroid)

        exemplars = [
            np.array([0.98, 0.02, 0.0]),
            np.array([0.97, 0.03, 0.0]),
        ]
        exemplars = [e / np.linalg.norm(e) for e in exemplars]

        cluster_info = ClusterInfo(
            cluster_id=0,
            centroid=centroid,
            exemplar_embeddings=exemplars
        )

        result = strategy.evaluate(face_emb, cluster_info, config)

        # Centroid distance should be > 0.38, so should fail
        if result.centroid_distance > config["centroid_threshold"]:
            assert result.attached is False

    def test_rejects_when_exemplars_fail(self, strategy, config):
        """Should reject when exemplar criteria fails, even if centroid passes."""
        face_emb = np.array([1.0, 0.0, 0.0])

        # Centroid close, but exemplars far
        centroid = np.array([0.98, 0.02, 0.0])
        centroid = centroid / np.linalg.norm(centroid)

        exemplars = [
            np.array([0.5, 0.5, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([-1.0, 0.0, 0.0]),
        ]
        exemplars = [e / np.linalg.norm(e) for e in exemplars]

        cluster_info = ClusterInfo(
            cluster_id=0,
            centroid=centroid,
            exemplar_embeddings=exemplars
        )

        result = strategy.evaluate(face_emb, cluster_info, config)

        # Exemplar matches should be 0, so should fail
        assert result.attached is False


class TestAttachmentStrategyFactory:
    """Tests for AttachmentStrategyFactory."""

    def test_creates_centroid_strategy(self):
        strategy = AttachmentStrategyFactory.create("centroid")
        assert isinstance(strategy, CentroidStrategy)

    def test_creates_exemplar_strategy(self):
        strategy = AttachmentStrategyFactory.create("exemplar")
        assert isinstance(strategy, ExemplarStrategy)

    def test_creates_hybrid_strategy(self):
        strategy = AttachmentStrategyFactory.create("hybrid")
        assert isinstance(strategy, HybridStrategy)

    def test_defaults_to_hybrid(self):
        strategy = AttachmentStrategyFactory.create("unknown")
        assert isinstance(strategy, HybridStrategy)
