"""Tests for identity refinement step."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from sim_bench.pipeline.steps.identity_refinement import IdentityRefinementStep


@dataclass
class MockFace:
    """Mock face object for testing."""
    original_path: Path
    face_index: int
    embedding: np.ndarray = None
    bbox: Dict[str, Any] = field(default_factory=dict)
    cluster_id: int = -1
    frontal_score: float = 0.5


@pytest.fixture
def step():
    """Create identity refinement step."""
    return IdentityRefinementStep()


@pytest.fixture
def mock_context():
    """Create mock pipeline context."""
    context = MagicMock()
    context.people_clusters = {}
    context.face_embeddings = {}
    context.insightface_faces = {}
    context.user_overrides = []
    context.refined_people_clusters = {}
    context.unassigned_faces = []
    context.cluster_exemplars = {}
    context.cluster_centroids = {}
    context.attachment_decisions = {}
    context.report_progress = MagicMock()
    return context


@pytest.fixture
def config():
    """Default config for testing."""
    return {
        "enabled": True,
        "centroid_threshold": 0.38,
        "exemplar_threshold": 0.40,
        "reject_threshold": 0.45,
        "exemplars_per_cluster": 5,
        "exemplar_min_frontal_score": 0.6,
        "exemplar_selection_method": "quality_diverse",
        "attachment_strategy": "hybrid",
        "small_cluster_threshold": 3,
        "small_cluster_min_matches": 1,
        "preserve_original_clusters": True,
        "apply_user_overrides": True,
    }


class TestSeparateNoise:
    """Tests for noise separation."""

    def test_separates_noise_from_core(self, step):
        """Should separate cluster -1 as noise."""
        face_a = MockFace(Path("a.jpg"), 0)
        face_b = MockFace(Path("b.jpg"), 0)
        face_noise = MockFace(Path("noise.jpg"), 0)

        clusters = {
            0: [face_a],
            1: [face_b],
            -1: [face_noise],
        }

        core, noise = step._separate_noise(clusters)

        assert 0 in core
        assert 1 in core
        assert -1 not in core
        assert len(noise) == 1
        assert noise[0] is face_noise

    def test_handles_no_noise(self, step):
        """Should handle case with no noise cluster."""
        face_a = MockFace(Path("a.jpg"), 0)

        clusters = {
            0: [face_a],
        }

        core, noise = step._separate_noise(clusters)

        assert 0 in core
        assert len(noise) == 0


class TestFaceKey:
    """Tests for face key generation."""

    def test_generates_correct_key(self, step):
        """Should generate consistent face key."""
        face = MockFace(Path("D:/photos/test.jpg"), 2)
        key = step._face_key(face)
        assert key == "D:/photos/test.jpg:face_2"

    def test_normalizes_backslashes(self, step):
        """Should normalize Windows backslashes."""
        face = MockFace(Path("D:\\photos\\test.jpg"), 0)
        key = step._face_key(face)
        assert "\\" not in key
        assert "D:/photos/test.jpg:face_0" == key


class TestComputeCentroid:
    """Tests for centroid computation."""

    def test_computes_normalized_centroid(self, step):
        """Should compute normalized centroid from embeddings."""
        face_a = MockFace(Path("a.jpg"), 0)
        face_b = MockFace(Path("b.jpg"), 0)

        embeddings = {
            "a.jpg:face_0": np.array([1.0, 0.0, 0.0]),
            "b.jpg:face_0": np.array([0.0, 1.0, 0.0]),
        }

        centroid = step._compute_centroid([face_a, face_b], embeddings)

        # Should be normalized
        assert np.abs(np.linalg.norm(centroid) - 1.0) < 1e-6

        # Should be average direction
        expected = np.array([0.5, 0.5, 0.0])
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_array_almost_equal(centroid, expected)

    def test_handles_missing_embeddings(self, step):
        """Should handle faces with missing embeddings."""
        face_a = MockFace(Path("a.jpg"), 0)
        face_b = MockFace(Path("b.jpg"), 0)

        embeddings = {
            "a.jpg:face_0": np.array([1.0, 0.0, 0.0]),
            # b.jpg missing
        }

        centroid = step._compute_centroid([face_a, face_b], embeddings)

        # Should use only available embedding
        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(centroid, expected)


class TestSelectExemplars:
    """Tests for exemplar selection."""

    def test_selects_high_quality_faces(self, step, config):
        """Should prefer faces with higher frontal scores."""
        faces = [
            MockFace(Path("a.jpg"), 0, frontal_score=0.9),
            MockFace(Path("b.jpg"), 0, frontal_score=0.3),
            MockFace(Path("c.jpg"), 0, frontal_score=0.8),
            MockFace(Path("d.jpg"), 0, frontal_score=0.2),
        ]

        embeddings = {
            "a.jpg:face_0": np.array([1.0, 0.0, 0.0]),
            "b.jpg:face_0": np.array([0.0, 1.0, 0.0]),
            "c.jpg:face_0": np.array([0.0, 0.0, 1.0]),
            "d.jpg:face_0": np.array([1.0, 1.0, 0.0]),
        }

        config["exemplars_per_cluster"] = 2
        config["exemplar_selection_method"] = "quality"

        exemplars = step._select_exemplars(faces, embeddings, config)

        assert len(exemplars) == 2
        # Should select the two highest quality faces
        frontal_scores = [e.frontal_score for e in exemplars]
        assert 0.9 in frontal_scores
        assert 0.8 in frontal_scores

    def test_returns_all_if_fewer_than_k(self, step, config):
        """Should return all faces if fewer than K available."""
        faces = [
            MockFace(Path("a.jpg"), 0, frontal_score=0.9),
            MockFace(Path("b.jpg"), 0, frontal_score=0.8),
        ]

        embeddings = {
            "a.jpg:face_0": np.array([1.0, 0.0, 0.0]),
            "b.jpg:face_0": np.array([0.0, 1.0, 0.0]),
        }

        config["exemplars_per_cluster"] = 5

        exemplars = step._select_exemplars(faces, embeddings, config)

        assert len(exemplars) == 2


class TestProcessDisabled:
    """Tests for disabled refinement."""

    def test_passes_through_when_disabled(self, step, mock_context, config):
        """Should pass through unchanged when disabled."""
        config["enabled"] = False

        face_a = MockFace(Path("a.jpg"), 0)
        mock_context.people_clusters = {0: [face_a]}

        step.process(mock_context, config)

        assert mock_context.refined_people_clusters == {0: [face_a]}


class TestProcessIntegration:
    """Integration tests for process method."""

    def test_attaches_close_noise_face(self, step, mock_context, config):
        """Should attach noise face that is close to a cluster."""
        # Create core cluster with one face
        core_face = MockFace(Path("core.jpg"), 0)
        core_emb = np.array([1.0, 0.0, 0.0])

        # Create noise face very close to core
        noise_face = MockFace(Path("noise.jpg"), 0)
        noise_emb = np.array([0.98, 0.02, 0.0])
        noise_emb = noise_emb / np.linalg.norm(noise_emb)

        mock_context.people_clusters = {
            0: [core_face],
            -1: [noise_face],
        }

        mock_context.face_embeddings = {
            "core.jpg:face_0": core_emb,
            "noise.jpg:face_0": noise_emb,
        }

        step.process(mock_context, config)

        # Noise face should be attached to cluster 0
        assert len(mock_context.refined_people_clusters[0]) == 2
        assert len(mock_context.unassigned_faces) == 0

    def test_rejects_distant_noise_face(self, step, mock_context, config):
        """Should not attach noise face that is far from all clusters."""
        # Create core cluster with one face
        core_face = MockFace(Path("core.jpg"), 0)
        core_emb = np.array([1.0, 0.0, 0.0])

        # Create noise face far from core
        noise_face = MockFace(Path("noise.jpg"), 0)
        noise_emb = np.array([0.0, 1.0, 0.0])  # Orthogonal

        mock_context.people_clusters = {
            0: [core_face],
            -1: [noise_face],
        }

        mock_context.face_embeddings = {
            "core.jpg:face_0": core_emb,
            "noise.jpg:face_0": noise_emb,
        }

        step.process(mock_context, config)

        # Noise face should remain unassigned
        assert len(mock_context.refined_people_clusters[0]) == 1
        assert len(mock_context.unassigned_faces) == 1
