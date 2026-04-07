"""Tests for extract_face_embeddings step.

This step requires pre-aligned faces from align_faces step.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image
import tempfile
import os

from sim_bench.pipeline.steps.extract_face_embeddings import ExtractFaceEmbeddingsStep
from sim_bench.pipeline.context import PipelineContext


@pytest.fixture
def temp_image():
    """Create a temporary test image."""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        img = Image.new('RGB', (640, 480), color='red')
        img.save(f.name, 'JPEG')
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def aligned_face_crop():
    """Create a 256x256 aligned face crop."""
    return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def context_with_aligned_faces(temp_image, aligned_face_crop):
    """Create a context with aligned faces from align_faces step."""
    context = PipelineContext(source_directory=Path(tempfile.gettempdir()))

    # Face metadata from insightface_detect_faces
    context.insightface_faces = {
        temp_image: {
            'faces': [
                {
                    'face_index': 0,
                    'bbox': {
                        'x': 0.2, 'y': 0.2, 'w': 0.3, 'h': 0.4,
                        'x_px': 128, 'y_px': 96, 'w_px': 192, 'h_px': 192
                    },
                    'confidence': 0.95,
                    'filter_passed': True,
                    'is_clusterable': True,
                },
                {
                    'face_index': 1,
                    'bbox': {
                        'x': 0.5, 'y': 0.2, 'w': 0.2, 'h': 0.3,
                        'x_px': 320, 'y_px': 96, 'w_px': 128, 'h_px': 144
                    },
                    'confidence': 0.87,
                    'filter_passed': True,
                    'is_clusterable': True,
                }
            ]
        }
    }

    # Aligned faces from align_faces step
    context.aligned_faces = {
        f"{temp_image}:face_0": aligned_face_crop.copy(),
        f"{temp_image}:face_1": aligned_face_crop.copy(),
    }

    return context


@pytest.fixture
def context_with_filtered_faces(temp_image, aligned_face_crop):
    """Create a context with some filtered faces."""
    context = PipelineContext(source_directory=Path(tempfile.gettempdir()))

    context.insightface_faces = {
        temp_image: {
            'faces': [
                {
                    'face_index': 0,
                    'bbox': {'x_px': 128, 'y_px': 96, 'w_px': 192, 'h_px': 192},
                    'confidence': 0.95,
                    'filter_passed': True,
                    'is_clusterable': True,
                },
                {
                    'face_index': 1,
                    'bbox': {'x_px': 320, 'y_px': 96, 'w_px': 128, 'h_px': 144},
                    'confidence': 0.87,
                    'filter_passed': False,  # Filtered out
                    'is_clusterable': True,
                },
                {
                    'face_index': 2,
                    'bbox': {'x_px': 400, 'y_px': 96, 'w_px': 100, 'h_px': 100},
                    'confidence': 0.80,
                    'filter_passed': True,
                    'is_clusterable': False,  # Not clusterable
                },
            ]
        }
    }

    # Only face_0 has alignment (others were filtered before align_faces)
    context.aligned_faces = {
        f"{temp_image}:face_0": aligned_face_crop.copy(),
    }

    return context


class TestExtractFaceEmbeddingsStep:
    """Tests for ExtractFaceEmbeddingsStep."""

    def test_step_metadata(self):
        """Test step metadata is correct."""
        step = ExtractFaceEmbeddingsStep()

        assert step._metadata.name == "extract_face_embeddings"
        assert "aligned_faces" in step._metadata.requires
        assert "align_faces" in step._metadata.depends_on

    def test_get_all_faces_from_aligned(self, context_with_aligned_faces):
        """Test that aligned faces are correctly retrieved."""
        step = ExtractFaceEmbeddingsStep()
        faces = step._get_all_faces(context_with_aligned_faces)

        assert len(faces) == 2
        assert faces[0].face_index == 0
        assert faces[1].face_index == 1
        assert faces[0].image is not None
        assert faces[1].image is not None
        assert faces[0].image.shape == (256, 256, 3)

    def test_get_all_faces_skips_filtered(self, context_with_filtered_faces):
        """Test that filtered faces are skipped."""
        step = ExtractFaceEmbeddingsStep()
        faces = step._get_all_faces(context_with_filtered_faces)

        # Only face_0 should be included (face_1 filtered, face_2 not clusterable)
        assert len(faces) == 1
        assert faces[0].face_index == 0

    def test_get_all_faces_empty_context(self):
        """Test with empty context."""
        context = PipelineContext(source_directory=Path(tempfile.gettempdir()))
        step = ExtractFaceEmbeddingsStep()
        faces = step._get_all_faces(context)

        assert faces == []

    def test_get_all_faces_no_aligned_faces(self, temp_image):
        """Test with context that has no aligned_faces."""
        context = PipelineContext(source_directory=Path(tempfile.gettempdir()))
        context.insightface_faces = {
            temp_image: {'faces': [{'face_index': 0, 'confidence': 0.9}]}
        }
        # No aligned_faces - align_faces step didn't run

        step = ExtractFaceEmbeddingsStep()
        faces = step._get_all_faces(context)

        assert faces == []

    def test_generate_cache_key(self, context_with_aligned_faces):
        """Test cache key generation."""
        step = ExtractFaceEmbeddingsStep()
        faces = step._get_all_faces(context_with_aligned_faces)

        key0 = step._generate_cache_key(faces[0])
        key1 = step._generate_cache_key(faces[1])

        assert 'face_0' in key0
        assert 'face_1' in key1
        assert key0 != key1

    def test_get_cache_config_returns_none_when_no_faces(self):
        """Test that cache config is None when no faces."""
        context = PipelineContext(source_directory=Path(tempfile.gettempdir()))
        step = ExtractFaceEmbeddingsStep()

        config = step._get_cache_config(context, {'backend': 'insightface'})
        assert config is None

    def test_get_cache_config_with_faces(self, context_with_aligned_faces):
        """Test cache config with valid faces."""
        step = ExtractFaceEmbeddingsStep()

        config = step._get_cache_config(
            context_with_aligned_faces,
            {'backend': 'insightface'}
        )

        assert config is not None
        assert config['feature_type'] == 'face_embedding'
        assert 'insightface' in config['model_name']
        assert len(config['items']) == 2


class TestExtractFaceEmbeddingsIntegration:
    """Integration tests (require model)."""

    def test_full_extraction_pipeline(self, context_with_aligned_faces):
        """Test full face embedding extraction."""
        step = ExtractFaceEmbeddingsStep()

        config = {
            'backend': 'insightface',
            'device': 'cpu'
        }

        # Run the step
        step.process(context_with_aligned_faces, config)

        # Check embeddings were stored
        assert hasattr(context_with_aligned_faces, 'face_embeddings')
        assert len(context_with_aligned_faces.face_embeddings) == 2

        # Check embedding shape (ArcFace produces 512-dim vectors)
        for key, embedding in context_with_aligned_faces.face_embeddings.items():
            assert isinstance(embedding, np.ndarray)
            assert len(embedding.shape) == 1
            assert embedding.shape[0] == 512
