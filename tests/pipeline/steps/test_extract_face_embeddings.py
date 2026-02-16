"""Tests for extract_face_embeddings step with InsightFace support."""

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
def context_with_insightface_faces(temp_image):
    """Create a context with InsightFace face data."""
    context = PipelineContext(source_directory=Path(tempfile.gettempdir()))
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
                    'landmarks': [[0, 0]] * 5,
                },
                {
                    'face_index': 1,
                    'bbox': {
                        'x': 0.5, 'y': 0.2, 'w': 0.2, 'h': 0.3,
                        'x_px': 320, 'y_px': 96, 'w_px': 128, 'h_px': 144
                    },
                    'confidence': 0.87,
                    'landmarks': [[0, 0]] * 5,
                }
            ]
        }
    }
    return context


@pytest.fixture
def context_with_invalid_faces(temp_image):
    """Create a context with some invalid InsightFace face data."""
    context = PipelineContext(source_directory=Path(tempfile.gettempdir()))
    context.insightface_faces = {
        temp_image: {
            'faces': [
                # Valid face
                {
                    'face_index': 0,
                    'bbox': {
                        'x': 0.2, 'y': 0.2, 'w': 0.3, 'h': 0.4,
                        'x_px': 128, 'y_px': 96, 'w_px': 192, 'h_px': 192
                    },
                    'confidence': 0.95,
                },
                # Invalid: zero width
                {
                    'face_index': 1,
                    'bbox': {
                        'x': 0.5, 'y': 0.2, 'w': 0, 'h': 0.3,
                        'x_px': 320, 'y_px': 96, 'w_px': 0, 'h_px': 144
                    },
                    'confidence': 0.87,
                },
                # Invalid: zero height
                {
                    'face_index': 2,
                    'bbox': {
                        'x': 0.5, 'y': 0.2, 'w': 0.2, 'h': 0,
                        'x_px': 320, 'y_px': 96, 'w_px': 128, 'h_px': 0
                    },
                    'confidence': 0.80,
                },
            ]
        },
        # Non-existent image
        '/nonexistent/image.jpg': {
            'faces': [
                {
                    'face_index': 0,
                    'bbox': {
                        'x': 0.2, 'y': 0.2, 'w': 0.3, 'h': 0.4,
                        'x_px': 128, 'y_px': 96, 'w_px': 192, 'h_px': 192
                    },
                    'confidence': 0.95,
                }
            ]
        }
    }
    return context


class TestExtractFaceEmbeddingsStep:
    """Tests for ExtractFaceEmbeddingsStep."""

    def test_get_all_faces_from_insightface(self, context_with_insightface_faces):
        """Test that InsightFace faces are correctly converted."""
        step = ExtractFaceEmbeddingsStep()
        faces = step._get_all_faces(context_with_insightface_faces)

        assert len(faces) == 2
        assert faces[0].face_index == 0
        assert faces[1].face_index == 1
        assert faces[0].image is not None
        assert faces[1].image is not None
        assert faces[0].detection_confidence == 0.95
        assert faces[1].detection_confidence == 0.87

    def test_get_all_faces_skips_invalid_bbox(self, context_with_invalid_faces):
        """Test that faces with invalid bbox are skipped."""
        step = ExtractFaceEmbeddingsStep()
        faces = step._get_all_faces(context_with_invalid_faces)

        # Should only have 1 valid face (skipped: 2 invalid bbox + 1 nonexistent image)
        assert len(faces) == 1
        assert faces[0].face_index == 0

    def test_get_all_faces_empty_context(self):
        """Test with empty context."""
        context = PipelineContext(source_directory=Path(tempfile.gettempdir()))
        step = ExtractFaceEmbeddingsStep()
        faces = step._get_all_faces(context)

        assert faces == []

    def test_get_all_faces_no_insightface_data(self):
        """Test with context that has no InsightFace data."""
        context = PipelineContext(source_directory=Path(tempfile.gettempdir()))
        context.faces = {}  # MediaPipe format, but empty
        step = ExtractFaceEmbeddingsStep()
        faces = step._get_all_faces(context)

        assert faces == []

    def test_face_image_is_rgb_array(self, context_with_insightface_faces):
        """Test that cropped face images are RGB numpy arrays."""
        step = ExtractFaceEmbeddingsStep()
        faces = step._get_all_faces(context_with_insightface_faces)

        for face in faces:
            assert isinstance(face.image, np.ndarray)
            assert len(face.image.shape) == 3  # H x W x C
            assert face.image.shape[2] == 3  # RGB

    def test_generate_cache_key(self, context_with_insightface_faces):
        """Test cache key generation."""
        step = ExtractFaceEmbeddingsStep()
        faces = step._get_all_faces(context_with_insightface_faces)

        key0 = step._generate_cache_key(faces[0])
        key1 = step._generate_cache_key(faces[1])

        assert 'face_0' in key0
        assert 'face_1' in key1
        assert key0 != key1

    def test_get_cache_config_returns_none_when_no_faces(self):
        """Test that cache config is None when no faces."""
        context = PipelineContext(source_directory=Path(tempfile.gettempdir()))
        step = ExtractFaceEmbeddingsStep()

        config = step._get_cache_config(context, {'checkpoint_path': 'model.pt'})
        assert config is None

    def test_get_cache_config_with_faces(self, context_with_insightface_faces):
        """Test cache config with valid faces."""
        step = ExtractFaceEmbeddingsStep()

        config = step._get_cache_config(
            context_with_insightface_faces,
            {'checkpoint_path': 'model.pt'}
        )

        assert config is not None
        assert config['feature_type'] == 'face_embedding'
        assert config['model_name'] == 'arcface'
        assert len(config['items']) == 2


class TestExtractFaceEmbeddingsIntegration:
    """Integration tests (require model checkpoint)."""

    @pytest.mark.skipif(
        not Path('models/album_app/arcface_resnet50.pt').exists(),
        reason='ArcFace model not found'
    )
    def test_full_extraction_pipeline(self, context_with_insightface_faces):
        """Test full face embedding extraction with real model."""
        step = ExtractFaceEmbeddingsStep()

        config = {
            'checkpoint_path': 'models/album_app/arcface_resnet50.pt',
            'device': 'cpu'
        }

        # Run the step
        step.process(context_with_insightface_faces, config)

        # Check embeddings were stored
        assert hasattr(context_with_insightface_faces, 'face_embeddings')
        assert len(context_with_insightface_faces.face_embeddings) == 2

        # Check embedding shape (ArcFace typically produces 512-dim vectors)
        for key, embedding in context_with_insightface_faces.face_embeddings.items():
            assert isinstance(embedding, np.ndarray)
            assert len(embedding.shape) == 1
            assert embedding.shape[0] > 0  # Has dimensions
