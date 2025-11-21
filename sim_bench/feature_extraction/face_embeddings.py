"""
Face embedding-based similarity method for image retrieval benchmarking.

Extracts face embeddings and uses them for similarity computation.
Treats face recognition as a feature extraction method like DINOv2, CLIP, etc.
"""

from typing import List, Dict, Any
import numpy as np
from pathlib import Path
import logging

from sim_bench.feature_extraction.base import BaseMethod
from sim_bench.photo_analysis.factory import create_photo_analyzer

logger = logging.getLogger(__name__)


class FaceEmbeddingMethod(BaseMethod):
    """
    Face embedding-based similarity method.

    Uses face detection + embedding extraction to compute image similarity.
    Images are similar if they contain the same people.

    Aggregation strategies for multiple faces per image:
    - 'average': Average all face embeddings (default)
    - 'max_confidence': Use embedding of most confident face detection
    - 'first': Use first detected face only
    - 'concat': Concatenate all embeddings (fixed-size via padding/truncation)
    """

    def __init__(self, method_config: Dict[str, Any]):
        """
        Initialize face embedding method.

        Args:
            method_config: Configuration dictionary with keys:
                - method: 'face_embeddings'
                - backend: Face detection backend (retinaface, opencv, etc.)
                - embedding_model: Face recognition model (VGG-Face, Facenet, etc.)
                - aggregation: How to handle multiple faces ('average', 'max_confidence', 'first', 'concat')
                - max_faces: Maximum faces to consider per image (for 'concat' mode)
                - embedding_dim: Embedding dimension (auto-detected if not specified)
                - device: 'cpu' or 'cuda'
        """
        super().__init__(method_config)

        self.backend = method_config.get('backend', 'retinaface')
        self.embedding_model = method_config.get('embedding_model', 'VGG-Face')
        self.aggregation = method_config.get('aggregation', 'average')
        self.max_faces = method_config.get('max_faces', 5)
        self.device = method_config.get('device', 'cpu')
        self.min_confidence = method_config.get('min_confidence', 0.9)

        # Initialize face analyzer
        self.face_analyzer = None
        self.embedding_dim = method_config.get('embedding_dim', None)

        logger.info(
            f"Initialized FaceEmbeddingMethod: backend={self.backend}, "
            f"model={self.embedding_model}, aggregation={self.aggregation}"
        )

    def _get_face_analyzer(self):
        """Lazy load face analyzer."""
        if self.face_analyzer is None:
            self.face_analyzer = create_photo_analyzer(
                analyzer_type='face',
                config={
                    'backend': self.backend,
                    'embedding_model': self.embedding_model,
                    'device': self.device
                }
            )
        return self.face_analyzer

    def extract_features(self, image_paths: List[str]) -> np.ndarray:
        """
        Extract face embeddings from images.

        Args:
            image_paths: List of image file paths

        Returns:
            Feature matrix [n_images, embedding_dim]
            For images without faces, returns zero vector
        """
        analyzer = self._get_face_analyzer()

        logger.info(
            f"Extracting face embeddings from {len(image_paths)} images "
            f"(aggregation={self.aggregation})"
        )

        all_embeddings = []

        for i, img_path in enumerate(image_paths):
            try:
                # Detect faces and extract embeddings
                result = analyzer.analyze_single(
                    img_path,
                    min_confidence=self.min_confidence
                )

                # Get embeddings
                embedding = self._aggregate_face_embeddings(result)

                # Set embedding dimension on first successful extraction
                if self.embedding_dim is None and embedding is not None:
                    self.embedding_dim = len(embedding)

                all_embeddings.append(embedding)

                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(image_paths)} images")

            except Exception as e:
                logger.warning(f"Failed to extract faces from {Path(img_path).name}: {e}")
                # Return zero vector for failed images
                all_embeddings.append(None)

        # Convert to numpy array, handling None values
        if self.embedding_dim is None:
            # No faces detected in any image - use default dimension
            self.embedding_dim = 512
            logger.warning(
                f"No faces detected in any image. Using default embedding_dim={self.embedding_dim}"
            )

        # Create feature matrix
        features = []
        for emb in all_embeddings:
            if emb is not None:
                features.append(emb)
            else:
                # Zero vector for images without faces
                features.append(np.zeros(self.embedding_dim))

        feature_matrix = np.vstack(features).astype('float32')

        logger.info(
            f"Extracted face embeddings: shape={feature_matrix.shape}, "
            f"images_with_faces={sum(1 for e in all_embeddings if e is not None)}/{len(image_paths)}"
        )

        return feature_matrix

    def _aggregate_face_embeddings(self, face_result: Dict) -> np.ndarray:
        """
        Aggregate multiple face embeddings into single vector.

        Args:
            face_result: Result from face analyzer with 'faces' list

        Returns:
            Aggregated embedding vector or None if no faces
        """
        faces = face_result.get('faces', [])

        if not faces:
            return None

        embeddings = [f['embedding'] for f in faces if 'embedding' in f]

        if not embeddings:
            return None

        if self.aggregation == 'average':
            # Average all face embeddings
            return np.mean(embeddings, axis=0)

        elif self.aggregation == 'max_confidence':
            # Use embedding from most confident detection
            confidences = [f.get('confidence', 0) for f in faces]
            max_idx = np.argmax(confidences)
            return embeddings[max_idx]

        elif self.aggregation == 'first':
            # Use first detected face
            return embeddings[0]

        elif self.aggregation == 'concat':
            # Concatenate up to max_faces embeddings
            # Pad with zeros if fewer faces, truncate if more
            emb_dim = len(embeddings[0])
            total_dim = emb_dim * self.max_faces

            concatenated = np.zeros(total_dim)

            for i, emb in enumerate(embeddings[:self.max_faces]):
                start_idx = i * emb_dim
                end_idx = start_idx + emb_dim
                concatenated[start_idx:end_idx] = emb

            return concatenated

        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

    def __str__(self) -> str:
        return (
            f"FaceEmbeddingMethod(backend={self.backend}, "
            f"model={self.embedding_model}, aggregation={self.aggregation})"
        )
