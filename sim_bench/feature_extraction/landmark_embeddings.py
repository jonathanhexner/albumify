"""
Landmark embedding-based similarity method for image retrieval benchmarking.

Extracts landmark/place embeddings and uses them for similarity computation.
Images are similar if they show the same landmark or location.
"""

from typing import List, Dict, Any
import numpy as np
from pathlib import Path
import logging

from sim_bench.feature_extraction.base import BaseMethod
from sim_bench.photo_analysis.factory import create_photo_analyzer

logger = logging.getLogger(__name__)


class LandmarkEmbeddingMethod(BaseMethod):
    """
    Landmark embedding-based similarity method.

    Uses landmark detection + embedding extraction to compute image similarity.
    Images are similar if they show the same landmark or location.

    Can use:
    - Visual embeddings from landmark recognition models
    - Categorical encoding of detected landmarks
    - Hybrid: embedding + landmark category
    """

    def __init__(self, method_config: Dict[str, Any]):
        """
        Initialize landmark embedding method.

        Args:
            method_config: Configuration dictionary with keys:
                - method: 'landmark_embeddings'
                - encoding: 'embedding', 'categorical', or 'hybrid'
                - embedding_dim: Embedding dimension
                - device: 'cpu' or 'cuda'
                - min_confidence: Minimum detection confidence
        """
        super().__init__(method_config)

        self.encoding = method_config.get('encoding', 'embedding')
        self.embedding_dim = method_config.get('embedding_dim', 512)
        self.device = method_config.get('device', 'cpu')
        self.min_confidence = method_config.get('min_confidence', 0.5)

        # Initialize landmark analyzer
        self.landmark_analyzer = None

        # For categorical encoding
        self.landmark_to_id = {}
        self.next_landmark_id = 0

        logger.info(
            f"Initialized LandmarkEmbeddingMethod: encoding={self.encoding}, "
            f"embedding_dim={self.embedding_dim}"
        )

    def _get_landmark_analyzer(self):
        """Lazy load landmark analyzer."""
        if self.landmark_analyzer is None:
            self.landmark_analyzer = create_photo_analyzer(
                analyzer_type='landmark',
                config={'device': self.device}
            )
        return self.landmark_analyzer

    def extract_features(self, image_paths: List[str]) -> np.ndarray:
        """
        Extract landmark embeddings from images.

        Args:
            image_paths: List of image file paths

        Returns:
            Feature matrix [n_images, embedding_dim]
            For images without landmarks, returns zero vector
        """
        analyzer = self._get_landmark_analyzer()

        logger.info(
            f"Extracting landmark embeddings from {len(image_paths)} images "
            f"(encoding={self.encoding})"
        )

        all_embeddings = []

        for i, img_path in enumerate(image_paths):
            try:
                # Detect landmark
                result = analyzer.analyze_single(
                    img_path,
                    min_confidence=self.min_confidence
                )

                # Extract embedding based on encoding type
                embedding = self._extract_embedding(result)
                all_embeddings.append(embedding)

                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(image_paths)} images")

            except Exception as e:
                logger.warning(f"Failed to extract landmark from {Path(img_path).name}: {e}")
                # Return zero vector for failed images
                all_embeddings.append(np.zeros(self.embedding_dim))

        feature_matrix = np.vstack(all_embeddings).astype('float32')

        detected_count = sum(1 for e in all_embeddings if not np.allclose(e, 0))

        logger.info(
            f"Extracted landmark embeddings: shape={feature_matrix.shape}, "
            f"images_with_landmarks={detected_count}/{len(image_paths)}"
        )

        if self.encoding == 'categorical':
            logger.info(f"Unique landmarks detected: {len(self.landmark_to_id)}")

        return feature_matrix

    def _extract_embedding(self, landmark_result: Dict) -> np.ndarray:
        """
        Extract embedding from landmark detection result.

        Args:
            landmark_result: Result from landmark analyzer

        Returns:
            Embedding vector
        """
        has_landmark = landmark_result.get('detected', False)

        if not has_landmark:
            return np.zeros(self.embedding_dim)

        if self.encoding == 'embedding':
            # Use visual embedding if available
            if 'embedding' in landmark_result:
                return np.array(landmark_result['embedding'])
            else:
                # Fallback to categorical if no embedding
                return self._categorical_encoding(landmark_result)

        elif self.encoding == 'categorical':
            return self._categorical_encoding(landmark_result)

        elif self.encoding == 'hybrid':
            # Combine visual embedding with categorical encoding
            if 'embedding' in landmark_result:
                visual_emb = np.array(landmark_result['embedding'])
                cat_emb = self._categorical_encoding(landmark_result)

                # Concatenate (resize to fit embedding_dim)
                visual_dim = min(len(visual_emb), self.embedding_dim - len(cat_emb))
                combined = np.concatenate([
                    visual_emb[:visual_dim],
                    cat_emb
                ])

                # Pad if needed
                if len(combined) < self.embedding_dim:
                    combined = np.pad(
                        combined,
                        (0, self.embedding_dim - len(combined)),
                        mode='constant'
                    )

                return combined[:self.embedding_dim]
            else:
                return self._categorical_encoding(landmark_result)

        else:
            raise ValueError(f"Unknown encoding type: {self.encoding}")

    def _categorical_encoding(self, landmark_result: Dict) -> np.ndarray:
        """
        Create categorical encoding for landmark.

        Uses one-hot encoding with learned landmark vocabulary.

        Args:
            landmark_result: Result with 'landmark' name

        Returns:
            One-hot encoded vector
        """
        landmark_name = landmark_result.get('landmark', 'Unknown')
        location = landmark_result.get('location', 'Unknown')

        # Use landmark name + location as unique key
        landmark_key = f"{landmark_name}|{location}"

        # Assign ID if not seen before
        if landmark_key not in self.landmark_to_id:
            self.landmark_to_id[landmark_key] = self.next_landmark_id
            self.next_landmark_id += 1

        landmark_id = self.landmark_to_id[landmark_key]

        # Create one-hot encoding
        # For efficiency, use fixed size and wrap around if needed
        max_landmarks = self.embedding_dim
        effective_id = landmark_id % max_landmarks

        encoding = np.zeros(self.embedding_dim)
        encoding[effective_id] = 1.0

        return encoding

    def __str__(self) -> str:
        return f"LandmarkEmbeddingMethod(encoding={self.encoding}, dim={self.embedding_dim})"
