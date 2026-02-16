"""Custom ArcFace extractor using trained checkpoint."""

import logging
from typing import List, Dict, Any

import numpy as np

from sim_bench.pipeline.face_embedding.base import BaseFaceEmbeddingExtractor

logger = logging.getLogger(__name__)


class CustomArcFaceExtractor(BaseFaceEmbeddingExtractor):
    """Extractor using custom-trained ArcFace model."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.checkpoint_path = config.get("checkpoint_path")
        if not self.checkpoint_path:
            raise ValueError("checkpoint_path required for CustomArcFaceExtractor")
        self._service = None

    def _get_service(self):
        """Lazy load face embedding service."""
        if self._service is None:
            from sim_bench.album.services.face_embedding_service import FaceEmbeddingService
            logger.info(f"Loading custom ArcFace model from {self.checkpoint_path}")
            svc_config = {
                'face': {'checkpoint_path': self.checkpoint_path},
                'device': self.device
            }
            self._service = FaceEmbeddingService(svc_config)
        return self._service

    def extract_batch(
        self,
        face_images: List[np.ndarray],
        face_metadata: List[Dict[str, Any]]
    ) -> List[np.ndarray]:
        """Extract embeddings using custom ArcFace model."""
        service = self._get_service()
        embeddings_array = service.extract_embeddings_batch(
            face_images,
            batch_size=32,
            show_progress=False
        )
        return [embeddings_array[i] for i in range(len(face_images))]

    def extract_single(
        self,
        face_image: np.ndarray,
        face_metadata: Dict[str, Any]
    ) -> np.ndarray:
        """Extract embedding for a single face."""
        return self.extract_batch([face_image], [face_metadata])[0]

    @property
    def embedding_dim(self) -> int:
        """Custom ArcFace produces 512-dim embeddings."""
        return 512

    @property
    def model_name(self) -> str:
        """Cache key identifier."""
        return "arcface_custom"
