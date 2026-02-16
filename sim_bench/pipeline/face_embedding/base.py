"""Base class for face embedding extraction strategies."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

import numpy as np


class BaseFaceEmbeddingExtractor(ABC):
    """Base class for face embedding extraction strategies."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize extractor with config dict.

        Args:
            config: Configuration dict with keys like:
                - device: "cpu" | "cuda" | "mps"
                - checkpoint_path: str (for custom backend)
                - model_name: str (for insightface backend)
        """
        self.config = config
        self.device = config.get("device", "cpu")

    @abstractmethod
    def extract_batch(
        self,
        face_images: List[np.ndarray],
        face_metadata: List[Dict[str, Any]]
    ) -> List[np.ndarray]:
        """Extract embeddings for a batch of face images.

        Args:
            face_images: List of face crops as RGB numpy arrays
            face_metadata: List of metadata dicts (path, face_index, etc.)

        Returns:
            List of embedding vectors (512-dim normalized)
        """
        pass

    @abstractmethod
    def extract_single(
        self,
        face_image: np.ndarray,
        face_metadata: Dict[str, Any]
    ) -> np.ndarray:
        """Extract embedding for a single face."""
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimension of output embeddings."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name identifier for caching."""
        pass
