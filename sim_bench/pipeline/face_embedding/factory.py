"""Factory for creating face embedding extractors."""

import logging
from typing import Dict, Any

from sim_bench.pipeline.face_embedding.base import BaseFaceEmbeddingExtractor

logger = logging.getLogger(__name__)


class FaceEmbeddingExtractorFactory:
    """Factory for creating face embedding extractors based on config."""

    _extractors = {}

    @classmethod
    def _ensure_registered(cls):
        """Ensure default extractors are registered."""
        if not cls._extractors:
            from sim_bench.pipeline.face_embedding.custom_arcface import CustomArcFaceExtractor
            from sim_bench.pipeline.face_embedding.insightface_native import InsightFaceNativeExtractor
            cls._extractors = {
                "custom": CustomArcFaceExtractor,
                "insightface": InsightFaceNativeExtractor,
            }

    @classmethod
    def create(cls, config: Dict[str, Any]) -> BaseFaceEmbeddingExtractor:
        """Create extractor based on config.

        Config keys:
            backend: "custom" | "insightface" (default: "insightface")
            checkpoint_path: str (required for custom backend)
            device: str ("cpu" | "cuda" | "mps", default: "cpu")
            model_name: str (for insightface backend, default "buffalo_l")

        Returns:
            BaseFaceEmbeddingExtractor instance

        Raises:
            ValueError: If unknown backend specified
        """
        cls._ensure_registered()

        backend = config.get("backend", "insightface")

        if backend not in cls._extractors:
            available = ", ".join(cls._extractors.keys())
            raise ValueError(f"Unknown face embedding backend: {backend}. Available: {available}")

        extractor_class = cls._extractors[backend]
        logger.info(f"Creating face embedding extractor: {backend}")
        return extractor_class(config)

    @classmethod
    def register(cls, name: str, extractor_class: type):
        """Register a new extractor type.

        Args:
            name: Backend name for config
            extractor_class: Class implementing BaseFaceEmbeddingExtractor
        """
        cls._ensure_registered()
        cls._extractors[name] = extractor_class
        logger.info(f"Registered face embedding extractor: {name}")

    @classmethod
    def available_backends(cls) -> list:
        """List available backend names."""
        cls._ensure_registered()
        return list(cls._extractors.keys())
