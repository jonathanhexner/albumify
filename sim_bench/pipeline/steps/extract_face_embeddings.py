"""Extract Face Embeddings step - ArcFace embeddings for face clustering."""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.serializers import Serializers

logger = logging.getLogger(__name__)


@register_step
class ExtractFaceEmbeddingsStep(BaseStep):
    """Extract face embeddings using trained ArcFace model."""

    def __init__(self):
        self._metadata = StepMetadata(
            name="extract_face_embeddings",
            display_name="Extract Face Embeddings",
            description="Extract ArcFace embeddings for face recognition and clustering.",
            category="people",
            requires=set(),  # faces is optional - step skips if none
            produces={"face_embeddings"},
            depends_on=["detect_faces"],
            config_schema={
                "type": "object",
                "properties": {
                    "checkpoint_path": {
                        "type": "string",
                        "description": "Path to trained ArcFace model checkpoint"
                    },
                    "device": {
                        "type": "string",
                        "enum": ["cpu", "cuda", "mps"],
                        "default": "cpu",
                        "description": "Device to run model on"
                    }
                },
                "required": ["checkpoint_path"]
            }
        )
        self._embedding_service = None

    def _get_service(self, checkpoint_path: str, device: str = "cpu"):
        """Lazy load face embedding service."""
        if self._embedding_service is None:
            from sim_bench.album.services.face_embedding_service import FaceEmbeddingService
            logger.info(f"Loading ArcFace model from {checkpoint_path}")
            config = {
                'face': {'checkpoint_path': checkpoint_path},
                'device': device
            }
            self._embedding_service = FaceEmbeddingService(config)
        return self._embedding_service

    def _generate_cache_key(self, face) -> str:
        """Generate unique cache key for a face."""
        return f"{face.original_path}:face_{face.face_index}"
    
    def _get_all_faces(self, context: PipelineContext) -> List:
        """Get all faces from context (either all_faces or flattened faces dict)."""
        if context.all_faces:
            return context.all_faces
        if context.faces:
            all_faces = []
            for faces_list in context.faces.values():
                all_faces.extend(faces_list)
            return all_faces
        return []

    def _get_cache_config(
        self,
        context: PipelineContext,
        config: dict
    ) -> Optional[Dict[str, Any]]:
        """Get cache configuration for face embedding extraction."""
        # Check for faces first - if no faces, skip gracefully
        all_faces = self._get_all_faces(context)
        if not all_faces:
            return None

        checkpoint_path = config.get("checkpoint_path")
        if not checkpoint_path:
            raise ValueError("checkpoint_path required in config for extract_face_embeddings step")

        # Use cache key strings as items
        cache_keys = [self._generate_cache_key(f) for f in all_faces]

        return {
            "items": cache_keys,
            "feature_type": "face_embedding",
            "model_name": "arcface",
            "metadata": {}
        }
    
    def _process_uncached(
        self,
        items: List[str],
        context: PipelineContext,
        config: dict
    ) -> Dict[str, np.ndarray]:
        """Process uncached items - extract face embeddings."""
        checkpoint_path = config.get("checkpoint_path")
        device = config.get("device", "cpu")

        all_faces = self._get_all_faces(context)

        # Map cache keys back to faces
        key_to_face = {
            self._generate_cache_key(f): f
            for f in all_faces
        }
        
        uncached_faces = [key_to_face[key] for key in items if key in key_to_face]
        
        if not uncached_faces:
            return {}
        
        service = self._get_service(checkpoint_path, device)
        
        # Extract in batch
        face_images = [face.image for face in uncached_faces]
        embeddings_array = service.extract_embeddings_batch(
            face_images,
            batch_size=32,
            show_progress=False
        )
        
        results = {}
        for i, (face, embedding) in enumerate(zip(uncached_faces, embeddings_array)):
            face.embedding = embedding
            key = self._generate_cache_key(face)
            results[key] = embedding
            
            progress = (i + 1) / len(uncached_faces)
            context.report_progress(
                "extract_face_embeddings", progress,
                f"Embedding {i + 1}/{len(uncached_faces)}"
            )
        
        return results
    
    def _serialize_for_cache(self, result: np.ndarray, item: str) -> bytes:
        """Serialize numpy array to bytes."""
        return Serializers.numpy_serialize(result)
    
    def _deserialize_from_cache(self, data: bytes, item: str) -> np.ndarray:
        """Deserialize bytes to numpy array."""
        return Serializers.numpy_deserialize(data)
    
    def _store_results(
        self,
        context: PipelineContext,
        results: Dict[str, np.ndarray],
        config: dict
    ) -> None:
        """Store face embeddings in context and update face objects."""
        all_faces = self._get_all_faces(context)

        # Update face objects with embeddings
        key_to_face = {
            self._generate_cache_key(f): f
            for f in all_faces
        }
        
        for key, embedding in results.items():
            if key in key_to_face:
                key_to_face[key].embedding = embedding
        
        # Store in context dict
        context.face_embeddings = {
            key: embedding
            for key, embedding in results.items()
        }
