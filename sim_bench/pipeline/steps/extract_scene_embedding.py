"""Extract scene embedding step - generate embeddings using DINOv2/OpenCLIP."""

import numpy as np
from typing import Dict, List, Any, Optional

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.serializers import Serializers


@register_step
class ExtractSceneEmbeddingStep(BaseStep):
    """Extract scene embeddings using DINOv2 or OpenCLIP."""

    def __init__(self):
        self._metadata = StepMetadata(
            name="extract_scene_embedding",
            display_name="Extract Scene Embeddings",
            description="Generate dense feature embeddings for scene similarity using DINOv2 or OpenCLIP.",
            category="embedding",
            requires={"active_images"},
            produces={"scene_embeddings"},
            depends_on=["filter_quality"],
            config_schema={
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": ["dinov2", "openclip", "resnet50"],
                        "default": "dinov2",
                        "description": "Feature extraction method"
                    },
                    "batch_size": {
                        "type": "integer",
                        "default": 32,
                        "minimum": 1,
                        "description": "Batch size for processing"
                    }
                }
            }
        )
        self._extractor = None
        self._current_method = None

    def _get_extractor(self, method: str, config: dict):
        if self._extractor is None or self._current_method != method:
            from sim_bench.feature_extraction.base import load_method
            extractor_config = {
                "method": method,
                "batch_size": config.get("batch_size", 32)
            }
            self._extractor = load_method(method, extractor_config)
            self._current_method = method
        return self._extractor

    def _get_cache_config(
        self,
        context: PipelineContext,
        config: dict
    ) -> Optional[Dict[str, Any]]:
        """Get cache configuration for scene embedding extraction."""
        method = config.get("method", "dinov2")
        image_paths = list(context.active_images)
        
        if not image_paths:
            return None
        
        return {
            "items": image_paths,
            "feature_type": "scene_embedding",
            "model_name": method,
            "metadata": {}
        }
    
    def _process_uncached(
        self,
        items: List[str],
        context: PipelineContext,
        config: dict
    ) -> Dict[str, np.ndarray]:
        """Process uncached items - extract embeddings."""
        method = config.get("method", "dinov2")
        
        context.report_progress(
            "extract_scene_embedding", 0.1,
            f"Extracting {method} features for {len(items)} images"
        )
        
        extractor = self._get_extractor(method, config)
        embeddings_list = extractor.extract_features(items)
        
        # Convert list to dict
        return {path: embeddings_list[i] for i, path in enumerate(items)}
    
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
        """Store scene embeddings in context."""
        context.scene_embeddings = results
