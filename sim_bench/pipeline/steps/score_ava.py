"""Score AVA step - aesthetic quality assessment."""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.serializers import Serializers

logger = logging.getLogger(__name__)


@register_step
class ScoreAVAStep(BaseStep):
    """Score images using AVA ResNet for aesthetic quality (1-10 scale)."""

    def __init__(self):
        self._metadata = StepMetadata(
            name="score_ava",
            display_name="Score Aesthetic Quality",
            description="Assess aesthetic quality using AVA ResNet model (requires trained checkpoint).",
            category="analysis",
            requires={"image_paths"},
            produces={"ava_scores"},
            depends_on=["discover_images"],
            config_schema={
                "type": "object",
                "properties": {
                    "checkpoint_path": {
                        "type": "string",
                        "description": "Path to trained AVA model checkpoint (best_model.pt)"
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
        self._ava_model = None
        self._checkpoint_path = None

    def _get_model(self, checkpoint_path: str, device: str = "cpu"):
        """Lazy load AVA model only when needed."""
        checkpoint_path_obj = Path(checkpoint_path)
        
        if not checkpoint_path_obj.exists():
            raise FileNotFoundError(f"AVA checkpoint not found: {checkpoint_path}")
        
        if self._ava_model is None or self._checkpoint_path != checkpoint_path:
            from sim_bench.image_quality_models.ava_model_wrapper import AVAQualityModel
            logger.info(f"Loading AVA model from {checkpoint_path}")
            self._ava_model = AVAQualityModel(checkpoint_path_obj, device=device)
            self._checkpoint_path = checkpoint_path
        
        return self._ava_model

    def _get_cache_config(
        self,
        context: PipelineContext,
        config: dict
    ) -> Optional[Dict[str, Any]]:
        """Get cache configuration for AVA scoring."""
        checkpoint_path = config.get("checkpoint_path")
        if not checkpoint_path:
            raise ValueError("checkpoint_path required in config for score_ava step")
        
        image_paths = [str(p) for p in context.image_paths]
        if not image_paths:
            return None
        
        return {
            "items": image_paths,
            "feature_type": "ava_score",
            "model_name": "ava_resnet",
            "metadata": {
                "model_version": None  # Could extract from checkpoint if needed
            }
        }
    
    def _process_uncached(
        self,
        items: List[str],
        context: PipelineContext,
        config: dict
    ) -> Dict[str, float]:
        """Process uncached items - compute AVA scores."""
        checkpoint_path = config.get("checkpoint_path")
        device = config.get("device", "cpu")
        
        model = self._get_model(checkpoint_path, device)
        results = {}
        
        for i, path_str in enumerate(items):
            score = model.score_image(Path(path_str))
            results[path_str] = score
            
            # Report progress for this batch
            progress = (i + 1) / len(items)
            context.report_progress(
                "score_ava", progress,
                f"Scoring {i + 1}/{len(items)}"
            )
        
        return results
    
    def _serialize_for_cache(self, result: float, item: str) -> bytes:
        """Serialize float score to JSON bytes."""
        return Serializers.json_serialize(result)
    
    def _deserialize_from_cache(self, data: bytes, item: str) -> float:
        """Deserialize JSON bytes to float score."""
        return float(Serializers.json_deserialize(data))
    
    def _store_results(
        self,
        context: PipelineContext,
        results: Dict[str, float],
        config: dict
    ) -> None:
        """Store AVA scores in context, normalized to 0-1 scale."""
        # AVA model returns 1-10 scale, normalize to 0-1 for consistency
        context.ava_scores = {path: score / 10.0 for path, score in results.items()}
