"""Score IQA step - technical quality assessment."""

from typing import Dict, List, Any, Optional

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.serializers import Serializers


@register_step
class ScoreIQAStep(BaseStep):
    """Score images using rule-based IQA (Image Quality Assessment)."""

    def __init__(self):
        self._metadata = StepMetadata(
            name="score_iqa",
            display_name="Score Technical Quality",
            description="Assess technical image quality (sharpness, exposure, contrast) using rule-based IQA.",
            category="analysis",
            requires={"image_paths"},
            produces={"iqa_scores", "sharpness_scores"},
            depends_on=["discover_images"],
            config_schema={
                "type": "object",
                "properties": {}
            }
        )
        self._iqa_model = None

    def _get_model(self):
        if self._iqa_model is None:
            from sim_bench.quality_assessment.rule_based import RuleBasedQuality
            self._iqa_model = RuleBasedQuality()
        return self._iqa_model

    def _get_cache_config(
        self,
        context: PipelineContext,
        config: dict
    ) -> Optional[Dict[str, Any]]:
        """Get cache configuration for IQA scoring."""
        image_paths = [str(p) for p in context.image_paths]
        if not image_paths:
            return None
        
        return {
            "items": image_paths,
            "feature_type": "iqa_scores",  # Cache both scores together
            "model_name": "rule_based",
            "metadata": {}
        }
    
    def _process_uncached(
        self,
        items: List[str],
        context: PipelineContext,
        config: dict
    ) -> Dict[str, Dict[str, float]]:
        """Process uncached items - compute IQA and sharpness scores."""
        model = self._get_model()
        results = {}
        
        for i, path_str in enumerate(items):
            scores = model.get_detailed_scores(path_str)
            results[path_str] = {
                "iqa": scores["overall"],
                "sharpness": scores["sharpness_normalized"]
            }
            
            progress = (i + 1) / len(items)
            context.report_progress(
                "score_iqa", progress,
                f"Scoring {i + 1}/{len(items)}"
            )
        
        return results
    
    def _serialize_for_cache(self, result: Dict[str, float], item: str) -> bytes:
        """Serialize scores dict to JSON bytes."""
        return Serializers.json_serialize(result)
    
    def _deserialize_from_cache(self, data: bytes, item: str) -> Dict[str, float]:
        """Deserialize JSON bytes to scores dict."""
        return Serializers.json_deserialize(data)
    
    def _store_results(
        self,
        context: PipelineContext,
        results: Dict[str, Dict[str, float]],
        config: dict
    ) -> None:
        """Store IQA and sharpness scores in context."""
        # Split combined results into separate dicts
        context.iqa_scores = {
            path: scores["iqa"]
            for path, scores in results.items()
        }
        context.sharpness_scores = {
            path: scores["sharpness"]
            for path, scores in results.items()
        }
