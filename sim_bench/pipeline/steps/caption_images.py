"""BLIP image-captioning step (spec-022, stage E; spec-094 storage refactor).

One caption per image. Thin translator over ``geo_cluster.captioning`` that
persists each caption through ``universal_cache`` via the BaseStep cache hooks
(same idiom as ``score_ava.py``) — re-runs hit the DB, no recompute.
"""

from typing import Any, Dict, List, Optional

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.serializers import Serializers

from geo_cluster.captioning import BlipCaptioner, CaptionInputs

_MODEL = "Salesforce/blip-image-captioning-base"


@register_step
class CaptionImagesStep(BaseStep):
    """BLIP captions -> context.image_captions (cached in universal_cache)."""

    def __init__(self):
        self._metadata = StepMetadata(
            name="caption_images",
            display_name="BLIP Captioning",
            description="Generate a short natural-language caption per image.",
            category="analysis",
            requires={"image_paths"},
            produces={"image_captions"},
            depends_on=["discover_images"],
            config_schema={
                "type": "object",
                "properties": {
                    "device": {"type": "string", "default": "cpu"},
                    "max_new_tokens": {"type": "integer", "default": 30},
                },
            },
        )

    def _get_cache_config(self, context: PipelineContext, config: dict) -> Optional[Dict[str, Any]]:
        image_paths = [str(p) for p in context.image_paths]
        if not image_paths:
            return None
        return {
            "items": image_paths,
            "feature_type": "blip_caption",
            "model_name": _MODEL,
            "metadata": {"model_version": "blip-base-v1"},
        }

    def _process_uncached(self, items: List[str], context: PipelineContext,
                          config: dict) -> Dict[str, str]:
        captioner = BlipCaptioner(
            device=config.get("device", "cpu"),
            max_new_tokens=config.get("max_new_tokens", 30),
        )
        return captioner.calc(CaptionInputs(image_paths=items)).captions

    def _serialize_for_cache(self, result: str, item: str) -> bytes:
        return Serializers.json_serialize(result)

    def _deserialize_from_cache(self, data: bytes, item: str) -> str:
        return str(Serializers.json_deserialize(data))

    def _store_results(self, context: PipelineContext, results: Dict[str, str],
                       config: dict) -> None:
        context.image_captions = results
