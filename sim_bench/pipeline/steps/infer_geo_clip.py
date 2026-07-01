"""StreetCLIP city geolocation step (spec-022, stage C; spec-094 storage refactor).

Guesses top-k cities per image for the GPS-less case. Thin translator over
``geo_cluster.streetclip`` that persists the full top-k through ``universal_cache``
via the BaseStep cache hooks — re-runs hit the DB, no recompute.
"""

from typing import Any, Dict, List, Optional

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.serializers import Serializers

from geo_cluster.streetclip import StreetClipInputs, StreetCLIPLocator

_MODEL = "geolocal/StreetCLIP"


@register_step
class InferGeoClipStep(BaseStep):
    """StreetCLIP top-k city predictions -> context.geo_clip_predictions."""

    def __init__(self):
        self._metadata = StepMetadata(
            name="infer_geo_clip",
            display_name="StreetCLIP Geolocation",
            description="Zero-shot top-k city guesses per image (GPS fallback).",
            category="analysis",
            requires={"image_paths"},
            produces={"geo_clip_predictions"},
            depends_on=["extract_geo_metadata"],
            config_schema={
                "type": "object",
                "properties": {
                    "device": {"type": "string", "default": "cpu"},
                    "top_k": {"type": "integer", "default": 3},
                    "only_missing_gps": {"type": "boolean", "default": False},
                },
            },
        )

    def _items(self, context: PipelineContext, config: dict) -> List[str]:
        paths = [str(p) for p in context.image_paths]
        if config.get("only_missing_gps") and context.geo_metadata:
            paths = [p for p in paths
                     if not (context.geo_metadata.get(p) and context.geo_metadata[p].has_geo)]
        return paths

    def _get_cache_config(self, context: PipelineContext, config: dict) -> Optional[Dict[str, Any]]:
        items = self._items(context, config)
        if not items:
            return None
        top_k = config.get("top_k", 3)
        return {
            "items": items,
            "feature_type": "geo_streetclip",
            "model_name": _MODEL,
            # top_k is part of the output shape -> include it in the version so a
            # changed top_k invalidates stale rows (spec-079 lesson).
            "metadata": {"model_version": f"streetclip-v1-k{top_k}"},
        }

    def _process_uncached(self, items: List[str], context: PipelineContext,
                          config: dict) -> Dict[str, list]:
        locator = StreetCLIPLocator(device=config.get("device", "cpu"),
                                    top_k=config.get("top_k", 3))
        return locator.calc(StreetClipInputs(image_paths=items)).predictions

    def _serialize_for_cache(self, result: list, item: str) -> bytes:
        return Serializers.json_serialize(result)

    def _deserialize_from_cache(self, data: bytes, item: str) -> list:
        return list(Serializers.json_deserialize(data))

    def _store_results(self, context: PipelineContext, results: Dict[str, list],
                       config: dict) -> None:
        context.geo_clip_predictions = results
