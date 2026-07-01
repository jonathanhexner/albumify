"""GeoCLIP coordinate geolocation step (spec-022, stage C; spec-094 storage refactor).

Predicts top-k (lat, lon) per image, reverse-geocoded to a place. Thin translator
over ``geo_cluster.geoclip_locator`` that persists the full top-k through
``universal_cache`` via the BaseStep cache hooks — re-runs hit the DB, no recompute.
"""

from typing import Any, Dict, List, Optional

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.serializers import Serializers

from geo_cluster.geoclip_locator import GeoCLIPLocator, GeoClipInputs

_MODEL = "geoclip"


@register_step
class InferGeoCoordsStep(BaseStep):
    """GeoCLIP top-k (lat, lon) predictions -> context.geo_coord_predictions."""

    def __init__(self):
        self._metadata = StepMetadata(
            name="infer_geo_coords",
            display_name="GeoCLIP Geolocation",
            description="Predict (lat, lon) per image directly, then reverse-geocode.",
            category="analysis",
            requires={"image_paths"},
            produces={"geo_coord_predictions"},
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
            "feature_type": "geo_geoclip",
            "model_name": _MODEL,
            "metadata": {"model_version": f"geoclip-v1-k{top_k}"},
        }

    def _process_uncached(self, items: List[str], context: PipelineContext,
                          config: dict) -> Dict[str, list]:
        locator = GeoCLIPLocator(top_k=config.get("top_k", 3),
                                 device=config.get("device", "cpu"))
        return locator.calc(GeoClipInputs(image_paths=items)).predictions

    def _serialize_for_cache(self, result: list, item: str) -> bytes:
        return Serializers.json_serialize(result)

    def _deserialize_from_cache(self, data: bytes, item: str) -> list:
        return list(Serializers.json_deserialize(data))

    def _store_results(self, context: PipelineContext, results: Dict[str, list],
                       config: dict) -> None:
        context.geo_coord_predictions = results
