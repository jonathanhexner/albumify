"""Extract EXIF geo/time metadata per image (spec-022 Slice 1; spec-094 storage refactor).

Thin translator over ``geo_cluster.exif_reader`` that persists each image's
``GeoMetadata`` through ``universal_cache`` via the BaseStep cache hooks — re-runs
hit the DB, no recompute.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.serializers import Serializers

from geo_cluster.exif_reader import ExifInputs, GeoMetadataExtractor
from geo_cluster.types import GeoMetadata


@register_step
class ExtractGeoMetadataStep(BaseStep):
    """Read EXIF GPS + DateTimeOriginal per image into ``context.geo_metadata``."""

    def __init__(self):
        self._metadata = StepMetadata(
            name="extract_geo_metadata",
            display_name="Extract Geo Metadata",
            description="Read EXIF GPS coordinates and capture time per image (spec-022).",
            category="analysis",
            requires={"image_paths"},
            produces={"geo_metadata"},
            depends_on=["discover_images"],
            config_schema={
                "type": "object",
                "properties": {
                    "min_year": {
                        "type": "integer",
                        "default": 1990,
                        "description": "Reject EXIF capture dates before this year",
                    }
                },
            },
        )

    def _get_cache_config(self, context: PipelineContext, config: dict) -> Optional[Dict[str, Any]]:
        image_paths = [str(p) for p in context.image_paths]
        if not image_paths:
            return None
        min_year = config.get("min_year", 1990)
        return {
            "items": image_paths,
            "feature_type": "geo_exif",
            "model_name": "exif",
            "metadata": {"model_version": f"exif-v1-min{min_year}"},
        }

    def _process_uncached(self, items: List[str], context: PipelineContext,
                          config: dict) -> Dict[str, GeoMetadata]:
        extractor = GeoMetadataExtractor(min_year=config.get("min_year", 1990))
        return extractor.calc(ExifInputs(image_paths=items)).metadata

    def _serialize_for_cache(self, result: GeoMetadata, item: str) -> bytes:
        return Serializers.json_serialize({
            "timestamp": result.timestamp.isoformat() if result.timestamp else None,
            "lat": result.lat,
            "lon": result.lon,
        })

    def _deserialize_from_cache(self, data: bytes, item: str) -> GeoMetadata:
        d = Serializers.json_deserialize(data)
        ts = datetime.fromisoformat(d["timestamp"]) if d.get("timestamp") else None
        return GeoMetadata(image_path=item, timestamp=ts, lat=d.get("lat"), lon=d.get("lon"))

    def _store_results(self, context: PipelineContext, results: Dict[str, GeoMetadata],
                       config: dict) -> None:
        context.geo_metadata = results
        n_geo = sum(1 for m in results.values() if m.has_geo)
        n_time = sum(1 for m in results.values() if m.has_time)
        context.report_progress(
            "extract_geo_metadata", 1.0,
            f"Geo: {n_geo}/{len(results)} with GPS, {n_time} with timestamp",
        )
