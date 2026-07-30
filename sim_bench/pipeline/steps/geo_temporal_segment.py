"""Group images into segments via the best-scoring axis (spec-022).

Thin translator: reads ``context.geo_metadata``, runs the home anchor + the
multi-axis competition (``geo_cluster.selector``), writes the winning segments
back to context. All domain logic lives in ``geo_cluster/``.
"""

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step

from geo_cluster.axes.base import AxisInputs
from geo_cluster.home import HomeAnchor
from geo_cluster.selector import SegmentationSelector


@register_step
class GeoTemporalSegmentStep(BaseStep):
    """Segment the album by the highest-scoring axis, or leave it FLAT."""

    def __init__(self):
        self._metadata = StepMetadata(
            name="geo_temporal_segment",
            display_name="Geo-Temporal Segment",
            description="Group images by the best-scoring axis (geo/time/identity) or decline.",
            category="clustering",
            requires={"geo_metadata"},
            produces={"geo_segments"},
            depends_on=["extract_geo_metadata"],
            config_schema={
                "type": "object",
                "properties": {
                    "geo_radius_km": {"type": "number", "default": 30.0},
                    "time_gap_hours": {"type": "number", "default": 8.0},
                    "floor": {"type": "number", "default": 0.45},
                    "enabled_axes": {"type": "array", "items": {"type": "string"}},
                },
            },
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        meta = context.geo_metadata or {}
        radius = config.get("geo_radius_km", 30.0)
        home = HomeAnchor(radius_km=radius).calc(meta)

        inputs = AxisInputs(
            metadata=meta,
            home=home,
            people_clusters=getattr(context, "people_clusters", None) or None,
            config={"geo_radius_km": radius, "time_gap_hours": config.get("time_gap_hours", 8.0)},
        )
        outcome = SegmentationSelector(
            floor=config.get("floor", 0.45),
            enabled=config.get("enabled_axes"),
        ).calc(inputs)

        context.geo_home = home
        context.geo_segments = outcome.winner.segments if outcome.winner else []
        verdict = "FLAT" if outcome.flat else outcome.winning_axis
        context.report_progress(
            "geo_temporal_segment", 1.0,
            f"{verdict}: {len(context.geo_segments)} segments",
        )
