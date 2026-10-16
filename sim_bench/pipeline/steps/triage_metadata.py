"""spec-107 Tier-1 triage step: set aside screenshots/downloads from metadata (free, no model).

Runs right after discover_images so obvious junk is removed BEFORE the expensive scoring/face
stack is ever spent on it. Thin wrapper over ``sim_bench.triage.metadata.classify_file``.
"""
from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.steps._triage_apply import partition_and_record
from sim_bench.triage.metadata import classify_file


@register_step
class TriageMetadataStep(BaseStep):
    """Set aside screenshots/downloads using file metadata (filename, EXIF, format, dimensions)."""

    def __init__(self):
        self._metadata = StepMetadata(
            name="triage_metadata",
            display_name="Triage — metadata",
            description="Set aside screenshots/downloads via filename + camera-EXIF + format/dims. "
                        "Pure, no model; runs before the expensive pipeline.",
            category="triage",
            requires={"image_paths"},
            produces={"set_aside", "active_images"},
            depends_on=["discover_images"],
            config_schema={"type": "object", "properties": {}},
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        from sim_bench.pipeline.steps.configs._validate import validate_step_config
        validate_step_config(self._metadata.name, config)
        decisions = {str(p): classify_file(p) for p in context.image_paths}
        n = partition_and_record(context, decisions, tier="metadata")
        context.report_progress(self._metadata.name, 1.0,
                                f"Set aside {n} (metadata); {len(context.image_paths)} remain")

    def validate(self, context: PipelineContext) -> list[str]:
        return []
