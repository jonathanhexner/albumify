"""Discover images step - scans directory for image files."""

from pathlib import Path

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step


@register_step
class DiscoverImagesStep(BaseStep):
    """Scan source directory for image files."""

    def __init__(self):
        self._metadata = StepMetadata(
            name="discover_images",
            display_name="Discover Images",
            description="Scan source directory for image files (jpg, jpeg, png, heic, raw).",
            category="discovery",
            requires=set(),
            produces={"image_paths"},
            depends_on=[],
            config_schema={
                "type": "object",
                "properties": {
                    "extensions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [".jpg", ".jpeg", ".png", ".heic", ".raw"],
                        "description": "File extensions to include"
                    }
                }
            }
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        extensions = config.get("extensions", [".jpg", ".jpeg", ".png", ".heic", ".raw"])

        images = []
        for ext in extensions:
            images.extend(context.source_directory.rglob(f"*{ext}"))
            images.extend(context.source_directory.rglob(f"*{ext.upper()}"))

        images = sorted(set(images))
        context.image_paths = images
        context.active_images = {str(p) for p in images}

        context.report_progress("discover_images", 1.0, f"Found {len(images)} images")

    def validate(self, context: PipelineContext) -> list[str]:
        errors = []
        if context.source_directory is None:
            errors.append("source_directory is not set")
        elif not context.source_directory.exists():
            errors.append(f"source_directory does not exist: {context.source_directory}")
        return errors
