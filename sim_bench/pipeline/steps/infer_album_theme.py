"""spec-108 P1 step: infer one album-level theme from the per-image semantic tags.

Tag-based baseline (pure, no model) — reuses `context.semantic_tags` from `derive_semantic_tags`.
A VLM refinement (MiniCPM-V, see `sim_bench/vlm/`) can later overwrite `album_theme` with
`source="vlm"` when the worker env is available; downstream reads the same `AlbumTheme` shape either way.
"""
from dataclasses import asdict

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.semantic.theme import infer_theme


@register_step
class InferAlbumThemeStep(BaseStep):
    """Infer the album type + motifs from semantic tags (tag-based baseline, no model)."""

    def __init__(self):
        self._metadata = StepMetadata(
            name="infer_album_theme",
            display_name="Infer album theme",
            description="Aggregate per-image semantic facets into one album theme (type + motifs). "
                        "Tag-based baseline; VLM refinement optional (spec-108).",
            category="analysis",
            requires={"semantic_tags"},
            produces={"album_theme"},
            depends_on=["derive_semantic_tags"],
            config_schema={"type": "object", "properties": {}},
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        from sim_bench.pipeline.steps.configs._validate import validate_step_config
        validate_step_config(self._metadata.name, config)
        theme = infer_theme(context.semantic_tags or {})
        context.album_theme = asdict(theme)
        context.report_progress(self._metadata.name, 1.0,
                                f"Album theme: {theme.album_type}")

    def validate(self, context: PipelineContext) -> list[str]:
        return []
