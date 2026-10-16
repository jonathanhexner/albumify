"""spec-108 P0 step: derive cleaned semantic tags from the cached RAM++ tags.

Reuses ``context.ram_tags`` (populated + cached by the spec-107 ``triage_content`` step) — no second
RAM++ pass. Pure transform; thin wrapper over ``sim_bench.semantic.tags.categorize``. Foundation for
thematic grouping, album-type inference, and search.
"""
from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.semantic.tags import categorize


@register_step
class DeriveSemanticTagsStep(BaseStep):
    """Clean + facet the RAM++ tags into per-image semantic labels (no model, reuses ram_tags)."""

    def __init__(self):
        self._metadata = StepMetadata(
            name="derive_semantic_tags",
            display_name="Derive semantic tags",
            description="Clean RAM++ content tags (drop verb/fragment noise) + bucket into coarse "
                        "facets. Reuses cached ram_tags from triage_content; no model.",
            category="analysis",
            requires={"ram_tags"},
            produces={"semantic_tags"},
            depends_on=["triage_content"],
            config_schema={"type": "object", "properties": {}},
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        from sim_bench.pipeline.steps.configs._validate import validate_step_config
        validate_step_config(self._metadata.name, config)
        for path, raw in (context.ram_tags or {}).items():
            st = categorize(raw)
            context.semantic_tags[path] = {
                "tags": st.tags, "facets": st.facets, "primary_facet": st.primary_facet,
            }
        context.report_progress(self._metadata.name, 1.0,
                                f"Derived semantic tags for {len(context.semantic_tags)} images")

    def validate(self, context: PipelineContext) -> list[str]:
        return []
