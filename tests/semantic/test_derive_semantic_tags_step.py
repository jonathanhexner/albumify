"""spec-108 P0 — derive_semantic_tags step reads cached ram_tags → semantic_tags."""
from __future__ import annotations

from pathlib import Path

from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.steps.derive_semantic_tags import DeriveSemanticTagsStep


class ut_DeriveSemanticTagsStep:
    def test_populates_semantic_tags_from_ram_tags(self):
        ctx = PipelineContext(source_directory=Path("."))
        ctx.ram_tags = {
            "a.jpg": ["mountain", "lake", "tree", "walk"],
            "b.jpg": ["person", "child", "family", "smile"],
        }
        DeriveSemanticTagsStep().process(ctx, {})
        assert set(ctx.semantic_tags) == {"a.jpg", "b.jpg"}
        assert ctx.semantic_tags["a.jpg"]["primary_facet"] == "nature"
        assert "walk" not in ctx.semantic_tags["a.jpg"]["tags"]        # verb dropped
        assert ctx.semantic_tags["b.jpg"]["primary_facet"] == "people"

    def test_empty_ram_tags_is_noop(self):
        ctx = PipelineContext(source_directory=Path("."))
        DeriveSemanticTagsStep().process(ctx, {})
        assert ctx.semantic_tags == {}
