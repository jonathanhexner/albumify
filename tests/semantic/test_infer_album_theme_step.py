"""spec-108 P1 — infer_album_theme step reads semantic_tags → album_theme."""
from __future__ import annotations

from pathlib import Path

from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.steps.infer_album_theme import InferAlbumThemeStep


class ut_InferAlbumThemeStep:
    def test_writes_album_theme(self):
        ctx = PipelineContext(source_directory=Path("."))
        ctx.semantic_tags = {
            "a": {"primary_facet": "people", "tags": ["family"], "facets": {"people": ["family"]}},
            "b": {"primary_facet": "nature", "tags": ["lake"], "facets": {"nature": ["lake"]}},
            "c": {"primary_facet": "urban", "tags": ["castle"], "facets": {"urban": ["castle"]}},
        }
        InferAlbumThemeStep().process(ctx, {})
        assert ctx.album_theme["album_type"] == "family trip — towns & nature"
        assert ctx.album_theme["source"] == "tags"
        assert ctx.album_theme["facet_distribution"] == {"people": 1, "nature": 1, "urban": 1}

    def test_empty_is_mixed(self):
        ctx = PipelineContext(source_directory=Path("."))
        InferAlbumThemeStep().process(ctx, {})
        assert ctx.album_theme["album_type"] == "mixed / general"
