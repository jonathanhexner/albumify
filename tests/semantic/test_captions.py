"""spec-108 P2 — caption helpers (pure) + caption steps (fake provider, no model)."""
from __future__ import annotations

from pathlib import Path

from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.steps.caption_keepers import CaptionKeepersStep
from sim_bench.pipeline.steps.caption_album import CaptionAlbumStep
from sim_bench.semantic.captions import clean_caption, build_album_prompt


class _FakeProvider:
    """Implements the VLMProvider surface the steps use — no model."""
    def __init__(self):
        self.caption_calls = []
        self.generate_calls = []

    def caption(self, image_path: str, prompt: str) -> str:
        self.caption_calls.append((image_path, prompt))
        return f"a scene at {Path(image_path).stem}"

    def caption_batch(self, image_paths, prompt):
        return [self.caption(p, prompt) for p in image_paths]

    def generate(self, prompt: str) -> str:
        self.generate_calls.append(prompt)
        return "A sunlit week among Slovakia's lakes and old towns."

    def close(self):
        pass


class ut_CleanCaption:
    def test_strips_boilerplate_lead_and_quotes(self):
        assert clean_caption('"This photo shows a lake."') == "A lake."

    def test_collapses_whitespace_and_caps(self):
        out = clean_caption("  a   rowboat  on\na lake ")
        assert out == "A rowboat on a lake"

    def test_empty(self):
        assert clean_caption("") == ""

    def test_length_cap(self):
        out = clean_caption("word " * 100, max_chars=40)
        assert len(out) <= 41 and out.endswith("…")


class ut_BuildAlbumPrompt:
    def test_includes_type_motifs_and_captions(self):
        p = build_album_prompt("family trip", ["lake", "castle"], ["kids by a creek", "a stone bridge"])
        assert "family trip" in p and "lake" in p
        assert "kids by a creek" in p and "stone bridge" in p


class ut_CaptionKeepersStep:
    def test_disabled_is_noop(self):
        ctx = PipelineContext(source_directory=Path("."))
        ctx.selected_images = ["a.jpg", "b.jpg"]
        CaptionKeepersStep().process(ctx, {"enabled": False})
        assert ctx.image_captions == {}

    def test_enabled_captions_each_keeper(self):
        ctx = PipelineContext(source_directory=Path("."))
        ctx.selected_images = ["x/one.jpg", "x/two.jpg"]
        step = CaptionKeepersStep()
        step._provider = _FakeProvider()
        step.process(ctx, {"enabled": True})
        assert set(ctx.image_captions) == {"x/one.jpg", "x/two.jpg"}
        assert ctx.image_captions["x/one.jpg"].startswith("A scene at one")  # cleaned (capitalized)


class ut_CaptionAlbumStep:
    def test_enabled_synthesizes_album_caption(self):
        ctx = PipelineContext(source_directory=Path("."))
        ctx.album_theme = {"album_type": "family trip — towns & nature", "motifs": ["lake", "castle"]}
        ctx.image_captions = {"a.jpg": "kids by a creek", "b.jpg": "a stone bridge at dusk"}
        step = CaptionAlbumStep()
        fake = _FakeProvider()
        step._provider = fake
        step.process(ctx, {"enabled": True})
        assert ctx.album_theme["caption"].startswith("A sunlit week")
        assert ctx.album_theme["source"] == "vlm"
        assert len(fake.generate_calls) == 1          # ONE text-only call, not per-image

    def test_disabled_is_noop(self):
        ctx = PipelineContext(source_directory=Path("."))
        ctx.album_theme = {"album_type": "x"}
        CaptionAlbumStep().process(ctx, {"enabled": False})
        assert "caption" not in ctx.album_theme
