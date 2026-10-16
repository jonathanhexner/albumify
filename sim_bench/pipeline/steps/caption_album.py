"""spec-108 P2 step: one album caption, synthesized from the keeper captions + theme.

A single text-only VLM call (``generate``) — no extra image passes. Writes the headline into
``context.album_theme["caption"]`` and sets ``source="vlm"`` (the theme was tag-based; the caption is
VLM-refined). Gated by ``enabled`` (default False).
"""
import logging
from typing import Optional

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.semantic.captions import build_album_prompt, clean_caption

logger = logging.getLogger(__name__)


@register_step
class CaptionAlbumStep(BaseStep):
    """One VLM-synthesized album headline from keeper captions + theme (opt-in via enabled)."""

    def __init__(self):
        self._metadata = StepMetadata(
            name="caption_album",
            display_name="Caption album (VLM)",
            description="Synthesize one album headline from the keeper captions + tag-based theme "
                        "via a single text-only VLM call. Opt-in (enabled=false by default).",
            category="semantic",
            requires={"album_theme"},
            produces={"album_theme"},
            depends_on=["caption_keepers", "infer_album_theme"],
            config_schema={"type": "object", "properties": {
                "enabled": {"type": "boolean", "default": False},
                "provider": {"type": "string", "default": "subprocess"},
                "device": {"type": "string", "default": "cpu"},
            }},
        )
        self._provider = None

    def _get_provider(self, config: dict):
        if self._provider is None:
            from sim_bench.vlm.subprocess_provider import SubprocessVLMProvider
            self._provider = SubprocessVLMProvider(
                stub=(config.get("provider") == "stub"), device=config.get("device", "cpu"))
        return self._provider

    def process(self, context: PipelineContext, config: dict) -> None:
        from sim_bench.pipeline.steps.configs._validate import validate_step_config
        validate_step_config(self._metadata.name, config)
        if not config.get("enabled", False):
            context.report_progress(self._metadata.name, 1.0, "disabled — skipped")
            return
        theme = context.album_theme or {}
        keeper_caps = [context.image_captions[k] for k in (context.image_captions or {})]
        prompt = build_album_prompt(theme.get("album_type", "a trip"),
                                    theme.get("motifs", []), keeper_caps)
        caption = clean_caption(self._get_provider(config).generate(prompt), max_chars=160)
        context.album_theme = {**theme, "caption": caption, "source": "vlm"}
        context.report_progress(self._metadata.name, 1.0, f"Album caption: {caption[:60]}")

    def release(self) -> None:
        if self._provider is not None and hasattr(self._provider, "close"):
            self._provider.close()
        self._provider = None

    def validate(self, context: PipelineContext) -> list[str]:
        return []
