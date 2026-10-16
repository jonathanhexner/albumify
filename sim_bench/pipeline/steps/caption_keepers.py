"""spec-108 P2 step: per-keeper captions via the VLM (MiniCPM-V through the seam).

Runs AFTER selection — captions only the ~20 selected keepers, never the whole roll (cost discipline).
Gated by ``enabled`` (default False) so CI / model-less envs skip it entirely. Writes one vivid
sentence per keeper into ``context.image_captions``.
"""
import logging
from typing import Optional

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.semantic.captions import KEEPER_PROMPT, clean_caption

logger = logging.getLogger(__name__)


@register_step
class CaptionKeepersStep(BaseStep):
    """One VLM sentence per selected keeper → context.image_captions (opt-in via enabled)."""

    def __init__(self):
        self._metadata = StepMetadata(
            name="caption_keepers",
            display_name="Caption keepers (VLM)",
            description="One vivid sentence per selected image via MiniCPM-V (through the VLM seam). "
                        "Keepers only; opt-in (enabled=false by default).",
            category="semantic",
            requires={"selected_images"},
            produces={"image_captions"},
            depends_on=["select_best"],
            config_schema={"type": "object", "properties": {
                "enabled": {"type": "boolean", "default": False},
                "provider": {"type": "string", "default": "subprocess"},  # subprocess | stub
                "device": {"type": "string", "default": "cpu"},
                "prompt": {"type": "string"},
            }},
        )
        self._provider = None  # injectable for tests

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
        keepers = list(context.selected_images or [])
        if not keepers:
            return
        prompt = config.get("prompt") or KEEPER_PROMPT
        provider = self._get_provider(config)
        for i, path in enumerate(keepers):
            context.image_captions[str(path)] = clean_caption(provider.caption(str(path), prompt))
            context.report_progress(self._metadata.name, (i + 1) / len(keepers),
                                    f"Captioning keeper {i + 1}/{len(keepers)}")

    def release(self) -> None:
        if self._provider is not None and hasattr(self._provider, "close"):
            self._provider.close()
        self._provider = None

    def validate(self, context: PipelineContext) -> list[str]:
        return []
