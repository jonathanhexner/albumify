"""spec-107 Tier-2 triage step: set aside junk metadata can't see (RAM++ content tags).

Photographed documents/posters/receipts/maps + re-shared screenshots. Runs after triage_metadata
(so RAM++ is spent only on images metadata didn't already remove) and before the face stack.
Caches RAM++ tags (feature_type ``ram_tags``) — reused by the spec-108 semantic layer. Thin wrapper
over ``sim_bench.triage.content`` (pure ``calc`` + lazy ``RamTagger``).
"""
import logging
from typing import Any, Dict, List, Optional

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.serializers import Serializers
from sim_bench.pipeline.steps._triage_apply import partition_and_record
from sim_bench.triage import content as content_triage

logger = logging.getLogger(__name__)


@register_step
class TriageContentStep(BaseStep):
    """Set aside documents/posters/re-shared screenshots via RAM++ content tags (cached)."""

    def __init__(self):
        self._metadata = StepMetadata(
            name="triage_content",
            display_name="Triage — content (RAM++)",
            description="Set aside photographed docs/posters/maps + re-shared screenshots using "
                        "RAM++ content tags. Cached; tags reused by the semantic layer.",
            category="triage",
            requires={"image_paths"},
            produces={"set_aside", "ram_tags", "active_images"},
            depends_on=["triage_metadata"],
            config_schema={"type": "object", "properties": {"device": {"type": "string", "default": "cpu"}}},
        )
        self._tagger = None

    def _get_cache_config(self, context: PipelineContext, config: dict) -> Optional[Dict[str, Any]]:
        paths = [str(p) for p in context.image_paths]
        if not paths:
            return None
        return {"items": paths, "feature_type": "ram_tags", "model_name": "ram_plus_swin_l_14m",
                "metadata": {"model_version": "ram_plus_v1"}}

    def _process_uncached(self, items: List[str], context: PipelineContext,
                          config: dict) -> Dict[str, list]:
        if self._tagger is None:
            self._tagger = content_triage.RamTagger(device=config.get("device", "cpu"))
        out: Dict[str, list] = {}
        for i, path in enumerate(items):
            try:
                out[path] = self._tagger.tag(path)
            except Exception as e:  # unreadable -> no tags, never raise
                logger.warning("triage_content RAM++ failed for %s: %s", path, e)
                out[path] = []
            context.report_progress(self._metadata.name, (i + 1) / len(items),
                                    f"RAM++ {i + 1}/{len(items)}")
        return out

    def _serialize_for_cache(self, result: list, item: str) -> bytes:
        return Serializers.json_serialize(result)

    def _deserialize_from_cache(self, data: bytes, item: str) -> list:
        return Serializers.json_deserialize(data)

    def _store_results(self, context: PipelineContext, results: Dict[str, list], config: dict) -> None:
        context.ram_tags.update(results)
        decisions = {path: content_triage.calc(tags) for path, tags in results.items()}
        n = partition_and_record(context, decisions, tier="content")
        context.report_progress(self._metadata.name, 1.0,
                                f"Set aside {n} (content); {len(context.image_paths)} remain")

    def release(self) -> None:
        self._release_models("_tagger")

    def validate(self, context: PipelineContext) -> list[str]:
        return []
