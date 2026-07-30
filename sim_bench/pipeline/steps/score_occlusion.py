"""Score occlusion step (spec-097 Stage 1) — thin wrapper over OcclusionScorer.

Runs the spec-096 winning detector (CLIP global+tile-max probe, 0.86 scene
PR-AUC on adjudicated labels) per image; writes ``context.occlusion_scores``
(P(occluded)) and ``context.occlusion_tiles`` (9 per-tile scores — the
localization signal for UI / Stage 2). Cached per image + artifact version.
"""

import logging
from typing import Any, Dict, List, Optional

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.serializers import Serializers

logger = logging.getLogger(__name__)


@register_step
class ScoreOcclusionStep(BaseStep):
    """P(lens occlusion) per image via the spec-096 CLIP probe artifact."""

    def __init__(self):
        self._metadata = StepMetadata(
            name="score_occlusion",
            display_name="Score Occlusion (CLIP probe)",
            description="P(finger/strap over lens) per image, + 9 tile scores for "
                        "localization. Model: spec-096 benchmark winner.",
            category="scoring",
            requires={"image_paths"},
            produces={"occlusion_scores", "occlusion_tiles"},
            depends_on=["discover_images"],
            config_schema={"type": "object", "properties": {
                "artifact_path": {"type": "string",
                                  "description": "Override the probe artifact (models/occlusion/...)"},
            }},
        )
        self._scorer = None

    def _get_scorer(self, config: dict):
        if self._scorer is None:
            from sim_bench.occlusion_bench.scorer import OcclusionScorer
            self._scorer = OcclusionScorer(config.get("artifact_path"))
            logger.info("occlusion scorer loaded: %s", self._scorer.version)
        return self._scorer

    def release(self) -> None:
        """SIGHTING-117: free the CLIP ViT-B/32 backbone after scoring."""
        self._release_models("_scorer")

    def _get_cache_config(self, context: PipelineContext, config: dict) -> Optional[Dict[str, Any]]:
        paths = [str(p) for p in context.image_paths]
        if not paths:
            return None
        return {"items": paths, "feature_type": "occlusion",
                "model_name": "clip-vitb32-gmax-probe",
                "metadata": {"model_version": self._get_scorer(config).version}}

    def _process_uncached(self, items: List[str], context: PipelineContext,
                          config: dict) -> Dict[str, dict]:
        from sim_bench.occlusion_bench.scorer import OcclusionInputs
        scorer = self._get_scorer(config)
        results: Dict[str, dict] = {}
        for i, path in enumerate(items):
            r = scorer.calc(OcclusionInputs(image_paths=[path]))
            # unreadable image -> score 0 (never penalized), recorded as skipped
            results[path] = ({"p": r.scores[path], "tiles": r.tiles[path]}
                             if path in r.scores else {"p": 0.0, "tiles": [0.0] * 9})
            context.report_progress("score_occlusion", (i + 1) / len(items),
                                    f"Occlusion {i + 1}/{len(items)}")
        return results

    def _serialize_for_cache(self, result: dict, item: str) -> bytes:
        return Serializers.json_serialize(result)

    def _deserialize_from_cache(self, data: bytes, item: str) -> dict:
        return Serializers.json_deserialize(data)

    def _store_results(self, context: PipelineContext, results: Dict[str, dict],
                       config: dict) -> None:
        context.occlusion_scores = {p: r["p"] for p, r in results.items()}
        context.occlusion_tiles = {p: r["tiles"] for p, r in results.items()}
