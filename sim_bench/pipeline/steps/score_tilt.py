"""Score tilt step (spec-099 Phase 1 / spec-100) — thin wrapper over TiltScorer.

Runs GeoCalib (learned single-image roll estimation, validated in spec-100) per
image; writes ``context.tilt_angles`` (signed roll deg, + = content clockwise)
and ``context.tilt_confidences`` ([0,1], from GeoCalib's roll uncertainty).
Cached per image + model version — GeoCalib is ~1.9 s/img on CPU, so the cache
makes repeat runs free. A guessed tilt comes back low-confidence and the
downstream penalty (tilt_penalty) ignores it.
"""

import logging
from typing import Any, Dict, List, Optional

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.serializers import Serializers

logger = logging.getLogger(__name__)


@register_step
class ScoreTiltStep(BaseStep):
    """Signed roll + confidence per image via the GeoCalib backend (spec-100)."""

    def __init__(self):
        self._metadata = StepMetadata(
            name="score_tilt",
            display_name="Score Tilt (GeoCalib)",
            description="Crooked-photo roll angle + confidence per image. "
                        "Learned single-image gravity estimation (spec-100).",
            category="scoring",
            requires={"image_paths"},
            produces={"tilt_angles", "tilt_confidences"},
            depends_on=["discover_images"],
            config_schema={"type": "object", "properties": {}},
        )
        self._scorer = None
        self._scorer_failed = False

    def _get_scorer(self, config: dict):
        if self._scorer is None and not self._scorer_failed:
            try:
                from sim_bench.quality_assessment.tilt_geocalib import TiltScorer
                self._scorer = TiltScorer()
                logger.info("tilt scorer loaded: %s", self._scorer.version)
            except Exception as exc:  # geocalib not installed / model load failed
                self._scorer_failed = True
                logger.warning("tilt scorer unavailable (%s); tilt penalty disabled", exc)
        return self._scorer

    def release(self) -> None:
        """SIGHTING-117: free the GeoCalib model after scoring. Leaves
        ``_scorer_failed`` set so a known-bad load is not retried next run."""
        self._release_models("_scorer")

    def _get_cache_config(self, context: PipelineContext, config: dict) -> Optional[Dict[str, Any]]:
        paths = [str(p) for p in context.image_paths]
        scorer = self._get_scorer(config)
        if not paths or scorer is None:
            return None
        return {"items": paths, "feature_type": "tilt",
                "model_name": "geocalib",
                "metadata": {"model_version": scorer.version}}

    def _process_uncached(self, items: List[str], context: PipelineContext,
                          config: dict) -> Dict[str, dict]:
        from sim_bench.quality_assessment.tilt_geocalib import TiltInputs
        scorer = self._get_scorer(config)
        results: Dict[str, dict] = {}
        for i, path in enumerate(items):
            r = scorer.calc(TiltInputs(image_paths=[path]))
            # unreadable image -> confidence 0 (never penalized)
            results[path] = ({"angle": r.angles[path], "conf": r.confidences[path]}
                             if path in r.angles else {"angle": 0.0, "conf": 0.0})
            context.report_progress("score_tilt", (i + 1) / len(items),
                                    f"Tilt {i + 1}/{len(items)}")
        return results

    def _serialize_for_cache(self, result: dict, item: str) -> bytes:
        return Serializers.json_serialize(result)

    def _deserialize_from_cache(self, data: bytes, item: str) -> dict:
        return Serializers.json_deserialize(data)

    def _store_results(self, context: PipelineContext, results: Dict[str, dict],
                       config: dict) -> None:
        context.tilt_angles = {p: r["angle"] for p, r in results.items()}
        context.tilt_confidences = {p: r["conf"] for p, r in results.items()}
