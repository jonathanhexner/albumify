"""Score Quality step (spec-093) - run N quality methods over the images.

Generic dispatcher: given ``methods: [...]`` it scores every image with each
method via ``image_quality_models.create_model`` and writes
``context.method_scores[path][method] = score`` (higher = better).

Unlike the single-feature BaseStep template, this step caches PER METHOD
(``feature_type="quality_<method>"``) so a run scoring 5 methods is 5 independent
cache namespaces and re-runs are incremental. Persistence is ``universal_cache``
(spec-093 locked decision); the context field is the in-run hand-off only.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.serializers import Serializers

logger = logging.getLogger(__name__)

# Sensible default: the pyiqa no-reference metrics plus rule-based IQA.
DEFAULT_METHODS = ["brisque", "niqe", "maniqa", "musiq", "hyperiqa", "clipiqa", "rule_based_iqa"]


@register_step
class ScoreQualityStep(BaseStep):
    """Score images with a configurable set of quality methods."""

    def __init__(self):
        self._metadata = StepMetadata(
            name="score_quality",
            display_name="Score Image Quality (multi-method)",
            description=(
                "Run one or more image-quality methods (MANIQA, MUSIQ, HyperIQA, "
                "BRISQUE, NIQE, CLIP-IQA, AVA, rule-based IQA) over all images and "
                "store per-method scores for comparison."
            ),
            category="analysis",
            requires={"image_paths"},
            produces={"method_scores"},
            depends_on=["discover_images"],
            config_schema={
                "type": "object",
                "properties": {
                    "methods": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Model-registry types to run (e.g. maniqa, niqe, ava).",
                    },
                    "device": {"type": "string", "enum": ["cpu", "cuda", "mps"], "default": "cpu"},
                    "model_configs": {
                        "type": "object",
                        "description": "Per-method extra config, e.g. {'ava': {'checkpoint': '...'}}.",
                    },
                },
            },
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        """Score every image with each requested method, caching per method."""
        methods: List[str] = config.get("methods") or DEFAULT_METHODS
        device: str = config.get("device", "cpu")
        model_configs: Dict[str, dict] = config.get("model_configs", {})

        image_paths = [str(p) for p in context.image_paths]
        if not image_paths:
            context.report_progress(self._metadata.name, 1.0, "No images to score")
            return

        results: Dict[str, Dict[str, float]] = {p: {} for p in image_paths}
        cache_handler = self._get_cache_handler(context)

        for m_idx, method in enumerate(methods):
            scores = self._score_method(
                method, image_paths, device, model_configs.get(method, {}), cache_handler, context
            )
            for path, score in scores.items():
                if score is not None:
                    results[path][method] = score
            context.report_progress(
                self._metadata.name, (m_idx + 1) / len(methods),
                f"Scored {method} ({m_idx + 1}/{len(methods)} methods)",
            )

        context.method_scores = results

    # ------------------------------------------------------------------

    def _score_method(
        self,
        method: str,
        image_paths: List[str],
        device: str,
        extra_config: dict,
        cache_handler: Optional[Any],
        context: PipelineContext,
    ) -> Dict[str, Optional[float]]:
        """Score all images with one method, using a per-method cache namespace."""
        feature_type = f"quality_{method}"
        cached = self._load_cached(method, feature_type, image_paths, cache_handler)
        misses = [p for p in image_paths if p not in cached]

        if misses:
            model = self._build_model(method, device, extra_config)
            if model is None:
                return cached  # method unavailable/failed — skip it, keep any hits
            for path in misses:
                entry = self._score_one(model, path)
                cached[path] = entry
                if cache_handler is not None and entry.get("score") is not None:
                    self._store(method, feature_type, path, entry, cache_handler)

        return {p: cached.get(p, {}).get("score") for p in image_paths}

    def _build_model(self, method: str, device: str, extra_config: dict):
        from sim_bench.image_quality_models import create_model
        try:
            return create_model({"type": method, "device": device, **extra_config})
        except Exception as e:  # unavailable dep, bad checkpoint, unknown type
            logger.warning("score_quality: skipping method '%s' (%s)", method, e)
            return None

    def _score_one(self, model, path: str) -> Dict[str, Optional[float]]:
        try:
            score = float(model.score_image(Path(path)))
            raw = float(model.raw_score(Path(path))) if hasattr(model, "raw_score") else score
            return {"score": score, "raw": raw}
        except Exception as e:
            logger.warning("score_quality: failed on %s (%s)", path, e)
            return {"score": None, "raw": None}

    def _load_cached(self, method, feature_type, image_paths, cache_handler) -> Dict[str, dict]:
        if cache_handler is None:
            return {}
        from sim_bench.pipeline.cache_handler import CacheKey
        keys = [CacheKey(image_path=p, feature_type=feature_type, model_name=method) for p in image_paths]
        loaded = cache_handler.load_from_cache(keys)
        out: Dict[str, dict] = {}
        for path, key in zip(image_paths, keys):
            hit = loaded.get(key.to_string())
            if hit is not None:
                out[path] = Serializers.json_deserialize(hit[0])
        return out

    def _store(self, method, feature_type, path, entry, cache_handler) -> None:
        from sim_bench.pipeline.cache_handler import CacheKey
        key = CacheKey(image_path=path, feature_type=feature_type, model_name=method)
        cache_handler.store_to_cache(key, Serializers.json_serialize(entry), {})
