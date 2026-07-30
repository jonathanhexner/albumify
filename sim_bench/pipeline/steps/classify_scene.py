"""Classify scene step — zero-shot CLIP scene tags (portrait/scenery/nature/...).

spec-094 follow-up (user request): categorical scene description per image via
CLIP text-image similarity. Stores the FULL ranked category list + softmax
confidence (FR-1: full top-k, not just top-1). Confidence is relative ranking,
not accuracy — same honesty rule as the geo methods.
"""

import logging
from typing import Any, Dict, List, Optional

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.serializers import Serializers

logger = logging.getLogger(__name__)

DEFAULT_CATEGORIES = [
    "a portrait photo of a person",
    "a group photo of people",
    "street photo with people",
    "scenery / landscape photo",
    "nature and plants",
    "architecture / historic building",
    "night life / evening city scene",
    "food photo",
    "indoor home scene",
    "a painting or artwork",
]


@register_step
class ClassifySceneStep(BaseStep):
    """Zero-shot scene categories from CLIP (cached per image + category-set)."""

    def __init__(self):
        self._metadata = StepMetadata(
            name="classify_scene",
            display_name="Classify Scene (CLIP zero-shot)",
            description="Tag each image with a scene category (portrait, scenery, nature, "
                        "night life, ...) via CLIP text-image similarity. Full ranked list stored.",
            category="analysis",
            requires={"image_paths"},
            produces={"scene_tags"},
            depends_on=["discover_images"],
            config_schema={"type": "object", "properties": {
                "categories": {"type": "array", "items": {"type": "string"}},
                "device": {"type": "string", "default": "cpu"},
            }},
        )
        self._model = None
        self._text_feats = None
        self._text_key: Optional[str] = None

    # ------------------------------------------------------------------ cache
    def _get_cache_config(self, context: PipelineContext, config: dict) -> Optional[Dict[str, Any]]:
        paths = [str(p) for p in context.image_paths]
        if not paths:
            return None
        cats = config.get("categories") or DEFAULT_CATEGORIES
        import hashlib
        cat_ver = hashlib.sha1("|".join(cats).encode()).hexdigest()[:10]
        return {"items": paths, "feature_type": "scene_tag",
                "model_name": "clip-vitb32-zeroshot",
                "metadata": {"model_version": f"v1-{cat_ver}"}}  # category change invalidates

    def _encoders(self, config: dict):
        import clip
        import torch
        device = config.get("device", "cpu")
        cats = config.get("categories") or DEFAULT_CATEGORIES
        key = "|".join(cats)
        if self._model is None:
            self._model, self._preprocess = clip.load("ViT-B/32", device=device)
            self._model.eval()
        if self._text_key != key:
            with torch.no_grad():
                t = self._model.encode_text(clip.tokenize(cats).to(device)).float()
                self._text_feats = t / t.norm(dim=-1, keepdim=True)
            self._text_key = key
        return cats

    def release(self) -> None:
        """SIGHTING-117: free the CLIP ViT-B/32 backbone after tagging. Also
        clears the cached text features + key so they recompute on reload."""
        self._release_models("_model", "_preprocess", "_text_feats", "_text_key")

    def _process_uncached(self, items: List[str], context: PipelineContext,
                          config: dict) -> Dict[str, list]:
        import torch
        from PIL import Image, ImageOps
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
        except ImportError:
            pass
        cats = self._encoders(config)
        results: Dict[str, list] = {}
        for i, path in enumerate(items):
            try:
                with Image.open(path) as im:
                    img = ImageOps.exif_transpose(im).convert("RGB")
                with torch.no_grad():
                    e = self._model.encode_image(self._preprocess(img).unsqueeze(0)).float()
                    e = e / e.norm(dim=-1, keepdim=True)
                    sims = (e @ self._text_feats.T).squeeze(0)
                    probs = sims.softmax(dim=-1)
                order = probs.argsort(descending=True)
                results[path] = [{"label": cats[int(j)], "score": float(probs[int(j)])}
                                 for j in order]
            except Exception as e:  # unreadable image -> empty tags, never raise
                logger.warning("classify_scene failed for %s: %s", path, e)
                results[path] = []
            context.report_progress("classify_scene", (i + 1) / len(items),
                                    f"Tagging {i + 1}/{len(items)}")
        return results

    def _serialize_for_cache(self, result: list, item: str) -> bytes:
        return Serializers.json_serialize(result)

    def _deserialize_from_cache(self, data: bytes, item: str) -> list:
        return Serializers.json_deserialize(data)

    def _store_results(self, context: PipelineContext, results: Dict[str, list],
                       config: dict) -> None:
        context.scene_tags = results
