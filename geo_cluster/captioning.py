"""BLIP image captioning (spec-022, stage E).

One short caption per image (e.g. "a group of people standing on a bridge").
These captions are the raw signal the semantic axis / LLM theming will later
consume; for now the standalone app just displays them.

Primary entry: ``BlipCaptioner.calc(CaptionInputs) -> CaptionResult``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_MODELS: dict = {}


@dataclass
class CaptionInputs:
    image_paths: list[str]


@dataclass
class CaptionResult:
    captions: dict[str, str] = field(default_factory=dict)


class BlipCaptioner:
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base",
                 device: str = "cpu", max_new_tokens: int = 30):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens

    def _ensure(self):
        if self.model_name in _MODELS:
            return _MODELS[self.model_name]
        from transformers import BlipForConditionalGeneration, BlipProcessor

        from geo_cluster._hf_compat import allow_unsafe_torch_load
        allow_unsafe_torch_load()  # BLIP-base ships .bin only; torch<2.6 (see shim)
        logger.info("loading BLIP %s (first time downloads ~1GB)...", self.model_name)
        proc = BlipProcessor.from_pretrained(self.model_name)
        model = BlipForConditionalGeneration.from_pretrained(self.model_name).to(self.device).eval()
        _MODELS[self.model_name] = (model, proc)
        return _MODELS[self.model_name]

    def _caption_one(self, path: str, model, proc) -> str:
        import torch
        from PIL import Image, ImageOps
        from pillow_heif import register_heif_opener
        register_heif_opener()
        with Image.open(path) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")
        with torch.no_grad():
            inp = proc(images=img, return_tensors="pt").to(self.device)
            out = model.generate(**inp, max_new_tokens=self.max_new_tokens)
        return proc.decode(out[0], skip_special_tokens=True).strip()

    def calc(self, inputs: CaptionInputs) -> CaptionResult:
        """Caption every given image. Caching is the caller's job (the pipeline
        step persists to ``universal_cache``); this helper always computes."""
        result = CaptionResult()
        model = proc = None
        for path in inputs.image_paths:
            if model is None:
                model, proc = self._ensure()
            try:
                cap = self._caption_one(path, model, proc)
            except Exception as e:
                logger.warning("BLIP failed on %s: %s", path, e)
                cap = ""
            result.captions[path] = cap
        return result
