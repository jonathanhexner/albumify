"""StreetCLIP zero-shot city geolocation (spec-022, stage C).

For photos without GPS, guess the top-k cities by scoring the image against
"(city, country)" text prompts with StreetCLIP (a CLIP model fine-tuned for
geolocalization). Text features for the candidate list are computed once and
reused for every image.

Primary entry: ``StreetCLIPLocator.calc(StreetClipInputs) -> StreetClipResult``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from geo_cluster.world_cities import city_prompts

logger = logging.getLogger(__name__)

# One loaded model per process, shared across StreetCLIPLocator instances.
_MODELS: dict = {}


@dataclass
class StreetClipInputs:
    image_paths: list[str]


@dataclass
class StreetClipResult:
    # path -> [{"label": "Budapest, Hungary", "score": 0.87}, ...]
    predictions: dict[str, list] = field(default_factory=dict)


class StreetCLIPLocator:
    def __init__(self, model_name: str = "geolocal/StreetCLIP", device: str = "cpu",
                 top_k: int = 3, cities=None):
        self.model_name = model_name
        self.device = device
        self.top_k = top_k
        self.labels, self.prompts = city_prompts(cities)

    def _ensure(self):
        if self.model_name in _MODELS:
            return _MODELS[self.model_name]
        import torch
        from transformers import CLIPModel, CLIPProcessor

        from geo_cluster._hf_compat import allow_unsafe_torch_load
        allow_unsafe_torch_load()  # StreetCLIP ships .bin only; torch<2.6 (see shim)
        logger.info("loading StreetCLIP %s (first time downloads ~600MB)...", self.model_name)
        model = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()
        proc = CLIPProcessor.from_pretrained(self.model_name)
        with torch.no_grad():
            tin = proc(text=self.prompts, return_tensors="pt", padding=True).to(self.device)
            tfeat = model.get_text_features(**tin)
            tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
        _MODELS[self.model_name] = (model, proc, tfeat)
        return _MODELS[self.model_name]

    def _predict_one(self, path: str, model, proc, tfeat) -> list:
        import torch
        from PIL import Image, ImageOps
        from pillow_heif import register_heif_opener
        register_heif_opener()
        with Image.open(path) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")
        with torch.no_grad():
            iin = proc(images=img, return_tensors="pt").to(self.device)
            ifeat = model.get_image_features(**iin)
            ifeat = ifeat / ifeat.norm(dim=-1, keepdim=True)
            sims = (ifeat @ tfeat.T).squeeze(0)
            probs = (sims * 100.0).softmax(dim=-1)
            vals, idx = probs.topk(self.top_k)
        return [{"label": self.labels[i], "score": round(float(v), 4)}
                for v, i in zip(vals.tolist(), idx.tolist())]

    def calc(self, inputs: StreetClipInputs) -> StreetClipResult:
        """Predict cities for every given image. Caching is the caller's job
        (the pipeline step persists to ``universal_cache``); this helper always
        computes."""
        result = StreetClipResult()
        model = proc = tfeat = None
        for path in inputs.image_paths:
            if model is None:
                model, proc, tfeat = self._ensure()
            try:
                preds = self._predict_one(path, model, proc, tfeat)
            except Exception as e:
                logger.warning("StreetCLIP failed on %s: %s", path, e)
                preds = []
            result.predictions[path] = preds
        return result
