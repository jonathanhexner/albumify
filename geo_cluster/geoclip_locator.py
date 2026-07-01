"""GeoCLIP coordinate geolocation (spec-022, stage C — coordinate-regression).

Unlike StreetCLIP (which picks from a fixed city list), GeoCLIP predicts an
actual (lat, lon) directly from the image, then we reverse-geocode that point to
a place name offline. No candidate list, so no "the right city wasn't an option"
failure mode.

Primary entry: ``GeoCLIPLocator.calc(GeoClipInputs) -> GeoClipResult``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_MODELS: dict = {}


@dataclass
class GeoClipInputs:
    image_paths: list[str]


@dataclass
class GeoClipResult:
    # path -> [{"lat":..,"lon":..,"prob":..,"place":"Budapest, HU"}, ...] top-k
    predictions: dict[str, list] = field(default_factory=dict)


def _reverse_geocode(lat: float, lon: float):
    try:
        import reverse_geocoder as rg
        r = rg.search([(lat, lon)], mode=1)[0]  # mode=1: single-process (Windows-safe)
        parts = [r.get("name", ""), r.get("admin1", ""), r.get("cc", "")]
        parts = [p for p in parts if p]
        return ", ".join(parts) if parts else None
    except Exception as e:  # reverse_geocoder missing / failed -> just show coords
        logger.debug("reverse geocode failed: %s", e)
        return None


class GeoCLIPLocator:
    def __init__(self, top_k: int = 3, device: str = "cpu"):
        self.top_k = top_k
        self.device = device

    def _ensure(self):
        if "geoclip" in _MODELS:
            return _MODELS["geoclip"]
        from geo_cluster._hf_compat import allow_unsafe_torch_load
        allow_unsafe_torch_load()
        from geoclip import GeoCLIP
        logger.info("loading GeoCLIP (first time downloads weights)...")
        model = GeoCLIP()
        try:
            model = model.to(self.device)
        except Exception:
            pass
        _MODELS["geoclip"] = model
        return model

    def _predict_one(self, path: str, model) -> list:
        gps, prob = model.predict(path, top_k=self.top_k)
        preds = []
        for i in range(len(prob)):
            lat, lon = float(gps[i][0]), float(gps[i][1])
            preds.append({
                "lat": round(lat, 5), "lon": round(lon, 5),
                "prob": round(float(prob[i]), 4),
                "place": _reverse_geocode(lat, lon),
            })
        return preds

    def calc(self, inputs: GeoClipInputs) -> GeoClipResult:
        """Predict coordinates for every given image. Caching is the caller's
        job (the pipeline step persists to ``universal_cache``); this helper
        always computes."""
        result = GeoClipResult()
        model = None
        for path in inputs.image_paths:
            if model is None:
                model = self._ensure()
            try:
                preds = self._predict_one(path, model)
            except Exception as e:
                logger.warning("GeoCLIP failed on %s: %s", path, e)
                preds = []
            result.predictions[path] = preds
        return result
