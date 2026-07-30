"""Spec-103 — fuse visual + capture-time + GPS into a scene-distance matrix.

The core of the time-backbone / optional-geo / visual-floor idea (spec-103). Framework-agnostic
(spec-053): config in ``__init__``, per-call data in ``SceneDistanceInputs``, ``calc()`` returns a
``SceneDistanceResult`` holding an NxN precomputed distance the clusterer consumes.

Degradation is per-PAIR and needs no imputation: each pair blends only the signals BOTH photos have,
then renormalizes the weights. A photo with no GPS simply contributes no geo term; a trip with no GPS
at all reduces the whole matrix to pure visual distance (the ``visual`` floor).

Distance terms, each mapped to [0,1] (0 = identical, 1 = far):
  visual : 1 - cosine_similarity(embeddings)
  time   : 1 - exp(-|dt| / time_scale)          # dt in minutes
  geo    : 1 - exp(-haversine_m / geo_scale)     # metres
Time comes from GeoMetadata.timestamp, else the ``YYYYMMDD_HHMMSS`` filename stem.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_STEM_TS = re.compile(r"(\d{8})[_-](\d{6})")  # 20250822_112331


def _time_from(image_id: str, meta) -> Optional[datetime]:
    """Capture time: EXIF timestamp if present, else parsed from the filename stem."""
    if meta is not None and getattr(meta, "timestamp", None) is not None:
        return meta.timestamp
    m = _STEM_TS.search(image_id)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _geo_from(meta) -> Optional[tuple]:
    if meta is not None and getattr(meta, "lat", None) is not None and getattr(meta, "lon", None) is not None:
        return float(meta.lat), float(meta.lon)
    return None


def _haversine_m(a: tuple, b: tuple) -> float:
    r = 6371000.0
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(h)))


@dataclass
class SceneDistanceInputs:
    embeddings: dict            # image_id -> np.ndarray
    geo_metadata: dict          # image_id -> GeoMetadata (entries may be missing / geo-less / time-less)
    image_ids: list             # order that fixes the matrix rows/cols


@dataclass
class SceneDistanceResult:
    image_ids: list
    distance_matrix: np.ndarray                 # NxN, symmetric, zero diagonal
    per_image_signal_used: dict = field(default_factory=dict)  # image_id -> ['visual','time','geo']


class SceneDistanceBuilder:
    """Path A (spec-103, the VALIDATED builder): visual cosine distance with a one-sided SHORT-RANGE
    capture-time boost. This is what ``build_scene_distance`` ships.

        d(i,j) = visual_cos(i,j) * (1 - boost * exp(-dt_seconds / tau_sec))

    Two photos taken <= ~tau_sec apart get their distance shrunk (a strong same-moment pull); photos
    minutes apart get discount ~= 1 so time does NOTHING and looks decide. Time only ever PULLS
    near-simultaneous photos together -- it never pushes distant photos apart. That one-sidedness is
    the fix for ``SceneDistanceFuser``'s additive blend, whose symmetric time term wrongly separated
    visually-similar shots a few minutes apart (spec-103 report, over-merge case).

    Capture time comes from ``GeoMetadata.timestamp`` else the ``YYYYMMDD_HHMMSS`` filename stem, so it
    works even when the working images are EXIF-stripped and ``geo_metadata`` is empty. No geo term here
    (Path A is visual+time; geo is a superseded axis, see ``SceneDistanceFuser``).
    """

    def __init__(self, boost: float = 0.6, tau_sec: float = 60.0):
        self.boost = float(boost)
        self.tau_sec = float(tau_sec)

    def calc(self, inputs: SceneDistanceInputs) -> SceneDistanceResult:
        ids = list(inputs.image_ids)
        n = len(ids)
        E = np.array([np.asarray(inputs.embeddings[i], dtype=np.float64) for i in ids])
        E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        cos = np.clip(E @ E.T, -1.0, 1.0)
        D = 1.0 - cos  # visual distance

        geo_md = inputs.geo_metadata or {}
        times = [_time_from(ids[k], geo_md.get(ids[k])) for k in range(n)]
        signal_used = {ids[k]: (["visual"] + (["time"] if times[k] else [])) for k in range(n)}

        for a in range(n):
            for b in range(a + 1, n):
                if times[a] and times[b]:
                    dt = abs((times[a] - times[b]).total_seconds())
                    discount = 1.0 - self.boost * math.exp(-dt / self.tau_sec)
                    D[a, b] = D[b, a] = D[a, b] * discount
        np.fill_diagonal(D, 0.0)
        return SceneDistanceResult(image_ids=ids, distance_matrix=D, per_image_signal_used=signal_used)


class SceneDistanceFuser:
    """Weighted visual+time+geo distance with per-pair signal-dropping (no imputation).

    NOTE (spec-103): SUPERSEDED by ``SceneDistanceBuilder`` (Path A) for production. Kept for the
    experiment sweep -- its symmetric additive time term over-merges (see the report's over-merge
    case); ``build_scene_distance`` ships the one-sided ``SceneDistanceBuilder`` instead.
    """

    def __init__(self, w_visual: float = 1.0, w_time: float = 1.0, w_geo: float = 1.0,
                 time_scale_min: float = 30.0, geo_scale_m: float = 200.0):
        self.w_visual = float(w_visual)
        self.w_time = float(w_time)
        self.w_geo = float(w_geo)
        self.time_scale_min = float(time_scale_min)
        self.geo_scale_m = float(geo_scale_m)

    def calc(self, inputs: SceneDistanceInputs) -> SceneDistanceResult:
        ids = list(inputs.image_ids)
        n = len(ids)
        # Unit-normalized embeddings -> cosine similarity via dot product.
        E = np.array([np.asarray(inputs.embeddings[i], dtype=np.float64) for i in ids])
        E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        cos = np.clip(E @ E.T, -1.0, 1.0)
        visual_d = 1.0 - cos  # [0,2] but ~[0,1] for these embeddings

        times = [_time_from(i, inputs.geo_metadata.get(i)) for i in ids]
        geos = [_geo_from(inputs.geo_metadata.get(i)) for i in ids]

        signal_used = {
            ids[k]: (["visual"] + (["time"] if times[k] else []) + (["geo"] if geos[k] else []))
            for k in range(n)
        }

        D = np.zeros((n, n), dtype=np.float64)
        for a in range(n):
            for b in range(a + 1, n):
                num = self.w_visual * visual_d[a, b]
                den = self.w_visual
                if times[a] and times[b]:
                    gap_min = abs((times[a] - times[b]).total_seconds()) / 60.0
                    num += self.w_time * (1.0 - math.exp(-gap_min / self.time_scale_min))
                    den += self.w_time
                if geos[a] and geos[b]:
                    dist_m = _haversine_m(geos[a], geos[b])
                    num += self.w_geo * (1.0 - math.exp(-dist_m / self.geo_scale_m))
                    den += self.w_geo
                D[a, b] = D[b, a] = num / den
        return SceneDistanceResult(image_ids=ids, distance_matrix=D, per_image_signal_used=signal_used)
