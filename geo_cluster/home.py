"""Home anchor — the location where most of life happens (spec-022).

Home = the GPS location cluster spanning the most distinct days across the
album. Anything far from home reads as travel; near home reads as routine.
Auto-derived per album; a manual-pin strategy can replace this later without
touching the axes (they just receive a ``home`` coordinate).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from geo_cluster._clustering import cluster_locations
from geo_cluster.types import GeoMetadata


class HomeAnchor:
    def __init__(self, radius_km: float = 30.0):
        self.radius_km = radius_km

    def calc(self, metadata: dict[str, GeoMetadata]) -> Optional[tuple[float, float]]:
        geo = [m for m in metadata.values() if m.has_geo]
        if not geo:
            return None
        latlon = np.array([[m.lat, m.lon] for m in geo])
        labels = cluster_locations(latlon, self.radius_km)

        best_centroid = None
        best_score = -1
        for c in set(labels):
            idx = [i for i in range(len(geo)) if labels[i] == c]
            days = {geo[i].timestamp.date() for i in idx if geo[i].has_time}
            score = len(days) if days else len(idx)  # distinct days, else photo count
            if score > best_score:
                best_score = score
                best_centroid = latlon[idx].mean(axis=0)
        return tuple(float(x) for x in best_centroid)
