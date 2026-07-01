"""GEO axis — group by location, then split by time within a place (spec-022)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from geo_cluster._clustering import cluster_locations
from geo_cluster.axes.base import AxisInputs, register_axis
from geo_cluster.types import Segment, Segmentation


@register_axis
class GeoAxis:
    name = "geo"
    needs = "GPS coordinates"

    def perturbed_config(self, config: dict) -> dict:
        c = dict(config)
        c["geo_radius_km"] = config.get("geo_radius_km", 30.0) * 1.25
        return c

    def propose(self, inputs: AxisInputs) -> Optional[Segmentation]:
        meta = inputs.metadata
        paths = [p for p, m in meta.items() if m.has_geo]
        if len(paths) < 2:
            return None

        radius = inputs.config.get("geo_radius_km", 30.0)
        latlon = np.array([[meta[p].lat, meta[p].lon] for p in paths])
        loc_labels = cluster_locations(latlon, radius)

        # GEO means *where* — one segment per location cluster. Splitting a
        # place by time is the TIME axis's job; doing it here fragments a
        # long-lived home cluster (the experiment caught this). Within a geo
        # segment, photos can still be time-sorted at display.
        score_space = {p: np.radians(ll) for p, ll in zip(paths, latlon)}
        segments: list[Segment] = []
        for loc in sorted(set(loc_labels)):
            members = [paths[i] for i in range(len(paths)) if loc_labels[i] == loc]
            centroid = latlon[loc_labels == loc].mean(axis=0).tolist()
            segments.append(Segment(image_paths=members, meta={"centroid": centroid}))

        unsorted = [p for p, m in meta.items() if not m.has_geo]
        return Segmentation(
            axis=self.name, segments=segments, unsorted=unsorted,
            score_space=score_space, metric="haversine",
        )
