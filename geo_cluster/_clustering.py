"""Low-level clustering helpers shared by the segmentation axes (spec-022).

Kept dependency-light (numpy + sklearn, both already project deps) and
framework-agnostic so the axes and notebooks can reuse them.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

EARTH_R_KM = 6371.0088


def cluster_locations(latlon_deg: np.ndarray, radius_km: float) -> np.ndarray:
    """Agglomerative clustering of GPS points -> integer label per point.

    Points within roughly ``radius_km`` merge (average linkage on great-circle
    distance), so neighbouring towns land in one location cluster.
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import haversine_distances

    n = len(latlon_deg)
    if n == 1:
        return np.zeros(1, dtype=int)

    dist_km = haversine_distances(np.radians(latlon_deg)) * EARTH_R_KM
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=float(radius_km),
    )
    return model.fit_predict(dist_km)


def split_on_time_gaps(times: list[datetime], gap_hours: float) -> np.ndarray:
    """Label each item (in input order) by which time-run it belongs to.

    Items are sorted by time; a new label starts whenever the gap to the
    previous photo exceeds ``gap_hours``. Returns labels aligned to the input
    order (not the sorted order).
    """
    labels = np.zeros(len(times), dtype=int)
    order = np.argsort([t.timestamp() for t in times])
    cur = 0
    prev: datetime | None = None
    for idx in order:
        t = times[idx]
        if prev is not None and (t - prev) > timedelta(hours=gap_hours):
            cur += 1
        labels[idx] = cur
        prev = t
    return labels
