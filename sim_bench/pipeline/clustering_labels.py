"""Single source of truth for cluster-label conventions.

The clustering algorithms used in the pipeline (HDBSCAN, DBSCAN, and the
face-clustering connected-components path) all reserve the integer ``-1`` for
"this item did not make it into any cluster." Until this module existed that
sentinel was a magic number repeated across ~9 pipeline-step sites, with
predictable drift: ``cluster_scenes`` excluded ``-1`` when counting real
clusters (``k >= 0``), while ``select_best`` and the
``test_selected_from_each_cluster`` integration test treated the same ``-1``
key as just another cluster.

This module pins the convention. Producers and consumers (both production
code and tests) import :data:`NOISE_LABEL` instead of writing ``-1`` literally.
The value matches HDBSCAN / sklearn so there is no translation at the
algorithm boundary.
"""
from __future__ import annotations

# Sentinel cluster id for items the clusterer could not group (HDBSCAN /
# sklearn convention). Items with this label are not part of any real
# cluster — they are outliers / noise. Pinned to -1 deliberately so we never
# need a translation layer at the algorithm boundary.
NOISE_LABEL: int = -1


def is_noise(cluster_id: int) -> bool:
    """True iff the given cluster id is the noise sentinel."""
    return cluster_id == NOISE_LABEL


__all__ = ["NOISE_LABEL", "is_noise"]
