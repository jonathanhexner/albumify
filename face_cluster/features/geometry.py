"""Groups B+C: cluster geometry, diameters, compactness, threshold features."""

from typing import List, Dict, Optional
import numpy as np


def _diameter(nodes: List[int], dm: np.ndarray) -> float:
    if len(nodes) < 2:
        return 0.0
    sub = dm[np.ix_(nodes, nodes)]
    return float(sub.max())


def _mean_intra(nodes: List[int], dm: np.ndarray) -> float:
    if len(nodes) < 2:
        return 0.0
    sub = dm[np.ix_(nodes, nodes)]
    upper = sub[np.triu_indices_from(sub, k=1)]
    return float(upper.mean()) if len(upper) > 0 else 0.0


def _t_local(exemplars: List[int], dm: np.ndarray) -> float:
    """P90 of intra-exemplar pairwise distances."""
    if len(exemplars) < 2:
        return 0.0
    sub = dm[np.ix_(exemplars, exemplars)]
    upper = sub[np.triu_indices_from(sub, k=1)]
    return float(np.percentile(upper, 90)) if len(upper) > 0 else 0.0


def compute_geometry_features(
    nodes_a: List[int],
    nodes_b: List[int],
    exemplars_a: List[int],
    exemplars_b: List[int],
    distance_matrix: np.ndarray,
    t_global: float,
    merge_exemplar_threshold: float,
) -> Dict[str, float]:
    """Compute cluster geometry and compactness features.

    Returns a flat dict of feature name -> value.
    """
    size_a = len(nodes_a)
    size_b = len(nodes_b)
    size_min = min(size_a, size_b)
    size_max = max(size_a, size_b)

    dia_a = _diameter(nodes_a, distance_matrix)
    dia_b = _diameter(nodes_b, distance_matrix)
    dia_max = max(dia_a, dia_b)
    dia_min = min(dia_a, dia_b)

    post_merge_diameter = _diameter(nodes_a + nodes_b, distance_matrix)
    diameter_expansion = post_merge_diameter / dia_max if dia_max > 0 else 1.0

    t_a = _t_local(exemplars_a, distance_matrix)
    t_b = _t_local(exemplars_b, distance_matrix)
    t_local = max(t_a, t_b)

    min_exemplar_dist = float(
        distance_matrix[np.ix_(exemplars_a, exemplars_b)].min()
    )

    return {
        "size_a": size_a,
        "size_b": size_b,
        "size_min": size_min,
        "size_ratio": size_max / size_min if size_min > 0 else 1.0,
        "size_sum": size_a + size_b,
        "diameter_a": dia_a,
        "diameter_b": dia_b,
        "diameter_max": dia_max,
        "diameter_ratio": dia_max / dia_min if dia_min > 0 else 1.0,
        "post_merge_diameter": post_merge_diameter,
        "diameter_expansion": diameter_expansion,
        "mean_intra_dist_a": _mean_intra(nodes_a, distance_matrix),
        "mean_intra_dist_b": _mean_intra(nodes_b, distance_matrix),
        "exemplar_count_a": len(exemplars_a),
        "exemplar_count_b": len(exemplars_b),
        "t_a": t_a,
        "t_b": t_b,
        "t_local": t_local,
        "t_global": t_global,
        "dist_to_threshold_ratio": (
            min_exemplar_dist / merge_exemplar_threshold
            if merge_exemplar_threshold > 0 else 0.0
        ),
    }
