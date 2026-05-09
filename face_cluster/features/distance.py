"""Group A: cross-cluster distance distribution features."""

from typing import List, Dict
import numpy as np


def compute_distance_features(
    nodes_a: List[int],
    nodes_b: List[int],
    exemplars_a: List[int],
    exemplars_b: List[int],
    distance_matrix: np.ndarray,
    support_threshold: float,
) -> Dict[str, float]:
    """Compute cross-cluster distance distribution features.

    Returns a flat dict of feature name -> value.
    """
    cross = distance_matrix[np.ix_(nodes_a, nodes_b)].flatten()
    ex_dists = distance_matrix[np.ix_(exemplars_a, exemplars_b)].flatten()

    p10, p25, p50, p75, p90 = np.percentile(cross, [10, 25, 50, 75, 90])
    ex_p10, ex_p25 = (
        np.percentile(ex_dists, [10, 25]) if len(ex_dists) > 1
        else (float(ex_dists[0]), float(ex_dists[0]))
    )

    return {
        "min_exemplar_dist": float(ex_dists.min()),
        "p10_exemplar_dist": float(ex_p10),
        "p25_exemplar_dist": float(ex_p25),
        "exemplar_dist_mean": float(ex_dists.mean()),
        "exemplar_dist_std": float(ex_dists.std()) if len(ex_dists) > 1 else 0.0,
        "min_cross_dist": float(cross.min()),
        "p10_cross_dist": float(p10),
        "p25_cross_dist": float(p25),
        "p50_cross_dist": float(p50),
        "p75_cross_dist": float(p75),
        "p90_cross_dist": float(p90),
        "cross_dist_iqr": float(p75 - p25),
        "support_fraction": float(np.mean(cross < support_threshold)),
        "n_cross_pairs_below_threshold": int(np.sum(cross < support_threshold)),
    }
