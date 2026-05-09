"""Feature engineering for cluster merge prediction.

Package structure:
  distance.py      — Group A: cross-cluster distance distribution
  geometry.py      — Groups B+C: cluster geometry and compactness
  source_images.py — Group D: shared source image diversity
  quality.py       — Group G: blur, pose, area statistics
  context.py       — Groups E+I: margin / global context (deferred)
  graph.py         — Group F: kNN topology features (deferred)
"""

import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from face_cluster.types import ClusterResult, FaceRecord, GraphResult
from face_cluster.features.distance import compute_distance_features
from face_cluster.features.geometry import compute_geometry_features
from face_cluster.features.source_images import compute_source_image_features
from face_cluster.features.quality import compute_quality_features

logger = logging.getLogger(__name__)

VERSION = 4


# ---------------------------------------------------------------------------
# Input container
# ---------------------------------------------------------------------------

@dataclass
class MergeFeatureContext:
    """All data needed to compute merge features for a run.

    distance_matrix must be face-list indexed (shape = len(faces) x len(faces)).
    graph_result is optional; when absent, Group F features are omitted.
    """
    cluster_result: ClusterResult
    faces: List[FaceRecord]
    distance_matrix: np.ndarray
    graph_result: Optional[GraphResult] = None

    def __post_init__(self):
        assert self.distance_matrix.shape[0] == len(self.faces), (
            f"distance_matrix shape {self.distance_matrix.shape} "
            f"does not match len(faces)={len(self.faces)}"
        )


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClusterPairFeatures:
    """Feature vector for a candidate cluster pair (V3).

    All fields are Optional so V1/V2 callers and partial computations remain
    backward compatible. Fields are named in snake_case throughout.
    """

    # --- Group A: cross-cluster distance distribution ---
    min_exemplar_dist: Optional[float] = None
    p10_exemplar_dist: Optional[float] = None
    p25_exemplar_dist: Optional[float] = None
    exemplar_dist_mean: Optional[float] = None
    exemplar_dist_std: Optional[float] = None
    min_cross_dist: Optional[float] = None
    p10_cross_dist: Optional[float] = None
    p25_cross_dist: Optional[float] = None
    p50_cross_dist: Optional[float] = None
    p75_cross_dist: Optional[float] = None
    p90_cross_dist: Optional[float] = None
    cross_dist_iqr: Optional[float] = None
    support_fraction: Optional[float] = None
    n_cross_pairs_below_threshold: Optional[int] = None

    # --- Groups B+C: geometry and compactness ---
    size_a: Optional[int] = None
    size_b: Optional[int] = None
    size_min: Optional[int] = None
    size_ratio: Optional[float] = None
    size_sum: Optional[int] = None
    diameter_a: Optional[float] = None
    diameter_b: Optional[float] = None
    diameter_max: Optional[float] = None
    diameter_ratio: Optional[float] = None
    post_merge_diameter: Optional[float] = None
    diameter_expansion: Optional[float] = None
    mean_intra_dist_a: Optional[float] = None
    mean_intra_dist_b: Optional[float] = None
    exemplar_count_a: Optional[int] = None
    exemplar_count_b: Optional[int] = None
    t_a: Optional[float] = None
    t_b: Optional[float] = None
    t_local: Optional[float] = None
    t_global: Optional[float] = None
    dist_to_threshold_ratio: Optional[float] = None

    # --- Group D: source image diversity ---
    n_images_a: Optional[int] = None
    n_images_b: Optional[int] = None
    shared_source_images: Optional[int] = None
    shared_source_ratio: Optional[float] = None
    same_image_min_dist: Optional[float] = None

    # --- Group G: quality and pose ---
    mean_blur_a: Optional[float] = None
    mean_blur_b: Optional[float] = None
    blur_min_a: Optional[float] = None
    blur_min_b: Optional[float] = None
    frontal_frac_a: Optional[float] = None
    frontal_frac_b: Optional[float] = None
    frontal_frac_min: Optional[float] = None
    pose_diff: Optional[float] = None
    yaw_std_a: Optional[float] = None
    yaw_std_b: Optional[float] = None
    mean_area_a: Optional[float] = None
    mean_area_b: Optional[float] = None
    area_ratio: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def to_array(self, feature_names: List[str]) -> np.ndarray:
        d = self.to_dict()
        return np.array([d.get(name, 0.0) or 0.0 for name in feature_names], dtype=float)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class FeatureComputer:
    """Compute V3 feature vectors for candidate cluster pairs.

    Usage:
        ctx = MergeFeatureContext(cluster_result, faces, distance_matrix)
        fc = FeatureComputer()
        pair_features = fc.compute_all_pairs(ctx, candidate_threshold=0.45)
        df = fc.to_dataframe(pair_features)
    """

    VERSION = VERSION

    def __init__(
        self,
        support_threshold: float = 0.35,
        frontal_threshold: float = 15.0,
        merge_exemplar_threshold: float = 0.35,
    ):
        self.support_threshold = support_threshold
        self.frontal_threshold = frontal_threshold
        self.merge_exemplar_threshold = merge_exemplar_threshold

    def compute_pair_features(
        self,
        cid_a: int,
        cid_b: int,
        ctx: MergeFeatureContext,
        t_global: float,
    ) -> ClusterPairFeatures:
        """Compute all features for a single candidate pair."""
        cr = ctx.cluster_result
        nodes_a = cr.clusters[cid_a]
        nodes_b = cr.clusters[cid_b]
        ex_a = cr.exemplars.get(cid_a, nodes_a)
        ex_b = cr.exemplars.get(cid_b, nodes_b)
        dm = ctx.distance_matrix

        dist_feats = compute_distance_features(
            nodes_a, nodes_b, ex_a, ex_b, dm, self.support_threshold
        )
        geom_feats = compute_geometry_features(
            nodes_a, nodes_b, ex_a, ex_b, dm, t_global, self.merge_exemplar_threshold
        )
        src_feats = compute_source_image_features(nodes_a, nodes_b, ctx.faces, dm)
        qual_feats = compute_quality_features(
            nodes_a, nodes_b, ctx.faces, self.frontal_threshold
        )

        combined = {**dist_feats, **geom_feats, **src_feats, **qual_feats}
        return ClusterPairFeatures(**{
            f.name: combined.get(f.name)
            for f in fields(ClusterPairFeatures)
        })

    def compute_all_pairs(
        self,
        ctx: MergeFeatureContext,
        candidate_threshold: float = 0.45,
    ) -> Dict[Tuple[int, int], ClusterPairFeatures]:
        """Compute features for all candidate cluster pairs.

        A pair is a candidate if their minimum exemplar distance is below
        candidate_threshold.

        Returns:
            Dict keyed by (min_cid, max_cid) tuple.
        """
        cr = ctx.cluster_result
        dm = ctx.distance_matrix
        cluster_ids = sorted(cr.clusters.keys())

        # Compute global T: median of per-cluster P90 intra-exemplar distances
        t_global = self._compute_t_global(cr, dm)

        result: Dict[Tuple[int, int], ClusterPairFeatures] = {}
        n_candidates = 0

        for i, cid_a in enumerate(cluster_ids):
            for cid_b in cluster_ids[i + 1:]:
                ex_a = cr.exemplars.get(cid_a, cr.clusters[cid_a])
                ex_b = cr.exemplars.get(cid_b, cr.clusters[cid_b])
                min_dist = float(dm[np.ix_(ex_a, ex_b)].min())
                if min_dist > candidate_threshold:
                    continue
                n_candidates += 1
                key = (min(cid_a, cid_b), max(cid_a, cid_b))
                result[key] = self.compute_pair_features(cid_a, cid_b, ctx, t_global)

        logger.debug(f"Computed features for {n_candidates} candidate pairs")
        return result

    def compute_top_n_pairs(
        self,
        ctx: MergeFeatureContext,
        max_dist: float = 0.80,
        top_n: int = 300,
    ) -> Dict[Tuple[int, int], ClusterPairFeatures]:
        """Compute features for the top-N closest cluster pairs.

        Ranks all cluster pairs by min exemplar distance, skips pairs whose
        min exemplar distance exceeds max_dist, and returns at most top_n
        pairs (closest first).

        Args:
            ctx: Feature computation context.
            max_dist: Skip pairs with min exemplar distance > this value.
            top_n: Maximum number of pairs to return.

        Returns:
            Dict keyed by (min_cid, max_cid), ordered by ascending exemplar dist.
        """
        cr = ctx.cluster_result
        dm = ctx.distance_matrix
        cluster_ids = sorted(cr.clusters.keys())
        t_global = self._compute_t_global(cr, dm)

        # Score all pairs — O(n_clusters^2) exemplar distance computation
        scored: List[Tuple[float, int, int]] = []
        for i, cid_a in enumerate(cluster_ids):
            for cid_b in cluster_ids[i + 1:]:
                ex_a = cr.exemplars.get(cid_a, cr.clusters[cid_a])
                ex_b = cr.exemplars.get(cid_b, cr.clusters[cid_b])
                min_dist = float(dm[np.ix_(ex_a, ex_b)].min())
                if min_dist <= max_dist:
                    scored.append((min_dist, cid_a, cid_b))

        scored.sort(key=lambda x: x[0])
        scored = scored[:top_n]

        result: Dict[Tuple[int, int], ClusterPairFeatures] = {}
        for _, cid_a, cid_b in scored:
            key = (min(cid_a, cid_b), max(cid_a, cid_b))
            result[key] = self.compute_pair_features(cid_a, cid_b, ctx, t_global)

        logger.debug(
            "compute_top_n_pairs: %d pairs (max_dist=%.2f, top_n=%d)",
            len(result), max_dist, top_n,
        )
        return result

    def to_dataframe(
        self,
        pair_features: Dict[Tuple[int, int], ClusterPairFeatures],
    ) -> pd.DataFrame:
        """Convert pair feature dict to a DataFrame (one row per pair)."""
        rows = []
        for (cid_a, cid_b), feat in pair_features.items():
            row = {"cluster_a": cid_a, "cluster_b": cid_b}
            row.update(feat.to_dict())
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------

    @staticmethod
    def _compute_t_global(cr: ClusterResult, dm: np.ndarray) -> float:
        """Global threshold: median of per-cluster P90 intra-exemplar distances."""
        t_values = []
        for cid, nodes in cr.clusters.items():
            exemplars = cr.exemplars.get(cid, nodes)
            if len(exemplars) < 2:
                continue
            sub = dm[np.ix_(exemplars, exemplars)]
            upper = sub[np.triu_indices_from(sub, k=1)]
            if len(upper) > 0:
                t_values.append(float(np.percentile(upper, 90)))
        return float(np.median(t_values)) if t_values else 0.0
