"""
Hybrid HDBSCAN + Local Cohesion clustering - Tcore2all variant.

Same as hybrid_hdbscan_knn but with different threshold computation:
- Still compute d3 and pick up to 10 exemplars (smallest d3)
- Instead of T from exemplar↔exemplar pairwise distances,
  compute T from exemplar→all-faces-in-cluster distances
- Pool distances from each exemplar to every face in its cluster
- T = percentile(pool, 95) with same clamping

Why: exemplar↔exemplar is "core-core" (too tight). exemplar→cluster captures
pose spread, so fewer splits.

Distance metric: Cosine distance = 1 - cosine_similarity, clipped to [0, 2].
"""

import logging
from typing import Dict, Any

import numpy as np

from sim_bench.clustering.hybrid_hdbscan_knn import HybridHDBSCANKNN, ClusterState
from sim_bench.clustering.distance_utils import cosine_distance_matrix

logger = logging.getLogger(__name__)


class HybridHDBSCANKNNTcore2all(HybridHDBSCANKNN):
    """Hybrid HDBSCAN + Local Cohesion with exemplar→all threshold computation."""

    doc_explanation = """
Variant of hybrid_hdbscan_knn with wider threshold computation.
Instead of T from exemplar↔exemplar (tight core-core), computes T from
exemplar→all-faces distances, capturing pose spread within the cluster.

Threshold T = percentile(exemplar_to_all_distances, 95), clamped to [floor, ceiling].
Uses same merge/attach logic as hybrid_hdbscan_knn but with larger T values.

Better for clusters with significant pose variation (frontal + profile faces).
"""

    # Inherit decision_parameters from parent, override threshold_percentile default
    decision_parameters = {
        **HybridHDBSCANKNN.decision_parameters,
        "threshold_percentile": {
            "description": "Percentile of exemplar→all distances for T",
            "default": 95,
            "decision_role": "Higher percentile (95 vs 90) gives wider T for pose variation"
        },
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Default to 95th percentile for exemplar→all (wider distribution)
        self.threshold_percentile = self.params.get('threshold_percentile', 95)

    def _compute_cluster_states(
        self,
        labels: np.ndarray,
        features: np.ndarray
    ) -> Dict[int, ClusterState]:
        """Compute threshold using exemplar→all-faces distances instead of exemplar↔exemplar."""
        cluster_states = {}

        for label in set(labels):
            if label == -1:
                continue

            indices = np.where(labels == label)[0]
            n_faces = len(indices)
            cluster_features = features[indices]

            if n_faces < 2:
                # Single-face cluster: use floor threshold, face is its own exemplar
                cluster_states[label] = ClusterState(
                    label=label,
                    indices=indices,
                    threshold=self.threshold_floor,
                    exemplar_indices=indices,
                    exemplar_embeddings=cluster_features,
                    q1=0.0,
                    q3=0.0,
                    iqr=0.0,
                    raw_threshold=self.threshold_floor
                )
                continue

            # Compute pairwise cosine distances
            distances = cosine_distance_matrix(cluster_features)

            # For each face, compute d3 (distance to 3rd nearest neighbor)
            k = min(self.knn_k, n_faces - 1)
            d3_values = []

            for i in range(n_faces):
                sorted_dists = np.sort(distances[i])[1:k + 1]  # Exclude self
                d3_values.append(sorted_dists[-1] if len(sorted_dists) > 0 else 0)

            d3_values = np.array(d3_values)

            # Select exemplars: faces with smallest d3 (most core-like)
            n_exemplars = min(self.max_exemplars, n_faces)
            exemplar_local_indices = np.argsort(d3_values)[:n_exemplars]
            exemplar_global_indices = indices[exemplar_local_indices]
            exemplar_embeddings = features[exemplar_global_indices]

            # CHANGE: Compute threshold from exemplar→all-faces distances
            # Pool distances from each exemplar to every face in the cluster
            if len(exemplar_embeddings) < 1:
                raw_threshold = self.threshold_floor
                q1 = q3 = iqr = 0.0
            else:
                # exemplar_to_all: [n_exemplars, n_faces] matrix
                exemplar_local_features = cluster_features[exemplar_local_indices]
                exemplar_to_all = cosine_distance_matrix(exemplar_local_features, cluster_features)

                # Flatten to get all exemplar→face distances (includes self-distances ~0)
                # Exclude diagonal-like self-distances by masking
                pool = []
                for ex_idx, ex_local in enumerate(exemplar_local_indices):
                    for face_idx in range(n_faces):
                        if face_idx != ex_local:  # Exclude self-distance
                            pool.append(exemplar_to_all[ex_idx, face_idx])

                if len(pool) == 0:
                    raw_threshold = self.threshold_floor
                    q1 = q3 = iqr = 0.0
                else:
                    pool = np.array(pool)
                    q1, q3 = np.percentile(pool, [25, 75])
                    iqr = q3 - q1
                    raw_threshold = float(np.percentile(pool, self.threshold_percentile))

            # Clamp to [floor, ceiling]
            threshold = max(raw_threshold, self.threshold_floor)
            threshold = min(threshold, self.threshold_ceiling)

            cluster_states[label] = ClusterState(
                label=label,
                indices=indices,
                threshold=float(threshold),
                exemplar_indices=exemplar_global_indices,
                exemplar_embeddings=exemplar_embeddings,
                q1=float(q1),
                q3=float(q3),
                iqr=float(iqr),
                raw_threshold=float(raw_threshold)
            )

            logger.debug(
                f"  Cluster {label}: {n_faces} faces, {len(exemplar_global_indices)} exemplars, "
                f"Q{self.threshold_percentile}(core→all)={raw_threshold:.3f} "
                f"(Q1={q1:.3f}, Q3={q3:.3f}, IQR={iqr:.3f}), T={threshold:.3f}"
            )

        return cluster_states
