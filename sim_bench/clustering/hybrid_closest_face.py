"""
Hybrid HDBSCAN+kNN with cross-cluster d3 matching.

Algorithm:
1. HDBSCAN → initial clusters
2. For each cluster:
   - Compute d3 (distance to 3rd nearest neighbor) for all faces
   - T = median(d3) + iqr_multiplier×IQR(d3)
3. Merge clusters A and B:
   - Stage 1: Early exit if min(exemplar distances) > 2×max(T_A, T_B)
   - Stage 2: For each face, compute d3 using other cluster's neighbors
   - Merge if ≥merge_min_faces faces fit into the other cluster
4. Attach: If noise point's d3 (using cluster neighbors) < T, otherwise leave as noise

Distance metric: Cosine distance = 1 - cosine_similarity, clipped to [0, 2].
Thresholds are calibrated for cosine distance (not Euclidean).
"""

import logging
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass, field
import numpy as np
from sim_bench.clustering.base import ClusteringMethod
from sim_bench.clustering.distance_utils import (
    cosine_distance_matrix,
    cosine_distance_to_set,
)

logger = logging.getLogger(__name__)


@dataclass
class ClusterD3State:
    """State for a cluster with d3-based threshold."""
    label: int
    indices: np.ndarray
    features: np.ndarray
    threshold: float
    d3_values: np.ndarray
    exemplar_indices: np.ndarray
    exemplar_features: np.ndarray
    # d3 stats (for debug)
    q1: float = 0.0
    q3: float = 0.0
    iqr: float = 0.0
    raw_threshold: float = 0.0


class HybridHDBSCANClosestFace(ClusteringMethod):
    """Hybrid HDBSCAN+kNN using closest-face matching."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # cluster_selection_epsilon in cosine distance space (was 0.3 Euclidean → 0.045 cosine)
        self.min_cluster_size = self.params.get('min_cluster_size', 2)
        self.min_samples = self.params.get('min_samples', 2)
        self.cluster_selection_epsilon = self.params.get('cluster_selection_epsilon', 0.045)

        self.knn_k = self.params.get('knn_k', 3)
        self.threshold_percentile = self.params.get('threshold_percentile', 90)

        # Thresholds calibrated for cosine distance (1 - cosine_similarity)
        # Converted from Euclidean thresholds using: t_c = (t_e²) / 2
        self.threshold_floor = self.params.get('threshold_floor', 0.045)  # was 0.30 Euclidean
        self.threshold_ceiling = self.params.get('threshold_ceiling', 0.405)  # was 0.90 Euclidean

        self.max_exemplars = self.params.get('max_exemplars', 10)

        self.merge_min_faces = self.params.get('merge_min_faces', 2)
        self.early_exit_multiplier = self.params.get('early_exit_multiplier', 2.0)

        self.attach_min_neighbors = self.params.get('attach_min_neighbors', 1)

    def cluster(
        self,
        features: np.ndarray,
        collect_debug_data: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run hybrid clustering with cross-cluster d3 matching."""
        n_samples = len(features)

        if n_samples == 0:
            return np.array([]), {'n_clusters': 0, 'n_noise': 0}
        if n_samples == 1:
            return np.array([0]), {'n_clusters': 1, 'n_noise': 0}

        features_norm = self.normalize_features(features)

        labels, hdbscan_stats = self._run_hdbscan(features_norm)
        cluster_states = self._compute_cluster_states(labels, features_norm)

        labels, merge_info, merge_decisions = self._merge_closest_face(
            labels, features_norm, cluster_states,
            collect_decisions=collect_debug_data,
        )
        cluster_states = self._compute_cluster_states(labels, features_norm)

        labels, attach_info, attach_decisions = self._attach_closest_face(
            labels, features_norm, cluster_states,
            collect_decisions=collect_debug_data,
        )

        stats = self._compute_stats(
            labels, hdbscan_stats, merge_info, attach_info,
            cluster_states, merge_decisions, attach_decisions,
            collect_debug_data,
        )
        return labels, stats

    # ------------------------------------------------------------------

    def _run_hdbscan(self, features: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        import hdbscan

        # Compute precomputed cosine distance matrix
        dist_matrix = cosine_distance_matrix(features)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='precomputed',
            cluster_selection_method='eom',
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            memory=None,  # Disable joblib caching
        )
        labels = clusterer.fit_predict(dist_matrix)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))
        logger.info(f"  HDBSCAN: {n_clusters} clusters, {n_noise} noise")
        return labels, {'n_clusters': n_clusters, 'n_noise': n_noise}

    def _compute_cluster_states(
        self,
        labels: np.ndarray,
        features: np.ndarray,
    ) -> Dict[int, ClusterD3State]:
        cluster_states = {}

        for label in set(labels):
            if label == -1:
                continue

            indices = np.where(labels == label)[0]
            n_faces = len(indices)
            cluster_features = features[indices]

            if n_faces < 2:
                cluster_states[label] = ClusterD3State(
                    label=label,
                    indices=indices,
                    features=cluster_features,
                    threshold=self.threshold_floor,
                    d3_values=np.array([0.0]),
                    exemplar_indices=indices,
                    exemplar_features=cluster_features,
                    q1=0.0, q3=0.0, iqr=0.0,
                    raw_threshold=self.threshold_floor,
                )
                continue

            distances = cosine_distance_matrix(cluster_features)
            k = min(self.knn_k, n_faces - 1)
            d3_values = []
            for i in range(n_faces):
                sorted_dists = np.sort(distances[i])[1:k + 1]
                d3_values.append(sorted_dists[-1] if len(sorted_dists) > 0 else 0)
            d3_values = np.array(d3_values)

            q1, q3 = float(np.percentile(d3_values, 25)), float(np.percentile(d3_values, 75))
            iqr = q3 - q1
            # Direct percentile of d3 values — same approach as hybrid_knn to avoid
            # high-variance clusters getting amplified thresholds via median+k*IQR.
            raw_threshold = float(np.percentile(d3_values, self.threshold_percentile))
            threshold = float(np.clip(raw_threshold, self.threshold_floor, self.threshold_ceiling))

            n_exemplars = min(self.max_exemplars, n_faces)
            exemplar_local = np.argsort(d3_values)[:n_exemplars]
            exemplar_global = indices[exemplar_local]

            cluster_states[label] = ClusterD3State(
                label=label,
                indices=indices,
                features=cluster_features,
                threshold=threshold,
                d3_values=d3_values,
                exemplar_indices=exemplar_global,
                exemplar_features=features[exemplar_global],
                q1=q1, q3=q3, iqr=iqr,
                raw_threshold=float(raw_threshold),
            )
            logger.debug(
                f"  Cluster {label}: {n_faces} faces, "
                f"Q{self.threshold_percentile}(d3)={raw_threshold:.3f} "
                f"(Q1={q1:.3f}, Q3={q3:.3f}, IQR={iqr:.3f}), T={threshold:.3f}"
            )

        return cluster_states

    def _merge_closest_face(
        self,
        labels: np.ndarray,
        features: np.ndarray,
        cluster_states: Dict[int, ClusterD3State],
        collect_decisions: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any], List[Dict]]:
        merged_labels = labels.copy()
        cluster_ids = sorted(cluster_states.keys())
        merge_decisions: List[Dict] = []

        if len(cluster_ids) <= 1:
            return merged_labels, {'n_merges': 0}, merge_decisions

        parent = {c: c for c in cluster_ids}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False

        merge_count = 0

        for i, c1 in enumerate(cluster_ids):
            for c2 in cluster_ids[i + 1:]:
                if find(c1) == find(c2):
                    continue

                state_a = cluster_states[c1]
                state_b = cluster_states[c2]

                exemplar_dists = cosine_distance_matrix(
                    state_a.exemplar_features, state_b.exemplar_features,
                )
                min_exemplar_dist = float(np.min(exemplar_dists))
                early_exit_threshold = self.early_exit_multiplier * max(
                    state_a.threshold, state_b.threshold
                )

                if min_exemplar_dist > early_exit_threshold:
                    if collect_decisions:
                        merge_decisions.append({
                            'cluster_a': c1,
                            'cluster_b': c2,
                            'threshold': min(state_a.threshold, state_b.threshold),
                            'n_pairs_within': 0,
                            'exemplars_a_involved': 0,
                            'exemplars_b_involved': 0,
                            'min_distance': min_exemplar_dist,
                            'merged': False,
                            'reason': 'early_exit_too_far',
                            'cross_distances': exemplar_dists.tolist(),
                        })
                    continue

                # Each cluster uses its OWN threshold — avoids tight clusters never merging
                # because min(T_A, T_B) dragged the bar down to the tighter cluster's level.
                n_fits = 0
                k_ab = min(self.knn_k, len(state_b.features))
                k_ba = min(self.knn_k, len(state_a.features))

                for face_feat in state_a.features:
                    dists_to_b = cosine_distance_to_set(face_feat, state_b.features)
                    d3_cross = np.sort(dists_to_b)[:k_ab][-1] if k_ab > 0 else float('inf')
                    if d3_cross <= state_a.threshold:
                        n_fits += 1

                for face_feat in state_b.features:
                    dists_to_a = cosine_distance_to_set(face_feat, state_a.features)
                    d3_cross = np.sort(dists_to_a)[:k_ba][-1] if k_ba > 0 else float('inf')
                    if d3_cross <= state_b.threshold:
                        n_fits += 1

                will_merge = n_fits >= self.merge_min_faces
                reason = 'merged' if will_merge else 'not_enough_fits'
                threshold_used = min(state_a.threshold, state_b.threshold)

                # Per-exemplar min distances for debug
                min_dists_a = np.min(exemplar_dists, axis=1).tolist()
                min_dists_b = np.min(exemplar_dists, axis=0).tolist()

                if collect_decisions:
                    merge_decisions.append({
                        'cluster_a': c1,
                        'cluster_b': c2,
                        'threshold': threshold_used,
                        'threshold_a': state_a.threshold,
                        'threshold_b': state_b.threshold,
                        'n_pairs_within': n_fits,
                        'exemplars_a_involved': n_fits,
                        'exemplars_b_involved': n_fits,
                        'min_distance': min_exemplar_dist,
                        'merged': will_merge,
                        'reason': reason,
                        'cross_distances': exemplar_dists.tolist(),
                        'min_dists_a': min_dists_a,
                        'min_dists_b': min_dists_b,
                    })

                if will_merge:
                    if union(c1, c2):
                        merge_count += 1

        # Apply merges
        if merge_count > 0:
            label_mapping: Dict[int, int] = {}
            for c in cluster_ids:
                root = find(c)
                if root not in label_mapping:
                    label_mapping[root] = len(label_mapping)
            for idx, label in enumerate(merged_labels):
                if label >= 0 and label in parent:
                    merged_labels[idx] = label_mapping[find(label)]

        logger.info(f"  Merged {merge_count} cluster pairs")
        return merged_labels, {'n_merges': merge_count}, merge_decisions

    def _attach_closest_face(
        self,
        labels: np.ndarray,
        features: np.ndarray,
        cluster_states: Dict[int, ClusterD3State],
        collect_decisions: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any], List[Dict]]:
        final_labels = labels.copy()
        noise_indices = np.where(labels == -1)[0]
        attach_decisions: List[Dict] = []

        if len(noise_indices) == 0 or len(cluster_states) == 0:
            return final_labels, {
                'n_attached': 0,
                'n_remaining_noise': int(len(noise_indices)),
            }, attach_decisions

        n_attached = 0

        for noise_idx in noise_indices:
            noise_feature = features[noise_idx:noise_idx + 1]
            best_cluster: Optional[int] = None
            best_d3 = float('inf')
            candidates: List[Dict] = []

            for label, state in cluster_states.items():
                dists = cosine_distance_to_set(noise_feature, state.features)
                k = min(self.knn_k, len(state.features))
                d3_cross = float(np.sort(dists)[:k][-1]) if k > 0 else float('inf')
                qualifies = d3_cross <= state.threshold

                if collect_decisions:
                    candidates.append({
                        'cluster': int(label),
                        'threshold': float(state.threshold),
                        'matches': 1 if qualifies else 0,
                        'required': self.attach_min_neighbors,
                        'min_dist': float(np.min(dists)),
                        'qualifies': qualifies,
                    })

                if qualifies and d3_cross < best_d3:
                    best_cluster = label
                    best_d3 = d3_cross

            if best_cluster is not None:
                final_labels[noise_idx] = best_cluster
                n_attached += 1

            if collect_decisions:
                attach_decisions.append({
                    'face_idx': int(noise_idx),
                    'attached_to': int(best_cluster) if best_cluster is not None else None,
                    'candidates': candidates,
                })

        n_remaining = int(len(noise_indices)) - n_attached
        logger.info(f"  Attached {n_attached} noise points, {n_remaining} remain as noise")
        return final_labels, {'n_attached': n_attached, 'n_remaining_noise': n_remaining}, attach_decisions

    def _compute_stats(
        self,
        labels: np.ndarray,
        hdbscan_stats: Dict[str, Any],
        merge_info: Dict[str, Any],
        attach_info: Dict[str, Any],
        cluster_states: Dict[int, ClusterD3State],
        merge_decisions: List[Dict],
        attach_decisions: List[Dict],
        collect_debug_data: bool,
    ) -> Dict[str, Any]:
        unique_labels = set(labels)
        n_clusters = len([l for l in unique_labels if l >= 0])
        n_noise = int(np.sum(labels == -1))

        cluster_sizes = {
            int(l): int(np.sum(labels == l))
            for l in unique_labels if l >= 0
        }

        stats: Dict[str, Any] = {
            'algorithm': 'hybrid_closest_face',
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'cluster_sizes': cluster_sizes,
            'hdbscan': hdbscan_stats,
            'merges': merge_info,
            'attachments': attach_info,
            'params': {
                'min_cluster_size': self.min_cluster_size,
                'knn_k': self.knn_k,
                'threshold_percentile': self.threshold_percentile,
                'threshold_floor': self.threshold_floor,
                'threshold_ceiling': self.threshold_ceiling,
                'max_exemplars': self.max_exemplars,
                'merge_min_faces': self.merge_min_faces,
                'early_exit_multiplier': self.early_exit_multiplier,
                'attach_min_neighbors': self.attach_min_neighbors,
            },
        }

        if collect_debug_data:
            cluster_thresholds: Dict[int, float] = {}
            cluster_exemplars: Dict[int, List[int]] = {}
            cluster_d3_stats: Dict[int, Dict] = {}

            for label, state in cluster_states.items():
                cluster_thresholds[int(label)] = state.threshold
                cluster_exemplars[int(label)] = state.exemplar_indices.tolist()
                cluster_d3_stats[int(label)] = {
                    'q1': state.q1,
                    'q3': state.q3,
                    'iqr': state.iqr,
                    'raw_threshold': state.raw_threshold,
                    'clamped_threshold': state.threshold,
                }

            stats['debug'] = {
                'cluster_thresholds': cluster_thresholds,
                'cluster_exemplars': cluster_exemplars,
                'cluster_d3_stats': cluster_d3_stats,
                'merge_decisions': merge_decisions,
                'attach_decisions': attach_decisions,
            }

        return stats
