"""
Hybrid HDBSCAN + Local Cohesion clustering for face identity recognition.

Algorithm:
1. HDBSCAN → initial clusters
2. For each cluster:
   - Compute d3 (distance to 3rd nearest neighbor) for each face
   - T = Q3(d3) + 1.5×IQR(d3), clamped to [0.30, 0.45]
   - Select E=10 exemplars (faces with smallest d3)
3. Iteratively:
   a. Attach: unassigned face → cluster if m≥2 exemplars within T
   b. Merge: clusters if L≥3 cross-exemplar pairs ≤ min(T_A, T_B), ≥2 distinct each
4. Repeat until no changes
"""

import logging
from typing import Dict, Any, Tuple, Set
from dataclasses import dataclass
import numpy as np
from scipy.spatial.distance import cdist

from sim_bench.clustering.base import ClusteringMethod

logger = logging.getLogger(__name__)


@dataclass
class ClusterState:
    """State for a single cluster during processing."""
    label: int
    indices: np.ndarray
    threshold: float
    exemplar_indices: np.ndarray
    exemplar_embeddings: np.ndarray


class HybridHDBSCANKNN(ClusteringMethod):
    """Hybrid HDBSCAN + Local Cohesion clustering using Tukey fence threshold."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # HDBSCAN parameters
        self.min_cluster_size = self.params.get('min_cluster_size', 2)
        self.min_samples = self.params.get('min_samples', 2)
        self.cluster_selection_epsilon = self.params.get('cluster_selection_epsilon', 0.3)

        # Local cohesion parameters
        self.knn_k = self.params.get('knn_k', 3)
        self.threshold_floor = self.params.get('threshold_floor', 0.50)
        self.threshold_ceiling = self.params.get('threshold_ceiling', 0.90)

        # Exemplar parameters
        self.max_exemplars = self.params.get('max_exemplars', 10)

        # Attachment parameters
        self.attach_min_exemplars = self.params.get('attach_min_exemplars', 2)

        # Merge parameters
        self.merge_min_pairs = self.params.get('merge_min_pairs', 3)
        self.merge_min_distinct = self.params.get('merge_min_distinct', 2)

        # Iteration
        self.max_iterations = self.params.get('max_iterations', 10)

    def cluster(self, features: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run hybrid clustering."""
        n_samples = len(features)

        if n_samples == 0:
            return np.array([]), {'n_clusters': 0, 'n_noise': 0}

        if n_samples == 1:
            return np.array([0]), {'n_clusters': 1, 'n_noise': 0}

        # Normalize for cosine distance
        features_norm = self.normalize_features(features)

        # Stage 1: HDBSCAN
        logger.info(f"Stage 1: HDBSCAN (min_cluster_size={self.min_cluster_size})")
        labels, hdbscan_stats = self._run_hdbscan(features_norm)

        # Stage 2: Iterative merge + attach
        total_merges = 0
        total_attached = 0

        for iteration in range(self.max_iterations):
            # Compute cluster states (threshold + exemplars)
            cluster_states = self._compute_cluster_states(labels, features_norm)

            if not cluster_states:
                logger.info(f"  Iteration {iteration + 1}: No clusters, stopping")
                break

            # Merge clusters
            labels, n_merges = self._merge_clusters(labels, cluster_states, features_norm)
            total_merges += n_merges

            # Recompute states after merge
            if n_merges > 0:
                cluster_states = self._compute_cluster_states(labels, features_norm)

            # Attach noise points
            labels, n_attached = self._attach_noise(labels, cluster_states, features_norm)
            total_attached += n_attached

            logger.info(f"  Iteration {iteration + 1}: {n_merges} merges, {n_attached} attached")

            if n_merges == 0 and n_attached == 0:
                logger.info(f"  Converged after {iteration + 1} iterations")
                break

        # Final stats
        stats = self._compute_final_stats(labels, hdbscan_stats, total_merges, total_attached)
        return labels, stats

    def _run_hdbscan(self, features: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run HDBSCAN to get initial clusters."""
        import hdbscan

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            cluster_selection_epsilon=self.cluster_selection_epsilon,
        )
        labels = clusterer.fit_predict(features)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))

        logger.info(f"  HDBSCAN: {n_clusters} clusters, {n_noise} noise points")
        return labels, {'n_clusters': n_clusters, 'n_noise': n_noise}

    def _compute_cluster_states(
        self,
        labels: np.ndarray,
        features: np.ndarray
    ) -> Dict[int, ClusterState]:
        """Compute threshold and exemplars for each cluster using d3 and Tukey fence."""
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
                    exemplar_embeddings=cluster_features
                )
                continue

            # Compute pairwise distances
            distances = cdist(cluster_features, cluster_features, metric='euclidean')

            # For each face, compute d3 (distance to 3rd nearest neighbor)
            k = min(self.knn_k, n_faces - 1)
            d3_values = []

            for i in range(n_faces):
                sorted_dists = np.sort(distances[i])[1:k + 1]  # Exclude self
                # Use the k-th neighbor distance (d3 when k=3)
                d3_values.append(sorted_dists[-1] if len(sorted_dists) > 0 else 0)

            d3_values = np.array(d3_values)

            # Tukey fence: T = Q3 + 1.5 × IQR
            q1, q3 = np.percentile(d3_values, [25, 75])
            iqr = q3 - q1
            threshold = q3 + 1.5 * iqr

            # Clamp to [floor, ceiling]
            threshold = max(threshold, self.threshold_floor)
            threshold = min(threshold, self.threshold_ceiling)

            # Select exemplars: faces with smallest d3 (most core-like)
            n_exemplars = min(self.max_exemplars, n_faces)
            exemplar_local_indices = np.argsort(d3_values)[:n_exemplars]
            exemplar_global_indices = indices[exemplar_local_indices]

            cluster_states[label] = ClusterState(
                label=label,
                indices=indices,
                threshold=float(threshold),
                exemplar_indices=exemplar_global_indices,
                exemplar_embeddings=features[exemplar_global_indices]
            )

            logger.debug(f"  Cluster {label}: {n_faces} faces, Q3={q3:.3f}, IQR={iqr:.3f}, "
                        f"T={threshold:.3f}, {len(exemplar_global_indices)} exemplars")

        return cluster_states

    def _merge_clusters(
        self,
        labels: np.ndarray,
        cluster_states: Dict[int, ClusterState],
        features: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """Merge clusters if L≥3 cross-exemplar pairs ≤ min(T_A, T_B), ≥2 distinct each."""
        merged_labels = labels.copy()
        cluster_ids = sorted(cluster_states.keys())

        if len(cluster_ids) <= 1:
            return merged_labels, 0

        # Union-find
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

        n_merges = 0

        for i, c1 in enumerate(cluster_ids):
            for c2 in cluster_ids[i + 1:]:
                if find(c1) == find(c2):
                    continue

                state_a = cluster_states[c1]
                state_b = cluster_states[c2]

                # Compute cross-distances between exemplars
                cross_dists = cdist(
                    state_a.exemplar_embeddings,
                    state_b.exemplar_embeddings,
                    metric='euclidean'
                )

                # Merge threshold = min of both
                merge_threshold = min(state_a.threshold, state_b.threshold)

                # Count pairs within threshold
                pairs_within: Set[Tuple[int, int]] = set()
                exemplars_a_involved: Set[int] = set()
                exemplars_b_involved: Set[int] = set()

                for idx_a in range(len(state_a.exemplar_indices)):
                    for idx_b in range(len(state_b.exemplar_indices)):
                        if cross_dists[idx_a, idx_b] <= merge_threshold:
                            pairs_within.add((idx_a, idx_b))
                            exemplars_a_involved.add(idx_a)
                            exemplars_b_involved.add(idx_b)

                # Check merge conditions
                should_merge = (
                    len(pairs_within) >= self.merge_min_pairs and
                    len(exemplars_a_involved) >= self.merge_min_distinct and
                    len(exemplars_b_involved) >= self.merge_min_distinct
                )

                if should_merge:
                    union(c1, c2)
                    n_merges += 1
                    logger.debug(f"  Merge {c1}+{c2}: {len(pairs_within)} pairs, "
                                f"{len(exemplars_a_involved)}/{len(exemplars_b_involved)} distinct, "
                                f"T={merge_threshold:.3f}")

        # Apply merges
        if n_merges > 0:
            label_mapping = {}
            for c in cluster_ids:
                root = find(c)
                if root not in label_mapping:
                    label_mapping[root] = len(label_mapping)

            for i, lbl in enumerate(merged_labels):
                if lbl >= 0 and lbl in parent:
                    merged_labels[i] = label_mapping[find(lbl)]

        return merged_labels, n_merges

    def _attach_noise(
        self,
        labels: np.ndarray,
        cluster_states: Dict[int, ClusterState],
        features: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """Attach noise points if m≥2 exemplars within T (or all if cluster < 4)."""
        final_labels = labels.copy()
        noise_indices = np.where(labels == -1)[0]

        if len(noise_indices) == 0 or len(cluster_states) == 0:
            return final_labels, 0

        n_attached = 0

        for noise_idx in noise_indices:
            noise_embedding = features[noise_idx:noise_idx + 1]
            best_cluster = None
            best_match_count = 0
            best_min_dist = float('inf')

            for label, state in cluster_states.items():
                # Compute distances to exemplars
                distances = cdist(noise_embedding, state.exemplar_embeddings, metric='euclidean')[0]

                # Count exemplars within threshold
                within_threshold = int(np.sum(distances <= state.threshold))
                min_dist = float(np.min(distances))

                # Determine required matches
                n_exemplars = len(state.exemplar_indices)
                if n_exemplars < self.attach_min_exemplars:
                    # Small cluster: require at least 1 exemplar
                    required_matches = 1
                else:
                    # Normal: require at least m exemplars
                    required_matches = self.attach_min_exemplars

                # Check if this cluster qualifies
                if within_threshold >= required_matches:
                    # Prefer more matches, then closer distance
                    if (within_threshold > best_match_count or
                        (within_threshold == best_match_count and min_dist < best_min_dist)):
                        best_cluster = label
                        best_match_count = within_threshold
                        best_min_dist = min_dist

            if best_cluster is not None:
                final_labels[noise_idx] = best_cluster
                n_attached += 1

        return final_labels, n_attached

    def _compute_final_stats(
        self,
        labels: np.ndarray,
        hdbscan_stats: Dict[str, Any],
        total_merges: int,
        total_attached: int
    ) -> Dict[str, Any]:
        """Compute final statistics."""
        unique_labels = set(labels)
        n_clusters = len([l for l in unique_labels if l >= 0])
        n_noise = int(np.sum(labels == -1))

        cluster_sizes = {}
        for label in unique_labels:
            if label >= 0:
                cluster_sizes[int(label)] = int(np.sum(labels == label))

        return {
            'algorithm': 'hybrid_hdbscan_knn',
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'cluster_sizes': cluster_sizes,
            'hdbscan': hdbscan_stats,
            'total_merges': total_merges,
            'total_attached': total_attached,
            'params': {
                'min_cluster_size': self.min_cluster_size,
                'min_samples': self.min_samples,
                'cluster_selection_epsilon': self.cluster_selection_epsilon,
                'knn_k': self.knn_k,
                'threshold_floor': self.threshold_floor,
                'threshold_ceiling': self.threshold_ceiling,
                'max_exemplars': self.max_exemplars,
                'attach_min_exemplars': self.attach_min_exemplars,
                'merge_min_pairs': self.merge_min_pairs,
                'merge_min_distinct': self.merge_min_distinct,
                'max_iterations': self.max_iterations,
            }
        }
