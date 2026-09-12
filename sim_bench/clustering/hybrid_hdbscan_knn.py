"""
Hybrid HDBSCAN + Local Cohesion clustering for face identity recognition.

Algorithm:
1. HDBSCAN → initial clusters + noise points
2. For each cluster:
   - Compute d3 (distance to 3rd nearest neighbor) for each face
   - Select E=10 exemplars (faces with smallest d3, most central)
   - Compute pairwise distances between exemplars
   - T = median(exemplar_pairwise) + iqr_multiplier×IQR, clamped to [floor, ceiling]
3. Iteratively:
   a. Merge: clusters if ≥3 cross-exemplar pairs ≤ min(T_A, T_B), with ≥2 distinct exemplars each
   b. Attach: unassigned face → cluster if ≥2 exemplars within T
4. Repeat until no changes

Distance metric: Cosine distance = 1 - cosine_similarity, clipped to [0, 2].
Thresholds are calibrated for cosine distance (not Euclidean).
"""

import logging
from typing import Dict, Any, Tuple, Set, List, Optional
from dataclasses import dataclass, field
import numpy as np

from sim_bench.clustering.base import ClusteringMethod
from sim_bench.clustering.distance_utils import (
    cosine_distance_matrix,
    cosine_distance_pairwise,
    cosine_distance_to_set,
)

logger = logging.getLogger(__name__)


@dataclass
class ClusterState:
    """State for a single cluster during processing."""
    label: int
    indices: np.ndarray
    threshold: float
    exemplar_indices: np.ndarray
    exemplar_embeddings: np.ndarray
    # d3 stats for debug/analysis
    q1: float = 0.0
    q3: float = 0.0
    iqr: float = 0.0
    raw_threshold: float = 0.0  # before clamping


@dataclass
class MergeDecision:
    """Record of a merge decision between two clusters."""
    cluster_a: int
    cluster_b: int
    threshold: float        # threshold that triggered/blocked the decision
    threshold_a: float      # T of cluster A
    threshold_b: float      # T of cluster B
    n_pairs_within: int
    exemplars_a_involved: int
    exemplars_b_involved: int
    min_distance: float
    merged: bool
    reason: str
    cross_distances: Optional[np.ndarray] = None
    min_dists_a: Optional[List[float]] = None   # per A-exemplar: min dist to any B-exemplar
    min_dists_b: Optional[List[float]] = None   # per B-exemplar: min dist to any A-exemplar


@dataclass
class AttachDecision:
    """Record of an attachment decision for a noise point."""
    face_idx: int
    attached_to: Optional[int]
    candidates: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SplitDecision:
    """Record of a split decision for a cluster."""
    cluster_id: int
    original_size: int
    n_components: int
    component_sizes: List[int]
    split: bool
    reason: str


class HybridHDBSCANKNN(ClusteringMethod):
    """Hybrid HDBSCAN + Local Cohesion clustering using Tukey fence threshold."""

    doc_explanation = """
HDBSCAN creates initial clusters, then iteratively merges and attaches using exemplar distances.
Per-cluster threshold T = percentile(exemplar_pairwise_distances), clamped to [floor, ceiling].

Merge Decision: Clusters A and B merge if >=merge_min_pairs exemplar pairs have distance
<= min(T_A, T_B), with >=merge_min_distinct exemplars involved from each side.

Attach Decision: A noise face attaches to a cluster if >=attach_min_exemplars exemplars
are within that cluster's threshold T.

Split Decision (post-processing): Large clusters are checked for internal connectivity using
kNN graph + threshold pruning. If a cluster has multiple disconnected components after pruning
edges below split_threshold, it gets split into separate clusters.
"""

    decision_parameters = {
        "threshold_floor": {
            "description": "Minimum allowed threshold T",
            "default": 0.125,
            "decision_role": "T cannot go below this; prevents over-splitting tight clusters"
        },
        "threshold_ceiling": {
            "description": "Maximum allowed threshold T",
            "default": 0.405,
            "decision_role": "T cannot exceed this; prevents loose clusters from over-merging"
        },
        "threshold_percentile": {
            "description": "Percentile of exemplar pairwise distances for T",
            "default": 90,
            "decision_role": "T = percentile(exemplar_dists, this); higher = looser threshold"
        },
        "merge_min_pairs": {
            "description": "Minimum exemplar pairs within T to merge clusters",
            "default": 3,
            "decision_role": "Merge if cross_pairs_within_T >= this"
        },
        "merge_min_distinct": {
            "description": "Minimum distinct exemplars from each side",
            "default": 2,
            "decision_role": "Both clusters must contribute >= this many exemplars to merge"
        },
        "attach_min_exemplars": {
            "description": "Minimum exemplars within T to attach a noise face",
            "default": 2,
            "decision_role": "Noise attaches if >= this many cluster exemplars within T"
        },
        "knn_k": {
            "description": "K for computing d3 (k-th nearest neighbor distance)",
            "default": 3,
            "decision_role": "Faces with smallest d3 become exemplars (most central)"
        },
        "split_enabled": {
            "description": "Enable post-split using kNN connected components",
            "default": True,
            "decision_role": "If True, large clusters are checked for internal connectivity"
        },
        "split_threshold": {
            "description": "Cosine similarity threshold for split edges (1 - cosine_distance)",
            "default": 0.65,
            "decision_role": "Edges below this similarity are pruned; lower = more aggressive splitting"
        },
        "split_min_cluster_size": {
            "description": "Minimum cluster size to consider for splitting",
            "default": 10,
            "decision_role": "Only clusters with >= this many faces are checked for splitting"
        },
        "split_k": {
            "description": "K neighbors for kNN graph in split phase",
            "default": 20,
            "decision_role": "Each face connects to k nearest neighbors before pruning"
        },
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # PCA dimensionality reduction (None = disabled)
        self.pca_dim = self.params.get('pca_dim', None)

        # HDBSCAN parameters
        # cluster_selection_epsilon in cosine distance space (was 0.3 Euclidean → 0.045 cosine)
        self.min_cluster_size = self.params.get('min_cluster_size', 2)
        self.min_samples = self.params.get('min_samples', 2)
        self.cluster_selection_epsilon = self.params.get('cluster_selection_epsilon', 0.045)

        # Local cohesion parameters
        # Thresholds calibrated for cosine distance (1 - cosine_similarity)
        # Converted from Euclidean thresholds using: t_c = (t_e²) / 2
        self.knn_k = self.params.get('knn_k', 3)
        self.threshold_percentile = self.params.get('threshold_percentile', 90)
        self.threshold_floor = self.params.get('threshold_floor', 0.125)  # was 0.50 Euclidean
        self.threshold_ceiling = self.params.get('threshold_ceiling', 0.405)  # was 0.90 Euclidean

        # Exemplar parameters
        self.max_exemplars = self.params.get('max_exemplars', 10)

        # Attachment parameters
        self.attach_min_exemplars = self.params.get('attach_min_exemplars', 2)

        # Merge parameters
        self.merge_min_pairs = self.params.get('merge_min_pairs', 3)
        self.merge_min_distinct = self.params.get('merge_min_distinct', 2)

        # Iteration
        self.max_iterations = self.params.get('max_iterations', 10)

        # Post-split parameters (kNN connected components)
        self.split_enabled = self.params.get('split_enabled', True)
        self.split_threshold = self.params.get('split_threshold', 0.65)
        self.split_min_cluster_size = self.params.get('split_min_cluster_size', 10)
        self.split_k = self.params.get('split_k', 20)

    def _apply_pca(self, features: np.ndarray) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Apply PCA dimensionality reduction if configured."""
        if self.pca_dim is None:
            return features, None

        from sklearn.decomposition import PCA

        n_samples, n_features = features.shape
        pca_dim = min(self.pca_dim, n_samples, n_features)

        if pca_dim >= n_features:
            logger.debug(f"PCA skipped: pca_dim={pca_dim} >= n_features={n_features}")
            return features, None

        logger.info(f"Applying PCA: {n_features}D -> {pca_dim}D for {n_samples} samples")
        pca = PCA(n_components=pca_dim)
        reduced = pca.fit_transform(features)
        variance_explained = float(np.sum(pca.explained_variance_ratio_))
        logger.info(f"PCA variance explained: {variance_explained:.2%}")

        return reduced, {'pca_dim': pca_dim, 'variance_explained': variance_explained}

    def cluster(
        self,
        features: np.ndarray,
        collect_debug_data: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run hybrid clustering.

        Args:
            features: Face embedding vectors (N x D)
            collect_debug_data: If True, collect detailed decision logs for debugging

        Returns:
            labels: Cluster assignments (-1 for noise)
            stats: Statistics dict, with debug data if collect_debug_data=True

        Raises:
            ValueError: If features contain NaN/Inf values or zero vectors
        """
        n_samples = len(features)

        if n_samples == 0:
            return np.array([]), {'n_clusters': 0, 'n_noise': 0}

        if n_samples == 1:
            return np.array([0]), {'n_clusters': 1, 'n_noise': 0}

        # Apply PCA if configured (before validation/normalization)
        features, pca_stats = self._apply_pca(features)

        # Input validation
        features = self._validate_features(features)

        # Normalize for cosine distance
        features_norm = self.normalize_features(features)

        # Stage 1: HDBSCAN
        logger.info(f"Stage 1: HDBSCAN (min_cluster_size={self.min_cluster_size})")
        labels, hdbscan_stats = self._run_hdbscan(features_norm)

        # Stage 2: Iterative merge + attach
        total_merges = 0
        total_attached = 0
        all_merge_decisions: List[MergeDecision] = []
        all_attach_decisions: List[AttachDecision] = []

        for iteration in range(self.max_iterations):
            # Compute cluster states (threshold + exemplars)
            cluster_states = self._compute_cluster_states(labels, features_norm)

            if not cluster_states:
                logger.info(f"  Iteration {iteration + 1}: No clusters, stopping")
                break

            # Merge clusters
            labels, n_merges, merge_decisions = self._merge_clusters(
                labels, cluster_states, features_norm, collect_decisions=collect_debug_data
            )
            total_merges += n_merges
            all_merge_decisions.extend(merge_decisions)

            # Recompute states after merge
            if n_merges > 0:
                cluster_states = self._compute_cluster_states(labels, features_norm)

            # Attach noise points
            labels, n_attached, attach_decisions = self._attach_noise(
                labels, cluster_states, features_norm, collect_decisions=collect_debug_data
            )
            total_attached += n_attached
            all_attach_decisions.extend(attach_decisions)

            logger.info(f"  Iteration {iteration + 1}: {n_merges} merges, {n_attached} attached")

            if n_merges == 0 and n_attached == 0:
                logger.info(f"  Converged after {iteration + 1} iterations")
                break

        # Stage 3: Split loosely-connected clusters using kNN connected components
        all_split_decisions: List[SplitDecision] = []
        total_splits = 0

        if self.split_enabled:
            logger.info(f"Stage 3: Split check (threshold={self.split_threshold}, k={self.split_k})")
            labels, total_splits, all_split_decisions = self._split_clusters(
                labels, features_norm, collect_decisions=collect_debug_data
            )
            if total_splits > 0:
                logger.info(f"  Split {total_splits} clusters")
            else:
                logger.info(f"  No clusters split")

        # Compute final cluster states for stats
        final_cluster_states = self._compute_cluster_states(labels, features_norm)

        # Final stats
        stats = self._compute_final_stats(
            labels, features_norm, hdbscan_stats, total_merges, total_attached, total_splits,
            final_cluster_states, all_merge_decisions, all_attach_decisions, all_split_decisions,
            collect_debug_data, pca_stats
        )
        return labels, stats

    def _validate_features(self, features: np.ndarray) -> np.ndarray:
        """Validate input features and handle edge cases.

        Args:
            features: Feature matrix [n_samples, n_features]

        Returns:
            Validated features (may have some rows removed)

        Raises:
            ValueError: If features is not a 2D array, or contains NaN/Inf or all zero vectors
        """
        # Shape guard: everything below assumes a 2D [n_samples, n_features] matrix
        # (e.g. the NaN check scans axis=1). Reject a 1D array here with a clear
        # message instead of letting numpy raise a cryptic AxisError downstream.
        if features.ndim != 2:
            raise ValueError(
                f"Features must be a 2D array [n_samples, n_features], "
                f"got {features.ndim}D with shape {features.shape}."
            )

        # Check for NaN/Inf
        if np.any(np.isnan(features)):
            nan_count = np.sum(np.isnan(features).any(axis=1))
            raise ValueError(
                f"Features contain {nan_count} rows with NaN values. "
                "Check embedding extraction for corrupted data."
            )

        if np.any(np.isinf(features)):
            inf_count = np.sum(np.isinf(features).any(axis=1))
            raise ValueError(
                f"Features contain {inf_count} rows with Inf values. "
                "Check embedding extraction for overflow."
            )

        # Check for zero vectors (would cause NaN after normalization)
        norms = np.linalg.norm(features, axis=1)
        zero_mask = norms < 1e-10
        zero_count = np.sum(zero_mask)

        if zero_count > 0:
            logger.warning(
                f"Found {zero_count} zero-vector embeddings. "
                "These may be cached from a previous bug - consider clearing cache."
            )
            if zero_count == len(features):
                raise ValueError(
                    "All embeddings are zero vectors. "
                    "Clear face embedding cache and re-run pipeline."
                )

        # Check dimensions
        if features.ndim != 2:
            raise ValueError(
                f"Features must be 2D array [n_samples, n_features], "
                f"got shape {features.shape}"
            )

        return features

    def _run_hdbscan(self, features: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run HDBSCAN to get initial clusters."""
        import hdbscan

        # Compute precomputed cosine distance matrix
        dist_matrix = cosine_distance_matrix(features)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='precomputed',
            cluster_selection_method='eom',
            cluster_selection_epsilon=self.cluster_selection_epsilon,
        )
        labels = clusterer.fit_predict(dist_matrix)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))

        logger.info(f"  HDBSCAN: {n_clusters} clusters, {n_noise} noise points")
        return labels, {'n_clusters': n_clusters, 'n_noise': n_noise}

    def _compute_cluster_states(
        self,
        labels: np.ndarray,
        features: np.ndarray
    ) -> Dict[int, ClusterState]:
        """Compute threshold and exemplars for each cluster using exemplar pairwise distances."""
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

            # Compute threshold from exemplar pairwise distances.
            # Use a direct percentile rather than median+k*IQR to avoid high-variance
            # clusters (noisy buckets) getting amplified thresholds that act as black holes.
            if len(exemplar_embeddings) < 2:
                raw_threshold = self.threshold_floor
                q1 = q3 = iqr = 0.0
            else:
                exemplar_dists = cosine_distance_pairwise(exemplar_embeddings)
                q1, q3 = np.percentile(exemplar_dists, [25, 75])
                iqr = q3 - q1
                raw_threshold = float(np.percentile(exemplar_dists, self.threshold_percentile))

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

            logger.debug(f"  Cluster {label}: {n_faces} faces, {len(exemplar_global_indices)} exemplars, "
                        f"Q{self.threshold_percentile}={raw_threshold:.3f} "
                        f"(Q1={q1:.3f}, Q3={q3:.3f}, IQR={iqr:.3f}), T={threshold:.3f}")

        return cluster_states

    def _check_merge(
        self,
        cross_dists: np.ndarray,
        t_a: float,
        t_b: float,
    ) -> Tuple[bool, str, float]:
        """Bidirectional merge check.

        Returns (should_merge, reason, threshold_used).
        Tries T_A first ("B fits A"), then T_B ("A fits B").
        """
        last_reason = 'not_enough_pairs'
        for threshold, direction in ((t_a, 'b_fits_a'), (t_b, 'a_fits_b')):
            pairs_within: Set[Tuple[int, int]] = set()
            involved_a: Set[int] = set()
            involved_b: Set[int] = set()
            for ia in range(cross_dists.shape[0]):
                for ib in range(cross_dists.shape[1]):
                    if cross_dists[ia, ib] <= threshold:
                        pairs_within.add((ia, ib))
                        involved_a.add(ia)
                        involved_b.add(ib)
            if len(pairs_within) < self.merge_min_pairs:
                last_reason = 'not_enough_pairs'
            elif len(involved_a) < self.merge_min_distinct:
                last_reason = 'not_enough_distinct_a'
            elif len(involved_b) < self.merge_min_distinct:
                last_reason = 'not_enough_distinct_b'
            else:
                return True, f'merged_{direction}', threshold
        return False, last_reason, min(t_a, t_b)

    def _merge_clusters(
        self,
        labels: np.ndarray,
        cluster_states: Dict[int, ClusterState],
        features: np.ndarray,
        collect_decisions: bool = False
    ) -> Tuple[np.ndarray, int, List[MergeDecision]]:
        """Merge clusters using bidirectional threshold check.

        A pair merges if EITHER direction satisfies the criteria:
          - "B fits A": ≥merge_min_pairs cross-pairs ≤ T_A, ≥merge_min_distinct from each side
          - "A fits B": ≥merge_min_pairs cross-pairs ≤ T_B, ≥merge_min_distinct from each side

        This prevents tight clusters (small T) from never merging due to min(T_A, T_B).
        """
        merged_labels = labels.copy()
        cluster_ids = sorted(cluster_states.keys())
        merge_decisions: List[MergeDecision] = []

        if len(cluster_ids) <= 1:
            return merged_labels, 0, merge_decisions

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

                cross_dists = cosine_distance_matrix(
                    state_a.exemplar_embeddings,
                    state_b.exemplar_embeddings,
                )
                min_distance = float(np.min(cross_dists))
                min_dists_a = np.min(cross_dists, axis=1).tolist()   # per A-exemplar
                min_dists_b = np.min(cross_dists, axis=0).tolist()   # per B-exemplar

                should_merge, reason, threshold_used = self._check_merge(
                    cross_dists, state_a.threshold, state_b.threshold
                )

                if collect_decisions:
                    merge_decisions.append(MergeDecision(
                        cluster_a=c1,
                        cluster_b=c2,
                        threshold=threshold_used,
                        threshold_a=state_a.threshold,
                        threshold_b=state_b.threshold,
                        n_pairs_within=0,
                        exemplars_a_involved=0,
                        exemplars_b_involved=0,
                        min_distance=min_distance,
                        merged=should_merge,
                        reason=reason,
                        cross_distances=cross_dists.copy(),
                        min_dists_a=min_dists_a,
                        min_dists_b=min_dists_b,
                    ))

                if should_merge:
                    union(c1, c2)
                    n_merges += 1
                    logger.debug(f"  Merge {c1}+{c2}: T={threshold_used:.3f} ({reason})")

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

        return merged_labels, n_merges, merge_decisions

    def _attach_noise(
        self,
        labels: np.ndarray,
        cluster_states: Dict[int, ClusterState],
        features: np.ndarray,
        collect_decisions: bool = False
    ) -> Tuple[np.ndarray, int, List[AttachDecision]]:
        """Attach noise points if m≥2 exemplars within T (or all if cluster < 4)."""
        final_labels = labels.copy()
        noise_indices = np.where(labels == -1)[0]
        attach_decisions: List[AttachDecision] = []

        if len(noise_indices) == 0 or len(cluster_states) == 0:
            return final_labels, 0, attach_decisions

        n_attached = 0

        for noise_idx in noise_indices:
            noise_embedding = features[noise_idx:noise_idx + 1]
            best_cluster = None
            best_match_count = 0
            best_min_dist = float('inf')
            candidates: List[Dict[str, Any]] = []

            for label, state in cluster_states.items():
                # Compute cosine distances to exemplars
                distances = cosine_distance_to_set(noise_embedding, state.exemplar_embeddings)

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

                qualifies = within_threshold >= required_matches

                # Collect candidate info for debugging
                if collect_decisions:
                    candidates.append({
                        'cluster': int(label),
                        'threshold': float(state.threshold),
                        'matches': within_threshold,
                        'required': required_matches,
                        'min_dist': min_dist,
                        'qualifies': qualifies,
                        'exemplar_distances': distances.tolist()
                    })

                # Check if this cluster qualifies
                if qualifies:
                    # Prefer more matches, then closer distance
                    if (within_threshold > best_match_count or
                        (within_threshold == best_match_count and min_dist < best_min_dist)):
                        best_cluster = label
                        best_match_count = within_threshold
                        best_min_dist = min_dist

            if best_cluster is not None:
                final_labels[noise_idx] = best_cluster
                n_attached += 1

            # Record decision
            if collect_decisions:
                attach_decisions.append(AttachDecision(
                    face_idx=int(noise_idx),
                    attached_to=int(best_cluster) if best_cluster is not None else None,
                    candidates=candidates
                ))

        return final_labels, n_attached, attach_decisions

    def _split_clusters(
        self,
        labels: np.ndarray,
        features: np.ndarray,
        collect_decisions: bool = False
    ) -> Tuple[np.ndarray, int, List[SplitDecision]]:
        """Split loosely-connected clusters using kNN + threshold connected components.

        For each cluster >= split_min_cluster_size:
        1. Build kNN graph (k = split_k neighbors per face)
        2. Prune edges where cosine_similarity < split_threshold
        3. Find connected components
        4. If multiple components, split into separate clusters

        Returns:
            new_labels: Updated cluster labels
            n_splits: Number of clusters that were split
            split_decisions: List of split decisions for debugging
        """
        from collections import defaultdict

        new_labels = labels.copy()
        split_decisions: List[SplitDecision] = []

        if not self.split_enabled:
            return new_labels, 0, split_decisions

        # Get current max label for assigning new cluster IDs
        max_label = max(labels) if len(labels) > 0 else -1
        next_label = max_label + 1

        unique_labels = set(labels) - {-1}
        n_splits = 0

        for cluster_label in sorted(unique_labels):
            cluster_mask = labels == cluster_label
            cluster_indices = np.where(cluster_mask)[0]
            cluster_size = len(cluster_indices)

            # Skip small clusters
            if cluster_size < self.split_min_cluster_size:
                if collect_decisions:
                    split_decisions.append(SplitDecision(
                        cluster_id=int(cluster_label),
                        original_size=cluster_size,
                        n_components=1,
                        component_sizes=[cluster_size],
                        split=False,
                        reason=f"size {cluster_size} < min {self.split_min_cluster_size}"
                    ))
                continue

            cluster_features = features[cluster_indices]

            # Build kNN graph within cluster
            # Use cosine similarity (1 - cosine_distance)
            sim_matrix = 1.0 - cosine_distance_matrix(cluster_features)

            # Build adjacency with pruning
            k = min(self.split_k, cluster_size - 1)
            adjacency = defaultdict(set)

            for i in range(cluster_size):
                # Get k nearest neighbors (excluding self)
                sims = sim_matrix[i].copy()
                sims[i] = -np.inf  # Exclude self
                top_k = np.argsort(sims)[-k:]

                for j in top_k:
                    # Only keep edge if similarity >= threshold
                    if sims[j] >= self.split_threshold:
                        adjacency[i].add(j)
                        adjacency[j].add(i)

            # Find connected components using BFS
            visited = set()
            components = []

            for start in range(cluster_size):
                if start in visited:
                    continue

                component = []
                queue = [start]

                while queue:
                    node = queue.pop(0)
                    if node in visited:
                        continue
                    visited.add(node)
                    component.append(node)

                    for neighbor in adjacency.get(node, []):
                        if neighbor not in visited:
                            queue.append(neighbor)

                components.append(component)

            # Check if we should split
            n_components = len(components)
            component_sizes = sorted([len(c) for c in components], reverse=True)

            if n_components > 1:
                # Split! Assign new labels to components
                # Keep the largest component with original label
                components_sorted = sorted(components, key=len, reverse=True)

                for comp_idx, component in enumerate(components_sorted):
                    if comp_idx == 0:
                        # Keep original label for largest component
                        assigned_label = cluster_label
                    else:
                        # Assign new label
                        assigned_label = next_label
                        next_label += 1

                    for local_idx in component:
                        global_idx = cluster_indices[local_idx]
                        new_labels[global_idx] = assigned_label

                n_splits += 1
                logger.info(f"  Split cluster {cluster_label} ({cluster_size} faces) "
                           f"into {n_components} components: {component_sizes}")

                if collect_decisions:
                    split_decisions.append(SplitDecision(
                        cluster_id=int(cluster_label),
                        original_size=cluster_size,
                        n_components=n_components,
                        component_sizes=component_sizes,
                        split=True,
                        reason=f"found {n_components} disconnected components"
                    ))
            else:
                if collect_decisions:
                    split_decisions.append(SplitDecision(
                        cluster_id=int(cluster_label),
                        original_size=cluster_size,
                        n_components=1,
                        component_sizes=[cluster_size],
                        split=False,
                        reason="single connected component"
                    ))

        return new_labels, n_splits, split_decisions

    def _compute_final_stats(
        self,
        labels: np.ndarray,
        features: np.ndarray,
        hdbscan_stats: Dict[str, Any],
        total_merges: int,
        total_attached: int,
        total_splits: int,
        final_cluster_states: Dict[int, ClusterState],
        all_merge_decisions: List[MergeDecision],
        all_attach_decisions: List[AttachDecision],
        all_split_decisions: List[SplitDecision],
        collect_debug_data: bool,
        pca_stats: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compute final statistics."""
        unique_labels = set(labels)
        n_clusters = len([l for l in unique_labels if l >= 0])
        n_noise = int(np.sum(labels == -1))

        cluster_sizes = {}
        for label in unique_labels:
            if label >= 0:
                cluster_sizes[int(label)] = int(np.sum(labels == label))

        stats = {
            'algorithm': 'hybrid_hdbscan_knn',
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'cluster_sizes': cluster_sizes,
            'hdbscan': hdbscan_stats,
            'total_merges': total_merges,
            'total_attached': total_attached,
            'total_splits': total_splits,
            'params': {
                'pca_dim': self.pca_dim,
                'min_cluster_size': self.min_cluster_size,
                'min_samples': self.min_samples,
                'cluster_selection_epsilon': self.cluster_selection_epsilon,
                'knn_k': self.knn_k,
                'threshold_percentile': self.threshold_percentile,
                'threshold_floor': self.threshold_floor,
                'threshold_ceiling': self.threshold_ceiling,
                'max_exemplars': self.max_exemplars,
                'attach_min_exemplars': self.attach_min_exemplars,
                'merge_min_pairs': self.merge_min_pairs,
                'merge_min_distinct': self.merge_min_distinct,
                'max_iterations': self.max_iterations,
                'split_enabled': self.split_enabled,
                'split_threshold': self.split_threshold,
                'split_min_cluster_size': self.split_min_cluster_size,
                'split_k': self.split_k,
            }
        }

        if pca_stats:
            stats['pca'] = pca_stats

        # Add debug data if requested
        if collect_debug_data:
            # Cluster thresholds and d3 stats
            cluster_thresholds = {}
            cluster_exemplars = {}
            cluster_d3_stats = {}

            for label, state in final_cluster_states.items():
                cluster_thresholds[int(label)] = state.threshold
                cluster_exemplars[int(label)] = state.exemplar_indices.tolist()
                cluster_d3_stats[int(label)] = {
                    'q1': state.q1,
                    'q3': state.q3,
                    'iqr': state.iqr,
                    'raw_threshold': state.raw_threshold,
                    'clamped_threshold': state.threshold
                }

            # Convert merge decisions to dicts (without numpy arrays for JSON serialization)
            merge_decisions_list = []
            for md in all_merge_decisions:
                merge_decisions_list.append({
                    'cluster_a': md.cluster_a,
                    'cluster_b': md.cluster_b,
                    'threshold': md.threshold,
                    'threshold_a': md.threshold_a,
                    'threshold_b': md.threshold_b,
                    'n_pairs_within': md.n_pairs_within,
                    'exemplars_a_involved': md.exemplars_a_involved,
                    'exemplars_b_involved': md.exemplars_b_involved,
                    'min_distance': md.min_distance,
                    'merged': md.merged,
                    'reason': md.reason,
                    'cross_distances': md.cross_distances.tolist() if md.cross_distances is not None else None,
                    'min_dists_a': md.min_dists_a,
                    'min_dists_b': md.min_dists_b,
                })

            # Convert attach decisions to dicts
            attach_decisions_list = []
            for ad in all_attach_decisions:
                attach_decisions_list.append({
                    'face_idx': ad.face_idx,
                    'attached_to': ad.attached_to,
                    'candidates': ad.candidates
                })

            # Convert split decisions to dicts
            split_decisions_list = []
            for sd in all_split_decisions:
                split_decisions_list.append({
                    'cluster_id': sd.cluster_id,
                    'original_size': sd.original_size,
                    'n_components': sd.n_components,
                    'component_sizes': sd.component_sizes,
                    'split': sd.split,
                    'reason': sd.reason
                })

            stats['debug'] = {
                'cluster_thresholds': cluster_thresholds,
                'cluster_exemplars': cluster_exemplars,
                'cluster_d3_stats': cluster_d3_stats,
                'merge_decisions': merge_decisions_list,
                'attach_decisions': attach_decisions_list,
                'split_decisions': split_decisions_list
            }

        # Store last run info for UI display
        cluster_thresholds_summary = {
            int(label): state.threshold
            for label, state in final_cluster_states.items()
        }
        self.last_run_info = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'total_merges': total_merges,
            'total_attached': total_attached,
            'total_splits': total_splits,
            'threshold_floor': self.threshold_floor,
            'threshold_ceiling': self.threshold_ceiling,
            'merge_min_pairs': self.merge_min_pairs,
            'attach_min_exemplars': self.attach_min_exemplars,
            'split_threshold': self.split_threshold,
            'cluster_thresholds': cluster_thresholds_summary,
        }

        return stats
