"""
Mutual K-Nearest Neighbors Two-Stage Clustering.

Algorithm:
1. Stage 1 - Initial Clustering:
   - Build mutual kNN graph (no/loose distance threshold)
   - Find connected components = initial clusters

2. Stage 2 - Iterative Refinement:
   - Prune: Validate each sample's membership using pluggable strategy
   - Reassign: Try to place unassigned samples in valid clusters
   - Repeat until convergence or max iterations

The pruning strategy is pluggable, allowing different membership criteria.
Default: RedundantSupportStrategy (support count + separation margin)
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import logging

from sim_bench.clustering.base import ClusteringMethod
from sim_bench.clustering.distance_utils import (
    cosine_distance_matrix,
    compute_cluster_members_from_labels,
)
from sim_bench.clustering.pruning_strategies import (
    PruningStrategy,
    RedundantSupportStrategy,
    create_pruning_strategy,
)

logger = logging.getLogger(__name__)


class MutualKNNTwoStageClusterer(ClusteringMethod):
    """
    Two-stage clustering: mutual kNN graph + iterative pruning/reassignment.

    Stage 1: Build clusters from mutual kNN connected components
    Stage 2: Iteratively prune weak members and reassign to valid clusters
    """

    doc_explanation = """
Two-stage clustering separates graph construction from membership validation.

Stage 1: Build mutual kNN graph (loose/no threshold) and find connected components.
This captures potential cluster structure without premature fragmentation.

Stage 2: Iteratively validate membership using pluggable pruning strategy.
Default strategy requires either:
- Redundant support: multiple neighbors within threshold, OR
- Clear separation: much closer to this cluster than any other

Unassigned samples try other clusters; if none fit, they become noise.
Repeats until convergence or max iterations.
"""

    decision_parameters = {
        "k": {
            "description": "Number of nearest neighbors for kNN graph",
            "default": 10,
            "decision_role": "Controls initial graph connectivity"
        },
        "initial_threshold": {
            "description": "Initial threshold for Stage 1 (None = no threshold)",
            "default": None,
            "decision_role": "Very loose filter for extreme outliers only"
        },
        "base_threshold": {
            "description": "Base distance threshold for pruning (X)",
            "default": 0.45,
            "decision_role": "Core threshold for membership decisions"
        },
        "relaxation_alpha": {
            "description": "Relaxation factor for base condition",
            "default": 1.1,
            "decision_role": "Allows up to alpha*X distance"
        },
        "support_beta": {
            "description": "Support radius factor",
            "default": 1.05,
            "decision_role": "Neighbors within beta*X count as support"
        },
        "min_support": {
            "description": "Minimum supporting neighbors",
            "default": 2,
            "decision_role": "Redundancy requirement"
        },
        "separation_delta": {
            "description": "Minimum separation to next cluster",
            "default": 0.15,
            "decision_role": "Alternative path: clear separation"
        },
        "max_iterations": {
            "description": "Maximum refinement iterations",
            "default": 10,
            "decision_role": "Prevents infinite loops"
        },
        "convergence_threshold": {
            "description": "Stop if changes <= this value",
            "default": 0,
            "decision_role": "Early termination condition"
        },
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Stage 1 params
        self.k = self.params.get('k', 10)
        self.initial_threshold = self.params.get('initial_threshold', None)

        # Stage 2 params
        self.max_iterations = self.params.get('max_iterations', 10)
        self.convergence_threshold = self.params.get('convergence_threshold', 0)

        # Build pruning strategy
        self.pruning_strategy = self._build_pruning_strategy()

    def _build_pruning_strategy(self) -> PruningStrategy:
        """Build the pruning strategy from config."""
        strategy_config = self.params.get('pruning_strategy', {})

        if isinstance(strategy_config, dict) and strategy_config:
            return create_pruning_strategy(strategy_config)

        # Default: RedundantSupportStrategy with params from main config
        return RedundantSupportStrategy(
            base_threshold=self.params.get('base_threshold', 0.45),
            relaxation_alpha=self.params.get('relaxation_alpha', 1.1),
            support_beta=self.params.get('support_beta', 1.05),
            min_support=self.params.get('min_support', 2),
            separation_delta=self.params.get('separation_delta', 0.15),
            use_distance=self.params.get('use_distance', True),
        )

    def cluster(
        self,
        features: np.ndarray,
        collect_debug_data: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Cluster features using two-stage mutual kNN algorithm.

        Args:
            features: Feature matrix [n_samples, n_features]
            collect_debug_data: If True, include distance_matrix and cluster_members

        Returns:
            labels: Cluster labels (-1 for noise)
            stats: Clustering statistics and debug data
        """
        n_samples = features.shape[0]

        if n_samples == 0:
            return np.array([], dtype=np.int32), self._empty_stats()

        if n_samples == 1:
            return np.array([0], dtype=np.int32), self._single_sample_stats()

        # L2-normalize
        normalized = self.normalize_features(features)

        # Compute distance matrix
        distance_matrix = cosine_distance_matrix(normalized)

        # Stage 1: Build initial clusters
        logger.info(f"Stage 1: Building mutual kNN graph (k={self.k})")
        initial_labels = self._build_initial_clusters(distance_matrix, n_samples)

        initial_n_clusters = len(set(initial_labels) - {-1})
        initial_noise = int(np.sum(initial_labels == -1))
        logger.info(f"Stage 1 result: {initial_n_clusters} clusters, {initial_noise} noise")

        # Stage 2: Iterative refinement
        logger.info("Stage 2: Iterative pruning and reassignment")
        labels, iterations_info = self._iterative_refinement(
            initial_labels, distance_matrix, collect_debug_data
        )

        # Compute final statistics
        stats = self._compute_stats(labels, iterations_info)

        # Add debug data if requested
        if collect_debug_data:
            cluster_members = compute_cluster_members_from_labels(labels)
            stats['debug'] = {
                'distance_matrix': distance_matrix,
                'cluster_members': cluster_members,
                'iterations': iterations_info,
                'initial_labels': initial_labels,
            }

        return labels, stats

    def _build_initial_clusters(
        self,
        distance_matrix: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """
        Stage 1: Build mutual kNN graph and find connected components.

        Uses very loose or no threshold to avoid premature fragmentation.
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        # Find k nearest neighbors for each sample
        k_actual = min(self.k, n_samples - 1)

        # Get top-k neighbors (smallest distances)
        # Set diagonal to inf to exclude self
        dist_for_knn = distance_matrix.copy()
        np.fill_diagonal(dist_for_knn, np.inf)

        top_k_indices = np.argsort(dist_for_knn, axis=1)[:, :k_actual]

        # Build sets for mutual check
        top_k_sets = [set(top_k_indices[i]) for i in range(n_samples)]

        # Build mutual kNN graph
        edges_row = []
        edges_col = []

        for i in range(n_samples):
            for j in top_k_sets[i]:
                if j > i:  # Process each pair once
                    # Check mutual: i in top-k of j AND j in top-k of i
                    if i in top_k_sets[j]:
                        # Apply initial threshold if specified
                        if self.initial_threshold is None or \
                           distance_matrix[i, j] <= self.initial_threshold:
                            edges_row.append(i)
                            edges_col.append(j)
                            edges_row.append(j)
                            edges_col.append(i)

        n_edges = len(edges_row) // 2
        logger.debug(f"Built mutual kNN graph: {n_edges} edges")

        # Find connected components
        if n_edges > 0:
            data = np.ones(len(edges_row), dtype=np.int8)
            adjacency = csr_matrix(
                (data, (edges_row, edges_col)),
                shape=(n_samples, n_samples)
            )
            n_components, labels = connected_components(
                adjacency, directed=False, return_labels=True
            )
        else:
            # No edges: each sample starts as its own cluster
            n_components = n_samples
            labels = np.arange(n_samples, dtype=np.int32)

        # Don't mark singletons as noise yet - let Stage 2 handle that
        return labels

    def _iterative_refinement(
        self,
        initial_labels: np.ndarray,
        distance_matrix: np.ndarray,
        collect_debug: bool
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Stage 2: Iteratively prune and reassign until convergence.
        """
        labels = initial_labels.copy()
        n_samples = len(labels)
        iterations_info = []

        for iteration in range(self.max_iterations):
            cluster_members = compute_cluster_members_from_labels(labels)
            n_changes = 0
            pruned_samples = []
            reassigned_samples = []

            # Phase 1: Prune - validate each sample's membership
            for sample_idx in range(n_samples):
                current_cluster = int(labels[sample_idx])

                # Skip noise samples (will be handled in reassignment)
                if current_cluster == -1:
                    continue

                # Skip singletons (can't validate against empty cluster)
                if len(cluster_members.get(current_cluster, [])) <= 1:
                    continue

                # Evaluate membership
                should_stay, confidence, details = self.pruning_strategy.evaluate_membership(
                    sample_idx=sample_idx,
                    candidate_cluster=current_cluster,
                    current_cluster=current_cluster,
                    cluster_members=cluster_members,
                    distance_matrix=distance_matrix
                )

                if not should_stay:
                    # Mark as unassigned
                    labels[sample_idx] = -1
                    n_changes += 1
                    if collect_debug:
                        pruned_samples.append({
                            'sample_idx': sample_idx,
                            'from_cluster': current_cluster,
                            'details': details
                        })

            # Rebuild cluster_members after pruning
            cluster_members = compute_cluster_members_from_labels(labels)

            # Phase 2: Reassign - try to place unassigned samples
            unassigned = [i for i in range(n_samples) if labels[i] == -1]

            for sample_idx in unassigned:
                best_cluster, confidence, details = self.pruning_strategy.find_best_cluster(
                    sample_idx=sample_idx,
                    cluster_members=cluster_members,
                    distance_matrix=distance_matrix,
                    exclude_clusters=[-1]
                )

                if best_cluster is not None:
                    labels[sample_idx] = best_cluster
                    cluster_members = compute_cluster_members_from_labels(labels)
                    n_changes += 1
                    if collect_debug:
                        reassigned_samples.append({
                            'sample_idx': sample_idx,
                            'to_cluster': best_cluster,
                            'confidence': confidence,
                            'details': details
                        })

            # Record iteration info
            remaining_unassigned = int(np.sum(labels == -1))
            iter_info = {
                'iteration': iteration,
                'n_changes': n_changes,
                'n_pruned': len(pruned_samples),
                'n_reassigned': len(reassigned_samples),
                'remaining_unassigned': remaining_unassigned,
                'n_clusters': len(set(labels) - {-1}),
            }
            if collect_debug:
                iter_info['pruned'] = pruned_samples
                iter_info['reassigned'] = reassigned_samples

            iterations_info.append(iter_info)

            logger.info(
                f"  Iteration {iteration + 1}: "
                f"pruned={len(pruned_samples)}, reassigned={len(reassigned_samples)}, "
                f"noise={remaining_unassigned}, clusters={iter_info['n_clusters']}"
            )

            # Check convergence
            if n_changes <= self.convergence_threshold:
                logger.info(f"Converged at iteration {iteration + 1}")
                break

        # Relabel clusters to be consecutive (0, 1, 2, ...)
        labels = self._relabel_consecutive(labels)

        return labels, iterations_info

    def _relabel_consecutive(self, labels: np.ndarray) -> np.ndarray:
        """Relabel clusters to consecutive integers starting from 0."""
        unique_labels = sorted(set(labels) - {-1})
        label_map = {old: new for new, old in enumerate(unique_labels)}
        label_map[-1] = -1

        return np.array([label_map[l] for l in labels], dtype=np.int32)

    def _compute_stats(
        self,
        labels: np.ndarray,
        iterations_info: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute clustering statistics."""
        unique_labels = set(labels) - {-1}
        n_clusters = len(unique_labels)
        n_noise = int(np.sum(labels == -1))
        n_total = len(labels)

        cluster_sizes = {}
        for label in unique_labels:
            cluster_sizes[int(label)] = int(np.sum(labels == label))

        stats = {
            'algorithm': 'mutual_knn_two_stage',
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_ratio': n_noise / n_total if n_total > 0 else 0,
            'cluster_sizes': cluster_sizes,
            'n_iterations': len(iterations_info),
            'converged': len(iterations_info) < self.max_iterations,
            'params': {
                'k': self.k,
                'initial_threshold': self.initial_threshold,
                'max_iterations': self.max_iterations,
                **self.pruning_strategy.get_params()
            }
        }

        # Store in last_run_info for UI
        self.last_run_info = {
            'n_iterations': len(iterations_info),
            'final_clusters': n_clusters,
            'final_noise': n_noise,
        }

        return stats

    def _empty_stats(self) -> Dict[str, Any]:
        """Return stats for empty input."""
        return {
            'algorithm': 'mutual_knn_two_stage',
            'n_clusters': 0,
            'n_noise': 0,
            'noise_ratio': 0,
            'cluster_sizes': {},
            'n_iterations': 0,
            'converged': True,
            'params': {
                'k': self.k,
                'initial_threshold': self.initial_threshold,
                'max_iterations': self.max_iterations,
                **self.pruning_strategy.get_params()
            }
        }

    def _single_sample_stats(self) -> Dict[str, Any]:
        """Return stats for single sample."""
        return {
            'algorithm': 'mutual_knn_two_stage',
            'n_clusters': 1,
            'n_noise': 0,
            'noise_ratio': 0,
            'cluster_sizes': {0: 1},
            'n_iterations': 0,
            'converged': True,
            'params': {
                'k': self.k,
                'initial_threshold': self.initial_threshold,
                'max_iterations': self.max_iterations,
                **self.pruning_strategy.get_params()
            }
        }
