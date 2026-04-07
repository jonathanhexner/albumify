"""Conservative cluster merging using multi-evidence approach."""

import logging
from typing import List, Dict, Set, Tuple
import numpy as np

from face_cluster.types import ClusterResult, GraphResult
from face_cluster.config import PipelineConfig

logger = logging.getLogger(__name__)


class ConservativeMerger:
    """Merge clusters using multiple pieces of evidence.

    Multi-evidence criteria:
    - (A) Exemplar agreement: min exemplar distance <= threshold
    - (B) Support count: sufficient cross-cluster pairs below threshold
    - (C) Margin vs next best: prevent ambiguous merges
    - (D) Post-merge diameter: don't create over-wide clusters
    """

    def __init__(self, config: PipelineConfig):
        """Initialize merger.

        Args:
            config: Pipeline configuration with merge parameters
        """
        self.config = config
        self.last_thresholds: Dict[int, float] = {}
        self.last_candidates: List[Tuple[int, int, float, Dict]] = []

    def merge_clusters_with_logging(
        self,
        cluster_result: ClusterResult,
        graph_result: GraphResult
    ) -> Tuple[ClusterResult, List[Dict]]:
        """Merge clusters and return decision log.

        Args:
            cluster_result: Current cluster result with exemplars
            graph_result: Graph result with distance matrix

        Returns:
            (updated_cluster_result, merge_log)
            merge_log: List of decision dictionaries with merge details
        """
        merge_log = []
        result = self._merge_clusters_internal(cluster_result, graph_result, merge_log)
        return result, merge_log

    def merge_clusters(
        self,
        cluster_result: ClusterResult,
        graph_result: GraphResult
    ) -> ClusterResult:
        """Merge clusters using conservative multi-evidence approach.

        Args:
            cluster_result: Current cluster result with exemplars
            graph_result: Graph result with distance matrix

        Returns:
            Updated ClusterResult with merged clusters
        """
        result, _ = self.merge_clusters_with_logging(cluster_result, graph_result)
        return result

    def _merge_clusters_internal(
        self,
        cluster_result: ClusterResult,
        graph_result: GraphResult,
        merge_log: List[Dict]
    ) -> ClusterResult:
        """Internal merge implementation that logs decisions.

        Args:
            cluster_result: Current cluster result with exemplars
            graph_result: Graph result with distance matrix
            merge_log: List to append merge decisions to

        Returns:
            Updated ClusterResult with merged clusters
        """
        if not self.config.merge_enabled:
            logger.info("Cluster merging disabled")
            return cluster_result

        if cluster_result.n_clusters <= 1:
            logger.info("Only 1 cluster, no merging needed")
            return cluster_result

        distance_matrix = graph_result.distance_matrix

        # Compute cluster thresholds once (for adaptive mode)
        if self.config.merge_use_adaptive_threshold:
            cluster_thresholds = self._compute_cluster_thresholds(
                cluster_result,
                distance_matrix
            )
            global_threshold = self._compute_global_threshold(cluster_thresholds)
            logger.info(
                f"Adaptive thresholds: global={global_threshold:.3f}, "
                f"local range=[{min(cluster_thresholds.values()):.3f}, "
                f"{max(cluster_thresholds.values()):.3f}]"
            )
        else:
            cluster_thresholds = {}
            global_threshold = self.config.merge_exemplar_threshold
            logger.info(f"Using fixed threshold: {global_threshold:.3f}")

        # Iterative merging
        iteration = 0
        max_iterations = cluster_result.n_clusters  # Prevent infinite loops

        while iteration < max_iterations:
            iteration += 1

            # Propose merge candidates
            merge_candidates = self._propose_merge_candidates(
                cluster_result,
                distance_matrix
            )

            if len(merge_candidates) == 0:
                logger.info(f"No merge candidates found after {iteration} iterations")
                break

            # Find best merge using multi-evidence
            best_merge, all_decisions = self._find_best_merge_with_decisions(
                merge_candidates,
                cluster_result,
                distance_matrix,
                cluster_thresholds,
                global_threshold,
                iteration
            )

            # Log all decisions for this iteration
            merge_log.extend(all_decisions)

            if best_merge is None:
                logger.info(f"No valid merges found after {iteration} iterations")
                break

            # Perform merge
            cluster_id_a, cluster_id_b = best_merge
            cluster_result = self._merge_two_clusters(
                cluster_result,
                cluster_id_a,
                cluster_id_b,
                distance_matrix
            )

            # Recompute cluster thresholds after merge (if adaptive)
            if self.config.merge_use_adaptive_threshold:
                cluster_thresholds = self._compute_cluster_thresholds(
                    cluster_result,
                    distance_matrix
                )
                global_threshold = self._compute_global_threshold(cluster_thresholds)

            logger.info(
                f"Iteration {iteration}: Merged clusters {cluster_id_a} and {cluster_id_b} "
                f"({cluster_result.n_clusters} clusters remaining)"
            )

        logger.info(f"Merging complete after {iteration} iterations")

        # Store decision metadata for analysis (last_candidates set in _find_best_merge)
        self.last_thresholds = cluster_thresholds.copy() if cluster_thresholds else {}

        return cluster_result

    def _propose_merge_candidates(
        self,
        cluster_result: ClusterResult,
        distance_matrix: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Propose merge candidates based on exemplar proximity.

        Args:
            cluster_result: Current cluster result
            distance_matrix: Distance matrix

        Returns:
            List of (cluster_id_a, cluster_id_b) tuples
        """
        candidates = []
        cluster_ids = sorted(cluster_result.clusters.keys())

        for i, cluster_id_a in enumerate(cluster_ids):
            for cluster_id_b in cluster_ids[i + 1:]:
                # Check if any exemplars are close
                exemplars_a = cluster_result.exemplars.get(cluster_id_a, [])
                exemplars_b = cluster_result.exemplars.get(cluster_id_b, [])

                if len(exemplars_a) == 0 or len(exemplars_b) == 0:
                    # Fall back to full cluster if no exemplars
                    exemplars_a = cluster_result.clusters[cluster_id_a]
                    exemplars_b = cluster_result.clusters[cluster_id_b]

                # Find minimum distance between exemplars
                min_dist = float('inf')
                for node_a in exemplars_a:
                    for node_b in exemplars_b:
                        dist = distance_matrix[node_a, node_b]
                        min_dist = min(min_dist, dist)

                if min_dist <= self.config.merge_candidate_threshold:
                    candidates.append((cluster_id_a, cluster_id_b))

        logger.debug(f"Proposed {len(candidates)} merge candidates")
        return candidates

    def _find_best_merge(
        self,
        candidates: List[Tuple[int, int]],
        cluster_result: ClusterResult,
        distance_matrix: np.ndarray,
        cluster_thresholds: Dict[int, float],
        global_threshold: float
    ) -> Tuple[int, int] | None:
        """Find best merge using multi-evidence criteria.

        Args:
            candidates: List of candidate pairs
            cluster_result: Current cluster result
            distance_matrix: Distance matrix

        Returns:
            (cluster_id_a, cluster_id_b) or None if no valid merge
        """
        valid_merges = []
        all_candidates_with_evidence = []

        for cluster_id_a, cluster_id_b in candidates:
            evidence = self._evaluate_merge_evidence(
                cluster_id_a,
                cluster_id_b,
                cluster_result,
                distance_matrix,
                cluster_thresholds,
                global_threshold
            )

            # Store all candidates with evidence for analysis
            exemplar_dist = evidence.get('exemplar_dist', float('inf'))
            all_candidates_with_evidence.append((
                cluster_id_a,
                cluster_id_b,
                exemplar_dist,
                evidence
            ))

            if evidence['valid']:
                valid_merges.append((cluster_id_a, cluster_id_b, evidence))

        # Store candidates for external analysis
        self.last_candidates = sorted(
            all_candidates_with_evidence,
            key=lambda x: x[2]  # Sort by exemplar distance
        )

        if len(valid_merges) == 0:
            return None

        # Select best merge (smallest exemplar distance)
        valid_merges.sort(key=lambda x: x[2]['exemplar_dist'])
        best_merge = valid_merges[0]

        logger.debug(
            f"Best merge: {best_merge[0]} + {best_merge[1]} "
            f"(ex_dist={best_merge[2]['exemplar_dist']:.3f}, "
            f"support={best_merge[2]['support']}, "
            f"post_diameter={best_merge[2]['post_diameter']:.3f})"
        )

        return (best_merge[0], best_merge[1])

    def _find_best_merge_with_decisions(
        self,
        candidates: List[Tuple[int, int]],
        cluster_result: ClusterResult,
        distance_matrix: np.ndarray,
        cluster_thresholds: Dict[int, float],
        global_threshold: float,
        iteration: int
    ) -> Tuple[Tuple[int, int] | None, List[Dict]]:
        """Find best merge and return all decisions for logging.

        Args:
            candidates: List of candidate pairs
            cluster_result: Current cluster result
            distance_matrix: Distance matrix
            cluster_thresholds: Per-cluster thresholds
            global_threshold: Global threshold
            iteration: Current iteration number

        Returns:
            (best_merge_pair, all_decisions)
            best_merge_pair: (cluster_id_a, cluster_id_b) or None
            all_decisions: List of decision dicts for all candidates
        """
        valid_merges = []
        all_decisions = []

        for cluster_id_a, cluster_id_b in candidates:
            evidence = self._evaluate_merge_evidence(
                cluster_id_a,
                cluster_id_b,
                cluster_result,
                distance_matrix,
                cluster_thresholds,
                global_threshold
            )

            # Create decision entry
            decision = {
                'iteration': iteration,
                'cluster_a': cluster_id_a,
                'cluster_b': cluster_id_b,
                'cluster_a_size': len(cluster_result.clusters[cluster_id_a]),
                'cluster_b_size': len(cluster_result.clusters[cluster_id_b]),
                'exemplar_dist': evidence['exemplar_dist'],
                'threshold_used': evidence['merge_threshold'],
                'support': evidence['support'],
                'required_support': evidence['required_support'],
                'post_diameter': evidence['post_diameter'],
                'max_allowed_diameter': evidence['max_allowed_diameter'],
                'action': 'merged' if evidence['valid'] else 'rejected',
                'passes_exemplar': evidence['passes_exemplar'],
                'passes_support': evidence['passes_support'],
                'passes_margin': evidence['passes_margin'],
                'passes_diameter': evidence['passes_diameter'],
            }

            # Add rejection reason if rejected
            if not evidence['valid']:
                reasons = []
                if not evidence['passes_exemplar']:
                    reasons.append(f"exemplar_dist {evidence['exemplar_dist']:.3f} > threshold {evidence['merge_threshold']:.3f}")
                if not evidence['passes_support']:
                    reasons.append(f"support {evidence['support']} < required {evidence['required_support']}")
                if not evidence['passes_margin']:
                    reasons.append("margin check failed")
                if not evidence['passes_diameter']:
                    reasons.append(f"diameter {evidence['post_diameter']:.3f} > max {evidence['max_allowed_diameter']:.3f}")
                decision['rejection_reason'] = "; ".join(reasons)
            else:
                decision['rejection_reason'] = None

            all_decisions.append(decision)

            if evidence['valid']:
                valid_merges.append((cluster_id_a, cluster_id_b, evidence))

        if len(valid_merges) == 0:
            return None, all_decisions

        # Select best merge (smallest exemplar distance)
        valid_merges.sort(key=lambda x: x[2]['exemplar_dist'])
        best_merge = valid_merges[0]

        # Mark which merge was actually performed
        for decision in all_decisions:
            if decision['cluster_a'] == best_merge[0] and decision['cluster_b'] == best_merge[1]:
                decision['actually_merged'] = True
            else:
                decision['actually_merged'] = False

        logger.debug(
            f"Best merge: {best_merge[0]} + {best_merge[1]} "
            f"(ex_dist={best_merge[2]['exemplar_dist']:.3f}, "
            f"support={best_merge[2]['support']}, "
            f"post_diameter={best_merge[2]['post_diameter']:.3f})"
        )

        return (best_merge[0], best_merge[1]), all_decisions

    def _evaluate_merge_evidence(
        self,
        cluster_id_a: int,
        cluster_id_b: int,
        cluster_result: ClusterResult,
        distance_matrix: np.ndarray,
        cluster_thresholds: Dict[int, float],
        global_threshold: float
    ) -> Dict:
        """Evaluate all evidence for merging two clusters.

        Args:
            cluster_id_a: First cluster ID
            cluster_id_b: Second cluster ID
            cluster_result: Current cluster result
            distance_matrix: Distance matrix

        Returns:
            Dictionary with evidence scores and validity
        """
        nodes_a = cluster_result.clusters[cluster_id_a]
        nodes_b = cluster_result.clusters[cluster_id_b]
        exemplars_a = cluster_result.exemplars.get(cluster_id_a, nodes_a)
        exemplars_b = cluster_result.exemplars.get(cluster_id_b, nodes_b)

        # (A) Exemplar agreement with adaptive threshold
        exemplar_dist = self._min_exemplar_distance(
            exemplars_a, exemplars_b, distance_matrix
        )

        # Compute merge threshold (adaptive or fixed)
        if self.config.merge_use_adaptive_threshold and cluster_thresholds:
            T_a = cluster_thresholds.get(cluster_id_a, self.config.merge_exemplar_threshold)
            T_b = cluster_thresholds.get(cluster_id_b, self.config.merge_exemplar_threshold)
            T_local = max(T_a, T_b)  # Use MAX (more permissive)
            T_global = global_threshold
            alpha = self.config.merge_threshold_alpha
            merge_threshold = alpha * T_local + (1 - alpha) * T_global
        else:
            merge_threshold = self.config.merge_exemplar_threshold

        passes_exemplar = exemplar_dist <= merge_threshold

        # (B) Support count (use merge_threshold for consistency)
        support = self._count_support(
            nodes_a, nodes_b, distance_matrix, merge_threshold
        )

        min_size = min(len(nodes_a), len(nodes_b))
        required_support = max(
            int(min_size * self.config.merge_support_frac),
            self.config.merge_support_min
        )
        passes_support = support >= required_support

        # (C) Margin vs next best
        passes_margin = self._check_margin(
            cluster_id_a, cluster_id_b, cluster_result, distance_matrix
        )

        # (D) Post-merge diameter (adaptive)
        post_diameter = self._compute_post_merge_diameter(
            nodes_a, nodes_b, distance_matrix
        )

        # Allow diameter to expand by factor
        current_max_diameter = max(
            cluster_result.cluster_stats[cluster_id_a].get('diameter', 0.0),
            cluster_result.cluster_stats[cluster_id_b].get('diameter', 0.0)
        )
        max_allowed_diameter = current_max_diameter * self.config.merge_diameter_expansion_factor
        passes_diameter = post_diameter <= max_allowed_diameter

        # All must pass
        valid = passes_exemplar and passes_support and passes_margin and passes_diameter

        return {
            'valid': valid,
            'exemplar_dist': exemplar_dist,
            'merge_threshold': merge_threshold,
            'passes_exemplar': passes_exemplar,
            'support': support,
            'required_support': required_support,
            'passes_support': passes_support,
            'passes_margin': passes_margin,
            'post_diameter': post_diameter,
            'max_allowed_diameter': max_allowed_diameter,
            'passes_diameter': passes_diameter,
        }

    def _min_exemplar_distance(
        self,
        exemplars_a: List[int],
        exemplars_b: List[int],
        distance_matrix: np.ndarray
    ) -> float:
        """Compute minimum distance between exemplars."""
        min_dist = float('inf')
        for node_a in exemplars_a:
            for node_b in exemplars_b:
                dist = distance_matrix[node_a, node_b]
                min_dist = min(min_dist, dist)
        return min_dist

    def _count_support(
        self,
        nodes_a: List[int],
        nodes_b: List[int],
        distance_matrix: np.ndarray,
        threshold: float
    ) -> int:
        """Count cross-cluster pairs below threshold."""
        support = 0
        for node_a in nodes_a:
            for node_b in nodes_b:
                if distance_matrix[node_a, node_b] <= threshold:
                    support += 1
        return support

    def _check_margin(
        self,
        cluster_id_a: int,
        cluster_id_b: int,
        cluster_result: ClusterResult,
        distance_matrix: np.ndarray
    ) -> bool:
        """Check margin to next-best cluster.

        For each exemplar in Ci, check that Cj is the nearest cluster
        by a margin.

        If merge_margin=0, this check is disabled (always returns True).
        """
        # Disable check if margin is 0
        if self.config.merge_margin == 0.0:
            return True

        exemplars_a = cluster_result.exemplars.get(
            cluster_id_a,
            cluster_result.clusters[cluster_id_a]
        )

        for node_a in exemplars_a:
            # Find distances to all clusters
            cluster_dists = {}
            for cid, nodes in cluster_result.clusters.items():
                if cid == cluster_id_a:
                    continue
                # Min distance to cluster
                min_dist = min(distance_matrix[node_a, node_b] for node_b in nodes)
                cluster_dists[cid] = min_dist

            # Check if cluster_b is closest by margin
            dist_to_b = cluster_dists.get(cluster_id_b, float('inf'))
            for cid, dist in cluster_dists.items():
                if cid != cluster_id_b:
                    if dist_to_b + self.config.merge_margin > dist:
                        # Another cluster is as close or closer
                        return False

        return True

    def _compute_post_merge_diameter(
        self,
        nodes_a: List[int],
        nodes_b: List[int],
        distance_matrix: np.ndarray
    ) -> float:
        """Compute diameter of hypothetical merged cluster."""
        merged_nodes = nodes_a + nodes_b
        max_dist = 0.0

        for i, node_i in enumerate(merged_nodes):
            for node_j in merged_nodes[i + 1:]:
                dist = distance_matrix[node_i, node_j]
                max_dist = max(max_dist, dist)

        return max_dist

    def _merge_two_clusters(
        self,
        cluster_result: ClusterResult,
        cluster_id_a: int,
        cluster_id_b: int,
        distance_matrix: np.ndarray
    ) -> ClusterResult:
        """Merge two clusters and update result.

        Args:
            cluster_result: Current cluster result
            cluster_id_a: First cluster ID (will be kept)
            cluster_id_b: Second cluster ID (will be merged into A)
            distance_matrix: Distance matrix

        Returns:
            Updated ClusterResult
        """
        # Merge nodes
        merged_nodes = cluster_result.clusters[cluster_id_a] + cluster_result.clusters[cluster_id_b]

        # Update clusters dict
        new_clusters = {}
        for cid, nodes in cluster_result.clusters.items():
            if cid == cluster_id_a:
                new_clusters[cid] = merged_nodes
            elif cid == cluster_id_b:
                continue  # Skip, merged into A
            else:
                new_clusters[cid] = nodes

        # Update labels
        new_labels = cluster_result.labels.copy()
        for i, label in enumerate(new_labels):
            if label == cluster_id_b:
                new_labels[i] = cluster_id_a

        # Recompute stats for merged cluster
        new_cluster_stats = cluster_result.cluster_stats.copy()
        new_cluster_stats[cluster_id_a] = self._compute_cluster_stats(
            merged_nodes, distance_matrix
        )
        if cluster_id_b in new_cluster_stats:
            del new_cluster_stats[cluster_id_b]

        # Recompute exemplars for merged cluster
        new_exemplars = cluster_result.exemplars.copy()
        # Merge exemplar lists and take top N
        exemplars_a = cluster_result.exemplars.get(cluster_id_a, [])
        exemplars_b = cluster_result.exemplars.get(cluster_id_b, [])
        combined_exemplars = list(set(exemplars_a + exemplars_b))

        # Compute d10 for combined exemplars and keep best
        if len(combined_exemplars) > self.config.N_exemplars_max:
            d10_values = []
            for node in combined_exemplars:
                # Distance to kth nearest in merged cluster
                k = min(self.config.d10_k, len(merged_nodes) - 1)
                dists = [distance_matrix[node, other] for other in merged_nodes if other != node]
                if len(dists) >= k:
                    d10 = sorted(dists)[k - 1]
                    d10_values.append((d10, node))

            d10_values.sort()
            new_exemplars[cluster_id_a] = [node for _, node in d10_values[:self.config.N_exemplars_max]]
        else:
            new_exemplars[cluster_id_a] = combined_exemplars

        if cluster_id_b in new_exemplars:
            del new_exemplars[cluster_id_b]

        return ClusterResult(
            labels=new_labels,
            clusters=new_clusters,
            cluster_stats=new_cluster_stats,
            exemplars=new_exemplars,
            n_clusters=len(new_clusters),
            n_noise=cluster_result.n_noise
        )

    def _compute_cluster_stats(
        self,
        cluster_nodes: List[int],
        distance_matrix: np.ndarray
    ) -> Dict[str, float]:
        """Compute statistics for a cluster."""
        size = len(cluster_nodes)

        if size < 2:
            return {
                'size': size,
                'diameter': 0.0,
                'median_dist': 0.0,
                'mean_dist': 0.0,
                'p95_dist': 0.0,
            }

        # Extract pairwise distances
        indices = np.array(cluster_nodes)
        cluster_dists = distance_matrix[np.ix_(indices, indices)]
        upper_tri = cluster_dists[np.triu_indices_from(cluster_dists, k=1)]

        return {
            'size': size,
            'diameter': float(upper_tri.max()),
            'median_dist': float(np.median(upper_tri)),
            'mean_dist': float(upper_tri.mean()),
            'p95_dist': float(np.percentile(upper_tri, 95)),
        }

    def _compute_cluster_thresholds(
        self,
        cluster_result: ClusterResult,
        distance_matrix: np.ndarray
    ) -> Dict[int, float]:
        """Compute adaptive threshold for each cluster.

        Uses P90 (or configured percentile) of exemplar pairwise distances.

        Args:
            cluster_result: Cluster result with exemplars
            distance_matrix: Distance matrix

        Returns:
            Dict mapping cluster_id -> threshold
        """
        cluster_thresholds = {}

        for cluster_id, nodes in cluster_result.clusters.items():
            exemplar_nodes = cluster_result.exemplars.get(cluster_id, [])

            # Fall back to all nodes if no exemplars
            if len(exemplar_nodes) < 2:
                exemplar_nodes = nodes

            if len(exemplar_nodes) < 2:
                # Too small, use fallback
                cluster_thresholds[cluster_id] = self.config.merge_exemplar_threshold
                continue

            # Compute pairwise distances between exemplars
            exemplar_dists = []
            for i, node_a in enumerate(exemplar_nodes):
                for node_b in exemplar_nodes[i + 1:]:
                    exemplar_dists.append(distance_matrix[node_a, node_b])

            # Use configured percentile
            if len(exemplar_dists) > 0:
                threshold = np.percentile(
                    exemplar_dists,
                    self.config.merge_exemplar_percentile
                )
                cluster_thresholds[cluster_id] = float(threshold)
            else:
                cluster_thresholds[cluster_id] = self.config.merge_exemplar_threshold

        return cluster_thresholds

    def _compute_global_threshold(
        self,
        cluster_thresholds: Dict[int, float]
    ) -> float:
        """Compute global threshold using configured percentile.

        Args:
            cluster_thresholds: Dict mapping cluster_id -> threshold

        Returns:
            Global threshold (percentile of all cluster thresholds)
        """
        if len(cluster_thresholds) == 0:
            return self.config.merge_exemplar_threshold

        return float(np.percentile(
            list(cluster_thresholds.values()),
            self.config.merge_global_percentile
        ))
