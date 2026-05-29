"""Conservative cluster merging using multi-evidence approach."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from face_cluster.types import ClusterResult, GraphResult
from face_cluster.config import PipelineConfig

logger = logging.getLogger(__name__)


# spec-053: typed boundary for ConservativeMerger.calc()
# (test files call it "SimplifiedMerger" historically; the class itself
# is ConservativeMerger).

@dataclass(frozen=True, slots=True)
class MergeInputs:
    """Per-call data for ConservativeMerger.calc()."""
    cluster_result: ClusterResult
    graph_result: GraphResult


@dataclass(frozen=True, slots=True)
class MergeResult:
    """Output of ConservativeMerger.calc()."""
    cluster_result: ClusterResult
    merge_log: List[Dict]
    merge_metadata: Dict


@dataclass
class MarginDetail:
    """Numeric detail for the margin gate check."""
    passed: bool
    worst_gap: float          # min(competitor_dist - dist_to_b) across exemplars; PASS if >= merge_margin
    worst_exemplar: int       # node index with the smallest gap
    competitor_id: int        # cluster ID that is the problematic competitor (-1 if passed)
    dist_to_b: float          # distance from worst_exemplar to cluster B
    competitor_dist: float    # distance from worst_exemplar to competitor cluster


@dataclass
class CandidateGroup:
    """A connected component of merge candidate pairs.

    Uses only primitive types — no dependency on UI dataclasses.
    Can be consumed by both the UI layer and (in future) the pipeline's
    ConservativeMerger to auto-approve high-cohesion groups.
    """
    group_id: int
    cluster_ids: List[int]             # sorted cluster IDs in this component
    pair_keys: List[Tuple[int, int]]   # (min_id, max_id) for each candidate pair
    pair_gate_counts: List[int]        # n_gates_passed per pair (parallel to pair_keys)
    cohesion: float                    # fraction of pairs with all 4 gates passing
    min_gates: int                     # weakest pair's gate count
    max_gates: int                     # strongest pair's gate count
    confidence: str                    # "auto_approve" | "review" | "auto_reject"


def group_merge_candidates(
    candidate_pairs: List[Tuple[int, int]],
    gate_counts: List[int],
    cohesion_threshold: float = 0.8,
    min_gates_for_promotion: int = 3,
) -> List["CandidateGroup"]:
    """Group candidate pairs into connected components and classify by confidence.

    Uses union-find to build transitive groups: if A+B and B+C are both
    candidates, {A, B, C} forms one group. Cohesion (fraction of 4/4-gate
    pairs) promotes borderline groups to auto_approve when the evidence is
    overwhelmingly consistent.

    Args:
        candidate_pairs: List of (cluster_a, cluster_b) candidate pairs.
        gate_counts: n_gates_passed (0-4) for each pair, parallel to candidate_pairs.
        cohesion_threshold: Fraction of 4/4 pairs required for cohesion promotion.
        min_gates_for_promotion: All pairs must pass >= this many gates for promotion.

    Returns:
        List of CandidateGroup, sorted: review first, then auto_approve, then
        auto_reject. Within each tier, sorted by group_id ascending.

    Confidence rules:
        auto_approve : all pairs 4/4  OR  cohesion >= threshold AND min_gates >= min_gates_for_promotion
        auto_reject  : all pairs <= 2/4
        review       : everything else
    """
    if not candidate_pairs:
        return []

    # Collect all cluster IDs
    all_ids: Set[int] = set()
    for a, b in candidate_pairs:
        all_ids.add(a)
        all_ids.add(b)

    # Union-find (path-compressed)
    parent: Dict[int, int] = {cid: cid for cid in all_ids}

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in candidate_pairs:
        _union(a, b)

    # Group pairs by root
    root_to_pairs: Dict[int, List[int]] = {}  # root -> indices into candidate_pairs
    for idx, (a, _) in enumerate(candidate_pairs):
        root = _find(a)
        root_to_pairs.setdefault(root, []).append(idx)

    # Build CandidateGroup for each component
    groups: List[CandidateGroup] = []
    for group_id, (root, pair_indices) in enumerate(sorted(root_to_pairs.items())):
        # Collect cluster IDs in this component
        component_ids: Set[int] = set()
        pair_keys: List[Tuple[int, int]] = []
        pair_gc: List[int] = []
        for idx in pair_indices:
            a, b = candidate_pairs[idx]
            component_ids.add(a)
            component_ids.add(b)
            pair_keys.append((min(a, b), max(a, b)))
            pair_gc.append(gate_counts[idx])

        n_pairs = len(pair_gc)
        n_full = sum(1 for g in pair_gc if g == 4)
        cohesion = n_full / n_pairs
        min_g = min(pair_gc)
        max_g = max(pair_gc)

        if min_g == 4:
            confidence = "auto_approve"
        elif cohesion >= cohesion_threshold and min_g >= min_gates_for_promotion:
            confidence = "auto_approve"
        elif max_g <= 2:
            confidence = "auto_reject"
        else:
            confidence = "review"

        groups.append(CandidateGroup(
            group_id=group_id,
            cluster_ids=sorted(component_ids),
            pair_keys=pair_keys,
            pair_gate_counts=pair_gc,
            cohesion=cohesion,
            min_gates=min_g,
            max_gates=max_g,
            confidence=confidence,
        ))

    # Sort: review first, then auto_approve, then auto_reject
    _order = {"review": 0, "auto_approve": 1, "auto_reject": 2}
    groups.sort(key=lambda g: (_order[g.confidence], g.group_id))
    return groups


def propose_merge_candidates(
    cluster_result: ClusterResult,
    distance_matrix: np.ndarray,
    merge_candidate_threshold: float,
) -> List[Tuple[int, int]]:
    """Return all cluster pairs whose min exemplar-exemplar distance is within threshold.

    Args:
        cluster_result: Current cluster state
        distance_matrix: Pairwise cosine distance matrix indexed by face index
        merge_candidate_threshold: Maximum distance to propose a pair as a candidate

    Returns:
        List of (cluster_id_a, cluster_id_b) pairs, sorted by cluster ID
    """
    candidates = []
    cluster_ids = sorted(cluster_result.clusters.keys())

    for i, cid_a in enumerate(cluster_ids):
        for cid_b in cluster_ids[i + 1:]:
            exemplars_a = cluster_result.exemplars.get(cid_a) or cluster_result.clusters[cid_a]
            exemplars_b = cluster_result.exemplars.get(cid_b) or cluster_result.clusters[cid_b]

            min_dist = min(
                distance_matrix[na, nb]
                for na in exemplars_a
                for nb in exemplars_b
            )
            if min_dist <= merge_candidate_threshold:
                candidates.append((cid_a, cid_b))

    logger.debug("Proposed %d merge candidates (threshold=%.3f)", len(candidates), merge_candidate_threshold)
    return candidates


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
        self.last_candidates: List[Tuple[int, int, float, Dict]] = []

    def calc(self, inputs: "MergeInputs") -> "MergeResult":
        """Single pipeline entry point (spec-053). Thin facade over
        ``merge_clusters_with_logging()``."""
        cr, log, meta = self.merge_clusters_with_logging(
            inputs.cluster_result, inputs.graph_result,
        )
        return MergeResult(cluster_result=cr, merge_log=log, merge_metadata=meta)

    def merge_clusters_with_logging(
        self,
        cluster_result: ClusterResult,
        graph_result: GraphResult
    ) -> Tuple[ClusterResult, List[Dict], Dict]:
        """Merge clusters and return decision log and metadata.

        Returns:
            (updated_cluster_result, merge_log, merge_metadata)
            merge_metadata: cluster_thresholds, global_threshold, iteration counts
        """
        merge_log: List[Dict] = []
        n_iterations = [0]
        result = self._merge_clusters_internal(cluster_result, graph_result, merge_log, n_iterations)
        merge_metadata = {
            "n_candidates_proposed": len({
                (min(e["cluster_a"], e["cluster_b"]), max(e["cluster_a"], e["cluster_b"]))
                for e in merge_log
            }),
            "n_iterations": n_iterations[0],
            "merge_exemplar_threshold":  self.config.merge_exemplar_threshold,
            "merge_candidate_threshold": self.config.merge_candidate_threshold,
            "config": {
                "merge_margin": self.config.merge_margin,
                "merge_support_min": self.config.merge_support_min,
                "merge_support_frac": self.config.merge_support_frac,
                "merge_diameter_expansion_factor": self.config.merge_diameter_expansion_factor,
            },
        }
        return result, merge_log, merge_metadata

    def merge_clusters(
        self,
        cluster_result: ClusterResult,
        graph_result: GraphResult
    ) -> ClusterResult:
        """Merge clusters using conservative multi-evidence approach."""
        result, _, _ = self.merge_clusters_with_logging(cluster_result, graph_result)
        return result

    def _merge_clusters_internal(
        self,
        cluster_result: ClusterResult,
        graph_result: GraphResult,
        merge_log: List[Dict],
        n_iterations_out: List[int],
    ) -> ClusterResult:
        """Internal merge implementation that logs decisions."""
        if not self.config.merge_enabled:
            logger.info("Cluster merging disabled")
            return cluster_result

        if cluster_result.n_clusters <= 1:
            logger.info("Only 1 cluster, no merging needed")
            return cluster_result

        distance_matrix = graph_result.distance_matrix

        # Fixed exemplar threshold (no adaptive computation)
        cluster_thresholds: Dict[int, float] = {}
        global_threshold = self.config.merge_exemplar_threshold
        logger.info(f"Merge exemplar threshold: {global_threshold:.3f}")

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

            logger.info(
                f"Iteration {iteration}: Merged clusters {cluster_id_a} and {cluster_id_b} "
                f"({cluster_result.n_clusters} clusters remaining)"
            )

        logger.info(f"Merging complete after {iteration} iterations")

        n_iterations_out[0] = iteration

        return cluster_result

    def _propose_merge_candidates(
        self,
        cluster_result: ClusterResult,
        distance_matrix: np.ndarray
    ) -> List[Tuple[int, int]]:
        return propose_merge_candidates(
            cluster_result, distance_matrix, self.config.merge_candidate_threshold
        )

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
                'T_a': evidence['T_a'],
                'T_b': evidence['T_b'],
                'T_global': evidence['T_global'],
                'p25_cross_dist': evidence['p25_cross_dist'],
                'passes_cross': evidence['passes_cross'],
                'support': evidence['support'],
                'unique_support': evidence['unique_support'],
                'required_support': evidence['required_support'],
                'post_diameter': evidence['post_diameter'],
                'max_allowed_diameter': evidence['max_allowed_diameter'],
                'action': 'passed' if evidence['valid'] else 'rejected',
                'passes_exemplar': evidence['passes_exemplar'],
                'passes_support': evidence['passes_support'],
                'passes_margin': evidence['passes_margin'],
                'margin_gap': evidence['margin_gap'],
                'margin_dist_to_b': evidence['margin_dist_to_b'],
                'margin_competitor_dist': evidence['margin_competitor_dist'],
                'margin_competitor_id': evidence['margin_competitor_id'],
                'passes_diameter': evidence['passes_diameter'],
            }

            # Add rejection reason if rejected
            if not evidence['valid']:
                reasons = []
                if not evidence['passes_exemplar']:
                    if self.config.merge_use_cross_gate:
                        msg = (
                            f"neither OR path passed: "
                            f"p25_exemplar {evidence['exemplar_dist']:.3f} > {evidence['merge_threshold']:.3f}, "
                            f"p25_cross {evidence['p25_cross_dist']:.3f} > {self.config.merge_cross_threshold:.3f}"
                        )
                    else:
                        msg = f"p25_exemplar {evidence['exemplar_dist']:.3f} > threshold {evidence['merge_threshold']:.3f}"
                    reasons.append(msg)
                if not evidence['passes_support']:
                    reasons.append(f"support {evidence['support']} < required {evidence['required_support']}")
                if not evidence['passes_margin']:
                    gap = evidence['margin_gap']
                    req = self.config.merge_margin
                    cid = evidence['margin_competitor_id']
                    reasons.append(
                        f"margin gap {gap:.3f} < required {req:.3f} "
                        f"(competitor C_{cid} at {evidence['margin_competitor_dist']:.3f})"
                    )
                if not evidence['passes_diameter']:
                    reasons.append(f"diameter {evidence['post_diameter']:.3f} > max {evidence['max_allowed_diameter']:.3f}")
                decision['rejection_reason'] = "; ".join(reasons)
            else:
                decision['rejection_reason'] = None

            all_decisions.append(decision)

            if evidence['valid']:
                valid_merges.append((cluster_id_a, cluster_id_b, evidence))

        if len(valid_merges) == 0:
            # Terminal iteration: no winner this round, but every row still needs
            # actually_merged=False to satisfy the merge_log row contract
            # (MergeDecisionRow.field_names — spec-030 / SIGHTING-058).
            for decision in all_decisions:
                decision['actually_merged'] = False
            return None, all_decisions

        # Select best merge (smallest exemplar distance)
        valid_merges.sort(key=lambda x: x[2]['exemplar_dist'])
        best_merge = valid_merges[0]

        # Mark which merge was actually performed — only the winner gets action="merged"
        for decision in all_decisions:
            if decision['cluster_a'] == best_merge[0] and decision['cluster_b'] == best_merge[1]:
                decision['action'] = 'merged'
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

        # (A) Exemplar agreement — fixed threshold
        exemplar_dist = self._p25_exemplar_distance(
            exemplars_a, exemplars_b, distance_matrix
        )
        merge_threshold = self.config.merge_exemplar_threshold
        T_a: Optional[float] = None
        T_b: Optional[float] = None
        T_global_val: Optional[float] = None

        passes_exemplar_only = exemplar_dist <= merge_threshold

        # (A') Cross-distance OR path — p25 over ALL node pairs
        # Only applies when the smaller cluster is <= merge_cross_max_size
        p25_cross_dist = self._p25_cross_distance(nodes_a, nodes_b, distance_matrix)
        passes_cross = p25_cross_dist <= self.config.merge_cross_threshold

        min_size = min(len(nodes_a), len(nodes_b))
        cross_gate_eligible = (
            self.config.merge_use_cross_gate
            and min_size <= self.config.merge_cross_max_size
        )
        if cross_gate_eligible:
            passes_exemplar = passes_exemplar_only or passes_cross
        else:
            passes_exemplar = passes_exemplar_only

        # (B) Support count (use merge_threshold for consistency)
        support = self._count_support(
            nodes_a, nodes_b, distance_matrix, merge_threshold
        )
        unique_support = self._count_unique_support(
            nodes_a, nodes_b, distance_matrix, merge_threshold
        )

        if self.config.merge_support_unique:
            effective_support = unique_support
        else:
            effective_support = support
        required_support = max(
            int(min_size * self.config.merge_support_frac),
            self.config.merge_support_min
        )
        passes_support = effective_support >= required_support

        # (C) Margin vs next best
        margin_detail = self._check_margin(
            cluster_id_a, cluster_id_b, cluster_result, distance_matrix
        )
        passes_margin = margin_detail.passed

        # (D) Post-merge diameter (adaptive)
        post_diameter = self._compute_post_merge_diameter(
            nodes_a, nodes_b, distance_matrix
        )

        # Allow diameter to expand by factor
        current_max_diameter = max(
            cluster_result.cluster_stats.get(cluster_id_a, {}).get('diameter', 0.0),
            cluster_result.cluster_stats.get(cluster_id_b, {}).get('diameter', 0.0)
        )
        max_allowed_diameter = current_max_diameter * self.config.merge_diameter_expansion_factor
        passes_diameter = post_diameter <= max_allowed_diameter

        # All must pass
        valid = passes_exemplar and passes_support and passes_margin and passes_diameter

        return {
            'valid': valid,
            'exemplar_dist': exemplar_dist,
            'merge_threshold': merge_threshold,
            'T_a': T_a,
            'T_b': T_b,
            'T_global': T_global_val,
            'passes_exemplar': passes_exemplar,
            'p25_cross_dist': p25_cross_dist,
            'passes_cross': passes_cross,
            'support': support,
            'unique_support': unique_support,
            'required_support': required_support,
            'passes_support': passes_support,
            'passes_margin': passes_margin,
            'margin_gap': margin_detail.worst_gap,
            'margin_dist_to_b': margin_detail.dist_to_b,
            'margin_competitor_dist': margin_detail.competitor_dist,
            'margin_competitor_id': margin_detail.competitor_id,
            'post_diameter': post_diameter,
            'max_allowed_diameter': max_allowed_diameter,
            'passes_diameter': passes_diameter,
        }

    def _p25_exemplar_distance(
        self,
        exemplars_a: List[int],
        exemplars_b: List[int],
        distance_matrix: np.ndarray
    ) -> float:
        """Compute 25th-percentile distance across all exemplar pairs."""
        dists = [
            distance_matrix[a, b]
            for a in exemplars_a
            for b in exemplars_b
        ]
        return float(np.percentile(dists, 25))

    def _p25_cross_distance(
        self,
        nodes_a: List[int],
        nodes_b: List[int],
        distance_matrix: np.ndarray,
    ) -> float:
        """Compute 25th-percentile distance across ALL cross-cluster node pairs."""
        idx_a = np.array(nodes_a)
        idx_b = np.array(nodes_b)
        cross_dists = distance_matrix[np.ix_(idx_a, idx_b)].ravel()
        return float(np.percentile(cross_dists, 25))

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

    def _count_unique_support(
        self,
        nodes_a: List[int],
        nodes_b: List[int],
        distance_matrix: np.ndarray,
        threshold: float,
    ) -> int:
        """Count unique cross-cluster pairs below threshold (greedy bipartite).

        Each node may participate in at most one counted pair.  Pairs are
        assigned greedily in ascending distance order.
        """
        pairs = []
        for na in nodes_a:
            for nb in nodes_b:
                d = distance_matrix[na, nb]
                if d <= threshold:
                    pairs.append((d, na, nb))
        pairs.sort()
        used_a: Set[int] = set()
        used_b: Set[int] = set()
        count = 0
        for _, na, nb in pairs:
            if na not in used_a and nb not in used_b:
                used_a.add(na)
                used_b.add(nb)
                count += 1
        return count

    def _check_margin(
        self,
        cluster_id_a: int,
        cluster_id_b: int,
        cluster_result: ClusterResult,
        distance_matrix: np.ndarray
    ) -> MarginDetail:
        """Check margin to next-best cluster.

        For each exemplar in A, verify that B is the nearest other cluster
        by at least merge_margin. Returns full numeric detail for logging.

        If merge_margin=0, this check is disabled (always passes).
        """
        _no_detail = MarginDetail(
            passed=True, worst_gap=float('inf'),
            worst_exemplar=-1, competitor_id=-1,
            dist_to_b=0.0, competitor_dist=0.0
        )
        if self.config.merge_margin == 0.0:
            return _no_detail

        exemplars_a = cluster_result.exemplars.get(
            cluster_id_a,
            cluster_result.clusters[cluster_id_a]
        )

        worst_gap = float('inf')
        worst_exemplar = -1
        worst_competitor_id = -1
        worst_dist_to_b = 0.0
        worst_competitor_dist = 0.0

        for node_a in exemplars_a:
            cluster_dists: Dict[int, float] = {}
            for cid, nodes in cluster_result.clusters.items():
                if cid == cluster_id_a:
                    continue
                cluster_dists[cid] = min(distance_matrix[node_a, nb] for nb in nodes)

            dist_to_b = cluster_dists.get(cluster_id_b, float('inf'))
            for cid, dist in cluster_dists.items():
                if cid == cluster_id_b:
                    continue
                gap = dist - dist_to_b          # positive = competitor is farther (good)
                if gap < worst_gap:
                    worst_gap = gap
                    worst_exemplar = node_a
                    worst_competitor_id = cid
                    worst_dist_to_b = dist_to_b
                    worst_competitor_dist = dist

        passed = worst_gap >= self.config.merge_margin
        return MarginDetail(
            passed=passed,
            worst_gap=worst_gap,
            worst_exemplar=worst_exemplar,
            competitor_id=worst_competitor_id,
            dist_to_b=worst_dist_to_b,
            competitor_dist=worst_competitor_dist,
        )

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

        # Recompute exemplars for merged cluster over ALL merged nodes so that
        # bridge nodes (never exemplars in their original cluster) can be discovered.
        new_exemplars = cluster_result.exemplars.copy()
        helper = _ClusterStatHelper(distance_matrix)
        new_exemplars[cluster_id_a] = helper.select_exemplars(
            merged_nodes,
            d10_k=self.config.d10_k,
            n_max=self.config.N_exemplars_max,
            suppression_radius=self.config.exemplar_suppression_radius,
        )
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


# ---------------------------------------------------------------------------
# Public utility: apply a user-specified set of merges to a base ClusterResult
# ---------------------------------------------------------------------------

def apply_manual_merges(
    base_cluster_result: ClusterResult,
    approved_pairs: List[Tuple[int, int]],
    distance_matrix: np.ndarray,
) -> ClusterResult:
    """Apply a user-approved list of cluster merges to a base ClusterResult.

    Handles transitive chains via union-find: if (A, B) and (B, C) are both
    approved, A, B, and C all end up in the same cluster.

    Args:
        base_cluster_result: The pre-merge ClusterResult (from kNN clustering).
        approved_pairs: List of (cluster_id_a, cluster_id_b) to merge.
            Unknown cluster IDs are silently ignored.
        distance_matrix: Full pairwise distance matrix (used to recompute stats
            and exemplars for merged clusters).

    Returns:
        New ClusterResult reflecting approved merges. Clusters not involved
        in any approved merge are returned unchanged.
    """
    if not approved_pairs:
        return base_cluster_result

    # Build union-find over known cluster IDs
    known_ids = set(base_cluster_result.clusters.keys())
    parent: Dict[int, int] = {cid: cid for cid in known_ids}

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra  # merge rb's tree under ra

    for a, b in approved_pairs:
        if a in known_ids and b in known_ids:
            _union(a, b)

    # Group cluster IDs by their root (representative)
    root_to_members: Dict[int, List[int]] = {}
    for cid in known_ids:
        root = _find(cid)
        root_to_members.setdefault(root, []).append(cid)

    # Build new cluster assignment
    # Use the smallest cluster ID within each group as the new canonical ID
    new_clusters: Dict[int, List[int]] = {}
    new_labels = base_cluster_result.labels.copy()

    for root, members in root_to_members.items():
        canonical_id = min(members)
        merged_nodes: List[int] = []
        for cid in members:
            merged_nodes.extend(base_cluster_result.clusters[cid])
            # Relabel all nodes from absorbed clusters to the canonical cluster
            for node in base_cluster_result.clusters[cid]:
                if base_cluster_result.labels[node] != -1:
                    new_labels[node] = canonical_id
        new_clusters[canonical_id] = merged_nodes

    # Recompute stats and exemplars for each new cluster
    # Reuse the private helper by instantiating a minimal merger
    _helper = _ClusterStatHelper(distance_matrix)
    new_cluster_stats: Dict[int, Dict[str, float]] = {}
    new_exemplars: Dict[int, List[int]] = {}

    for cid, nodes in new_clusters.items():
        new_cluster_stats[cid] = _helper.compute_stats(nodes)
        new_exemplars[cid] = _helper.select_exemplars(
            nodes, d10_k=3, n_max=10, suppression_radius=0.2
        )

    return ClusterResult(
        labels=new_labels,
        clusters=new_clusters,
        cluster_stats=new_cluster_stats,
        exemplars=new_exemplars,
        n_clusters=len(new_clusters),
        n_noise=base_cluster_result.n_noise,
    )


class _ClusterStatHelper:
    """Minimal helper to recompute cluster stats and exemplars without full config."""

    def __init__(self, distance_matrix: np.ndarray) -> None:
        self._dm = distance_matrix

    def compute_stats(self, nodes: List[int]) -> Dict[str, float]:
        size = len(nodes)
        if size < 2:
            return {'size': size, 'diameter': 0.0, 'median_dist': 0.0,
                    'mean_dist': 0.0, 'p95_dist': 0.0}
        indices = np.array(nodes)
        sub = self._dm[np.ix_(indices, indices)]
        upper = sub[np.triu_indices_from(sub, k=1)]
        return {
            'size': size,
            'diameter': float(upper.max()),
            'median_dist': float(np.median(upper)),
            'mean_dist': float(upper.mean()),
            'p95_dist': float(np.percentile(upper, 95)),
        }

    def select_exemplars(
        self,
        nodes: List[int],
        d10_k: int,
        n_max: int,
        suppression_radius: float,
    ) -> List[int]:
        """Select up to n_max exemplars by d10 score (kth-NN distance)."""
        if len(nodes) <= 1:
            return list(nodes)
        scored: List[Tuple[float, int]] = []
        k = min(d10_k, len(nodes) - 1)
        for node in nodes:
            dists = sorted(self._dm[node, other] for other in nodes if other != node)
            d10 = dists[k - 1] if len(dists) >= k else dists[-1]
            scored.append((d10, node))
        scored.sort()
        # Greedy suppression
        selected: List[int] = []
        for _, node in scored:
            if len(selected) >= n_max:
                break
            if all(self._dm[node, s] >= suppression_radius for s in selected):
                selected.append(node)
        return selected
