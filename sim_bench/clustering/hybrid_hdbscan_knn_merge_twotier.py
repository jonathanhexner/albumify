"""
Hybrid HDBSCAN + Local Cohesion clustering - Two-tier merge variant.

Same as hybrid_hdbscan_knn but with additional secondary merge rule:
- Primary (unchanged): merge if ≥3 cross-exemplar pairs pass <= min(T_A, T_B)
  with ≥2 distinct exemplars each side
- Secondary (new): merge if ≥5 cross-exemplar pairs pass <= max(T_A, T_B)
  with ≥2 distinct exemplars each side

Why: this merges pose-mode splits when one cluster's T is slightly tighter,
without over-merging unrelated clusters.

Distance metric: Cosine distance = 1 - cosine_similarity, clipped to [0, 2].
"""

import logging
from typing import Dict, Any, Tuple, Set

from sim_bench.clustering.hybrid_hdbscan_knn import HybridHDBSCANKNN

logger = logging.getLogger(__name__)


class HybridHDBSCANKNNMergeTwotier(HybridHDBSCANKNN):
    """Hybrid HDBSCAN + Local Cohesion with two-tier merge rules."""

    doc_explanation = """
Variant of hybrid_hdbscan_knn with two-tier merge rules for pose-split recovery.

Primary Rule: Merge if >=merge_min_pairs (3) pairs pass <= min(T_A, T_B).
Secondary Rule: Merge if >=merge_secondary_min_pairs (5) pairs pass <= max(T_A, T_B).

The secondary rule catches pose-mode splits where one cluster has a tighter T
but both clusters genuinely belong to the same person. Requires more evidence
(5 pairs vs 3) to use the looser threshold.
"""

    decision_parameters = {
        **HybridHDBSCANKNN.decision_parameters,
        "merge_secondary_min_pairs": {
            "description": "Pairs required for secondary (looser) merge rule",
            "default": 5,
            "decision_role": "Secondary merge if pairs <= max(T_A,T_B) >= this"
        },
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Secondary merge requires more pairs but uses looser threshold
        self.merge_secondary_min_pairs = self.params.get('merge_secondary_min_pairs', 5)

    def _check_merge(
        self,
        cross_dists,
        t_a: float,
        t_b: float,
    ) -> Tuple[bool, str, float]:
        """Two-tier merge check.

        Primary: ≥3 pairs pass <= min(T_A, T_B) with ≥2 distinct each side
        Secondary: ≥5 pairs pass <= max(T_A, T_B) with ≥2 distinct each side

        Returns (should_merge, reason, threshold_used).
        """
        # Primary check: try min(T_A, T_B) first, then individual T_A and T_B
        # This is the original bidirectional logic
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

            if (len(pairs_within) >= self.merge_min_pairs and
                len(involved_a) >= self.merge_min_distinct and
                len(involved_b) >= self.merge_min_distinct):
                return True, f'merged_primary_{direction}', threshold

        # Secondary check: use max(T_A, T_B) but require more pairs
        threshold_max = max(t_a, t_b)
        pairs_within = set()
        involved_a = set()
        involved_b = set()

        for ia in range(cross_dists.shape[0]):
            for ib in range(cross_dists.shape[1]):
                if cross_dists[ia, ib] <= threshold_max:
                    pairs_within.add((ia, ib))
                    involved_a.add(ia)
                    involved_b.add(ib)

        if (len(pairs_within) >= self.merge_secondary_min_pairs and
            len(involved_a) >= self.merge_min_distinct and
            len(involved_b) >= self.merge_min_distinct):
            return True, 'merged_secondary_max_threshold', threshold_max

        # No merge
        return False, 'not_enough_pairs', min(t_a, t_b)
