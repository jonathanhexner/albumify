"""
Pruning strategies for two-stage clustering algorithms.

Strategies evaluate whether a sample should belong to a cluster based on
distance relationships, support counts, and separation margins.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import logging

from sim_bench.clustering.distance_utils import (
    closest_distance_to_cluster,
    support_count,
    separation_margin,
)

logger = logging.getLogger(__name__)


class PruningStrategy(ABC):
    """
    Abstract base class for pruning strategies.

    A pruning strategy decides whether a sample should belong to a cluster
    based on the distance matrix and cluster membership information.
    """

    # Override in subclasses: name for identification
    name: str = "base_strategy"

    # Override in subclasses: parameter documentation
    decision_parameters: Dict[str, Dict[str, Any]] = {}

    def __init__(self, use_distance: bool = True):
        """
        Args:
            use_distance: If True, lower values = closer (distance mode).
                          If False, higher values = closer (similarity mode).
        """
        self.use_distance = use_distance

    @abstractmethod
    def evaluate_membership(
        self,
        sample_idx: int,
        candidate_cluster: int,
        current_cluster: int,
        cluster_members: Dict[int, List[int]],
        distance_matrix: np.ndarray
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Evaluate whether a sample should belong to a candidate cluster.

        Args:
            sample_idx: Index of the sample to evaluate
            candidate_cluster: ID of the cluster to evaluate membership for
            current_cluster: Sample's current cluster (-1 if unassigned)
            cluster_members: Dict mapping cluster_id -> list of sample indices
            distance_matrix: Pairwise distance matrix (n_samples, n_samples)

        Returns:
            Tuple of:
            - should_assign: Whether the sample should be in this cluster
            - confidence: Confidence score (higher = more confident)
            - details: Dict with evaluation details for debugging
        """
        pass

    def find_best_cluster(
        self,
        sample_idx: int,
        cluster_members: Dict[int, List[int]],
        distance_matrix: np.ndarray,
        exclude_clusters: Optional[List[int]] = None
    ) -> Tuple[Optional[int], float, Dict[str, Any]]:
        """
        Find the best cluster for a sample among all valid clusters.

        Args:
            sample_idx: Index of the sample
            cluster_members: Dict mapping cluster_id -> list of sample indices
            distance_matrix: Pairwise distance matrix
            exclude_clusters: Cluster IDs to skip (e.g., noise cluster -1)

        Returns:
            Tuple of:
            - best_cluster: ID of best cluster (None if no valid cluster)
            - best_confidence: Confidence score for best cluster
            - details: Evaluation details
        """
        exclude = set(exclude_clusters or [-1])
        best_cluster = None
        best_confidence = -np.inf
        best_details = {}

        for cluster_id in cluster_members.keys():
            if cluster_id in exclude:
                continue
            if not cluster_members[cluster_id]:
                continue

            should_assign, confidence, details = self.evaluate_membership(
                sample_idx=sample_idx,
                candidate_cluster=cluster_id,
                current_cluster=-1,
                cluster_members=cluster_members,
                distance_matrix=distance_matrix
            )

            if should_assign and confidence > best_confidence:
                best_cluster = cluster_id
                best_confidence = confidence
                best_details = details

        return best_cluster, best_confidence, best_details


class RedundantSupportStrategy(PruningStrategy):
    """
    Pruning strategy based on redundant support and separation margin.

    A sample belongs to a cluster if:
    1. Base condition: closest_dist <= alpha * base_threshold
    2. AND one of:
       - Redundant support: >= min_support neighbors within beta * base_threshold
       - Separation: distance to next-best cluster is >= separation_delta farther

    This allows:
    - Larger distances if there are multiple nearby neighbors (redundant support)
    - Points far from other clusters even with weaker support (clear separation)
    """

    name = "redundant_support"

    decision_parameters = {
        "base_threshold": {
            "description": "Base distance threshold (X)",
            "default": 0.45,
            "decision_role": "Core threshold for membership decisions"
        },
        "relaxation_alpha": {
            "description": "Relaxation factor for base condition",
            "default": 1.1,
            "decision_role": "Allows up to alpha*X distance for base check"
        },
        "support_beta": {
            "description": "Support radius factor",
            "default": 1.05,
            "decision_role": "Neighbors within beta*X count as support"
        },
        "min_support": {
            "description": "Minimum supporting neighbors required",
            "default": 2,
            "decision_role": "Redundancy requirement for support path"
        },
        "separation_delta": {
            "description": "Minimum separation margin to next cluster",
            "default": 0.15,
            "decision_role": "Alternative to support: clear cluster separation"
        },
    }

    def __init__(
        self,
        base_threshold: float = 0.45,
        relaxation_alpha: float = 1.1,
        support_beta: float = 1.05,
        min_support: int = 2,
        separation_delta: float = 0.15,
        use_distance: bool = True
    ):
        """
        Args:
            base_threshold: Base distance threshold (X)
            relaxation_alpha: Relaxation factor (alpha) for base condition
            support_beta: Support radius factor (beta)
            min_support: Minimum neighbors for redundant support (m)
            separation_delta: Separation margin requirement (delta)
            use_distance: True for distance mode, False for similarity mode
        """
        super().__init__(use_distance=use_distance)
        self.base_threshold = base_threshold
        self.relaxation_alpha = relaxation_alpha
        self.support_beta = support_beta
        self.min_support = min_support
        self.separation_delta = separation_delta

    def evaluate_membership(
        self,
        sample_idx: int,
        candidate_cluster: int,
        current_cluster: int,
        cluster_members: Dict[int, List[int]],
        distance_matrix: np.ndarray
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Evaluate membership using redundant support + separation criteria.
        """
        details = {
            "sample_idx": sample_idx,
            "candidate_cluster": candidate_cluster,
            "current_cluster": current_cluster,
        }

        # Get cluster members (excluding self)
        members = cluster_members.get(candidate_cluster, [])
        other_members = [m for m in members if m != sample_idx]

        if not other_members:
            details["reason"] = "empty_cluster"
            return False, 0.0, details

        # Compute closest distance to cluster
        closest_dist = float(np.min(distance_matrix[sample_idx, other_members]))
        details["closest_dist"] = closest_dist

        # Thresholds
        max_allowed_dist = self.relaxation_alpha * self.base_threshold
        support_radius = self.support_beta * self.base_threshold

        details["max_allowed_dist"] = max_allowed_dist
        details["support_radius"] = support_radius

        # Check base condition
        if closest_dist > max_allowed_dist:
            details["reason"] = "exceeds_max_distance"
            details["base_condition_passed"] = False
            return False, 0.0, details

        details["base_condition_passed"] = True

        # Check redundant support
        n_support = support_count(
            sample_idx, candidate_cluster, distance_matrix,
            cluster_members, support_radius
        )
        details["support_count"] = n_support
        details["min_support_required"] = self.min_support

        has_support = n_support >= self.min_support
        details["has_redundant_support"] = has_support

        # Check separation
        margin, next_best = separation_margin(
            sample_idx, candidate_cluster, distance_matrix, cluster_members
        )
        details["separation_margin"] = margin
        details["next_best_cluster"] = next_best
        details["separation_delta_required"] = self.separation_delta

        has_separation = margin >= self.separation_delta
        details["has_separation"] = has_separation

        # Decision
        if has_support or has_separation:
            # Confidence based on distance (closer = higher confidence)
            # Normalize to [0, 1] range based on max_allowed_dist
            confidence = max(0.0, 1.0 - closest_dist / max_allowed_dist)

            # Boost confidence if both conditions are met
            if has_support and has_separation:
                confidence = min(1.0, confidence * 1.2)
                details["reason"] = "both_support_and_separation"
            elif has_support:
                details["reason"] = "redundant_support"
            else:
                details["reason"] = "separation"

            details["decision"] = "accept"
            return True, confidence, details
        else:
            details["reason"] = "no_support_no_separation"
            details["decision"] = "reject"
            return False, 0.0, details

    def get_params(self) -> Dict[str, Any]:
        """Get current parameter values."""
        return {
            "base_threshold": self.base_threshold,
            "relaxation_alpha": self.relaxation_alpha,
            "support_beta": self.support_beta,
            "min_support": self.min_support,
            "separation_delta": self.separation_delta,
            "use_distance": self.use_distance,
        }


def create_pruning_strategy(config: Dict[str, Any]) -> PruningStrategy:
    """
    Factory function to create a pruning strategy from config.

    Args:
        config: Dictionary with 'strategy' key and strategy-specific params

    Returns:
        Instantiated PruningStrategy
    """
    strategy_name = config.get("strategy", "redundant_support")

    if strategy_name == "redundant_support":
        return RedundantSupportStrategy(
            base_threshold=config.get("base_threshold", 0.45),
            relaxation_alpha=config.get("relaxation_alpha", 1.1),
            support_beta=config.get("support_beta", 1.05),
            min_support=config.get("min_support", 2),
            separation_delta=config.get("separation_delta", 0.15),
            use_distance=config.get("use_distance", True),
        )
    else:
        raise ValueError(f"Unknown pruning strategy: {strategy_name}")
