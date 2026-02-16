"""Attachment strategies for face-to-cluster assignment in identity refinement."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AttachmentResult:
    """Result of an attachment evaluation."""
    attached: bool
    cluster_id: Optional[int] = None
    confidence: float = 0.0
    centroid_distance: Optional[float] = None
    best_exemplar_distance: Optional[float] = None
    exemplar_matches: int = 0
    min_required_matches: int = 0
    reason: str = ""


@dataclass
class ClusterInfo:
    """Information about a cluster for attachment evaluation."""
    cluster_id: int
    centroid: np.ndarray
    exemplar_embeddings: List[np.ndarray]


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two vectors."""
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return float(1.0 - np.dot(a_norm, b_norm))


class AttachmentStrategy(ABC):
    """Base class for attachment strategies."""

    @abstractmethod
    def evaluate(
        self,
        face_embedding: np.ndarray,
        cluster_info: ClusterInfo,
        config: Dict[str, Any]
    ) -> AttachmentResult:
        """Evaluate whether a face should attach to a cluster.

        Args:
            face_embedding: The face embedding vector
            cluster_info: Information about the candidate cluster
            config: Configuration parameters

        Returns:
            AttachmentResult with decision and metadata
        """
        pass

    def _compute_min_required_matches(self, num_exemplars: int, config: Dict[str, Any]) -> int:
        """Compute minimum required exemplar matches."""
        small_cluster_threshold = config.get("small_cluster_threshold", 3)
        small_cluster_min_matches = config.get("small_cluster_min_matches", 1)

        if num_exemplars <= small_cluster_threshold:
            return small_cluster_min_matches

        return max(2, int(np.ceil(0.3 * num_exemplars)))


class CentroidStrategy(AttachmentStrategy):
    """Attach based on centroid distance only."""

    def evaluate(
        self,
        face_embedding: np.ndarray,
        cluster_info: ClusterInfo,
        config: Dict[str, Any]
    ) -> AttachmentResult:
        centroid_threshold = config.get("centroid_threshold", 0.38)
        reject_threshold = config.get("reject_threshold", 0.45)

        centroid_dist = cosine_distance(face_embedding, cluster_info.centroid)

        if centroid_dist > reject_threshold:
            return AttachmentResult(
                attached=False,
                cluster_id=cluster_info.cluster_id,
                centroid_distance=centroid_dist,
                reason=f"Centroid distance {centroid_dist:.3f} > reject threshold {reject_threshold}"
            )

        if centroid_dist <= centroid_threshold:
            confidence = 1.0 - (centroid_dist / reject_threshold)
            return AttachmentResult(
                attached=True,
                cluster_id=cluster_info.cluster_id,
                confidence=confidence,
                centroid_distance=centroid_dist,
                reason=f"Centroid distance {centroid_dist:.3f} <= threshold {centroid_threshold}"
            )

        return AttachmentResult(
            attached=False,
            cluster_id=cluster_info.cluster_id,
            centroid_distance=centroid_dist,
            reason=f"Centroid distance {centroid_dist:.3f} > threshold {centroid_threshold}"
        )


class ExemplarStrategy(AttachmentStrategy):
    """Attach based on exemplar matches only."""

    def evaluate(
        self,
        face_embedding: np.ndarray,
        cluster_info: ClusterInfo,
        config: Dict[str, Any]
    ) -> AttachmentResult:
        exemplar_threshold = config.get("exemplar_threshold", 0.40)
        reject_threshold = config.get("reject_threshold", 0.45)

        if not cluster_info.exemplar_embeddings:
            return AttachmentResult(
                attached=False,
                cluster_id=cluster_info.cluster_id,
                reason="No exemplars available"
            )

        exemplar_dists = [
            cosine_distance(face_embedding, ex_emb)
            for ex_emb in cluster_info.exemplar_embeddings
        ]
        best_dist = min(exemplar_dists)

        if best_dist > reject_threshold:
            return AttachmentResult(
                attached=False,
                cluster_id=cluster_info.cluster_id,
                best_exemplar_distance=best_dist,
                reason=f"Best exemplar distance {best_dist:.3f} > reject threshold {reject_threshold}"
            )

        matches = sum(1 for d in exemplar_dists if d <= exemplar_threshold)
        min_required = self._compute_min_required_matches(len(exemplar_dists), config)

        if matches >= min_required:
            confidence = 1.0 - (best_dist / reject_threshold)
            return AttachmentResult(
                attached=True,
                cluster_id=cluster_info.cluster_id,
                confidence=confidence,
                best_exemplar_distance=best_dist,
                exemplar_matches=matches,
                min_required_matches=min_required,
                reason=f"Exemplar matches {matches} >= required {min_required}"
            )

        return AttachmentResult(
            attached=False,
            cluster_id=cluster_info.cluster_id,
            best_exemplar_distance=best_dist,
            exemplar_matches=matches,
            min_required_matches=min_required,
            reason=f"Exemplar matches {matches} < required {min_required}"
        )


class HybridStrategy(AttachmentStrategy):
    """Require both centroid AND exemplar criteria."""

    def evaluate(
        self,
        face_embedding: np.ndarray,
        cluster_info: ClusterInfo,
        config: Dict[str, Any]
    ) -> AttachmentResult:
        centroid_threshold = config.get("centroid_threshold", 0.38)
        exemplar_threshold = config.get("exemplar_threshold", 0.40)
        reject_threshold = config.get("reject_threshold", 0.45)

        centroid_dist = cosine_distance(face_embedding, cluster_info.centroid)

        if not cluster_info.exemplar_embeddings:
            return AttachmentResult(
                attached=False,
                cluster_id=cluster_info.cluster_id,
                centroid_distance=centroid_dist,
                reason="No exemplars available for hybrid evaluation"
            )

        exemplar_dists = [
            cosine_distance(face_embedding, ex_emb)
            for ex_emb in cluster_info.exemplar_embeddings
        ]
        best_exemplar_dist = min(exemplar_dists)

        if best_exemplar_dist > reject_threshold:
            return AttachmentResult(
                attached=False,
                cluster_id=cluster_info.cluster_id,
                centroid_distance=centroid_dist,
                best_exemplar_distance=best_exemplar_dist,
                reason=f"Best distance {best_exemplar_dist:.3f} > reject threshold {reject_threshold}"
            )

        centroid_pass = centroid_dist <= centroid_threshold

        matches = sum(1 for d in exemplar_dists if d <= exemplar_threshold)
        min_required = self._compute_min_required_matches(len(exemplar_dists), config)
        exemplar_pass = matches >= min_required

        if centroid_pass and exemplar_pass:
            confidence = 1.0 - (centroid_dist / reject_threshold)
            return AttachmentResult(
                attached=True,
                cluster_id=cluster_info.cluster_id,
                confidence=confidence,
                centroid_distance=centroid_dist,
                best_exemplar_distance=best_exemplar_dist,
                exemplar_matches=matches,
                min_required_matches=min_required,
                reason=f"Hybrid pass: centroid={centroid_dist:.3f}, exemplar_matches={matches}/{min_required}"
            )

        reasons = []
        if not centroid_pass:
            reasons.append(f"centroid {centroid_dist:.3f} > {centroid_threshold}")
        if not exemplar_pass:
            reasons.append(f"exemplar_matches {matches} < {min_required}")

        return AttachmentResult(
            attached=False,
            cluster_id=cluster_info.cluster_id,
            centroid_distance=centroid_dist,
            best_exemplar_distance=best_exemplar_dist,
            exemplar_matches=matches,
            min_required_matches=min_required,
            reason=f"Hybrid fail: {', '.join(reasons)}"
        )


class AttachmentStrategyFactory:
    """Factory for creating attachment strategies."""

    _strategies = {
        "centroid": CentroidStrategy,
        "exemplar": ExemplarStrategy,
        "hybrid": HybridStrategy,
    }

    @classmethod
    def create(cls, strategy_name: str) -> AttachmentStrategy:
        """Create attachment strategy by name."""
        strategy_class = cls._strategies.get(strategy_name, HybridStrategy)
        logger.info(f"Created attachment strategy: {strategy_name}")
        return strategy_class()

    @classmethod
    def register(cls, name: str, strategy_class: type):
        """Register a new attachment strategy."""
        cls._strategies[name] = strategy_class
        logger.info(f"Registered attachment strategy: {name}")
