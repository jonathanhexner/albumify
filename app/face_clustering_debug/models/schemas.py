"""Data models for face clustering debug app.

These dataclasses define the data structures passed between layers.
They are independent of storage format (DB vs files).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np


@dataclass
class FaceInfo:
    """Single face metadata."""
    index: int
    image_path: str
    bbox: Tuple[float, float, float, float]  # (x, y, w, h) - relative coordinates
    confidence: float
    crop_path: Optional[str] = None
    landmarks: Optional[List[Tuple[float, float]]] = None  # 5 points: eyes, nose, mouth
    pose_angles: Optional[Tuple[float, float, float]] = None  # pitch, yaw, roll
    # Quality metrics
    frontal_score: float = 0.0
    eye_bbox_ratio: float = 0.0
    asymmetry_ratio: float = 0.0


@dataclass
class ClusterInfo:
    """Single cluster with its faces and statistics."""
    cluster_id: int
    face_indices: List[int]
    threshold: float
    exemplar_indices: List[int]
    # Threshold computation stats
    q1: float = 0.0
    q3: float = 0.0
    iqr: float = 0.0
    raw_threshold: float = 0.0  # Before clamping to floor/ceiling


@dataclass
class MergeDecision:
    """Record of a merge decision between two clusters.

    Works with both hybrid_hdbscan_knn (exemplar-based) and hybrid_closest_face (face-based).
    """
    cluster_a: int
    cluster_b: int
    threshold_a: float          # T of cluster A
    threshold_b: float          # T of cluster B
    merged: bool
    reason: str
    min_distance: float = 0.0   # Min distance (exemplar for knn, exemplar early-exit for closest)

    # Common fields (used by both algorithms)
    cross_distances: Optional[np.ndarray] = None  # Full exemplar cross-distance matrix

    # hybrid_hdbscan_knn specific (exemplar-based merge)
    threshold_used: float = 0.0  # The threshold that was applied (min or specific direction)
    pairs_within_threshold: int = 0
    exemplars_a_involved: int = 0
    exemplars_b_involved: int = 0
    min_dists_a: Optional[List[float]] = None     # Per A-exemplar: min dist to any B-exemplar
    min_dists_b: Optional[List[float]] = None     # Per B-exemplar: min dist to any A-exemplar

    # hybrid_closest_face specific (face-based d3_cross merge)
    d3_cross_a: Optional[List[float]] = None      # Per A-face: d3_cross to B
    d3_cross_b: Optional[List[float]] = None      # Per B-face: d3_cross to A
    fits_a: int = 0                               # How many A faces fit into B
    fits_b: int = 0                               # How many B faces fit into A
    n_fits_total: int = 0                         # Total fits (fits_a + fits_b)
    effective_threshold_a: float = 0.0            # T_A * merge_threshold_multiplier
    effective_threshold_b: float = 0.0            # T_B * merge_threshold_multiplier
    merge_threshold_multiplier: float = 1.0
    merge_min_faces: int = 2
    early_exit_threshold: float = 0.0             # early_exit_multiplier * max(T_A, T_B)


@dataclass
class AttachDecision:
    """Record of an attachment decision for a noise point."""
    face_index: int
    attached_to: Optional[int]  # Cluster ID or None if stayed noise
    reason: str  # 'attached', 'no_cluster_qualified', 'below_threshold', etc.
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    # Each candidate: {cluster_id, distance, qualified, reason}


@dataclass
class ClusteringRequest:
    """Container for a clustering run request."""
    algorithm: str
    params: Dict[str, Any]
    embeddings: np.ndarray
    faces: List[FaceInfo]
    collect_debug_data: bool = True


@dataclass
class ClusteringResult:
    """Complete clustering result with all debug data."""
    labels: np.ndarray  # Cluster label per face (-1 for noise)
    embeddings: np.ndarray  # Face embeddings [N, 512]
    faces: List[FaceInfo]
    clusters: List[ClusterInfo]
    merge_decisions: List[MergeDecision]
    attach_decisions: List[AttachDecision]
    algorithm: str
    params: Dict[str, Any]
    # Summary stats
    n_clusters: int = 0
    n_noise: int = 0

    def __post_init__(self):
        """Compute summary stats after initialization."""
        if self.n_clusters == 0 and len(self.clusters) > 0:
            self.n_clusters = len(self.clusters)
        if self.n_noise == 0 and len(self.labels) > 0:
            self.n_noise = int(np.sum(self.labels == -1))
