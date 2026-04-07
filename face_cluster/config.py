"""Configuration dataclass for face clustering pipeline."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PipelineConfig:
    """Configuration for face clustering pipeline.

    Core clustering parameters:
        K: Number of nearest neighbors for mutual kNN graph
        distance_threshold: Maximum cosine distance for edge creation
        min_cluster_size: Minimum cluster size (mark smaller as noise)

    Quality gating parameters:
        yaw_max: Maximum absolute yaw angle (degrees) for core set
        pitch_max: Maximum absolute pitch angle (degrees) for core set
        roll_max: Maximum absolute roll angle (degrees) for core set
        blur_min: Minimum blur score (Laplacian variance) for core set
        max_faces_per_image_core: Keep only top N faces by area per image
        min_face_area: Minimum face area in pixels (optional)

    Exemplar selection parameters:
        d10_k: K for d10 computation (kth nearest neighbor distance)
        exemplars_d10_threshold: Max d10 value to be exemplar candidate
        N_exemplars_max: Maximum number of exemplars per cluster
        exemplar_suppression_radius: Min distance between exemplars

    Cluster splitting parameters (optional):
        split_enabled: Whether to run split safeguard
        split_diameter_threshold: Max diameter before trying to split
        split_distance_threshold: Distance threshold for internal splitting
        split_K: K for internal mutual kNN graph when splitting

    Conservative merge parameters (optional):
        merge_enabled: Whether to run conservative merge
        merge_use_adaptive_threshold: Use adaptive per-cluster thresholds (recommended)
        merge_exemplar_percentile: Percentile for cluster threshold (e.g., 90 = P90)
        merge_global_percentile: Percentile for global threshold (50 = median, 75 = more permissive)
        merge_threshold_alpha: Weight for local vs global (0.7 = 70% local, 30% global)
        merge_candidate_threshold: Loose threshold for proposing candidates
        merge_exemplar_threshold: Fallback threshold if adaptive disabled
        merge_support_frac: Fraction of min(|Ci|,|Cj|) for cross-cluster support
        merge_support_min: Absolute minimum support (fallback)
        merge_margin: Margin to next-best cluster (avoid ambiguous merges)
        merge_diameter_expansion_factor: Allow diameter growth by this factor

    Holdout attachment parameters (optional):
        attach_enabled: Whether to attach holdout faces to clusters
        attach_distance_threshold: Max distance to attach
        K_attach: Number of nearest neighbors to check for voting
        vote_min: Minimum votes required from K_attach neighbors
        margin: Minimum margin between best and second-best cluster
    """
    # Core clustering
    K: int = 5
    distance_threshold: float = 0.35
    min_cluster_size: int = 2

    # Quality gating
    yaw_max: float = 30.0
    pitch_max: float = 25.0
    roll_max: float = 25.0
    blur_min: float = 50.0
    max_faces_per_image_core: int = 3
    min_face_area: Optional[int] = None
    require_pose: bool = False
    """If True, faces without pose go to holdout. If False (default), pose filter is skipped when pose is None."""

    # Exemplar selection
    d10_k: int = 3
    exemplars_d10_threshold: float = 0.35
    N_exemplars_max: int = 10
    exemplar_suppression_radius: float = 0.2

    # Splitting safeguards
    split_enabled: bool = False
    split_diameter_threshold: float = 0.6
    split_distance_threshold: float = 0.30
    split_K: int = 3

    # Conservative merge (optional)
    merge_enabled: bool = False

    # Adaptive thresholds (recommended)
    merge_use_adaptive_threshold: bool = True  # Use adaptive per-cluster thresholds
    merge_exemplar_percentile: int = 90        # Percentile for cluster threshold (P90)
    merge_global_percentile: int = 50          # Percentile for global threshold (50=median)
    merge_threshold_alpha: float = 0.7         # Weight: α×local + (1-α)×global

    # Fallback thresholds (used if adaptive disabled or for proposals)
    merge_candidate_threshold: float = 0.45    # Loose threshold for proposing candidates
    merge_exemplar_threshold: float = 0.35     # Fallback: min exemplar distance

    # Support count
    merge_support_frac: float = 0.3            # Fraction of min(|Ci|,|Cj|)
    merge_support_min: int = 2                 # Absolute minimum (fallback)

    # Safety constraints
    merge_margin: float = 0.05                 # Margin to next-best cluster
    merge_diameter_expansion_factor: float = 1.5  # Allow diameter to grow by this factor

    # Holdout attachment
    attach_enabled: bool = False
    attach_distance_threshold: float = 0.35
    K_attach: int = 5
    vote_min: int = 3
    margin: float = 0.1

    def __post_init__(self):
        """Validate configuration."""
        assert self.K > 0, "K must be positive"
        assert 0 <= self.distance_threshold <= 2, "distance_threshold must be in [0, 2]"
        assert self.min_cluster_size >= 1, "min_cluster_size must be >= 1"
        assert self.d10_k > 0, "d10_k must be positive"
        assert self.N_exemplars_max > 0, "N_exemplars_max must be positive"
