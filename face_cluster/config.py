"""Configuration dataclass for face clustering pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional


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
        merge_candidate_threshold: Loose threshold for proposing candidates
        merge_exemplar_threshold: Fixed threshold for exemplar gate (p25 exemplar distance must be <= this)
        merge_use_cross_gate: Enable p25_cross_dist OR path for Gate A
        merge_cross_threshold: Threshold for p25 all-pairs cross-distance OR gate
        merge_cross_max_size: OR path only applies when min(|A|,|B|) <= this value
        merge_support_frac: Fraction of min(|Ci|,|Cj|) for cross-cluster support
        merge_support_min: Absolute minimum support (fallback)
        merge_support_unique: Use greedy bipartite matching for support count (each node used at most once)
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
    det_score_min: Optional[float] = None
    """Minimum InsightFace detection confidence (0–1). None = disabled.
    Faces with det_score below this threshold go to holdout.
    When det_score is unavailable for a face, the gate passes (permissive).
    Recommended starting value: 0.7."""

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

    # Merge thresholds
    merge_candidate_threshold: float = 0.45    # Loose threshold for proposing candidates
    merge_exemplar_threshold: float = 0.35     # Fixed threshold: p25 exemplar distance must be <= this

    # Adaptive merge threshold (alternative to fixed merge_exemplar_threshold)
    use_adaptive_merge_threshold: bool = True
    merge_exemplar_percentile: int = 90        # Percentile of intra-cluster exemplar dists → T_local
    merge_global_percentile: int = 75          # Percentile across all cluster thresholds → T_global
    merge_threshold_alpha: float = 1.0         # Weight of T_local in adaptive formula
    merge_threshold_beta: float = 0.5          # Weight of T_global in adaptive formula

    # Cross-distance OR gate (Gate A alternative path — small clusters only)
    merge_use_cross_gate: bool = True          # Enable p25_cross_dist OR path for Gate A
    merge_cross_threshold: float = 0.40        # Threshold for p25 all-pairs cross-distance
    merge_cross_max_size: int = 5              # OR path only when min(|A|,|B|) <= this

    # Support count
    merge_support_frac: float = 0.3            # Fraction of min(|Ci|,|Cj|)
    merge_support_min: int = 2                 # Absolute minimum (fallback)
    merge_support_unique: bool = False         # Use greedy bipartite matching (each node used once)

    # Safety constraints
    merge_margin: float = 0.05                 # Margin to next-best cluster
    merge_diameter_expansion_factor: float = 1.5  # Allow diameter to grow by this factor

    # Holdout attachment
    attach_enabled: bool = False
    attach_distance_threshold: float = 0.35
    K_attach: int = 5
    vote_min: int = 3
    margin: float = 0.1

    # Embed cache — skip InsightFace inference when images haven't changed
    embed_cache_enabled: bool = True

    # Pipeline execution — which stages to run and where
    stages: Optional[List[str]] = None
    """Ordered list of stage names to execute. None means full run (all stages)."""
    source_dir: Optional[str] = None
    """Input directory: raw images (full run), previous run output (recluster/remerge)."""
    output_dir: Optional[str] = None
    """Output directory for this run."""
    on_progress: Optional[Callable] = field(default=None, repr=False)
    """Progress callback (stage, fraction, message). Not serialized."""

    def __post_init__(self):
        """Validate configuration."""
        assert self.K > 0, "K must be positive"
        assert 0 <= self.distance_threshold <= 2, "distance_threshold must be in [0, 2]"
        assert self.min_cluster_size >= 1, "min_cluster_size must be >= 1"
        assert self.d10_k > 0, "d10_k must be positive"
        assert self.N_exemplars_max > 0, "N_exemplars_max must be positive"

    # -- Preset factories ----------------------------------------------------

    @classmethod
    def full_run(cls, source_dir, output_dir, **kwargs) -> "PipelineConfig":
        """Full pipeline: discover -> embed -> quality -> crops -> cluster -> exemplars -> merge -> export."""
        return cls(
            stages=["discover", "embed", "quality", "crops",
                    "cluster", "exemplars", "merge", "export"],
            source_dir=str(source_dir),
            output_dir=str(output_dir),
            **kwargs,
        )

    @classmethod
    def recluster(cls, source_dir, output_dir, **kwargs) -> "PipelineConfig":
        """Recluster: reuse existing crops/embeddings, re-run cluster -> exemplars -> merge -> export."""
        return cls(
            stages=["cluster", "exemplars", "merge", "export"],
            source_dir=str(source_dir),
            output_dir=str(output_dir),
            **kwargs,
        )

    @classmethod
    def remerge(cls, source_dir, output_dir, *, with_exemplars: bool = False, **kwargs) -> "PipelineConfig":
        """Remerge: start from existing cluster snapshot, run merge -> export.

        Args:
            with_exemplars: If True, re-run exemplar selection before merge
                            (exemplars -> merge -> export).
        """
        stages = (["exemplars", "merge", "export"] if with_exemplars
                  else ["merge", "export"])
        return cls(
            stages=stages,
            source_dir=str(source_dir),
            output_dir=str(output_dir),
            **kwargs,
        )
