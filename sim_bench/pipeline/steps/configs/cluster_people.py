"""Typed config for cluster_people step (spec-033 P-G)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ClusterPeopleConfig(BaseModel):
    """Config for `cluster_people` step.

    Method-specific fields are validated only when used — the model itself
    permits the union of all method's keys so a profile can switch methods
    without re-validation. Hard contract: `extra="forbid"` still rejects
    typo'd keys.
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal[
        "face_cluster_knn",
        "hdbscan",
        "hdbscan_pca",
        "hybrid_hdbscan_knn",
        "mutual_knn",
        "agglomerative",
    ] = Field(
        default="face_cluster_knn",
        description=(
            "Clustering algorithm. 'face_cluster_knn' is the FC App algorithm "
            "(mutual kNN + connected components + optional merge/attach); the "
            "others are legacy."
        ),
    )

    # --- face_cluster_knn parameters ---
    K: int = Field(
        default=5, ge=1,
        description="Mutual kNN neighbours for edge creation.",
    )
    distance_threshold: float = Field(
        default=0.35, ge=0.0, le=2.0,
        description="Max cosine distance for kNN edge. Lower = stricter.",
    )
    min_cluster_size: int = Field(
        default=2, ge=1,
        description="Components below this become noise.",
    )
    merge_enabled: bool = Field(
        default=False,
        description="Run conservative cluster merge after base clustering.",
    )
    attach_enabled: bool = Field(
        default=False,
        description="Try to attach quality-failed (holdout) faces to clusters.",
    )
    split_enabled: bool = Field(
        default=False,
        description="Run cluster split safeguard.",
    )
    export_for_analysis: bool = Field(
        default=False,
        description="Write FC App artifacts (faces.csv, etc.) to the run dir.",
    )

    # --- bridge-disabled quality gates (spec-033 P-C will unblock these) ---
    yaw_max: float = Field(
        default=999.0, ge=0.0,
        description=(
            "degrees. Max absolute yaw. Currently force-disabled on Albumify "
            "via face_cluster_bridge.py:66-71 (SIGHTING-059). Spec-033 P-C unblocks."
        ),
    )
    pitch_max: float = Field(
        default=999.0, ge=0.0,
        description=(
            "degrees. Max absolute pitch. Currently force-disabled (SIGHTING-059)."
        ),
    )
    roll_max: float = Field(
        default=999.0, ge=0.0,
        description=(
            "degrees. Max absolute roll. Currently force-disabled (SIGHTING-059)."
        ),
    )
    blur_min: float = Field(
        default=0.0, ge=0.0,
        description=(
            "Min Laplacian variance. Currently force-disabled (SIGHTING-059)."
        ),
    )
    det_score_min: Optional[float] = Field(
        default=None,
        description=(
            "Min InsightFace det_score. Currently force-disabled (SIGHTING-059)."
        ),
    )

    # --- merge thresholds (used when merge_enabled=True) ---
    merge_candidate_threshold: float = Field(
        default=0.45, ge=0.0, le=2.0,
        description="Loose threshold for proposing merge candidates.",
    )
    merge_exemplar_threshold: float = Field(
        default=0.35, ge=0.0, le=2.0,
        description="Fixed exemplar threshold (used when adaptive is disabled).",
    )
    merge_use_cross_gate: bool = Field(
        default=True,
        description="Allow Gate A to pass via p25 all-pairs cross-distance (small clusters).",
    )
    merge_cross_threshold: float = Field(
        default=0.40, ge=0.0, le=2.0,
        description="Threshold for p25 of all cross-cluster node-pair distances.",
    )
    merge_cross_max_size: int = Field(
        default=5, ge=1,
        description="Cross-gate OR path only when min(|A|,|B|) <= this.",
    )
    merge_support_frac: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="Minimum fraction of smaller cluster supporting the merge.",
    )
    merge_support_min: int = Field(
        default=2, ge=0,
        description="Minimum absolute support count.",
    )
    merge_support_unique: bool = Field(
        default=False,
        description="Use greedy bipartite matching (each node used at most once).",
    )
    merge_margin: float = Field(
        default=0.05, ge=0.0,
        description="Min margin between best and second-best candidate distance.",
    )
    merge_diameter_expansion_factor: float = Field(
        default=1.5, ge=1.0,
        description="Max allowed post-merge diameter relative to larger input.",
    )

    # --- diameter cap (spec-031) ---
    cluster_diameter_cap_enabled: bool = Field(
        default=False,
        description="Post-merge absolute diameter ceiling. Reverts violators.",
    )
    max_full_diameter: float = Field(
        default=1.2, ge=0.0, le=2.0,
        description="Max pairwise cosine distance across all faces in cluster.",
    )
    max_exemplar_diameter: float = Field(
        default=0.8, ge=0.0, le=2.0,
        description="Max pairwise cosine distance across cluster exemplars only.",
    )

    # --- legacy method parameters (passed through to non-face_cluster_knn methods) ---
    n_clusters: Optional[int] = Field(
        default=None, ge=1,
        description="Fixed cluster count (agglomerative only).",
    )
    cluster_selection_epsilon: float = Field(
        default=0.3, ge=0.0,
        description="HDBSCAN cluster-selection epsilon.",
    )
    pca_components: int = Field(
        default=128, ge=1,
        description="PCA dims (hdbscan_pca only).",
    )
    min_samples: int = Field(
        default=2, ge=1,
        description="HDBSCAN min_samples.",
    )
    k: int = Field(
        default=10, ge=1,
        description="kNN k (mutual_knn legacy method only).",
    )
    similarity_threshold: float = Field(
        default=0.70, ge=0.0, le=1.0,
        description="Similarity threshold (mutual_knn legacy method).",
    )

    # Legacy hybrid/HDBSCAN tuning (kept for backward compat with profiles)
    knn_k: int = Field(default=3, ge=1, description="Legacy hybrid kNN k.")
    threshold_floor: float = Field(
        default=0.125, ge=0.0,
        description="Min per-cluster merge/attach threshold (legacy).",
    )
    threshold_ceiling: float = Field(
        default=0.405, ge=0.0,
        description="Max per-cluster merge/attach threshold (legacy).",
    )
    max_exemplars: int = Field(default=10, ge=1, description="Legacy exemplar cap.")
    attach_min_exemplars: int = Field(default=2, ge=1, description="Legacy attach minimum.")
    merge_min_pairs: int = Field(default=3, ge=1, description="Legacy merge minimum pair count.")
    max_faces_per_image_core: int = Field(
        default=3, ge=1,
        description="Keep N largest faces per image in core set.",
    )

    # Adaptive merge formulation
    use_adaptive_merge_threshold: bool = Field(
        default=True,
        description="Use adaptive per-cluster thresholds instead of fixed merge_exemplar_threshold.",
    )
    merge_exemplar_percentile: int = Field(
        default=90, ge=0, le=100,
        description="Percentile of intra-cluster exemplar distances → T_local.",
    )
    merge_global_percentile: int = Field(
        default=75, ge=0, le=100,
        description="Percentile across all cluster thresholds → T_global.",
    )
    merge_threshold_alpha: float = Field(
        default=1.0, ge=0.0,
        description="Weight of T_local in adaptive formula.",
    )
    merge_threshold_beta: float = Field(
        default=0.5, ge=0.0,
        description="Weight of T_global in adaptive formula.",
    )
