"""spec-041 — single contract for FC App v2 configuration.

The validation contract — defaults, range bounds (``ge``/``le``), and
human-readable descriptions. **No UI metadata lives here.** This module
is Streamlit-unaware; it could be consumed by a CLI-only or library-only
caller.

UI rendering metadata for the v2 Streamlit app lives in
``app/face_clustering_v2/ui_spec.py``. That module references FCParams
fields by name and pulls ``ge``/``le``/default at render time — there
is no second declaration of any range or default. See spec-041
``CONTRACT_OPTIONS.html`` and the follow-up that introduced the split.

Drift policed by:

* ``tests/architecture/test_fcparams_fcconfig_parity.py`` — every
  FCParams field has a matching field on ``FCConfig`` (minus runtime
  fields).
* ``tests/architecture/test_ui_spec_references_fcparams.py`` — every
  ``UI_SPEC`` entry references a real FCParams field, and every
  field whose ``zero_is_none`` is set is genuinely Optional.

Fields without a ``UI_SPEC`` entry stay invisible to the v2 UI — they
keep their FCParams defaults (e.g., ``d10_k``, ``K_attach``,
``embed_cache_enabled``).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from face_cluster.config import PipelineConfig as FCConfig


# Runtime fields on FCConfig that are NOT user-tunable knobs and therefore
# are NOT mirrored on FCParams. Excluded from the parity test.
RUNTIME_FIELDS = frozenset({"stages", "source_dir", "output_dir", "on_progress"})


class FCParams(BaseModel):
    """Single configuration container for the FC App v2 pipeline.

    Each field mirrors a tunable knob on
    ``face_cluster.config.PipelineConfig``. UI rendering hints live in
    ``app/face_clustering_v2/ui_spec.py`` — never here.
    """
    model_config = ConfigDict(extra="forbid")

    # --- Core clustering ----------------------------------------------------
    K: int = Field(5, ge=1, le=100,
                   description="Mutual kNN edge requires both nodes in each other's top-K.")
    distance_threshold: float = Field(
        0.35, ge=0.01, le=1.0,
        description="Max cosine distance for an edge. Lower = stricter, more clusters.",
    )
    min_cluster_size: int = Field(
        2, ge=1, le=50,
        description="Components below this become noise.",
    )

    # --- Quality gating -----------------------------------------------------
    yaw_max: float = Field(
        30.0, ge=5.0, le=999.0,
        description="Max yaw angle in degrees. Set to 999 to disable the gate.",
    )
    pitch_max: float = Field(
        25.0, ge=5.0, le=999.0,
        description="Max pitch angle in degrees. Set to 999 to disable.",
    )
    roll_max: float = Field(
        25.0, ge=5.0, le=999.0,
        description="Max roll angle in degrees. Set to 999 to disable.",
    )
    blur_min: float = Field(
        50.0, ge=0.0, le=500.0,
        description=("Min Laplacian variance. NO EFFECT on Albumify today — the "
                     "InsightFace pipeline has no blur scorer."),
    )
    max_faces_per_image_core: int = Field(
        3, ge=1, le=50,
        description="Keep N largest faces per image in the core set.",
    )
    min_face_area: Optional[int] = Field(
        None, ge=0,
        description="Minimum face area in pixels. None disables the gate.",
    )
    require_pose: bool = Field(
        False,
        description="If on, faces without pose data go to holdout.",
    )
    det_score_min: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Min InsightFace detection confidence. Recommended: 0.7.",
    )

    # --- Exemplar selection -------------------------------------------------
    # d10_k is algorithm-only — no UI binding.
    d10_k: int = Field(3, ge=1, le=20)
    exemplars_d10_threshold: float = Field(
        0.35, ge=0.01, le=1.0,
        description="Max d10 (10th-NN cosine distance) for a node to be exemplar-eligible.",
    )
    N_exemplars_max: int = Field(
        10, ge=1, le=100,
        description="Max exemplars selected per cluster.",
    )
    exemplar_suppression_radius: float = Field(
        0.2, ge=0.01, le=1.0,
        description="Min cosine distance between selected exemplars.",
    )

    # --- Splitting safeguards ----------------------------------------------
    split_enabled: bool = Field(
        False,
        description="Run cluster split safeguard after base clustering.",
    )
    split_diameter_threshold: float = Field(0.6, ge=0.0, le=2.0)
    split_distance_threshold: float = Field(0.30, ge=0.01, le=1.0)
    split_K: int = Field(3, ge=1, le=50)

    # --- Conservative merge -------------------------------------------------
    merge_enabled: bool = Field(
        False,
        description="Run conservative merge after exemplar selection.",
    )
    merge_candidate_threshold: float = Field(
        0.45, ge=0.01, le=1.5,
        description="Loose threshold for proposing merge candidates.",
    )
    merge_exemplar_threshold: float = Field(
        0.35, ge=0.01, le=1.5,
        description="Fixed exemplar threshold used when adaptive is disabled.",
    )
    use_adaptive_merge_threshold: bool = Field(
        True,
        description="Use adaptive per-cluster thresholds instead of a fixed value.",
    )
    merge_exemplar_percentile: int = Field(
        90, ge=0, le=100,
        description="Percentile of intra-cluster exemplar distances used as T_local.",
    )
    merge_global_percentile: int = Field(
        75, ge=0, le=100,
        description="Percentile across all cluster thresholds used as T_global.",
    )
    merge_threshold_alpha: float = Field(
        1.0, ge=0.0, le=3.0,
        description="Weight of T_local in adaptive formula.",
    )
    merge_threshold_beta: float = Field(
        0.5, ge=0.0, le=3.0,
        description="Weight of T_global in adaptive formula.",
    )
    merge_use_cross_gate: bool = Field(
        True,
        description="Allow Gate A to pass via p25 all-pairs cross-distance.",
    )
    merge_cross_threshold: float = Field(
        0.40, ge=0.01, le=1.5,
        description="Threshold for p25 of all cross-cluster node-pair distances.",
    )
    merge_cross_max_size: int = Field(
        5, ge=1, le=20,
        description="OR path only when min(|A|,|B|) <= this.",
    )
    merge_support_frac: float = Field(
        0.3, ge=0.0, le=1.0,
        description="Min fraction of the smaller cluster in the cross-cluster K-NN.",
    )
    merge_support_min: int = Field(
        2, ge=1, le=20,
        description="Min absolute number of supporting cross-cluster pairs.",
    )
    merge_support_unique: bool = Field(
        False,
        description="Greedy bipartite matching: each node used at most once.",
    )
    merge_margin: float = Field(
        0.05, ge=0.0, le=0.5,
        description="Min gap between best and second-best candidate distances.",
    )
    merge_diameter_expansion_factor: float = Field(
        1.5, ge=1.0, le=5.0,
        description="Max allowed post-merge diameter relative to larger input cluster.",
    )

    # --- Diameter cap (spec-031) -------------------------------------------
    cluster_diameter_cap_enabled: bool = Field(
        False,
        description="Post-merge safety net: reject clusters whose diameter exceeds the ceiling.",
    )
    max_full_diameter: float = Field(
        1.2, ge=0.3, le=2.0,
        description="Max pairwise cosine distance across all faces in the cluster.",
    )
    max_exemplar_diameter: float = Field(
        0.8, ge=0.2, le=2.0,
        description="Max pairwise cosine distance restricted to the cluster's exemplars.",
    )

    # --- Holdout attachment ------------------------------------------------
    attach_enabled: bool = Field(
        False,
        description="Try to attach quality-failed (holdout) faces to clusters.",
    )
    attach_distance_threshold: float = Field(0.35, ge=0.01, le=1.0)
    K_attach: int = Field(5, ge=1, le=50)
    vote_min: int = Field(3, ge=1, le=50)
    margin: float = Field(0.1, ge=0.0, le=1.0)

    # --- Embed cache --------------------------------------------------------
    embed_cache_enabled: bool = True

    # ------------------------------------------------------------------ API
    def to_fc_config(self) -> "FCConfig":
        """Translate to the algorithm-layer dataclass."""
        from face_cluster.config import PipelineConfig as FCConfig
        return FCConfig(**self.model_dump())

    def to_step_configs(self) -> Dict[str, Dict[str, Any]]:
        """Broadcast over the unified clustering step chain."""
        from face_cluster.fc_app_runner import UNIFIED_CLUSTERING_STEPS
        d = self.model_dump()
        return {name: dict(d) for name in UNIFIED_CLUSTERING_STEPS}

    @classmethod
    def load(cls, path) -> "FCParams":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")


__all__ = ["FCParams", "RUNTIME_FIELDS"]
