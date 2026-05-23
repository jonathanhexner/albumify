"""spec-041 — single Pydantic container for FC App v2 configuration.

The user-facing contract between:

* the FC App v2 Streamlit UI (``app/face_clustering_v2/tabs/run_tab.py``),
* the headless CLI (``scripts/run_v2.py``),
* profile JSON files (``~/.sim_bench/profiles_v2/<name>.json``),
* the integration tests, and
* the algorithm layer (via the one boundary translator ``to_fc_config()``).

Each UI-bound field carries a ``json_schema_extra`` block holding the
UI hints (widget type, display range, step, label, help, group, order).
The widget factory at ``app/face_clustering_v2/widget_factory.py`` reads
these hints — no widget literal is hand-written in run_tab.py.

Two layers of metadata per field:

* ``ge`` / ``le``  — the **validation contract**. Any value in this range
  is legal to pass to the algorithm layer (``FCConfig``). Sentinels like
  ``yaw_max=999.0`` ("disable pose gating") live here.
* ``json_schema_extra.ui_*`` — the **display hint**. What range the UI
  exposes via slider min/max, what step to advance, what to call it.
  These can be tighter than ``ge``/``le`` (the typical use range);
  consumers that want a sentinel value can type it explicitly via
  ``st.number_input``.

The drift between FCParams and FCConfig field sets is policed by
``tests/architecture/test_fcparams_fcconfig_parity.py``. The drift
between ``ge``/``le`` and ``ui_min``/``ui_max`` is policed by
``tests/architecture/test_fcparams_ui_hints_consistent.py``.

Fields without ``json_schema_extra`` are intentionally NOT bound to the
UI — they stay at their FCParams default and are invisible to the user
(e.g. ``d10_k``, ``K_attach``, ``embed_cache_enabled``).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from face_cluster.config import PipelineConfig as FCConfig


# Runtime fields on FCConfig that are NOT user-tunable knobs and therefore
# are NOT mirrored on FCParams. The parity test in
# tests/architecture/test_fcparams_fcconfig_parity.py excludes these.
RUNTIME_FIELDS = frozenset({"stages", "source_dir", "output_dir", "on_progress"})


def _ui(
    widget: str,
    label: str,
    group: str,
    order: int,
    help: str = "",
    *,
    min: Any = None,
    max: Any = None,
    step: Any = None,
    zero_is_none: bool = False,
) -> Dict[str, Any]:
    """Pack a json_schema_extra payload. Keeps Field(...) declarations short.

    Conventions:
      widget: "number_input" | "slider" | "checkbox"
      group:  "cluster" | "quality" | "exemplars" | "optional" | "merge" | "cap"
      order:  display order within the group (10, 20, 30, ...) — gaps make
              future inserts cheap.
      zero_is_none: for Optional[int|float] fields where 0 / 0.0 is the
              "disabled" sentinel at the UI layer but None at the contract.
    """
    out: Dict[str, Any] = {
        "ui_widget": widget,
        "ui_label": label,
        "ui_group": group,
        "ui_order": order,
        "ui_help": help,
    }
    if min is not None:
        out["ui_min"] = min
    if max is not None:
        out["ui_max"] = max
    if step is not None:
        out["ui_step"] = step
    if zero_is_none:
        out["ui_zero_is_none"] = True
    return out


class FCParams(BaseModel):
    """Single configuration container for the FC App v2 pipeline.

    Each field mirrors a tunable knob on ``face_cluster.config.PipelineConfig``.
    UI-bound fields carry ``json_schema_extra`` with display hints; see
    module docstring for the contract.
    """
    model_config = ConfigDict(extra="forbid")

    # --- Core clustering ----------------------------------------------------
    K: int = Field(
        5, ge=1, le=100,
        json_schema_extra=_ui(
            "slider", "K (kNN neighbours)", "cluster", 10,
            help="Mutual kNN edge requires both nodes in each other's top-K.",
            min=1, max=100, step=1,
        ),
    )
    distance_threshold: float = Field(
        0.35, ge=0.01, le=1.0,
        json_schema_extra=_ui(
            "slider", "distance_threshold", "cluster", 20,
            help="Max cosine distance for an edge. Lower = stricter, more clusters.",
            min=0.01, max=1.0, step=0.01,
        ),
    )
    min_cluster_size: int = Field(
        2, ge=1, le=50,
        json_schema_extra=_ui(
            "slider", "min_cluster_size", "cluster", 30,
            help="Components below this become noise.",
            min=1, max=50, step=1,
        ),
    )

    # --- Quality gating -----------------------------------------------------
    yaw_max: float = Field(
        30.0, ge=5.0, le=999.0,
        json_schema_extra=_ui(
            "number_input", "yaw_max °", "quality", 70,
            help="Max yaw angle in degrees. Set to 999 to disable the gate entirely.",
            min=5.0, max=90.0, step=1.0,
        ),
    )
    pitch_max: float = Field(
        25.0, ge=5.0, le=999.0,
        json_schema_extra=_ui(
            "number_input", "pitch_max °", "quality", 80,
            help="Max pitch angle in degrees. Set to 999 to disable.",
            min=5.0, max=90.0, step=1.0,
        ),
    )
    roll_max: float = Field(
        25.0, ge=5.0, le=999.0,
        json_schema_extra=_ui(
            "number_input", "roll_max °", "quality", 90,
            help="Max roll angle in degrees. Set to 999 to disable.",
            min=5.0, max=90.0, step=1.0,
        ),
    )
    blur_min: float = Field(
        50.0, ge=0.0, le=500.0,
        json_schema_extra=_ui(
            "slider", "blur_min", "quality", 10,
            help=("Min Laplacian variance. NO EFFECT on Albumify today — the "
                  "InsightFace pipeline has no blur scorer."),
            min=0.0, max=500.0, step=5.0,
        ),
    )
    max_faces_per_image_core: int = Field(
        3, ge=1, le=50,
        json_schema_extra=_ui(
            "slider", "max_faces_per_image_core", "quality", 20,
            help="Keep N largest faces per image in the core set.",
            min=1, max=50, step=1,
        ),
    )
    min_face_area: Optional[int] = Field(
        None, ge=0,
        json_schema_extra=_ui(
            "number_input", "min_face_area px (0=off)", "quality", 30,
            help="Minimum face area in pixels. 0 disables the gate.",
            min=0, max=100000, step=500, zero_is_none=True,
        ),
    )
    require_pose: bool = Field(
        False,
        json_schema_extra=_ui(
            "checkbox", "require_pose", "quality", 60,
            help="If on, faces without pose data go to holdout.",
        ),
    )
    det_score_min: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        json_schema_extra=_ui(
            "number_input", "det_score_min (0=off)", "quality", 40,
            help="Min InsightFace detection confidence. Recommended: 0.7.",
            min=0.0, max=1.0, step=0.05, zero_is_none=True,
        ),
    )

    # --- Exemplar selection -------------------------------------------------
    # d10_k is algorithm-only — not exposed in the UI.
    d10_k: int = Field(3, ge=1, le=20)
    exemplars_d10_threshold: float = Field(
        0.35, ge=0.01, le=1.0,
        json_schema_extra=_ui(
            "slider", "exemplars_d10_threshold", "exemplars", 20,
            help="Max d10 (10th-NN cosine distance) for a node to be exemplar-eligible.",
            min=0.01, max=1.0, step=0.01,
        ),
    )
    N_exemplars_max: int = Field(
        10, ge=1, le=100,
        json_schema_extra=_ui(
            "slider", "N_exemplars_max", "exemplars", 10,
            help="Max exemplars selected per cluster.",
            min=1, max=100, step=1,
        ),
    )
    exemplar_suppression_radius: float = Field(
        0.2, ge=0.01, le=1.0,
        json_schema_extra=_ui(
            "slider", "exemplar_suppression_radius", "exemplars", 30,
            help="Min cosine distance between selected exemplars.",
            min=0.01, max=1.0, step=0.01,
        ),
    )

    # --- Splitting safeguards (toggle only in UI for now) -------------------
    split_enabled: bool = Field(
        False,
        json_schema_extra=_ui(
            "checkbox", "split_enabled", "optional", 10,
            help="Run cluster split safeguard after base clustering.",
        ),
    )
    split_diameter_threshold: float = Field(0.6, ge=0.0, le=2.0)
    split_distance_threshold: float = Field(0.30, ge=0.01, le=1.0)
    split_K: int = Field(3, ge=1, le=50)

    # --- Conservative merge -------------------------------------------------
    merge_enabled: bool = Field(
        False,
        json_schema_extra=_ui(
            "checkbox", "merge_enabled", "optional", 20,
            help="Run conservative merge after exemplar selection.",
        ),
    )
    merge_candidate_threshold: float = Field(
        0.45, ge=0.01, le=1.5,
        json_schema_extra=_ui(
            "slider", "merge_candidate_threshold", "merge", 10,
            help="Loose threshold for proposing merge candidates.",
            min=0.01, max=1.5, step=0.01,
        ),
    )
    merge_exemplar_threshold: float = Field(
        0.35, ge=0.01, le=1.5,
        json_schema_extra=_ui(
            "slider", "merge_exemplar_threshold", "merge", 20,
            help="Fixed exemplar threshold used when adaptive is disabled.",
            min=0.01, max=1.5, step=0.01,
        ),
    )

    # Adaptive merge threshold.
    use_adaptive_merge_threshold: bool = Field(
        True,
        json_schema_extra=_ui(
            "checkbox", "use_adaptive_merge_threshold", "merge", 30,
            help="Use adaptive per-cluster thresholds instead of a fixed value.",
        ),
    )
    merge_exemplar_percentile: int = Field(
        90, ge=0, le=100,
        json_schema_extra=_ui(
            "slider", "exemplar_percentile", "merge", 40,
            help="Percentile of intra-cluster exemplar distances used as T_local.",
            min=0, max=100, step=1,
        ),
    )
    merge_global_percentile: int = Field(
        75, ge=0, le=100,
        json_schema_extra=_ui(
            "slider", "global_percentile", "merge", 50,
            help="Percentile across all cluster thresholds used as T_global.",
            min=0, max=100, step=1,
        ),
    )
    merge_threshold_alpha: float = Field(
        1.0, ge=0.0, le=3.0,
        json_schema_extra=_ui(
            "slider", "alpha", "merge", 60,
            help="Weight of T_local in adaptive formula.",
            min=0.0, max=3.0, step=0.05,
        ),
    )
    merge_threshold_beta: float = Field(
        0.5, ge=0.0, le=3.0,
        json_schema_extra=_ui(
            "slider", "beta", "merge", 70,
            help="Weight of T_global in adaptive formula.",
            min=0.0, max=3.0, step=0.05,
        ),
    )

    # Cross-distance OR gate.
    merge_use_cross_gate: bool = Field(
        True,
        json_schema_extra=_ui(
            "checkbox", "merge_use_cross_gate", "merge", 80,
            help="Allow Gate A to pass via p25 all-pairs cross-distance.",
        ),
    )
    merge_cross_threshold: float = Field(
        0.40, ge=0.01, le=1.5,
        json_schema_extra=_ui(
            "slider", "merge_cross_threshold", "merge", 90,
            help="Threshold for p25 of all cross-cluster node-pair distances.",
            min=0.01, max=1.5, step=0.01,
        ),
    )
    merge_cross_max_size: int = Field(
        5, ge=1, le=20,
        json_schema_extra=_ui(
            "slider", "merge_cross_max_size", "merge", 100,
            help="OR path only when min(|A|,|B|) <= this. Prevents loosening for large merges.",
            min=1, max=20, step=1,
        ),
    )

    # Support count.
    merge_support_frac: float = Field(
        0.3, ge=0.0, le=1.0,
        json_schema_extra=_ui(
            "slider", "merge_support_frac", "merge", 110,
            help="Min fraction of the smaller cluster that must be in the cross-cluster K-NN.",
            min=0.0, max=1.0, step=0.05,
        ),
    )
    merge_support_min: int = Field(
        2, ge=1, le=20,
        json_schema_extra=_ui(
            "slider", "merge_support_min", "merge", 120,
            help="Min absolute number of supporting cross-cluster pairs.",
            min=1, max=20, step=1,
        ),
    )
    merge_support_unique: bool = Field(
        False,
        json_schema_extra=_ui(
            "checkbox", "merge_support_unique", "merge", 130,
            help="Greedy bipartite matching: each node used at most once.",
        ),
    )

    # Safety constraints.
    merge_margin: float = Field(
        0.05, ge=0.0, le=0.5,
        json_schema_extra=_ui(
            "slider", "merge_margin", "merge", 140,
            help="Min gap between best and second-best candidate distances.",
            min=0.0, max=0.5, step=0.01,
        ),
    )
    merge_diameter_expansion_factor: float = Field(
        1.5, ge=1.0, le=5.0,
        json_schema_extra=_ui(
            "slider", "diameter_expansion_factor", "merge", 150,
            help="Max allowed post-merge diameter relative to larger input cluster.",
            min=1.0, max=5.0, step=0.05,
        ),
    )

    # --- Diameter cap (spec-031) -------------------------------------------
    cluster_diameter_cap_enabled: bool = Field(
        False,
        json_schema_extra=_ui(
            "checkbox", "cluster_diameter_cap_enabled", "cap", 10,
            help="Post-merge safety net: reject clusters whose diameter exceeds the ceiling.",
        ),
    )
    max_full_diameter: float = Field(
        1.2, ge=0.3, le=2.0,
        json_schema_extra=_ui(
            "slider", "max_full_diameter", "cap", 20,
            help="Max pairwise cosine distance across all faces in the cluster.",
            min=0.3, max=2.0, step=0.05,
        ),
    )
    max_exemplar_diameter: float = Field(
        0.8, ge=0.2, le=2.0,
        json_schema_extra=_ui(
            "slider", "max_exemplar_diameter", "cap", 30,
            help="Max pairwise cosine distance restricted to the cluster's exemplars.",
            min=0.2, max=2.0, step=0.05,
        ),
    )

    # --- Holdout attachment ------------------------------------------------
    attach_enabled: bool = Field(
        False,
        json_schema_extra=_ui(
            "checkbox", "attach_enabled", "optional", 30,
            help="Try to attach quality-failed (holdout) faces to clusters.",
        ),
    )
    # Sub-fields of attach not exposed in UI yet.
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

    # --- UI helpers ---------------------------------------------------------
    @classmethod
    def ui_fields_by_group(cls) -> Dict[str, list[str]]:
        """Return {group_name: [field_name, ...]} sorted by ui_order.

        Fields without ``json_schema_extra`` are excluded — they have no
        UI binding.
        """
        groups: Dict[str, list[tuple[int, str]]] = {}
        for name, info in cls.model_fields.items():
            extra = info.json_schema_extra
            if not isinstance(extra, dict):
                continue
            g = extra.get("ui_group")
            o = extra.get("ui_order", 999)
            if g is None:
                continue
            groups.setdefault(g, []).append((o, name))
        return {g: [n for _, n in sorted(v)] for g, v in groups.items()}


__all__ = ["FCParams", "RUNTIME_FIELDS"]
