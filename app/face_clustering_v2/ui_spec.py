"""spec-041 follow-up — Streamlit UI metadata for FCParams, kept separate from
the validation contract.

``face_cluster/fc_params.py`` defines the *contract*: validation bounds
(``ge``/``le``), defaults, and descriptions. That module is
Streamlit-unaware and could be consumed by a CLI-only world.

This module defines *how to render* a contract field in the Streamlit
v2 UI: widget choice, layout group, display order, step granularity,
and (for ``Optional`` numeric fields) whether 0 is the "off" sentinel.

What is **NOT** here, by design:

* ``min`` / ``max`` — the legal range. The widget factory reads
  ``ge`` / ``le`` from ``FCParams.model_fields[name].metadata``. Declaring
  a range in two places is the drift bug this whole refactor exists to
  eliminate.
* ``default`` — also lives on FCParams (``Field(default, ...)``).
* ``help`` — populated from ``FCParams.description``.

Adding a new UI surface (e.g., a mobile app, an argparse CLI) means
adding a parallel ``*_spec.py`` in that app — never touching FCParams.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional


@dataclass(frozen=True)
class FieldUI:
    """How to render one FCParams field in the v2 Streamlit UI.

    Structurally has no ``min`` / ``max`` / ``default`` fields — those
    come from ``FCParams.model_fields[name]`` at render time. The type
    system makes range duplication impossible.
    """
    widget: Literal["slider", "number_input", "checkbox"]
    label: str
    group: Literal["cluster", "quality", "exemplars", "optional", "merge", "cap"]
    order: int
    step: Optional[Any] = None
    zero_is_none: bool = False


# Single dict: field name → UI rendering instructions. Field names MUST
# match a real entry on ``FCParams.model_fields``; enforced by
# ``tests/architecture/test_ui_spec_references_fcparams.py``.
UI_SPEC: dict[str, FieldUI] = {
    # --- Cluster (3) -----------------------------------------------------
    "K":                  FieldUI("slider",       "K (kNN neighbours)",      "cluster", 10, step=1),
    "distance_threshold": FieldUI("slider",       "distance_threshold",      "cluster", 20, step=0.01),
    "min_cluster_size":   FieldUI("slider",       "min_cluster_size",        "cluster", 30, step=1),

    # --- Quality gate (8) ------------------------------------------------
    "blur_min":                 FieldUI("slider",       "blur_min",                  "quality", 10, step=5.0),
    "max_faces_per_image_core": FieldUI("slider",       "max_faces_per_image_core",  "quality", 20, step=1),
    "min_face_area":            FieldUI("number_input", "min_face_area px (0=off)",  "quality", 30, step=500, zero_is_none=True),
    "det_score_min":            FieldUI("number_input", "det_score_min (0=off)",     "quality", 40, step=0.05, zero_is_none=True),
    "require_pose":             FieldUI("checkbox",     "require_pose",              "quality", 60),
    "yaw_max":                  FieldUI("number_input", "yaw_max °",                 "quality", 70, step=1.0),
    "pitch_max":                FieldUI("number_input", "pitch_max °",               "quality", 80, step=1.0),
    "roll_max":                 FieldUI("number_input", "roll_max °",                "quality", 90, step=1.0),

    # --- Exemplars (3) ---------------------------------------------------
    "N_exemplars_max":             FieldUI("slider", "N_exemplars_max",             "exemplars", 10, step=1),
    "exemplars_d10_threshold":     FieldUI("slider", "exemplars_d10_threshold",     "exemplars", 20, step=0.01),
    "exemplar_suppression_radius": FieldUI("slider", "exemplar_suppression_radius", "exemplars", 30, step=0.01),

    # --- Optional stage toggles (3) --------------------------------------
    "split_enabled":  FieldUI("checkbox", "split_enabled",  "optional", 10),
    "merge_enabled":  FieldUI("checkbox", "merge_enabled",  "optional", 20),
    "attach_enabled": FieldUI("checkbox", "attach_enabled", "optional", 30),

    # --- Merge sub-panel (15) --------------------------------------------
    "merge_candidate_threshold":       FieldUI("slider",       "merge_candidate_threshold",       "merge",  10, step=0.01),
    "merge_exemplar_threshold":        FieldUI("slider",       "merge_exemplar_threshold",        "merge",  20, step=0.01),
    "use_adaptive_merge_threshold":    FieldUI("checkbox",     "use_adaptive_merge_threshold",    "merge",  30),
    "merge_exemplar_percentile":       FieldUI("slider",       "exemplar_percentile",             "merge",  40, step=1),
    "merge_global_percentile":         FieldUI("slider",       "global_percentile",               "merge",  50, step=1),
    "merge_threshold_alpha":           FieldUI("slider",       "alpha",                           "merge",  60, step=0.05),
    "merge_threshold_beta":            FieldUI("slider",       "beta",                            "merge",  70, step=0.05),
    "merge_use_cross_gate":            FieldUI("checkbox",     "merge_use_cross_gate",            "merge",  80),
    "merge_cross_threshold":           FieldUI("slider",       "merge_cross_threshold",           "merge",  90, step=0.01),
    "merge_cross_max_size":            FieldUI("slider",       "merge_cross_max_size",            "merge", 100, step=1),
    "merge_support_frac":              FieldUI("slider",       "merge_support_frac",              "merge", 110, step=0.05),
    "merge_support_min":               FieldUI("slider",       "merge_support_min",               "merge", 120, step=1),
    "merge_support_unique":            FieldUI("checkbox",     "merge_support_unique",            "merge", 130),
    "merge_margin":                    FieldUI("slider",       "merge_margin",                    "merge", 140, step=0.01),
    "merge_diameter_expansion_factor": FieldUI("slider",       "diameter_expansion_factor",       "merge", 150, step=0.05),

    # --- Diameter cap (spec-031, 3) --------------------------------------
    "cluster_diameter_cap_enabled": FieldUI("checkbox", "cluster_diameter_cap_enabled", "cap", 10),
    "max_full_diameter":            FieldUI("slider",   "max_full_diameter",            "cap", 20, step=0.05),
    "max_exemplar_diameter":        FieldUI("slider",   "max_exemplar_diameter",        "cap", 30, step=0.05),
}


def fields_by_group() -> dict[str, list[str]]:
    """Return ``{group: [field_name, ...]}`` sorted by FieldUI.order."""
    grouped: dict[str, list[tuple[int, str]]] = {}
    for name, spec in UI_SPEC.items():
        grouped.setdefault(spec.group, []).append((spec.order, name))
    return {g: [n for _, n in sorted(v)] for g, v in grouped.items()}


__all__ = ["FieldUI", "UI_SPEC", "fields_by_group"]
