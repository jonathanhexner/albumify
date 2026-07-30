# spec-073 — Area-% quality gate (resolution-independent face-size filter)

**Created**: 2026-06-05
**Status**: Implemented
**Priority**: P2
**Predecessors**: spec-053 (consolidated quality gate), spec-072 (Area % display metric), spec-041 (FCParams/UI_SPEC registry)

---

## Problem

The only face-size gate is `min_face_area` **in pixels** — resolution-dependent
(a 50 px face is large in a 1 MP photo, tiny in a 4 K one). The pipeline already
computes `area_ratio` (set at detection, `insightface_detect_faces.py:187`) and
spec-072 surfaces it as **Area %**. Add the matching **filter**: drop faces whose
bbox is below X % of the image.

## What we build (mirror `min_face_area`)

| Layer | Change |
|---|---|
| `face_cluster/fc_params.py` | add `min_face_area_pct: Optional[float]` (0–100, None=off) |
| `face_cluster/config.py` (`PipelineConfig`) | add `min_face_area_pct: Optional[float] = None` (FCParams↔FCConfig parity) |
| `app/face_clustering_v2/ui_spec.py` | add `min_face_area_pct` to the **quality** group → auto-renders in the Run tab |
| `face_cluster/quality.py` | new `_add_area_pct_gate`: `area_ratio*100 ≥ thr`; permissive when `area_ratio` is None; added to gate set + rejection priority |

Wiring is automatic: `to_fc_config()`/`to_step_configs()` use `model_dump()`, so the
new field reaches the quality-gate step config and `QualityGater.config` with no
extra plumbing. None → gate disabled (vacuous pass), matching the pixel gate.

## Acceptance criteria

| # | Criterion | Verified by |
|---|---|---|
| AC1 | `min_face_area_pct` on FCParams **and** PipelineConfig; parity tests green | pytest test_fcparams_fcconfig_parity |
| AC2 | UI_SPEC entry references the field; renders in Run-tab Quality group | pytest test_ui_spec_references_fcparams + AppTest |
| AC3 | Gate rejects a face with `area_ratio*100 < thr`; passes when ≥ thr; disabled when None; permissive when area_ratio None | unit test |
| AC4 | Verdict persists as `filter_name="area_pct"` in `filter_decisions` (rides spec-093 G1 writer) | unit/integration |
| AC5 | budapest baseline unaffected by default (field defaults None) | Scenario A/D unaffected |

## Non-goals
- Replacing the pixel gate (`min_face_area` stays; both can be set).
- Touching `cluster_by_identity`'s separate `min_face_area_ratio` (different step).
