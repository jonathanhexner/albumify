# Feature Specification: InsightFace Blur Scoring Step

**Created**: 2026-05-15
**Status**: Draft
**Resolves**: FR-033-3; lifts the SIGHTING-061 workaround pin

## Problem Statement

The active InsightFace pipeline does not compute a per-face blur score. The FC App standalone path does (Laplacian variance on the aligned crop). The bridge between the two assumed it could plumb `blur_score` from `context.insightface_faces`, but no upstream step writes it. Spec-033 P-C C-1 removed the bridge's `blur_min=0.0` force-disable on that incorrect assumption, causing SIGHTING-061 (100% face rejection).

This spec adds the missing step so the bridge can stop pinning `blur_min`.

## User Stories

### US1 — Blur score available on every detected face (Priority: P1)
As the QualityGater, I want a numeric blur score on every face entering quality gating, so that the `blur_min` config knob has meaningful data to gate on.

**Acceptance Criteria**:
- After the `insightface_score_blur` step runs, every face in `context.insightface_faces[path]["faces"]` has a non-None `scores["blur_score"]` field.
- The score is the Laplacian variance of the aligned face crop, matching what `face_cluster.quality` computes on the FC App standalone path (so FC App and Albumify produce comparable scores).
- Faces below `min_face_size` skip the computation; their `blur_score` remains None and the gate is permissive on None (matching the pose/det convention).

### US2 — SIGHTING-061 pin is removed (Priority: P1)
As the bridge, I want to honor `cluster_people.blur_min` from yaml again, so that quality gating works on Albumify the way it does on FC App standalone.

**Acceptance Criteria**:
- `face_cluster_bridge.build_fc_config` reads `blur_min` from config (no more pin).
- spec-036's `GATE_WAIVERS["cluster_people.blur_min"]` is removed.
- spec-035's E2E test still passes (blur_min=50 from yaml is honored, but real blur scores pass the gate).

### US3 — FC App ↔ Albumify produce comparable blur (Priority: P2)
As an analyst comparing FC App and Albumify outputs, I want the blur score to mean the same thing on both paths.

**Acceptance Criteria**:
- An identical input image processed by both paths produces blur scores within 5% of each other.
- The computation is colocated in a single helper used by both paths (de-dupe the FC App's `face_cluster.quality._compute_blur` if it exists).

## Edge Cases

- Aligned face crop missing → step skips with a warning; blur_score stays None; gate is permissive.
- Very small crop (<32px on any edge) → score may be unreliable; skip and log.
- Greyscale image → Laplacian works; no special path needed.

## Requirements (brief)

- FR-001: New step at `sim_bench/pipeline/steps/insightface_score_blur.py` following the BaseStep + template-method pattern (cacheable).
- FR-002: Registered in `configs/pipeline.yaml::default_pipeline`, placed after `align_faces` and before `filter_quality`.
- FR-003: Typed config via `sim_bench/pipeline/steps/configs/insightface_score_blur.py` (Pydantic, `extra="forbid"`). Single field: `min_face_size: int` (skip below).
- FR-004: Writes to `context.insightface_faces[path]["faces"][i]["scores"]["blur_score"]`.
- FR-005: Bridge `_lookup_insightface_face` already reads `if_scores.get("blur_score")` — verify it picks up the new value once the step runs.
- FR-006: `face_cluster_bridge.build_fc_config` reverts `blur_min=0.0` pin.
- FR-007: spec-036's waiver for `cluster_people.blur_min` is removed.
- FR-008: spec-035 E2E test still green after this lands.

## Non-Goals

- Not introducing a new ML model — Laplacian variance is fine, matches FC App.
- Not a perceptual blur model (BRISQUE etc.) — overkill for face filtering.

## Open Questions

- [NEEDS CLARIFICATION] Should the step write to `scores["blur_score"]` (nested under `scores`) or to a top-level `blur_score` field on the face dict? Bridge currently reads `if_scores.get("blur_score")`. Confirm path before T010.
