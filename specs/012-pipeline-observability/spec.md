# Feature Specification: Pipeline Run Observability

**Feature Branch**: `012-pipeline-observability`
**Created**: 2026-04-22
**Status**: Implemented
**Resolves**: SIGHTING-022 (quality gating opaque), SIGHTING-024 (no cluster provenance)
**Input**: Three sightings against `results/Noa2_5_1/merge_remerge_1` revealed that pipeline stages make decisions but don't explain them. Quality gating, exemplar selection, and cluster formation are all opaque — the user cannot answer "why did this face pass?" or "how did this cluster form?" from the run artifacts.

## Problem Statement

The face clustering pipeline computes rich per-face and per-cluster decisions at every stage but only persists minimal final-state outputs. When the user encounters a suspicious result (e.g., a low-quality face in a cluster, or a suspiciously large cluster), they cannot trace back to the decision that caused it without replaying the entire pipeline.

Three independent sightings trace to this root cause:

1. **SIGHTING-022**: Cluster 8 contains obviously bad faces. `faces.csv` has `is_core` but no rejection reason, no gate verdicts, and no thresholds. The user cannot determine why a bad face passed quality gating.
2. **SIGHTING-024**: A large cluster exists but there is no way to tell if it was formed by base clustering, auto-merge, manual merge, or remerge. `merge_log.json` is legitimately empty (zero candidates at 0.45 threshold), but there are no stage snapshots showing intermediate cluster states.
3. **Cross-cutting**: Exemplar d10 scores are computed then discarded. Detection confidence (`det_score`) is available from InsightFace but never extracted. Split signal statistics are ephemeral (computed in UI, never saved).

## User Scenarios & Testing

### User Story 1 — Per-Face Quality Audit Trail (Priority: P1)

After a pipeline run completes, the user inspects a face that looks low-quality and wants to know: what quality criteria were evaluated, what were the thresholds, and which gates did this face pass or fail? Every face in `faces.csv` (both core and holdout) carries a structured quality verdict showing each gate's metric value, threshold, and pass/fail.

The run directory also contains a `quality_config.json` snapshot of the exact quality thresholds in force, and a run-level quality summary (counts rejected per gate, counts passing by thin margin).

**Why this priority**: Directly unblocks SIGHTING-022 — the most urgent pain point. Users need to debug quality gating to tune thresholds.

**Independent Test**: Run the pipeline on test data, open `faces.csv`, verify every face has columns for `quality_blur_pass`, `quality_blur_value`, `quality_pose_pass`, `quality_pose_value`, `quality_area_pass`, `quality_area_value`, and `quality_rejection_reason` (null for core faces, enum for holdout).

**Acceptance Scenarios**:

1. **Given** a completed pipeline run, **When** the user opens `faces.csv`, **Then** every face row has quality gate columns: for each gate (blur, pose, area), a value column and a pass/fail column; holdout faces have a `quality_rejection_reason` column with the specific gate that failed.
2. **Given** the run directory, **When** the user opens `quality_config.json`, **Then** it contains the exact blur_min, yaw_max, pitch_max, area_min thresholds used for that run.
3. **Given** a completed run loaded in the app, **When** the user expands a face in the Face Analysis tab, **Then** a "Quality Report" section shows each gate with metric value, threshold, and pass/fail badge — identical to the Merge Analysis gate display style.
4. **Given** the run directory, **When** the user opens `pipeline_run.json`, **Then** a `quality_summary` block records: total faces detected, N faces rejected per gate, N faces passing within 10% of a threshold. (UI display of this summary is spec 013's responsibility.)

---

### User Story 2 — Cluster Provenance & Stage Snapshots (Priority: P1)

After a pipeline run, the user sees a suspiciously large cluster and wants to know: was it always this big from base clustering, or did it grow through merges? Every cluster carries an `origin` field indicating how it was formed, and the run directory contains stage snapshots showing the cluster state at key points.

**Why this priority**: Directly unblocks SIGHTING-024. Without provenance, the user cannot audit merge quality or identify over-merging.

**Independent Test**: Run a pipeline that performs base clustering then auto-merge. Verify that `clusters.csv` has an `origin` column, that base clusters have `origin=base`, merged clusters have `origin=auto_merge` with `parent_ids`, and that `clusters_stage_base.csv` exists alongside `clusters_stage_merged.csv`.

**Acceptance Scenarios**:

1. **Given** a completed pipeline run with merges, **When** the user opens `clusters.csv`, **Then** every cluster has `origin` (one of: `base`, `auto_merge`, `manual_merge`, `remerge`) and `parent_cluster_ids` (JSON list of cluster IDs that were merged to form it; empty for `base`).
2. **Given** the run directory, **When** the user lists files, **Then** `clusters_stage_base.csv` (cluster state after initial clustering, before merge) exists alongside the final `clusters.csv`.
3. **Given** a run loaded in the app, **When** the user selects a merged cluster in Cluster Analysis, **Then** a "Provenance" section shows the origin, parent cluster IDs, and the merge decision that caused it (link to merge_log entry).
4. **Given** a remerge run where zero candidates were found, **When** the user or spec 013's History tab consumes the run, **Then** `merge_metadata.json` carries enough data (candidate threshold, `n_candidates_proposed=0`) to render "0 candidates at threshold X" without guesswork. (UI display is spec 013's responsibility.)

---

### User Story 3 — Enriched Face Metadata (Priority: P2)

InsightFace computes detection confidence (`det_score`) and the quality gating stage computes per-face d10 exemplar scores, but neither is persisted. Adding these to `faces.csv` gives the user richer signals for debugging without any runtime cost (the data is already computed).

**Why this priority**: Low implementation effort, high diagnostic value, but less urgent than the gate verdicts (US1) and provenance (US2).

**Independent Test**: Run the pipeline, verify `faces.csv` contains `det_score` and `d10_score` columns with numeric values.

**Acceptance Scenarios**:

1. **Given** a completed pipeline run, **When** the user opens `faces.csv`, **Then** every face has a `det_score` column (float, InsightFace detection confidence) and core faces have a `d10_score` column (float, d10 centrality metric from exemplar selection; null for holdout faces).
2. **Given** the Face Analysis tab in the app, **When** the user views a face, **Then** the detection confidence and d10 score are shown alongside blur, pose, and area.

---

> **NOTE**: The prior "User Story 4 — Run-Level Observability Dashboard" has been **moved to spec 013** (Run History & Comments). Spec 012 persists the underlying data (`quality_summary`, extended `merge_metadata.json`, cluster provenance). Spec 013 renders the cross-run summary dashboard in the new History tab.

### Edge Cases

- **Old runs without quality columns**: `faces.csv` from runs before this feature will lack quality gate columns. The app must handle missing columns gracefully (show "N/A" rather than crashing).
- **Remerge runs with zero candidates**: These have a legitimately empty merge_log. The UI must not treat this as an error — show "0 candidates found at threshold X" with the actual threshold value.
- **Manual merge snapshots**: `save_manual_merge_snapshot()` already writes `clusters.csv` — it must set `origin=manual_merge` and `parent_cluster_ids` for the merged cluster.
- **Stage snapshots for recluster**: `recluster()` skips detect/embed/quality but runs cluster+merge. Stage snapshot should still be written for the cluster stage.

## Requirements

### Functional Requirements

- **FR-001**: `faces.csv` MUST include per-gate quality columns: `quality_blur_value`, `quality_blur_pass`, `quality_pose_value`, `quality_pose_pass`, `quality_area_value`, `quality_area_pass`, `quality_rejection_reason`.
- **FR-002**: Run directory MUST contain `quality_config.json` with all quality thresholds used for that run.
- **FR-003**: `clusters.csv` MUST include `origin` and `parent_cluster_ids` columns for every cluster.
- **FR-004**: Run directory MUST contain `clusters_stage_base.csv` — the cluster state before merge.
- **FR-005**: `faces.csv` MUST include `det_score` (InsightFace detection confidence).
- **FR-006**: `faces.csv` MUST include `d10_score` for core faces (null for holdout).
- **FR-007**: The app MUST show a per-face Quality Report panel with gate verdicts and thresholds.
- **FR-008**: The app MUST show a per-cluster Provenance section with origin and parent IDs.
- **FR-009**: `merge_metadata.json` MUST include `merge_candidate_threshold` and `n_candidates_proposed` so downstream UIs (spec 013) can distinguish "zero candidates found" from "merge stage never ran".
- **FR-010**: Old runs without new columns MUST load without error (graceful degradation).
- **FR-011**: `pipeline_run.json` MUST include a `quality_summary` block (total detected, rejected per gate, survived) — the data surface for spec 013's History dashboard.

### Key Entities

- **QualityVerdict**: Per-face structured result — gate name, metric value, threshold, pass/fail.
- **ClusterOrigin**: Enum — `base`, `auto_merge`, `manual_merge`, `remerge`.
- **StageSnapshot**: A point-in-time `clusters.csv` saved at a named pipeline stage.

## Success Criteria

### Measurable Outcomes

- **SC-001**: User can determine why any specific face was included or rejected within 10 seconds of opening its detail view.
- **SC-002**: User can determine the origin of any cluster (base vs merge) within 5 seconds of viewing it.
- **SC-003**: All persisted data (quality verdicts, provenance, det_score, d10_score) adds less than 5% to pipeline runtime.
- **SC-004**: Old runs (without new columns) still load and display without errors.

## Assumptions

- Quality gating currently evaluates blur, pose (yaw/pitch), and area. If new gates are added later, the schema should be extensible (quality_<gate>_value / quality_<gate>_pass pattern).
- Stage snapshots use the same CSV format as `clusters.csv` with a `_stage_<name>` suffix. No additional schema needed.
- `det_score` is available from InsightFace's `face.det_score` attribute. If a different detector is used, this field may be null.
- The provenance field `parent_cluster_ids` is a JSON-encoded list (e.g., `[3, 7]`). For `origin=base`, it is an empty list `[]`.
