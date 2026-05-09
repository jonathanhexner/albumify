# Feature Specification: Force Merge (User-Driven Cluster Merge)

**Feature**: `014-force-merge`
**Created**: 2026-04-22
**Status**: Implemented
**Extracted from**: spec 011 US3 (moved here because it has no ML dependency)

---

## Problem

Some clusters that should be merged are never offered as candidates because their exemplar distance exceeds `merge_candidate_threshold` (default 0.45). The user has no way to correct these false negatives — they can only accept whatever the algorithm proposes.

---

## User Story — Manual Merge for Any Two Clusters (Priority: P1)

The user selects any two cluster IDs from the Cluster Analysis tab or Merge Analysis tab. A preview shows exemplar face crops from both clusters plus the same merge evidence metrics shown in a normal Merge Analysis card (exemplar distance vs threshold, support count, 4-gate breakdown, margin gap, post-merge diameter), plus a note if the pair was not a candidate. The user confirms, and the merge is applied immediately as a manual merge step identical to an approved candidate merge.

**Acceptance Scenarios**:

1. **Given** two cluster IDs not in the merge candidate list, **When** the user enters both in the force-merge widget, **Then** a preview shows exemplar crops from both clusters with full merge metrics and the note "Not a merge candidate (exemplar dist = X > threshold Y)."
2. **Given** the preview is shown, **When** the user clicks "Confirm Merge", **Then** the merge is applied to the current loaded result (same path as "Apply Only" in Merge Analysis), clusters unified in display.
3. **Given** a force merge has been applied, **When** session records the step, **Then** step metadata includes `action: "force_merge"`, `cluster_a`, `cluster_b`, `exemplar_dist`, gate results, `source: "human"`.
4. **Given** two cluster IDs that ARE already merge candidates, **When** the user enters them, **Then** the same widget works — no special-casing required.
5. **Given** a force-merged pair, **When** training data discipline runs, **Then** the pair is eligible as a human-labeled positive sample (`source="human"`, `label=1`).

---

## Scope

**In scope**:
- UI widget (cluster ID selector + preview card + confirm button) in Cluster Analysis tab
- Compute merge metrics for any arbitrary pair on demand
- Apply merge via existing `save_manual_merge_snapshot()` path
- Session step recording

**Out of scope**:
- Undo / split (spec 013)
- Auto-harvesting negatives (spec 011 US4)

---

## Non-Functional Requirements

- Computing metrics for a non-candidate pair must complete in < 2s (embeddings already loaded)
- No new models required — reuse `FeatureComputer` and existing merge evidence computation

---

## Dependencies

- `face_cluster/features.py` — `FeatureComputer` for merge metrics
- `face_cluster/manual_merge_snapshot.py` — apply the merge
- `app/face_clustering.py` — Cluster Analysis tab UI
