# Feature Specification: Session Operation Pipeline

**Feature Branch**: `009-session-op-pipeline`
**Created**: 2026-04-21
**Revised**: 2026-04-21
**Status**: Draft (v2 — base+chain model)

## Overview

Currently, every interactive action in the face clustering app (recluster, merge, split) creates a new output folder, causing folder proliferation and confusion about which folder represents the "current" state.

This feature replaces that pattern with a **session** composed of two layers:

1. **Base step** — the expensive, one-time work: face detection, embedding extraction, quality gating. Done once per album, never re-executed.
2. **Clustering chain** — an ordered list of clustering step definitions (initial clustering, merge, recluster, split). Each step is cheap to execute because it reuses the base embeddings. The chain is re-executable from any point.

**Branching** is achieved by creating a new chain: copy the step definitions up to step N, then define new steps from N+1 onward. Each chain lives in its own subfolder. Old chains are preserved intact.

**Merge labeling** supports two actions:
- **Apply only** (quick): save label decisions (approve/reject pairs) to a pending labels file. No new step folder is created — labels accumulate across multiple "apply only" clicks.
- **Apply + Remerge** (default): consume all pending labels, remerge clusters, and create a new materialized step folder. This catches cascading effects from centroid shifts.

A **step** in the chain represents a **materialized clustering state** (a folder with `clusters.csv`), not every individual user click. Labels accumulate as cheap metadata until the user decides to materialize by remerging.

---

## User Scenarios & Testing

### User Story 1 — View the Chain as a Pipeline (Priority: P1)

A user runs a base pipeline, then performs clustering, a merge, and a recluster. Instead of browsing folders, they see:

```
Base (embeddings)     2:14pm   120 images, 342 faces
Chain: austria_v1
  [0] Cluster          2:15pm   K=5, dist=0.35 → 28 clusters
  [1] Merge (8 pairs)  2:31pm   apply+remerge → 20 clusters
  [2] Merge (5 pairs)  2:45pm   apply+remerge → 15 clusters
```

**Acceptance Scenarios**:

1. **Given** a base run and chain exist, **When** the user views the session, **Then** each step displays its type, timestamp, parameters, and result summary.
2. **Given** the app is restarted, **When** the user reopens the session, **Then** the full chain is still shown (persisted in `session.json`).

---

### User Story 2 — Re-execute Chain from a Point (Priority: P1)

A user decides step [2] was wrong. They want to go back to step [1] and try different merge labels.

**Flow**: User selects step [1] → clicks "Branch from here" → a new chain is created with steps [0] and [1] copied → user adds new steps from [2] onward.

**Acceptance Scenarios**:

1. **Given** chain_v1 has steps [0,1,2], **When** user branches from step [1], **Then** a new chain_v2 is created containing steps [0,1].
2. **Given** chain_v2 exists, **When** user applies a new merge, **Then** chain_v2 grows to [0,1,2_new]. chain_v1 remains unchanged.
3. **Given** multiple chains exist, **When** user views the session, **Then** they can switch between chains.

---

### User Story 3 — Incremental Merge Labeling (Priority: P1)

A user sees 20 merge candidates. They label 8 and click "Apply only" (saving labels without remerging). They label 5 more and click "Apply only" again. Then they click "Apply + Remerge" — all 13 labels are consumed, clusters are remerged, and a new step is created.

**Acceptance Scenarios**:

1. **Given** the user clicks "Apply only" twice (8 pairs, then 5 pairs), **When** they view the chain, **Then** no new step folder exists, but a "13 pending labels" indicator is shown.
2. **Given** 13 pending labels exist, **When** the user clicks "Apply + Remerge", **Then** a new step is created containing all 13 labels and the remerged clustering state.
3. **Given** pending labels exist, **When** the user tries to branch or switch chain, **Then** a warning is shown: "N pending labels haven't been remerged. Discard or remerge first?"

---

### User Story 4 — Compare Chains (Priority: P2)

A user has two chains with different clustering parameters and wants to compare results side by side.

**Acceptance Scenarios**:

1. **Given** two chains exist, **When** the user switches between them, **Then** the analysis tabs update to reflect the selected chain's final state.

---

### Edge Cases

- Base folder deleted externally → session loads but marks base as missing, prevents chain execution.
- Two steps applied in the same second → steps are ordered by sequence number, not timestamp.
- Session folder moved or renamed → `session.json` uses relative paths internally, so this works as long as internal structure is preserved.
- First use (no session) → base run creates the session automatically.
- Chain with zero clustering steps → valid (just the base output with initial clustering from the pipeline).

---

## Requirements

### Functional Requirements

- **FR-001**: A session MUST consist of a base step (detect/embed/quality) and one or more clustering chains.
- **FR-002**: Each chain MUST be an ordered list of step definitions, each with type, parameters, and timestamp.
- **FR-003**: Step types: `cluster`, `merge`, `recluster`, `split`. Each step definition includes its parameters.
- **FR-004**: "Apply only" MUST accumulate labels in a pending file without creating a new step folder. "Apply + Remerge" MUST consume all pending labels, remerge, and create a new materialized step.
- **FR-005**: A step in the chain represents a materialized clustering state (folder with output files), not an individual user action. Labels accumulate until materialized.
- **FR-006**: Branching MUST create a new chain subfolder. The original chain is preserved intact (no renaming, no `_` prefix).
- **FR-007**: Chains MUST be re-executable from any step — the system re-runs steps N+1..end using base embeddings as input.
- **FR-008**: The session MUST be persisted in `session.json` and survive app restarts.
- **FR-009**: The UI MUST show the chain as an ordered pipeline list, not raw folder paths.
- **FR-010**: The UI MUST allow switching between chains for comparison.
- **FR-011**: The UI MUST NOT require the user to type folder paths for derivative operations.

### Key Entities

- **Session**: Root folder containing `session.json`, the `base/` folder, and chain subfolders.
- **Base**: The expensive one-time output — detected faces, embeddings, quality scores. Shared by all chains.
- **Chain**: A subfolder (e.g., `chain_01/`) containing an ordered sequence of clustering step outputs. Defined by a list of step definitions in `session.json`.
- **Step**: A single clustering operation within a chain. Has a type, parameters, and output state.
- **session.json**: The persistent record of the base run and all chains with their step definitions.

---

## Success Criteria

### Measurable Outcomes

- **SC-001**: A user performing 5 "apply only" clicks and 2 "apply + remerge" clicks sees exactly 2 merge steps in the chain (not 7), with no folder paths exposed.
- **SC-002**: Branching from step N and applying a new step takes under 3 seconds (re-execution is cheap).
- **SC-003**: Two chains from the same base share the embedding data (no duplication of the expensive base output).
- **SC-004**: The chain list survives an app restart.
- **SC-005**: Zero manual folder management is required from the user.

---

## Assumptions

- Each session corresponds to one source album. Sessions are not merged or combined.
- The base step is never re-executed within a session. To re-run detection/embedding, start a new session.
- Split is a step type placeholder — implementation deferred but the model supports it.
- The existing History tab continues to work for browsing past sessions and pre-session runs.
- Users work on one session at a time; multi-session parallel editing is out of scope.
- Clustering steps are cheap enough (~seconds) that re-execution from any point is acceptable UX.
