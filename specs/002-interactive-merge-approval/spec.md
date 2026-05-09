# Feature Specification: Interactive Merge Approval + ML Merge Classifier

**Feature Branch**: `002-interactive-merge-approval`
**Created**: 2026-04-11
**Status**: Draft
**Input**: User description: "Interactive merge approval UI that doubles as labeling for ML merge classifier. Simplify heuristic merger by dropping adaptive thresholds. Train ML classifier on accept/reject decisions."

## Design Decisions (confirmed with user 2026-04-12)

- **Location**: Approval controls live inside the existing "Merge Analysis" tab — no new tab. Candidates list is already shown there; add Approve/Reject buttons inline.
- **Source**: Approval always operates on **base clusters** (pre-merge), not on the heuristic-merged result. Ensures consistent feature vectors across sessions and unambiguous training samples.
- **Pre-population**: Heuristic-accepted candidates default to "Approved"; heuristic-rejected default to "Rejected". User only changes the ones they disagree with.
- **Sorting / filtering**: Candidates sorted by ambiguity (1–3 gates passing = contested, shown first). Filter toggle: "Show all" vs "Show contested only".
- **Bulk actions**: "Accept all heuristic decisions" and "Reset all" buttons for efficiency.
- **After applying**: Result is shown as an updated merged cluster view within the same tab (in-memory, same session). Not a new pipeline run — no model inference or re-embedding.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Review and Approve/Reject Proposed Merges (Priority: P1)

After running the face clustering pipeline, the user opens the "Merge Analysis" tab. They see a list of all merge candidates (both heuristic-accepted and heuristic-rejected), pre-populated with the heuristic's decision, each showing:
- Exemplar face crops from both clusters side by side
- Key distances (min exemplar distance, p10 cross distance)
- Gate pass/fail summary (exemplar, support, margin, diameter)
- Cluster sizes

Contested candidates (1–3 gates passing) appear first. The user reviews, overrides any decisions they disagree with, then clicks "Apply Approved Merges". The merged cluster view updates in the same tab — no re-run required.

**Why this priority**: This is the core interaction — without it, nothing else works. It delivers immediate value (manual merge correction) and generates labeled data for ML training.

**Independent Test**: Can be tested by loading any completed run, reviewing proposals in Merge Analysis, and verifying that approved merges produce correct cluster reassignments.

**Acceptance Scenarios**:

1. **Given** a completed pipeline run, **When** the user opens Merge Analysis, **Then** all candidate pairs are listed with exemplar crops, distances, gate results, and default decisions pre-populated from the heuristic.
2. **Given** candidates are listed, **Then** contested candidates (1–3 gates passing) appear before clear-pass and clear-fail candidates.
3. **Given** a candidate pair, **When** the user clicks "Approve" or "Reject", **Then** the decision updates immediately with no page reload.
4. **Given** the user has reviewed proposals, **When** they click "Apply Approved Merges", **Then** clusters are recomputed from the base clustering (not the heuristic-merged result) applying only approved merges, and the cluster view updates in the same tab.
5. **Given** a transitive chain (A+B approved, B+C approved), **When** applying merges, **Then** A, B, and C all end up in the same cluster.
6. **Given** no changes were made, **When** "Apply Approved Merges" is clicked, **Then** the result equals the heuristic merged output.

---

### User Story 2 - Save Merge Decisions as Training Labels (Priority: P1)

After the user finishes reviewing merge proposals and applies their decisions, the accept/reject labels are persisted to disk alongside the run artifacts. The file format includes cluster pair IDs, the decision (approve/reject), all computed features for the pair, and a timestamp. This data can later be aggregated across multiple runs to train a classifier.

**Why this priority**: Equal to P1 because this is the entire point — the approval UI exists to generate training data. If decisions aren't saved, the ML path is blocked.

**Independent Test**: Can be tested by making decisions in the UI, then reading the saved file and verifying all fields are present with correct values.

**Acceptance Scenarios**:

1. **Given** the user has made approve/reject decisions and clicked "Apply", **When** decisions are saved, **Then** a `merge_decisions.json` file is written to the run output directory.
2. **Given** merge_decisions.json is saved, **Then** each entry contains: cluster_a, cluster_b, decision (approve/reject), all ClusterPairFeatures fields, timestamp, and run_id.
3. **Given** the user loads a run that already has merge_decisions.json, **When** they open the Merge Approval tab, **Then** previous decisions are pre-loaded and displayed.

---

### User Story 3 - Simplify Heuristic Merger (Priority: P2)

The adaptive per-cluster threshold system (alpha, beta, percentile, global_percentile) is replaced with a single fixed merge distance threshold. The merger still uses the 4-gate approach (exemplar distance, support count, margin, diameter) but the exemplar gate uses a simple fixed threshold instead of the adaptive formula. This reduces configuration from 6 interacting knobs to 1 threshold value.

**Why this priority**: Prerequisite for clean ML training — the adaptive thresholds add noise to the feature space and make it harder to interpret what the heuristic does. Simpler baseline = cleaner comparison with ML.

**Independent Test**: Can be tested by running the merger on existing test data with the simplified config and verifying merge outcomes are reasonable (no regression in quality).

**Acceptance Scenarios**:

1. **Given** merge is enabled with default config, **When** the merger runs, **Then** it uses a fixed `merge_exemplar_threshold` (no adaptive computation).
2. **Given** the adaptive threshold config fields are removed, **When** existing code references them, **Then** no runtime errors occur (backward-compatible deprecation or clean removal).
3. **Given** a run with the simplified merger, **When** comparing to the same run with adaptive thresholds, **Then** merge decisions are at least as interpretable (each decision log entry shows threshold used = fixed value).

---

### User Story 4 - Train ML Merge Classifier (Priority: P3)

After labeling merge decisions across 2+ runs, the user runs a training script that:
- Aggregates all `merge_decisions.json` files from specified run directories
- Extracts features using `FeatureComputer` / `ClusterPairFeatures`
- Trains a logistic regression (or gradient boosted) model
- Reports accuracy, precision, recall, and feature importances
- Saves the trained model to a standard location

**Why this priority**: Depends on having enough labeled data from P1/P2. Medium-term goal — the heuristic merger with manual approval is sufficient for near-term use.

**Independent Test**: Can be tested with synthetic labeled data (mock merge_decisions.json files) to verify the training pipeline runs end-to-end.

**Acceptance Scenarios**:

1. **Given** merge_decisions.json files from 2+ runs, **When** the training script runs, **Then** it produces a trained model file and a metrics report.
2. **Given** a trained model, **When** it is loaded in the merger, **Then** it can score candidate pairs and produce merge/no-merge predictions.
3. **Given** a trained model is available, **When** the user opens the Merge Approval tab, **Then** they see the ML prediction alongside the heuristic decision for each candidate.

---

### Edge Cases

- What happens when a run has no merge candidates (all clusters are far apart)? The Merge Analysis tab shows "No merge candidates for this run" with an explanation.
- What happens when the user approves a merge that creates a very large cluster (>50 faces)? Show a warning but allow it — user judgment overrides heuristics.
- What happens when a run was done with merge disabled? The tab shows all base clusters and computes candidates on-the-fly from the loaded distance matrix.
- What happens when the user re-opens a run and changes previous decisions? Previous decisions are overwritten; the new file replaces the old one.
- What happens when merge_decisions.json references cluster IDs that no longer match (data was re-clustered)? Decisions are keyed by run_id + cluster pair; stale decisions for a different run are discarded with a warning.
- What is shown after "Apply Approved Merges"? An in-memory merged cluster view within the same tab — no new files written until the user also saves decisions.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST display all merge candidates (both heuristic-accepted and heuristic-rejected) in the Merge Analysis tab with exemplar face crops from both clusters.
- **FR-002**: System MUST show key merge evidence for each candidate: min exemplar distance, support count, margin gap, post-merge diameter, and pass/fail per gate.
- **FR-003**: Candidates MUST be sorted by ambiguity (contested = 1–3 gates passing) first, with a "Show contested only" filter toggle.
- **FR-004**: System MUST pre-populate each candidate's decision from the heuristic outcome (accepted → Approved, rejected → Rejected).
- **FR-005**: Users MUST be able to override any candidate's decision with a single click (Approve / Reject buttons).
- **FR-006**: System MUST provide bulk actions: "Accept all heuristic decisions" and "Reset all to unreviewed".
- **FR-007**: System MUST apply approved merges starting from base clusters (pre-merge), not from the heuristic-merged result.
- **FR-008**: System MUST handle transitive merge chains correctly (if A+B and B+C are both approved, A, B, and C merge into one cluster).
- **FR-009**: After applying, System MUST show the resulting merged cluster view inline in the Merge Analysis tab without triggering a pipeline re-run.
- **FR-010**: System MUST persist merge decisions to `merge_decisions.json` in the run output directory, including all ClusterPairFeatures fields, decision, timestamp, and run identifier.
- **FR-011**: System MUST pre-load previous decisions when opening a run that already has merge_decisions.json.
- **FR-012**: System MUST replace the adaptive threshold system (alpha, beta, percentile, global_percentile) with a single fixed threshold for the exemplar gate.
- **FR-013**: The simplified merger MUST still enforce all 4 gates (exemplar distance, support count, margin, diameter) — only the threshold computation changes.
- **FR-014**: System MUST provide a training script that aggregates merge_decisions.json files, trains a classifier, and reports metrics.
- **FR-015**: The Merge Approval UI MUST remain responsive (non-blocking) while loading face crops — heavy work dispatched via the existing async pattern.

### Key Entities

- **MergeCandidate**: A pair of clusters proposed for merging, with computed features and gate results.
- **MergeDecision**: A user's approve/reject judgment on a MergeCandidate, with timestamp and associated features.
- **MergeDecisionStore**: The persisted collection of MergeDecisions for a run (`merge_decisions.json`).
- **ClusterPairFeatures**: Existing dataclass in `features.py` — the feature vector for a candidate pair, used both for display and as ML training input.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can review and decide on all merge candidates for a typical run (20-50 candidates) in under 10 minutes.
- **SC-002**: Merge decisions are persisted and correctly reloaded on subsequent app opens — zero data loss.
- **SC-003**: Applied merges produce correct cluster assignments (verified by unit test: approved pairs end up in same cluster, rejected pairs remain separate).
- **SC-004**: The simplified merger produces equivalent or better clustering quality compared to adaptive thresholds on existing test datasets.
- **SC-005**: After labeling 3+ runs, the ML classifier achieves >85% accuracy on held-out merge decisions.
- **SC-006**: Merger configuration is reduced from 6 adaptive-threshold parameters to 1 fixed threshold.

## Assumptions

- Existing `merge_log.json` and `merge_metadata.json` from pipeline runs contain sufficient information to reconstruct merge candidates for display.
- The `ClusterPairFeatures` dataclass in `features.py` already covers the core features needed for ML training; additional features (image count, pose distribution) will be added in a later iteration.
- Face crop images are available on disk (via `crop_manifest.json`) for display in the approval UI.
- The existing `_AsyncState` pattern in `app/face_clustering.py` is sufficient for keeping the UI responsive during merge computation.
- Runs with the old adaptive-threshold config will still load correctly (backward compatibility for reading old merge_log.json files).
