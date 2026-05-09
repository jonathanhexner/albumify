# Feature Specification: Config-Driven Pipeline + Iterative Manual Merge

**Feature Branch**: `007-config-driven-pipeline-iterative-merge`
**Created**: 2026-04-17
**Status**: Draft

## Summary

Two related changes delivered together because the second depends on the first:

1. **Config-driven pipeline**: The face clustering pipeline becomes a single generic stage executor driven entirely by config. No special-case methods per mode.
2. **Iterative manual merge**: After a user approves merge candidates, the result is persisted as a new run and the merge stage re-runs automatically on top of it, presenting a fresh round of candidates. This repeats until no candidates remain.

---

## User Scenarios & Testing

### User Story 1 — Config-Driven Pipeline (Priority: P1)

A developer running the pipeline always calls the same method regardless of whether they are doing a full run, a recluster, or a remerge. The configuration object carries everything: which stages to run, where to read input, where to write output, and all algorithm parameters. Named preset factories make the common cases easy to construct.

**Why this priority**: All other stories depend on this. The iterative merge requires a remerge preset. Without the unified pipeline, adding remerge would require a third ad-hoc method.

**Independent Test**: A developer can construct a full-run config, a recluster config, and a remerge config and call `pipeline.run(config)` for each. All three produce valid output directories.

**Acceptance Scenarios**:

1. **Given** a directory of raw images, **When** `pipeline.run(PipelineConfig.full_run(source, output))` is called, **Then** all stages from discovery through export execute and results are written to the output directory.
2. **Given** a completed full-run output directory, **When** `pipeline.run(PipelineConfig.recluster(source, output))` is called, **Then** only cluster/exemplars/merge/export stages run, reusing the existing crops and embeddings.
3. **Given** a saved cluster snapshot directory, **When** `pipeline.run(PipelineConfig.remerge(source, output))` is called, **Then** only merge/export stages run, starting from the existing cluster assignments.
4. **Given** any preset config, **When** the run completes, **Then** exactly one row is written to the run history database.

---

### User Story 2 — Iterative Manual Merge (Priority: P2)

After reviewing merge candidates in the Merge Analysis tab and clicking "Apply Approved Merges", the user sees a new set of merge candidates automatically — computed on the freshly merged clusters — without having to manually trigger any additional steps. The user can keep approving rounds until the merge stage finds no further candidates, at which point the tab shows a "no more candidates" message.

**Why this priority**: The current one-shot apply misses transitive merges (cluster AB formed by merging A+B may now be close enough to merge with C). This is the core feature request.

**Independent Test**: A user can apply one round of merges and immediately see new candidates (if any) in the Merge Analysis tab, with the run visible in History.

**Acceptance Scenarios**:

1. **Given** a run is loaded and merge candidates are visible, **When** the user approves at least one pair and clicks Apply, **Then** a snapshot of the merged state is saved to disk as a new run entry in History.
2. **Given** the snapshot is saved, **When** the merge stage runs on the snapshot, **Then** the Merge Analysis tab resets and shows only candidates from the new run (previously approved pairs are gone).
3. **Given** the remerge produces no new candidates, **When** the Merge Analysis tab renders, **Then** a clear message indicates no further merges are possible.
4. **Given** a user rejects all candidates or there are none, **When** they click Apply, **Then** no new snapshot is created and the tab state is unchanged.

---

### User Story 3 — History Includes All Run Types (Priority: P3)

The History tab shows manual merge snapshots and remerge runs alongside full runs and reclusters, so the user can load any point in the merge iteration history.

**Why this priority**: Without this, the iterative merge history is invisible to the user. Lower priority because the merge flow still works even if History doesn't show it.

**Independent Test**: After an iterative merge cycle (full run → apply → remerge → apply → remerge), the History tab lists all entries with their type labels.

**Acceptance Scenarios**:

1. **Given** a completed iterative merge cycle, **When** the History tab is opened, **Then** all run types (full run, recluster, manual merge snapshot, remerge) appear with distinct type labels.
2. **Given** any run in History is selected, **When** the user loads it, **Then** the Clusters tab and Merge Analysis tab populate correctly from that run's output directory.

---

### Edge Cases

- What happens if the user closes the app mid-Apply before the remerge completes? The snapshot is already on disk, so it can be loaded from History. The incomplete remerge output dir may be partial.
- What if the snapshot source directory is deleted or moved before remerge starts? Remerge fails with a clear error message; the snapshot row in History is marked failed.
- What if the user approves 0 pairs? No snapshot is created; the tab remains unchanged.
- What if a recluster config points to a source directory that has no embeddings? The pipeline fails at the source loader stage with a descriptive error before any stage executes.

---

## Requirements

### Functional Requirements

- **FR-001**: The pipeline MUST expose a single public `run(config)` method. No separate `recluster()` or `remerge()` methods on the pipeline class.
- **FR-002**: `PipelineConfig` MUST carry the stage list, source directory, and output directory in addition to algorithm parameters.
- **FR-003**: Named preset factories MUST exist for full run, recluster, and remerge, each setting the correct stage list and source loader.
- **FR-004**: Each preset's source loader MUST validate that the required inputs exist before any stage executes, and MUST fail with a clear error if they do not.
- **FR-005**: Clicking "Apply Approved Merges" with at least one approved pair MUST save a snapshot directory to disk containing updated cluster assignments, embeddings reference, crop manifest, and metadata (approved/rejected pairs, parent run id, thresholds).
- **FR-006**: After saving the snapshot, the pipeline MUST automatically run the remerge preset on it and load the result into the active session.
- **FR-007**: The Merge Analysis tab MUST reset to the new run's candidates after Apply; previously approved pair keys MUST NOT carry over.
- **FR-008**: If Apply produces a remerge with zero new candidates, the Merge Analysis tab MUST display a message indicating no further merges are possible.
- **FR-009**: Each pipeline run (full, recluster, remerge) MUST write exactly one row to the run history database. Duplicate writes are not permitted.
- **FR-010**: The History tab MUST display manual merge snapshot and remerge run types alongside existing types.
- **FR-011**: The existing `recluster()` method MUST be replaced by `PipelineConfig.recluster(...)` preset. External callers (app, tests) MUST be updated accordingly.
- **FR-012**: The Apply UI MUST offer a checkbox to include exemplar re-selection before merge (`exemplars → merge → export`) instead of the default `merge → export` only. The preset used for remerge is determined by this checkbox at apply time.

### Key Entities

- **PipelineConfig**: Carries stage list, source dir, output dir, and all algorithm parameters. Preset factories produce pre-configured instances.
- **ManualMergeSnapshot**: A directory written after manual approvals, containing enough state to run remerge on top. Contains updated `faces.csv`, embeddings reference, crop manifest, `pipeline_run.json` with approval metadata.
- **Run history row**: A record in the history database for each pipeline execution, carrying run type, source, output, status, and result metrics.

---

## Success Criteria

### Measurable Outcomes

- **SC-001**: All three presets (full run, recluster, remerge) produce valid output directories when called via `pipeline.run(config)`.
- **SC-002**: After clicking Apply with N approved pairs, a new run appears in History within the same app session, without any manual refresh.
- **SC-003**: The History tab correctly shows run type labels for all four run types (full, recluster, manual merge snapshot, remerge).
- **SC-004**: No pipeline execution writes more than one history row to the database.
- **SC-005**: All existing tests continue to pass after the `recluster()` method is removed and replaced by the preset factory.

---

## Assumptions

- The existing `recluster()` callers are limited to `app/face_clustering.py` and the test suite — no external consumers.
- Manual merge snapshots are stored under the same `results/` directory as other runs.
- The default remerge preset runs `merge → export` only. A user-selectable option (checkbox in the Apply UI) enables re-running exemplar selection first (`exemplars → merge → export`), which may improve merge quality after cluster shapes change significantly.
- Algorithm parameters for remerge are inherited from the parent run's config, not re-specified by the user at apply time.
- The in-memory iterative round rendering (`_render_next_round_section`) is removed as it is superseded by the snapshot + remerge flow.
