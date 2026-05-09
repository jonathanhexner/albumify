# Feature Specification: Run History & Run Annotations

**Feature Branch**: `013-run-history-annotations`
**Created**: 2026-04-23
**Status**: Implemented
**Spec**: This file

---

## User Scenarios & Testing

### User Story 1 — Source-album identity across derived runs (Priority: P1)

After running base clustering on an album, the user runs recluster and remerge several times. Every derived run must appear under the **same album name** in the History tab — the name of the original source image folder — not the output directory name. The user can filter the History table by album to see all variants at once.

**Why this priority**: This is the primary reported problem. Without it, the History tab is unusable for multi-run workflows: every variant looks like a separate, unrelated album.

**Independent Test**: Create one base run and three derived runs (remerge × 2, recluster × 1) from the same source folder. Open the History tab; all four rows show the same value in the Album column. Apply the Album filter: selecting that album shows exactly the four rows; selecting any other album shows zero rows.

**Acceptance Scenarios**:

1. **Given** a base run registered for source album `Noa2_5`, **When** the user runs Apply + Remerge, **Then** the new run row in `action_log` has `source_album = 'Noa2_5'` (not the output path).
2. **Given** a recluster action on the same source, **When** the run completes, **Then** `action_log.source_album` equals the original `session.json.source_album` value.
3. **Given** rows from two different source albums in the History table, **When** the user selects "Album = Noa2_5" in the filter dropdown, **Then** only runs whose `source_album = 'Noa2_5'` are shown.
4. **Given** a pre-013 row where `source_album` is NULL, **When** the History table is displayed, **Then** the Album cell shows a fallback label "(unknown)" rather than crashing.

---

### User Story 2 — Overwrite protection and unique output directories (Priority: P1)

When the user triggers Apply + Remerge (or any action that creates an output run directory) the system must never silently overwrite an existing directory. A central path allocator guarantees a unique output path and shows it to the user before dispatching.

**Why this priority**: Silent overwrites destroy previous results without warning. Combined with US1, this is part of the P0 problem: output-folder chaos caused both the naming mess and the data-loss risk.

**Independent Test**: From the app, trigger Apply + Remerge twice in a row from the same parent. Confirm that two separate directories exist on disk and both appear as distinct rows in the History table; neither directory overwrites the other.

**Acceptance Scenarios**:

1. **Given** `results/Noa2_5/remerge_1` already exists, **When** `allocate_run_dir` is called with the same source album and kind, **Then** it returns `results/Noa2_5/remerge_2`.
2. **Given** the UI is about to dispatch an action, **When** the target path is computed, **Then** the UI displays the full output path before the action starts.
3. **Given** concurrent calls to `allocate_run_dir` with the same base path, **When** both calls are resolved, **Then** they return different paths (atomicity test).

---

### User Story 3 — Per-run free-text comment (Priority: P1)

Every run row can carry a short free-text comment entered by the user. The comment is editable from the History table and from the run header when a run is loaded. It persists across app restarts. The History table can be searched by comment text.

**Why this priority**: Without comments the user cannot distinguish runs like `remerge_3` vs `remerge_4` after the fact.

**Independent Test**: Open the History tab, type a comment on a run, reload the app, verify the comment is still present. Run a free-text search that matches only the comment substring; verify only that row is returned.

**Acceptance Scenarios**:

1. **Given** a run is shown in the History table, **When** the user types a comment and presses Enter, **Then** the comment is saved to `action_log.comment` immediately.
2. **Given** a saved comment, **When** the app is restarted, **Then** the comment is still displayed.
3. **Given** two runs where only one has a comment containing "tight threshold", **When** the user searches for "tight", **Then** only that run appears.
4. **Given** a comment longer than 2 048 characters, **When** the user attempts to save it, **Then** the input is rejected with a visible error message.

---

### User Story 4 — Searchable, filterable History table (Priority: P1)

The History tab renders a single flat table with one row per run. The table has an Album filter dropdown, a date-range picker, and a free-text search bar. The user can sort by any column and click a row to load that run.

**Why this priority**: The flat unfiltered list is already unmanageable with ten runs. This is the primary navigation improvement.

**Independent Test**: Populate the DB with runs across 3 albums and a date range of 2 weeks. Verify: (a) Album filter narrows to the correct subset; (b) date range hides runs outside the window; (c) text search on run name substring narrows correctly; (d) clicking a row loads that run.

**Acceptance Scenarios**:

1. **Given** 12 runs across 3 albums, **When** the History tab opens with no filters, **Then** all 12 rows are visible.
2. **Given** the Album filter is set to album A, **When** the table refreshes, **Then** only the 4 rows for album A are shown.
3. **Given** a free-text query that matches one run's comment, **When** the search is submitted, **Then** exactly that row is returned.
4. **Given** the user clicks a row in the History table, **When** the click is registered, **Then** the app loads that run and navigates to the Clusters tab.

---

### User Story 5 — Parent pointer and config diff in run header (Priority: P2)

When a run is loaded the page header shows the source album name, the run name, and the parent run (if any). Expanding the "Config Δ" control reveals a diff table showing only the config fields that differ from the parent run. If no parent exists, the header shows "Base run".

**Why this priority**: Needed to understand what changed between iterations, but not as urgent as the navigation and data-integrity fixes.

**Independent Test**: Load a remerge run. Verify the header shows the correct source album and parent run label. Expand Config Δ; verify it lists only fields whose values differ (e.g., `distance_threshold`) and hides identical fields.

**Acceptance Scenarios**:

1. **Given** a remerge run with a known parent, **When** the run is loaded, **Then** the header shows "Parent: <parent run name>" as a clickable link.
2. **Given** the parent config had `distance_threshold=0.35` and the child has `0.5`, **When** Config Δ is expanded, **Then** the table shows one row: `distance_threshold | 0.35 → 0.5`.
3. **Given** a base run (no parent), **When** the run is loaded, **Then** the header shows "Base run" with no Config Δ control.

---

### User Story 6 — Run Summary dashboard (Priority: P2)

When a run is selected or loaded, a Run Summary panel shows: a quality funnel (detected → passed quality gates), cluster formation counts (before/after merge), stage durations, and a clear "0 merge candidates" message when no merges were attempted.

**Why this priority**: Depends on spec 012 data being present; useful but not blocking navigation or data integrity.

**Independent Test**: Load a run that produced no merge candidates. Verify the panel shows "0 candidates at threshold X" using the value from `merge_metadata.json`, not a generic "no merges" label.

**Acceptance Scenarios**:

1. **Given** a run with `merge_log.json = []` and `merge_metadata.json.merge_candidate_threshold = 0.45`, **When** the Run Summary is viewed, **Then** it displays "0 candidates at threshold 0.45".
2. **Given** a run with quality gating data, **When** the quality funnel is displayed, **Then** it shows numeric counts for each gate stage.
3. **Given** stage timing data in `pipeline_run.json.stages`, **When** the Run Summary is viewed, **Then** each stage name and duration in seconds is listed.

---

### Edge Cases

- Runs registered before spec 013 have `source_album = NULL` — display as "(unknown)" in Album column; do not crash.
- `allocate_run_dir` called in rapid succession (race condition) — must atomically reserve the path.
- Config stored in `payload_json` may be missing keys present in the parent — treat missing keys as unchanged (show only keys present in both configs).
- Run Summary panel requested for a run missing `merge_metadata.json` (pre-012 run) — show "Data not available (pre-012 run)".
- Comment save while DB is temporarily locked — surface an error to the user, do not silently drop the comment.

---

## Requirements

### Functional Requirements

- **FR-001**: The `action_log` table MUST be extended with columns: `source_album`, `run_name`, `parent_run_id`, `run_kind`, `comment`, `config_json`, `n_core`.
- **FR-002**: `source_album` MUST be written as the original input image directory name. For derived runs it MUST be inherited from the parent run's `source_album`, never derived from the output path.
- **FR-003**: A `allocate_run_dir(source_album, kind, label) -> Path` helper MUST guarantee a unique output directory before any action dispatches.
- **FR-004**: All action-dispatch paths in the app (Apply + Remerge, recluster, manual merge) MUST call `allocate_run_dir`; none may compute the output path locally.
- **FR-005**: The History tab MUST render a flat table with columns: Album, Run name, Output folder, Kind, Parent, Created at, n_faces/n_core, n_clusters, Key config, Comment.
- **FR-006**: The History tab MUST provide: Album filter dropdown, date-range filter, free-text search across album + run name + comment.
- **FR-007**: Every run row MUST have an editable Comment field, persisted to `action_log.comment`, with a hard limit of 2 048 characters.
- **FR-008**: When a run is loaded, the page header MUST display: source album, run name, parent run (clickable), and an expandable Config Δ diff.
- **FR-009**: Config Δ MUST show only fields that differ between parent and child config; identical fields MUST be collapsed by default.
- **FR-010**: The Run Summary panel MUST consume `pipeline_run.json.quality_summary`, `merge_metadata.json`, and `pipeline_run.json.stages` (spec 012 outputs) to render quality funnel, cluster counts, and stage timeline.
- **FR-011**: When `merge_log.json` is empty, the Run Summary MUST display "0 candidates at threshold X" using `merge_metadata.json.merge_candidate_threshold`.
- **FR-012**: A DB migration MUST add new columns to existing rows; pre-013 rows get `source_album = NULL`, `parent_run_id = NULL`, `comment = NULL`.

### Key Entities

- **ActionLog row** (`action_log`): represents one user-initiated operation; extended with album identity, lineage, and annotation fields.
- **RunRow** (Python dataclass): typed projection of an `action_log` row used by the History search helper and the UI.
- **HistoryFilters** (Python dataclass): encapsulates album, date range, and text search inputs for the `run_history.search()` call.
- **ConfigDelta** (Python dataclass): one row in the config diff — `(field, parent_value, child_value)`.

---

## Success Criteria

### Measurable Outcomes

- **SC-001**: In the History tab, all runs derived from the same source album share the same Album cell value. Zero runs appear under an output-directory-based album name.
- **SC-002**: Running Apply + Remerge N times from the same parent produces N distinct output directories; zero overwrites.
- **SC-003**: A comment saved on a run is still present after app restart with 100% reliability.
- **SC-004**: Album filter, date filter, and free-text search each independently narrow the History table to the correct subset, verified by automated test.
- **SC-005**: Config Δ shows exactly the fields that differ and hides fields that are identical.
- **SC-006**: Run Summary "0 candidates" message quotes the correct threshold value from `merge_metadata.json`.

---

## Assumptions

- Spec 012 must be deployed before US6 (Run Summary) is functional; US1–US5 have no dependency on spec 012 outputs.
- The `session.json.source_album` field (already present in spec 009) is the authoritative source-album value for new base runs.
- `pipeline_run.json` is written by the pipeline and can be extended with `source_album`, `run_name`, `comment` fields without breaking existing readers (additive only).
- The app runs as a single-user local tool; multi-user concurrency for comment writes is not a concern. Concurrent `allocate_run_dir` calls within one session must still be safe.
- Existing DB rows predating spec 013 have no parent-run inference; `parent_run_id = NULL` is acceptable for all pre-existing rows.
