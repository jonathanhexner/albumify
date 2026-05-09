# Feature Specification: Unified Merge Decision Gallery

**Feature Branch**: `003-unified-merge-gallery`
**Created**: 2026-04-14
**Status**: Draft
**Input**: User description: "In Merge Analysis tab, All Merge Decisions needs to show images in each cluster (like Rejected Candidates does). Merge/combine the four separate sections (All Merge Decisions, Near Misses, Merged Pairs, Rejected Candidates) into a unified filterable gallery."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Review merge decisions with visual context (Priority: P1)

A user opens the Merge Analysis tab after running a face clustering pipeline. They want to evaluate whether the heuristic merge decisions are correct by seeing the actual face crops from each cluster pair side-by-side, alongside the gate metrics that drove the decision.

**Why this priority**: The current "All Merge Decisions" table shows only numbers (cluster IDs, gate deltas, pass/fail) with no face images. The user must scroll to separate sections (Merged Pairs, Rejected Candidates) to see what the clusters actually look like. This disconnects the decision-making context from the visual evidence.

**Independent Test**: Load a completed pipeline run with merge results. Open Merge Analysis tab. Every merge decision row shows exemplar face crops for both clusters alongside gate metrics. User can evaluate correctness without scrolling to other sections.

**Acceptance Scenarios**:

1. **Given** a pipeline run with merge results, **When** user opens Merge Analysis tab, **Then** each merge decision displays exemplar face crops for cluster A and cluster B inline with the gate metrics.
2. **Given** a merge decision row, **When** user views it, **Then** the row shows: face crops for both clusters, the 4 gate values with pass/fail indication, the heuristic outcome (merged/rejected), and approve/reject buttons.

---

### User Story 2 - Filter decisions by category (Priority: P1)

A user wants to focus on specific subsets of merge decisions -- e.g., only rejected pairs, or only contested decisions (1-3 gates passing) -- without scrolling through everything.

**Why this priority**: The current UI has four separate sections for different decision categories. Replacing them with a single gallery requires filters to preserve the ability to focus on subsets.

**Independent Test**: Use filter controls to select "Rejected" only. Verify only rejected pairs are shown. Switch to "Near Misses (3/4 gates)". Verify only 3/4 gate pairs are shown.

**Acceptance Scenarios**:

1. **Given** the unified gallery, **When** user selects "All" filter, **Then** all merge decisions (merged + rejected) are displayed.
2. **Given** the unified gallery, **When** user selects "Merged" filter, **Then** only pairs that were merged are displayed.
3. **Given** the unified gallery, **When** user selects "Rejected" filter, **Then** only rejected pairs are displayed.
4. **Given** the unified gallery, **When** user selects "Near Misses" filter, **Then** only pairs with exactly 3/4 gates passing are displayed.
5. **Given** the unified gallery, **When** user selects "Contested" filter, **Then** only pairs with 1-3 gates passing are displayed.

---

### User Story 3 - Sort and paginate decisions (Priority: P2)

A user has many merge candidates (50+) and wants to sort by exemplar distance or gates passed, and paginate to avoid a slow, overwhelming page.

**Why this priority**: Important for usability but not core to the visual context problem being solved. The current UI already renders all rows which can be slow for large runs.

**Independent Test**: Sort by exemplar distance ascending. Verify order is correct. Navigate to page 2. Verify different pairs are shown.

**Acceptance Scenarios**:

1. **Given** 50+ merge candidates, **When** user views the gallery, **Then** results are paginated (default 10 per page).
2. **Given** the gallery, **When** user selects "Sort by: exemplar distance", **Then** pairs are ordered by exemplar distance ascending.
3. **Given** the gallery, **When** user selects "Sort by: gates passed", **Then** pairs are ordered by number of gates passed descending (contested first).

---

### Edge Cases

- What happens when a pipeline run has no merge candidates? Display an info message ("No merge candidates found").
- What happens when face crop images are missing from disk? Show a placeholder with the face_id label (current `_render_pair_crops` already handles this).
- What happens with 100+ candidates? Pagination prevents rendering all at once.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST display exemplar face crops for both clusters in every merge decision row.
- **FR-002**: System MUST show the 4 gate metrics (Exemplar, Support, Margin, Diameter) with pass/fail indication and delta values per row.
- **FR-003**: System MUST provide filter controls to show: All, Merged, Rejected, Near Misses (3/4), Contested (1-3).
- **FR-004**: System MUST provide sort options: by exemplar distance, by gates passed.
- **FR-005**: System MUST paginate results (default 10 per page) with page navigation controls.
- **FR-006**: System MUST preserve approve/reject buttons per decision row.
- **FR-007**: System MUST replace the four existing sections (All Merge Decisions, Near Misses, Merged Pairs, Rejected Candidates) with the unified gallery. The summary table, approval controls, and absorbed clusters sections remain unchanged.

### Key Entities

- **MergeDecisionRow**: A cluster pair candidate with gate metrics, heuristic outcome, user decision, and exemplar face IDs for both clusters. Already exists in `MergeAnalysisView.merges` and `MergeAnalysisView.rejections`.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can evaluate any merge decision (see faces + metrics) without scrolling to a different section.
- **SC-002**: Users can filter to any decision category in one click.
- **SC-003**: Page with 50+ merge candidates loads and renders without noticeable lag (under 2 seconds).
- **SC-004**: Zero regressions: all existing approve/reject functionality continues to work.

## Assumptions

- Existing `_render_pair_crops` function and `MergeAnalysisView` data structures are reused without modification.
- The summary table (`_render_all_decisions_table` styled dataframe) is kept as a read-only reference below the gallery, or removed if the gallery makes it redundant.
- Gate Bottleneck, Threshold Distribution, and Absorbed Clusters sections are not affected by this change.
- The `merge_approval_decisions` session state mechanism is preserved as-is.
