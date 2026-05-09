# Feature Specification: Cluster Gallery UX

**Feature Branch**: `004-cluster-gallery-ux`
**Created**: 2026-04-14
**Status**: Implemented
**Input**: User description: "Clusters (Merged) isn't very UI friendly. For every cluster we need to select and open. Would be more convenient if it was in an expandable table showing some thumbnails so I don't need to review everything."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Scan all clusters at a glance (Priority: P1)

A user has just applied manual merges and wants to quickly scan all resulting clusters to spot any obvious mistakes (wrong faces in a cluster, a person split across two clusters) — without having to open each one individually via a dropdown.

**Why this priority**: The core complaint. The current select-then-open workflow requires O(n) interactions to review n clusters. A thumbnail gallery reduces this to a single scroll.

**Independent Test**: Load a merged result with 10+ clusters. Without clicking any button, thumbnails for every cluster are visible on the page. User can visually assess all clusters without any selection step.

**Acceptance Scenarios**:

1. **Given** a loaded merged result, **When** user opens Clusters (Merged) tab, **Then** all clusters are displayed as expandable rows with thumbnail previews visible without opening each row.
2. **Given** the cluster gallery, **When** a cluster row is collapsed, **Then** it shows cluster ID, size, diameter, and up to 3 exemplar face thumbnails inline.
3. **Given** the cluster gallery, **When** a cluster row is expanded, **Then** it shows the full cluster detail view (all faces, stats) — same content as the current "Open Merged Cluster Analysis" flow.

---

### User Story 2 - Sort and filter clusters (Priority: P2)

A user wants to focus on large clusters or clusters with high diameter (potentially merged incorrectly) without scrolling through small singleton-like clusters.

**Why this priority**: Useful for QA of merge results but not core to the thumbnail request.

**Independent Test**: Sort by size descending — largest clusters appear first. Filter to clusters with size >= 5 — small clusters hidden.

**Acceptance Scenarios**:

1. **Given** the gallery, **When** user selects "Sort by: size", **Then** clusters are ordered by size descending.
2. **Given** the gallery, **When** user selects "Sort by: diameter", **Then** clusters are ordered by diameter descending.

---

### Edge Cases

- What if a cluster has no exemplar crops on disk? Show placeholder boxes with face_id labels.
- What if there are 100+ clusters? Render all rows collapsed by default; expanding is opt-in. No pagination needed since collapsed rows are cheap.
- What if merged result has 0 clusters? Show info message.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST display all clusters on page load without requiring any selection step.
- **FR-002**: Each cluster row MUST show, when collapsed: cluster ID, size, diameter, and up to 3 exemplar face thumbnails (small, ~80px).
- **FR-003**: Each cluster row MUST be expandable to show the full existing cluster detail view.
- **FR-004**: Clusters MUST be sorted by size descending by default.
- **FR-005**: System MUST provide sort options: size (default), diameter.
- **FR-006**: The summary metrics (cluster count, noise count, clustered faces) at the top of the tab MUST be preserved.
- **FR-007**: The same gallery design MUST apply to both the Clusters (Base) tab and the Clusters (Merged) tab.

### Key Entities

- **ClusterRow** (from `RunOverview`): already contains `cluster_id`, `size`, `diameter`, `n_exemplars`, `exemplar_face_ids`. Used directly.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: User can visually assess all clusters without clicking any button (thumbnails visible on page load in collapsed state).
- **SC-002**: Expanding a cluster shows the same detail as the current "Open Merged Cluster Analysis" flow — no regression in detail view.
- **SC-003**: Tab loads within 3 seconds for runs with up to 50 clusters (collapsed rows with 3 thumbnails each).

## Assumptions

- `RunOverview.cluster_rows` already contains `exemplar_face_ids` per cluster (the face IDs whose crops can be loaded via `_crop_for_face`). If this field is absent, fall back to no thumbnails with a caption.
- The Clusters (Base) tab has the same select-then-open pattern and will receive the same treatment (FR-007).
- The existing `_render_cluster_detail()` function is reused unchanged for the expanded body — no regression risk on detail view content.
- Collapsed rows with 3 small images (80px) are cheap enough to render all at once without pagination.
