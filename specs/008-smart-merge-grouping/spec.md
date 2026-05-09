# Feature Specification: Smart Merge Grouping

**Feature Branch**: `008-smart-merge-grouping`
**Created**: 2026-04-20
**Status**: Draft
**Input**: User description: "Reduce merge analysis review burden via transitive grouping of merge candidates, component cohesion scoring, and smart auto-approve/reject. Filed as SIGHTING-021."

## User Scenarios & Testing

### User Story 1 - Smart Approve Reduces Review to Near-Misses Only (Priority: P1)

A user runs face clustering on a large album (e.g., 750+ faces, 39 clusters). The merge pipeline produces ~226 candidate pairs. Instead of reviewing all 226 individually, the system groups transitively connected candidates into connected components, classifies each group by confidence, and auto-resolves obvious decisions. The user only reviews borderline groups.

**Why this priority**: This is the core value proposition. Without this, the merge analysis tab is impractical for albums with more than ~10 clusters. The user reported spending 30+ minutes on 23 pages of mostly obvious decisions.

**Independent Test**: Run pipeline on a multi-person album, open Merge Analysis tab, click "Smart Approve", verify that only review-worthy groups remain for manual inspection and total decisions dropped by 90%+.

**Acceptance Scenarios**:

1. **Given** a completed pipeline run with 226 merge candidate pairs across 39 clusters, **When** the user opens Merge Analysis, **Then** candidates are grouped into connected components (significantly fewer than 226 groups) and each group is labeled with a confidence tier (auto_approve / review / auto_reject).
2. **Given** the grouped view is displayed, **When** the user clicks "Smart Approve", **Then** all auto_approve groups are set to "approve", all auto_reject groups are set to "reject", and only review groups remain for manual inspection.
3. **Given** Smart Approve has been applied, **When** the user clicks "Apply Approved Merges", **Then** the remerge pipeline runs successfully with the same outcome as if pairs had been approved individually.

---

### User Story 2 - Group-Level Review with Per-Pair Override (Priority: P1)

A user sees a group containing clusters A, B, C. They recognize A and B as the same person but C is different. They need to approve A+B but reject A+C (which would also remove C from the group).

**Why this priority**: Equal to P1 because without per-pair control, grouping could force wrong merges. The group view must be a convenience layer, not a constraint.

**Independent Test**: Open a review group, expand individual pairs, override one pair's decision, verify the pair-level override is respected when applying merges.

**Acceptance Scenarios**:

1. **Given** a group {A, B, C} with 3 pairs displayed, **When** the user clicks "Approve Group", **Then** all 3 pairs (A+B, A+C, B+C) are set to "approve" in the decisions dict.
2. **Given** all pairs in a group are approved via "Approve Group", **When** the user expands individual pairs and clicks "Reject" on the A+C pair, **Then** A+C is set to "reject" while A+B and B+C remain "approve".
3. **Given** mixed per-pair decisions within a group, **When** the user applies merges, **Then** only approved pairs are passed to the snapshot/remerge pipeline. The downstream union-find handles transitive closure correctly (A+B approved, B+C approved, A+C rejected → A, B, C still merge together because of transitive path through B).

---

### User Story 3 - Component Cohesion as Confidence Signal (Priority: P2)

When classifying groups, the system uses component cohesion — the fraction of pairs within a connected component that pass all 4 gates. If cluster B and C are both independently strong matches with cluster A (both pass 4/4 gates), the weak direct B+C link (3/4 gates) is likely noise. High cohesion (e.g., 80%+ of pairs pass 4/4) boosts confidence in the entire group.

**Why this priority**: P2 because the basic grouping + gate-count classification (P1) already delivers most of the value. Cohesion scoring refines the boundary between "review" and "auto_approve" — it catches groups where one borderline pair shouldn't trigger manual review of the entire well-connected component.

**Independent Test**: Create a test group where 5 of 6 pairs pass 4/4 gates and 1 pair passes 3/4 gates. With basic classification this is "review". With cohesion scoring (83% cohesion > 80% threshold), it becomes "auto_approve". Verify the classification changes.

**Acceptance Scenarios**:

1. **Given** a group with 6 pairs where 5 pass 4/4 gates and 1 passes 3/4, **When** cohesion is computed, **Then** cohesion = 5/6 = 83%.
2. **Given** cohesion >= 80%, **When** the group is classified, **Then** it is classified as "auto_approve" despite having one sub-4/4 pair.
3. **Given** a group with 3 pairs where 1 passes 4/4 and 2 pass 3/4, **When** cohesion is computed (33%), **Then** the group remains classified as "review" because cohesion is below threshold.

---

### User Story 4 - Flat Pair View Fallback (Priority: P3)

Users who prefer the original per-pair gallery can toggle off the group view. This provides backward compatibility and may be preferred for small datasets where grouping adds no value.

**Why this priority**: P3 because most users will benefit from the group view. This is a safety net for edge cases and user preference.

**Independent Test**: Toggle to "Individual pairs" view, verify the original paginated pair gallery renders identically to the pre-feature behavior.

**Acceptance Scenarios**:

1. **Given** the group view is active, **When** the user unchecks "Group view", **Then** the original flat pair gallery is displayed with all existing filter/sort/pagination functionality intact.
2. **Given** old runs from before this feature, **When** loaded in Merge Analysis, **Then** groups are computed dynamically from the existing merge_log.json data with no format changes required.

---

### Edge Cases

- **Single-pair group**: A candidate pair where neither cluster appears in any other candidate. Forms a group of 1 pair, 2 clusters. Behaves identically to the current per-pair card.
- **Very large component**: 15+ clusters forming one connected component (e.g., one person photographed many times producing many similar clusters). The group card must handle this gracefully — show representative thumbnails (capped), collapse pair detail by default.
- **All pairs in one component**: Degenerate case where every cluster is connected. Produces 1 giant group. Still classifiable by cohesion. UI must not break.
- **Empty merge log**: No candidates → no groups → show "No merge candidates found" (existing behavior preserved).
- **Mixed iteration data**: merge_log.json contains entries from multiple merge iterations. The grouping operates on de-duplicated pairs (existing `_parse_merge_log` already de-duplicates rejected pairs).

## Requirements

### Functional Requirements

- **FR-001**: System MUST group merge candidate pairs into connected components using transitive closure (union-find) on cluster IDs.
- **FR-002**: System MUST classify each group into exactly one confidence tier: `auto_approve` (all pairs pass 4/4 gates), `auto_reject` (all pairs pass <=2/4 gates), or `review` (any pair passes 3+ gates but not all 4/4).
- **FR-003**: System MUST compute component cohesion for each group: the fraction of pairs that pass all 4 gates. Groups with cohesion >= 80% and no pair below 3/4 gates SHOULD be promoted to `auto_approve`.
- **FR-004**: System MUST provide a "Smart Approve" action that sets all auto_approve pairs to "approve" and all auto_reject pairs to "reject" in one click, leaving review pairs unset.
- **FR-005**: System MUST display groups as cards showing: all cluster IDs with face counts, representative exemplar crops from each cluster, confidence tier, cohesion percentage, pair count, and group-level approve/reject buttons.
- **FR-006**: System MUST allow expanding a group card to see individual pair details with per-pair approve/reject override buttons.
- **FR-007**: System MUST support toggling between group view (default) and flat pair view (original gallery).
- **FR-008**: The pair-based `merge_approval_decisions` session state dict and `merge_decisions.json` file format MUST remain unchanged. Grouping is a presentation layer only.
- **FR-009**: System MUST show a summary line with group counts by tier and pair counts: "N groups: A auto-approve (X pairs) | B review (Y pairs) | C auto-reject (Z pairs)".
- **FR-010**: System MUST paginate the group gallery. Default page size: 10 groups per page.

### Key Entities

- **MergeGroup**: A connected component of merge candidate clusters. Contains cluster IDs, all candidate pairs within the component, confidence tier, cohesion score, and face count totals.
- **MergeDecisionRow**: (existing) One candidate pair with gate pass/fail data. Now also belongs to exactly one MergeGroup.
- **MergeAnalysisView**: (existing, extended) Now includes a list of MergeGroups alongside the existing merges/rejections lists.

## Success Criteria

### Measurable Outcomes

- **SC-001**: For a dataset with 226 candidate pairs (Austria24_2), the number of manual review decisions is reduced by at least 90% (from 226 to fewer than 23).
- **SC-002**: Time to complete merge review for a 39-cluster dataset is under 5 minutes (down from 30+ minutes).
- **SC-003**: Smart Approve produces identical merge outcomes to manual all-approve on datasets where the user would have approved all heuristic merges (no behavioral regression).
- **SC-004**: Existing merge_decisions.json files from prior runs load correctly with no migration needed.
- **SC-005**: The group view is the default, but the flat pair view is accessible within one click.

## Assumptions

- The 80% cohesion threshold for auto_approve promotion is a reasonable default. This can be tuned based on user feedback but does not need to be a configurable parameter in v1.
- Groups classified as `auto_reject` (all pairs <=2/4 gates) are safe to reject without review. The 4-gate system already captures the key merge quality signals.
- The existing `merge_log.json` format contains all data needed for grouping — no new pipeline output is required.
- The existing union-find pattern in `merge.py` (used by `apply_manual_merges`) validates that transitive closure is the correct grouping strategy.
- Maximum expected group count for a 39-cluster dataset is 15-30 groups. Page size of 10 is adequate.
