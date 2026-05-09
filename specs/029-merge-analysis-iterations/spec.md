# Feature Specification: Merge Analysis — Iteration Visibility

**Created**: 2026-04-30
**Status**: Draft
**Sighting**: SIGHTING-029
**UI Mock**: [mock.html](mock.html)

## Problem Statement

The Merge Analysis tab deduplicates pairs across all iterations, keeping only the
iteration-1 entry for each rejected pair. A pair like C1 vs C10 that is re-evaluated
5 times (with C1 growing 135→143) appears as a single row with size=135, making it
look like C1 never grew. There is no way to see whether a rejection reason persisted,
improved, or worsened across iterations.

## Objectives

1. Show how many iterations ran and what was merged at each step (timeline strip).
2. Let users filter the pair list to a specific iteration.
3. Show per-pair evaluation history inside the expander — one row per iteration.
4. Default view ("Latest") shows the last evaluation of each pair, not the first.

## Design (see mock.html)

### Data layer
- Add `iteration: int = 0` to `MergeDecisionRow`.
- In `_parse_merge_log_to_rows()`: remove the `seen_rejected` dedup set. Keep ALL
  entries from the log with their iteration number. For `rejections`, keep all rows
  (not just the first per pair).
- Add helper `_latest_per_pair(rows)` → deduped list keeping the last iteration per
  pair (used for the "Latest" default view).

### Iteration Timeline Strip (new component)
Above the pair list. One cell per iteration showing:
- Which pair was merged (or "No merge")
- Number of candidates evaluated
- The cluster that absorbed (its updated size)

Clicking a cell sets the iteration filter to that number.

### Iteration Pill Filter (addition to existing controls)
`All | Latest | 1 | 2 | 3 | …` pills alongside the existing Filter/Sort row.
- **Latest** (default): last evaluation per pair — same count as before, but correct sizes
- **N**: all pairs from iteration N
- **All**: every log entry, undeduped (may have many duplicate pairs)

### Multi-iteration badge in pair header
Pairs with >1 evaluation show a `N iters ↻` badge. Header shows LATEST iteration sizes.

### History table inside expander
"Evaluation history" table: one row per iteration showing cluster sizes, exemplar dist,
and action. Latest row highlighted. Includes a diagnostic note if exemplar dist
never changed across iterations (suggests SIGHTING-030 bridge node blind spot).

### Summary metrics
Replace "Rejected candidates" with two metrics:
- **Iterations run** (new)
- **Unique rejected pairs** (deduped count, same as current)

## Acceptance Criteria

1. Default view shows LATEST iteration sizes for each pair (not iter-1 sizes).
2. Iteration Timeline Strip correctly shows which pair was merged per iteration.
3. Selecting iteration N filters pair list to only that iteration's candidates.
4. Rejected pair expander shows history table with one row per iteration.
5. Multi-iter badge appears for pairs evaluated in >1 iteration.
6. "Iterations run" metric is displayed.
7. Existing approve/reject decision flow is unaffected.
