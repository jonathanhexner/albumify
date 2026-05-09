# Technical Plan: Interactive Merge Approval + ML Merge Classifier

**Spec**: spec.md
**Created**: 2026-04-12
**Status**: Approved

## Architecture Overview

Six self-contained changes. Each can be reviewed and tested independently.

```
face_cluster/config.py        ← [1] Remove 5 adaptive-threshold fields
face_cluster/merge.py         ← [1] Remove adaptive logic; [2] Add apply_manual_merges()
face_cluster/pipeline.py      ← [3] Add merge_decisions field to PipelineResult
face_cluster/export.py        ← [3] Add save/load merge_decisions.json
face_cluster/loader.py        ← [3] Load merge_decisions.json on run load
app/face_clustering.py        ← [4] Approval UI in Merge Analysis tab
tests/face_clustering/        ← [5] Unit tests for merge + export contracts
```

## Phase 1: Simplify ConservativeMerger

**Files**: `face_cluster/config.py`, `face_cluster/merge.py`

Remove from `PipelineConfig`:
- `merge_use_adaptive_threshold` (bool)
- `merge_exemplar_percentile` (int)
- `merge_global_percentile` (int)
- `merge_threshold_alpha` (float)
- `merge_threshold_beta` (float)

Remove from `ConservativeMerger`:
- `_compute_cluster_thresholds()`
- `_compute_global_threshold()`
- Adaptive branch in `_merge_clusters_internal()` (lines 105-115, 164-170)
- `last_thresholds`, `last_global_threshold` instance state (replaced by fixed value)

Simplify `_evaluate_merge_evidence()`: gate A threshold = `config.merge_exemplar_threshold` (fixed).

Update `merge_metadata` dict to drop `cluster_thresholds` / `global_threshold` keys (no longer computed).

**Backward compat**: `_render_threshold_distribution()` in app already guards on `view.cluster_thresholds is None` — no app change needed for old runs.

## Phase 2: apply_manual_merges()

**File**: `face_cluster/merge.py`

New public function:
```python
def apply_manual_merges(
    base_cluster_result: ClusterResult,
    approved_pairs: List[Tuple[int, int]],
    distance_matrix: np.ndarray,
) -> ClusterResult:
```

Implementation:
1. Union-find over all cluster IDs
2. For each approved `(a, b)` pair: union(a, b)
3. Build new cluster assignment from roots
4. Recompute stats and exemplars for merged clusters using existing `_compute_cluster_stats()`
5. Return new `ClusterResult`

Note: `approved_pairs` use cluster IDs from `base_cluster_result.clusters`, NOT from merged result.

## Phase 3: MergeDecision Persistence

**Files**: `face_cluster/pipeline.py`, `face_cluster/export.py`, `face_cluster/loader.py`

### merge_decisions.json schema (writer-owns-contract)

```json
[
  {
    "cluster_a": 3,
    "cluster_b": 7,
    "decision": "approve",
    "n_gates_passed": 2,
    "exemplar_dist": 0.312,
    "threshold_used": 0.35,
    "support": 4,
    "required_support": 2,
    "margin_gap": 0.08,
    "post_diameter": 0.41,
    "run_id": "Germany_run_7",
    "timestamp": "2026-04-12T14:30:00"
  }
]
```

### PipelineResult (pipeline.py)
Add field: `merge_decisions: Optional[List[Dict]] = None`

### export.py additions
- `save_merge_decisions(decisions: List[Dict], output_dir: Path) -> None`
  — writes `merge_decisions.json`; uses existing `_NumpyEncoder`
- `load_merge_decisions(run_dir: Path) -> Optional[List[Dict]]`
  — returns `None` if file absent; raises on malformed JSON

### loader.py
In `load_pipeline_result()`, after loading merge_metadata, add:
```python
merge_decisions_path = run_dir / "merge_decisions.json"
merge_decisions = load_merge_decisions(run_dir) if merge_decisions_path.exists() else None
```
Pass to `PipelineResult(... merge_decisions=merge_decisions)`.

## Phase 4: Merge Approval UI

**File**: `app/face_clustering.py`

### Session state keys (new)
- `merge_approval_decisions`: `Dict[Tuple[int,int], str]` — `(min_id, max_id) → "approve"/"reject"`
- `merge_approval_applied_result`: `Optional[ClusterResult]` — result after Apply

### Pre-population
When a run loads (in `render_merge_analysis_tab()`), populate `merge_approval_decisions`:
1. If `result.merge_decisions` exists → load from file
2. Else → derive from `result.merge_log`: `action=="merged"` → `"approve"`, `"rejected"` → `"reject"`

Call `_invalidate_approval_state()` whenever a new run is loaded.

### `_render_all_decisions_table()` changes
- Sort by `n_gates_passed` ascending (contested = 1–3 gates first), then by `exemplar_dist`
- Add "Show contested only" toggle (checkbox) — filters to `n_gates_passed` in 1..3
- For each row, add Approve/Reject buttons (replacing static `action` column)
- Button click updates `st.session_state.merge_approval_decisions[(min,max)]`

### New `_render_approval_controls()` section (inserted before decisions table)
- Bulk: "Accept all heuristic" / "Reset all" buttons
- Tally: "N approved, M rejected, K unreviewed"
- "Apply Approved Merges" button → calls `apply_manual_merges()` on base clusters
- After apply: show cluster count delta + inline cluster gallery (reuse existing `_render_cluster_grid()`)
- "Save Decisions" button → calls `save_merge_decisions()` + success toast

### run_merge_analysis_tab() guard update
Remove hard block on `result.merged_cluster_result is None` — the tab is now useful even without
a prior merge run (it can propose candidates from merge_log and allow manual approval).

## Phase 5: Tests

**File**: `tests/face_clustering/test_merge.py` (new)
- `ut_ApplyManualMerges::test_simple_pair` — approve one pair → one merged cluster
- `ut_ApplyManualMerges::test_transitive_chain` — approve A+B and B+C → single cluster
- `ut_ApplyManualMerges::test_no_approvals` → cluster count unchanged
- `ut_ApplyManualMerges::test_unknown_pair_ignored` — bad cluster id doesn't crash

**File**: `tests/face_clustering/test_export.py` (existing — add tests)
- `ut_MergeDecisions::test_save_load_roundtrip` — write then read, assert all fields present with correct types
- `ut_MergeDecisions::test_load_missing_returns_none` — absent file → None

**File**: `tests/face_clustering/test_merge_stage.py` (existing)
- Add `ut_SimplifiedMerger::test_fixed_threshold_used` — verify no adaptive computation
- Add `ut_SimplifiedMerger::test_four_gates_still_enforced` — all 4 gates still block

## Risks

- `_render_threshold_distribution()` tab section references `merge_threshold_alpha`/`beta` from run summary config — will show "?" after adaptive removal for new runs. Acceptable; section is for diagnostic use only.
- Runs saved before this change have `cluster_thresholds` in merge_metadata.json — loader and display code already handle this gracefully via `None` checks.
