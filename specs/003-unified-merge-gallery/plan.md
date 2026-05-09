# Technical Plan: Unified Merge Decision Gallery

**Spec**: spec.md
**Created**: 2026-04-14
**Status**: Approved

## Architecture Overview

Pure UI change in `app/face_clustering.py`. No backend, data model, or face_cluster/ changes needed — all required data already exists in `MergeAnalysisView` and `MergeDecisionRow`.

```
app/face_clustering.py
  - ADD: _render_unified_merge_gallery(view, result)   ← new combined gallery
  - REMOVE: _render_all_decisions_table(view)           ← replaced
  - REMOVE: _render_near_miss_gallery(view, result)     ← absorbed into gallery
  - REMOVE: _render_merged_pairs_gallery(view, result)  ← absorbed into gallery
  - REMOVE: _render_rejected_candidates_gallery(view, result) ← absorbed into gallery
  - UPDATE: _render_merge_analysis() call order
  - UPDATE: _invalidate_run_caches() to reset gallery pagination state
```

## Data Available Per Row

`MergeDecisionRow` (from `face_cluster/analysis_views.py`) already has:
- `cluster_a`, `cluster_b`, `cluster_a_size`, `cluster_b_size`
- `exemplar_dist`, `threshold_used`, `n_gates_passed`, `action`
- `passes_exemplar`, `passes_support`, `passes_margin`, `passes_diameter`
- `support`, `required_support`, `margin_gap`, `post_diameter`, `max_allowed_diameter`
- `exemplar_face_ids_a`, `exemplar_face_ids_b` — used by existing `_render_pair_crops()`
- `rejection_reason`

`MergeAnalysisView.merges + MergeAnalysisView.rejections` = all decisions.

## New Session State Keys

```python
"merge_gallery_filter":    str   # "All"|"Merged"|"Rejected"|"Near Misses"|"Contested"  default "All"
"merge_gallery_sort":      str   # "exemplar_dist"|"gates_passed"                        default "exemplar_dist"
"merge_gallery_page":      int   # current page (0-indexed)                              default 0
```

Add these to `_DEFAULT_SESSION_STATE` and reset `merge_gallery_page` to 0 in `_invalidate_run_caches()`.

## `_render_unified_merge_gallery(view, result)` Design

### 1. Build the row pool

```python
all_rows = view.merges + view.rejections
```

### 2. Filter

| Filter label | Condition |
|---|---|
| All | no filter |
| Merged | `d.action == "merged"` |
| Rejected | `d.action == "rejected"` |
| Near Misses | `d.n_gates_passed == 3` |
| Contested | `1 <= d.n_gates_passed <= 3` |

### 3. Sort

| Sort option | Key |
|---|---|
| Exemplar distance | `d.exemplar_dist` ascending |
| Gates passed | `d.n_gates_passed` ascending (most contested first), then `d.exemplar_dist` |

### 4. Pagination

Page size: 10. Navigation: Prev / Next buttons + "Page N of M" label.
Reset page to 0 when filter or sort changes (detect via `st.session_state` comparison before selectbox renders, or use `on_change` callback).

### 5. Per-pair card (one `st.expander` per row)

**Expander header** (collapsed by default):
```
C{a} ({sz_a}) vs C{b} ({sz_b})  |  dist={exemplar_dist:.3f}  |  gates={n}/4  |  [MERGED / REJECTED badge]  |  decision: [approve/reject/-]
```

**Expander body:**
1. `_render_pair_crops(row, result, symbol)` — reuse unchanged
2. Gate badges row (4 columns):
   - Each badge: `[PASS] Exemplar: 0.312/0.350` (green) or `[FAIL] Exemplar: 0.412/0.350 (+0.062)` (red)
   - Use `st.markdown` with inline HTML color spans (same color scheme as existing `_gate_bg()`)
3. Approve / Reject buttons (same logic as in existing `_render_all_decisions_table()`)
4. (optional) Rejection reason caption if `row.rejection_reason`

## Sections Retained Unchanged

- `_render_criteria_reference()`
- `_render_merge_summary(view)`
- `_render_gate_bottleneck(view)`
- `_render_threshold_distribution(view, result)`
- `_render_approval_controls(view, result)` — bulk approve/reject + tally + Apply + Save
- `_render_absorbed_clusters(view)`

The styled read-only summary dataframe (currently at bottom of `_render_all_decisions_table`) is **dropped** — the gallery provides the same information visually.

## New call order in `_render_merge_analysis()`

```python
_render_criteria_reference()
_render_merge_summary(view)
_render_gate_bottleneck(view)
_render_threshold_distribution(view, result)
_render_approval_controls(view, result)
_render_unified_merge_gallery(view, result)   # ← NEW (replaces 4 old calls)
_render_absorbed_clusters(view)
```

## Risks

- Streamlit `on_change` callbacks for filter/sort require care — resetting page index must happen before `st.session_state.merge_gallery_page` is read for slicing. Simplest approach: detect change by comparing current selectbox value to stored state value at the top of the function.
- If `exemplar_face_ids_a` / `exemplar_face_ids_b` are empty (old runs), `_render_pair_crops` already handles gracefully (shows no images with no crash).
- `merge_approval_decisions` session state key is preserved unchanged — approve/reject button logic is identical.
