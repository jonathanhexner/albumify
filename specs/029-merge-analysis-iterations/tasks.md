# Tasks: Merge Analysis — Iteration Visibility

**Spec**: [spec.md](spec.md)
**Sighting**: SIGHTING-029

## Checklist

- [ ] **T1 — Data layer: add `iteration` field + remove dedup**
  - Add `iteration: int = 0` to `MergeDecisionRow` (face_cluster/views/merge_view.py)
  - In `_parse_merge_log_to_rows()`: remove `seen_rejected` set; keep ALL entries with
    `entry["iteration"]` populated
  - Add `_latest_per_pair(rows)` helper that returns the last iteration entry per
    (cluster_a, cluster_b) key

- [ ] **T2 — `MergeAnalysisView`: expose iteration data**
  - Add `n_iterations: int` field
  - Add `iter_timeline: List[dict]` — one entry per iteration:
    `{iteration, merged_pair, n_candidates, cluster_grown, size_after}`
  - Populate in `build_merge_analysis_view()` by scanning `actually_merged=True` entries

- [ ] **T3 — UI: Iteration Timeline Strip**
  - New function `_render_iteration_timeline(view)` in `_merge_decisions_panel.py`
  - One `st.columns` cell per iteration; click sets `st.session_state.merge_iter_filter`

- [ ] **T4 — UI: Iteration pill filter + apply to pair list**
  - Add iteration pills (All / Latest / 1 / 2 / …) to controls row in
    `_render_flat_merge_gallery()` and `_render_grouped_merge_gallery()`
  - Before rendering pairs: filter/dedup rows based on selected iteration

- [ ] **T5 — UI: Multi-iter badge + history table in expander**
  - In pair header: show `N iters ↻` badge if pair appears in >1 iteration
  - In `_render_gate_badges()` / expander body: add history table from all rows
    for that pair; highlight latest row
  - Add diagnostic note if exemplar dist is unchanged across all iterations

- [ ] **T6 — UI: Summary metric update**
  - Replace "Rejected candidates" metric with "Iterations run" + "Unique rejected pairs"

- [ ] **T7 — Tests**
  - Unit test: `_parse_merge_log_to_rows` with a 3-iteration log returns all rows
    (not deduped); iteration field is populated
  - Unit test: `_latest_per_pair` returns correct last entry per pair
