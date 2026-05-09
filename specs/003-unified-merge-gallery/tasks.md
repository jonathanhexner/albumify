# Tasks: Unified Merge Decision Gallery

**Plan**: plan.md
**Created**: 2026-04-14

## Phase 1 — Session State & Scaffolding

- [ ] T001 Add `merge_gallery_filter`, `merge_gallery_sort`, `merge_gallery_page` keys to `_DEFAULT_SESSION_STATE` in `app/face_clustering.py`
- [ ] T002 Reset `merge_gallery_page` to 0 in `_invalidate_run_caches()` in `app/face_clustering.py`

## Phase 2 — Unified Gallery (P1: visual context per row)

- [ ] T003 [US1] Implement `_render_unified_merge_gallery(view, result)` in `app/face_clustering.py`: build row pool from `view.merges + view.rejections`, apply filter/sort/pagination controls
- [ ] T004 [US1] Add per-pair expander card in gallery: header with cluster IDs, dist, gates count, heuristic/decision badge
- [ ] T005 [US1] Inline `_render_pair_crops()` call inside each expander card
- [ ] T006 [US1] Add 4-gate badge row inside each expander (green=PASS, red=FAIL, with actual/threshold values)
- [ ] T007 [US1] Add Approve/Reject buttons per card (same session state logic as removed table)

## Phase 3 — Filter & Sort (P1)

- [ ] T008 [US2] Add filter selectbox (All / Merged / Rejected / Near Misses / Contested) with page reset on change
- [ ] T009 [US2] Add sort selectbox (Exemplar distance / Gates passed) with page reset on change

## Phase 4 — Pagination (P2)

- [ ] T010 [US3] Add pagination controls (Prev/Next buttons + "Page N of M" label, 10 per page)

## Phase 5 — Cleanup & Wiring

- [ ] T011 Replace the 4 old section calls in `_render_merge_analysis()` with single `_render_unified_merge_gallery()` call
- [ ] T012 Remove `_render_all_decisions_table()`, `_render_near_miss_gallery()`, `_render_merged_pairs_gallery()`, `_render_rejected_candidates_gallery()` functions
- [ ] T013 Update `docs/FEATURE_REQUESTS.md` to mark feature done and `CHANGES_LOG.md` entry
