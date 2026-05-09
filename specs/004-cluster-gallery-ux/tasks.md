# Tasks: Cluster Gallery UX

**Feature**: 004-cluster-gallery-ux
**Created**: 2026-04-14

## Task List

- [x] Add `_CLUSTER_THUMB_SIZE` and `_CLUSTER_SORT_OPTIONS` constants
- [x] Implement `_render_cluster_gallery(overview, result, tab_key)` helper function
- [x] Update `_invalidate_run_caches()` to clear `base_gallery_*` and `merged_gallery_*` keys
- [x] Update `render_run_overview_tab()`: replace dataframe+selectbox+button with `_render_cluster_gallery`
- [x] Update `render_merged_clusters_tab()`: replace dataframe+selectbox+button+detail with `_render_cluster_gallery`
- [x] Update `specs/004-cluster-gallery-ux/spec.md` status to Implementation
- [x] Update `docs/FEATURE_REQUESTS.md` status to Done
- [x] Append entry to `CHANGES_LOG.md`
