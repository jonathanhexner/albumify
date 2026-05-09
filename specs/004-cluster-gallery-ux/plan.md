# Implementation Plan: Cluster Gallery UX

**Feature**: 004-cluster-gallery-ux
**Created**: 2026-04-14
**Status**: Approved

## Overview

Replace the dataframe + selectbox + button pattern in both Clusters (Base) and Clusters (Merged) tabs with a scrollable gallery. Each cluster row always shows thumbnails + key stats. An expand button reveals the full cluster detail inline.

## Design Decisions

### Streamlit Constraint: Thumbnails in Collapsed State
`st.expander` body content is always rendered in Python but only *visible* when expanded. FR-002 requires thumbnails visible **without** opening each row — i.e., they must always be visible in the "collapsed" state. This rules out `st.expander`.

**Solution**: Custom card layout using `st.columns` + `st.divider`. Each cluster gets a fixed-column row (always visible): info | 3 thumb slots | Details button. The button toggles a session_state boolean. When toggled on, the full `_render_cluster_detail()` is rendered directly below in full width.

### Lazy-Loading Detail View
`_render_cluster_detail` fires an async `ClusterView.compute` worker. Firing it for all N clusters simultaneously would be wasteful. Detail is loaded only when the user clicks "Details" for that cluster (session_state toggle gate).

### Shared Helper
A single `_render_cluster_gallery(overview, result, tab_key)` function serves both tabs. `tab_key` namespaces all session_state keys (e.g., `base_gallery_*` vs `merged_gallery_*`).

## Data Flow

```
RunOverview.cluster_rows       → row stats (cluster_id, size, diameter)
result.cluster_result.exemplars → Dict[cid, List[face_id]] → up to 3 face IDs
_crop_for_face(face_id, output_dir) → PIL Image → st.image(width=80)
session_state[f"{tab_key}_expanded_{cid}"] → bool → trigger _render_cluster_detail
_render_cluster_detail(result, cid, worker_key=f"{tab_key}_worker_{cid}") → ClusterView
```

## Components

### 1. `_render_cluster_gallery(overview, result, tab_key)` (new)
- Sort selectbox (size/diameter), keyed `{tab_key}_sort`
- For each cluster row (sorted):
  - Fixed 5-column row: `[3, 1, 1, 1, 1]` → info | 3 thumb slots | button
  - Info: "**Cluster {cid}** &nbsp; {size} faces &nbsp; diam `{diameter:.3f}`"
  - Thumb slots: load `_crop_for_face` for up to 3 exemplars; show caption "#{fid}" on miss
  - Button: "Details" / "Collapse" toggling `{tab_key}_expanded_{cid}`
  - If expanded: full-width `_render_cluster_detail(result, cid, worker_key=f"{tab_key}_worker_{cid}", debug_worker_key=None)`
  - `st.divider()` between rows

### 2. `render_run_overview_tab()` changes
- Remove: `st.dataframe(cdf)` + `st.selectbox("Select cluster to analyse")` + `st.button("Open Cluster Analysis")`
- Keep: all summary metrics, stage timings, Cluster Size Distribution chart, UMAP, Face Mapping Tables
- Add: `st.subheader("All Clusters")` → `_render_cluster_gallery(overview, result, tab_key="base_gallery")`

### 3. `render_merged_clusters_tab()` changes
- Remove: `st.dataframe(cdf)` + `st.selectbox` + `st.button("Open Merged Cluster Analysis")` + inline `_render_cluster_detail` at bottom
- Keep: all summary metrics
- Add: `_render_cluster_gallery(overview, merged_result, tab_key="merged_gallery")`
- Remove `selected_merged_cluster` logic (replaced by gallery toggle)

### 4. `_invalidate_run_caches()` changes
- Add loop to delete all `st.session_state` keys with prefix `"base_gallery_"` or `"merged_gallery_"`

## Constants (add near other `_GALLERY_*` constants)
```python
_CLUSTER_THUMB_SIZE = 80
_CLUSTER_SORT_OPTIONS = ["size", "diameter"]
```

## Edge Cases
- **No exemplars**: Thumbnail slots show caption "No exemplars" for that cluster
- **Crop file missing**: `_crop_for_face` returns None → show caption `#{fid}`
- **0 clusters**: `st.info("No clusters found.")` early return
- **100+ clusters**: All rows render collapsed (thumbnails loaded), which is acceptable per spec assumption

## No New Tests Required
- Gallery is a pure Streamlit rendering function; existing `test_streamlit_app.py` AppTest suite covers tab rendering
- No new algorithmic logic — only rearranges how existing data is displayed
- `_render_cluster_detail` and `_crop_for_face` already have coverage
