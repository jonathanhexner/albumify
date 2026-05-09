# spec-023: Image Detail View & Face Clustering App Integration

**Status**: In Progress
**Date**: 2026-05-02
**Sightings**: 039, 046, 052, 053, 055

## Problem

1. **No unified image detail view.** When clicking an image anywhere in the app, the user should see ALL information about that image in one place: bounding boxes, quality metrics, filter/selection decisions, cluster membership, face assignments. Currently fragmented or missing.

2. **Face Clustering App integration broken.** Exports from the main app are missing data the FC app needs: face crop images, merged cluster CSVs, base cluster exemplars. Face thumbnails don't appear in Merge Analysis.

3. **Performance.** Image detail must be instant — all data precomputed and cached, not fetched on demand.

## User Stories

1. As a user, I click any image in the app and immediately see:
   - The image with face bounding boxes drawn on it
   - Whether it was filtered out or selected, and WHY (decision reason from pipeline)
   - All quality metrics: IQA, AVA, sharpness, composite score
   - Which scene cluster it belongs to, and what other images are in that cluster
   - Which face(s) belong to which person(s), with face cluster info
   - Per-face scores: pose, eyes, expression, detection confidence

2. As a user, I open the Face Clustering App via the deep-link and see:
   - Face crop thumbnails in every tab (Clusters, Merge Analysis, Cluster Analysis)
   - Merge Analysis with face thumbnails for both clusters in each merge pair
   - Clusters (Merged) showing the post-merge state when merges occurred

3. As a user, the image detail popup opens instantly (<200ms) with no loading delay.

## Scope

### Part A: Image Detail Popup (main app)

**Where it appears** (every location that shows an image):
- Results gallery — click any image
- Explore tabs — click any thumbnail row
- People & Faces — click any image in person detail
- Scene Clustering — click any cluster image

**Content layout** (two columns):
- Left: image with face bounding boxes (green for selected person, gray for others)
- Right: all metadata organized in sections:
  - Quality Scores: IQA, AVA, sharpness, composite (with visual bars)
  - Status: selected/rejected + reason (from StepDecision)
  - Pipeline Decisions: filter_quality, select_best decisions for this image
  - Scene Cluster: cluster ID, cluster size, link to other images in cluster
  - Faces: per-face table with pose/eyes/expression scores, person assignment

**Data source**: `image_metrics` JSON in PipelineResult (already includes most fields). Face bbox data added via pipeline_service fix (bbox field in filter_scores).

**Performance**: All data comes from a single API call (`get_result` or `list_results`). No additional API calls when popup opens. Image thumbnail cached via `@st.cache_data`.

### Part B: Face Clustering App Integration

**Export must include**:
- `faces.csv` — base (pre-merge) cluster assignments
- `clusters.csv` — base cluster metadata with exemplars
- `faces_merged.csv` — post-merge cluster assignments (when merges occurred)
- `clusters_merged.csv` — post-merge cluster metadata
- `clusters_stage_base.csv` — snapshot of pre-merge clusters (for provenance)
- `crop_manifest.json` — maps face_id → crop file path
- `crops/` — 112x112 face crop JPEGs
- `merge_log.json` — with proper float types (not strings)
- `merge_metadata.json`
- `pipeline_run.json` — with `stages.merge.status: done` when merge ran

**Root cause of missing face thumbnails**: Export wrote post-merge `clusters.csv` as base. The FC app's `_exemplar_face_ids()` looks up cluster IDs from merge_log in base cluster exemplars. Post-merge cluster IDs don't match base IDs → empty exemplar lists → no face thumbnails.

**Fix**: Export base cluster result BEFORE merging, then export merged result separately via `export_merged_results()`.

### Part C: Bounding Boxes Everywhere

Bounding boxes must appear in:
1. Image detail popup (Part A) — ✓ code exists, needs bbox data in API response
2. People & Faces person detail — ✓ implemented
3. Explore → Face Detection tab — face crops with bboxes per row
4. Results gallery — optional overlay on multi-face images

**Technical**: `bbox_overlay.py` utility exists. Draws after resize (not before) so lines are visible. Supports both normalized (0-1) and pixel coordinates.

## Acceptance Criteria

1. Click any image in Results → popup shows with bounding boxes, all scores, selection reason
2. Click any image in Explore tabs → same popup
3. Click any image in People detail → same popup
4. Open FC app via deep-link → Clusters (Base) shows face crops
5. FC app Merge Analysis → face thumbnails visible for both clusters in each pair
6. FC app Clusters (Merged) → shows merged cluster state when merges occurred
7. Image detail popup opens in <200ms (all data precomputed)
8. Bounding boxes visible with clear colored rectangles around detected faces

## Implementation Status

| Task | Status | Notes |
|------|--------|-------|
| `bbox_overlay.py` utility | Done | Resize-first, normalized+pixel coords |
| People detail bbox | Done | Green rectangle visible |
| Face crop generation in export | Done | 411 crops, manifest populated |
| Merge log with proper types | Done | `_numpy_safe` serializer |
| Base vs merged cluster export | Done | `deepcopy` before merge, separate exports |
| `pipeline_run.json` stages | Done | Includes merge stage |
| Bbox data in API image_metrics | Done | `filter_scores[].bbox` field added |
| Image popup component | Partial | Component exists, not wired to all galleries |
| Popup with StepDecision data | Not done | Needs to read step_decisions for this image |
| Popup in Results gallery | Partial | Button exists but Streamlit dialog click issues |
| Popup in Explore tabs | Partial | Detail button exists |
| FC app face thumbnails in Merge Analysis | Blocked | Requires new run with base cluster export |
| Explore Face Detection tab with bboxes | Not done | |
