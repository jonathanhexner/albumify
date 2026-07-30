# spec-082 — Clickable thumbnails (Images + Gallery)

**Status:** Implemented
**Date:** 2026-06-05

## Problem (user, verbatim)
> "The Images page is crap. IQA/Composite are None, I can't click on an image.
> how many times did I ask that thumbnails be clickable, opening either face or
> image analysis?!?!!!"

Three concrete defects on the v2 **Images** page, plus a parallel gap on the
**Gallery**:

1. **Wall of None.** Images table offered Composite / IQA / AVA / Sharpness
   columns — all `None` for face-clustering runs (those are Albumify
   image-scoring outputs, never computed here).
2. **No thumbnails / not clickable.** The Images table was text-only; you
   could not click an image to open its analysis.
3. **Cold-start blank.** First render resized 122 large source photos inline
   (~12 s) with no feedback → the table looked broken (blank).
4. **Gallery faces not clickable.** Gallery drew faces with bare `st.image`;
   the only drill-in was the cluster-level "Open in Cluster Analysis" button.
   Per-face → Face Analysis existed only in the Face Metrics table.

## Goal
"Thumbnails are clickable, opening either face or image analysis" — everywhere
a thumbnail appears:

| Thumbnail location | Click target            | Status |
|--------------------|-------------------------|--------|
| Images table row   | **Image Analysis**      | spec-082 (new) |
| Gallery face       | **Face Analysis**       | spec-082 (new) |
| Face Metrics row   | Face Analysis           | spec-080 (already) |

## Changes
- `face_cluster/views/image_metrics.py` — new `populated_columns(rows)`:
  drops every column that is `None` across all rows. Streamlit-free, unit-
  tested. (Service decides which columns have data; tab just renders.)
- `app/face_clustering_v2/tabs/images_tab.py` —
  - default + offered columns come from `populated_columns` (no more None wall);
  - `st.column_config.ImageColumn` thumbnails (110 px, EXIF-corrected, base64);
  - `_encode_thumb` + `ThreadPoolExecutor(max_workers=8)` cold-start **12 s → 3.3 s**
    (PIL releases the GIL during JPEG decode/encode); wrapped in `st.spinner`;
  - row-select → `render_image_analysis(image_detail(path))`.
- `app/face_clustering_v2/components/cluster_strip.py` — each Gallery face gets
  an "Open" button → `selected_face_id` + `navigate_to("Face Analysis")` (same
  drill-in the Face Metrics rows use).

## Out of scope
- Disqualify-from-gallery action (spec-069).
- Albumify image scores for face runs (they are genuinely absent; we hide the
  columns rather than fabricate them).

## Acceptance
- Images table shows only populated columns (no None wall). ✓ (screenshot)
- Images thumbnails render; clicking a row opens Image Analysis. ✓ (screenshot:
  `SHOT_image_analysis.png` — "20250822_112331.jpg — 2 faces · 1 clustered · 1
  noise · 0 filtered").
- Gallery face button → `selected_face_id` set + view switches to Face
  Analysis. ✓ (AppTest: fid=6 → active_page=Face Analysis, 0 exc).
- `populated_columns` unit-tested. ✓ (4 passed).
- Both pages AppTest 0 exceptions. ✓

## E2E
Canvas-rendered `st.dataframe` row-pick is not Playwright-addressable
(SIGHTING-091) — Images row-click verified by real-browser **screenshot** +
AppTest, consistent with how Face Metrics / History drill-in are validated.

Budapest baseline gate (affected tabs Gallery + Merged Clusters): **g, e, i
green**. Scenario I had counted crop `<img>`s before Streamlit's async media
endpoint loaded them (naturalWidth==0 → "not visible"); fixed with a
`wait_for_function(naturalWidth>0)` — the same single-page-render timing class
spec-080 fixed for H's plotly mount. Images has no dedicated scenario yet
(canvas thumbnails un-addressable); the table + thumbnail render is screenshot-
verified. **Follow-up:** add a real-DOM Images scenario (assert None columns
absent + ≥1 thumbnail loads).
