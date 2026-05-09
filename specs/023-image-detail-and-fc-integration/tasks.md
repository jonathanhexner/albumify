# spec-023 Tasks

## Part A: Image Detail Popup
- [x] A1: Create `bbox_overlay.py` — draw bboxes with resize-first approach
- [x] A2: Create `image_popup.py` — `@st.dialog` component
- [x] A3: Wire popup into `main.py` via `maybe_show_image_popup()`
- [x] A4: Add `image_detail_btn` to Results gallery (`gallery.py`)
- [x] A5: Add `image_detail_btn` to Explore tab rows
- [x] A6: Popup shows bounding boxes from `filter_scores[].bbox` (code done — needs new pipeline run)
- [x] A7: Popup shows StepDecision reasons (Decision tab reads `_popup_step_decisions`)
- [x] A8: Popup shows scene cluster peers (Scene tab reads `_popup_all_images`)
- [x] A9: Popup shows per-face scores (Faces tab with pose/eyes/expression chips)
- [ ] A10: Verify popup works with real click (Playwright can't trigger @st.dialog)

## Part B: Face Clustering App Integration
- [x] B1: Generate face crops during export (`_generate_crops_from_bboxes`)
- [x] B2: Write `crop_manifest.json`
- [x] B3: Fix merge_log serialization (`_numpy_safe` instead of `default=str`)
- [x] B4: Save base cluster_result before merging (`deepcopy`)
- [x] B5: Export base clusters as `clusters.csv` (pre-merge)
- [x] B6: Export merged clusters via `export_merged_results` (when merges occurred)
- [x] B7: Include merge stage in `pipeline_run.json` stages
- [ ] B8: Verify FC app Merge Analysis shows face thumbnails (needs run with relaxed thresholds that produce actual merges)
- [ ] B9: Verify FC app Clusters (Merged) shows merged state (same as B8)

## Part C: Bounding Boxes Everywhere
- [x] C1: People & Faces person detail — bbox on representative image
- [x] C2: `bbox_overlay.py` supports normalized + pixel coords
- [x] C3: Image popup — bbox drawn from filter_scores
- [x] C4: Explore Face Detection tab — images with face bboxes + per-face score chips (verified: 119 imgs, 80 detail btns)
- [ ] C5: Verify bboxes visible in all 4 locations with screenshots (needs new pipeline run for popup bbox data)
