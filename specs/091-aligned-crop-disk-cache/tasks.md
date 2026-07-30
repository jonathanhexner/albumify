# spec-091 tasks

## Phase 1a — SHIPPED (export reuse + band-aid; fixes the timeout)
- [x] T3 `face_cluster_export`: reuse the in-memory `face.aligned_face` crop
      (`_crop_from_aligned`); `Image.open(source)` only as fallback (`_crop_from_source`).
      No more per-face source decode in the export.
- [x] Band-aid: `_pipeline_progress_fragment` catches `ApiError` → "still working, retrying"
      instead of crashing (SIGHTING-110).
- [x] Tests: `tests/pipeline/test_export_crop_reuse.py` (3, green).

## Phase 1b — DEFERRED (align_faces disk cache; clustering-adjacent, needs e2e gate)
- [ ] T1 `feature_type="aligned_face_crop"` + `ALIGN_VERSION` (in model_name).
- [ ] T2 `align_faces`: load/store aligned crops via the cache (LOSSLESS — npz/PNG — so
      embeddings are bit-identical); on hit, populate `aligned_faces`/`record.aligned_face`
      with no source decode/warp.
- [ ] T4 Tests: cache-hit avoids `Image.open` (AC1), mtime invalidation (AC3), cache==fresh
      bytes (AC4).
- [ ] GATE: run budapest e2e (`tests/face_clustering/e2e_budapest/`) — clusters/embeddings
      MUST be unchanged before shipping.
- [ ] T5 CHANGES_LOG; `/code-review`; flip status.

## Phase 2 — People & Faces thumbnails reuse (optional, later)
- [ ] T6 Serve the representative face's cached crop so `people_browser` stops cropping from
      source at display time.
