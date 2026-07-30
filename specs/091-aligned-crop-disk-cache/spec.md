# spec-091 — Cache aligned face crops to disk (stop re-decoding source photos every run)

**Created**: 2026-06-26 · **Status**: Draft · **Priority**: P1
**Source**: user — re-running a pipeline (e.g. to tune clustering params) re-decodes the
full-size source photos to make face crops, even though detections/embeddings/scores are
already cached. This also blocks the API and causes the Configure & Run status-poll
`ReadTimeout` (the spec-088 export step re-crops every face from source).

## Background — what's already cached vs not
**Already cached** in `universal_cache` (`sim_bench.db`), keyed by image + model + mtime, reused
across runs: `insightface_detection`, `person_detection`, `face_embedding`, `scene_embedding`,
and all scores (iqa/ava/expression/eyes/pose/smile/quality). So re-running with new clustering
params does NOT recompute embeddings.

**NOT cached — recomputed every run:**
- **aligned face crops** — the `align_faces` step decodes each source photo, warps each face to
  112×112, keeps it in memory (`record.aligned_face` / `context.aligned_faces`), never persisted.
- **export crops** — `face_cluster_analysis_export` re-opens each source photo *per face* and
  writes 112×112 JPEGs. The single most expensive avoidable work; it starves the API → timeouts.

## What we build
**One disk cache for aligned crops, three consumers.**

1. **Cache aligned crops** in `universal_cache` as a new `feature_type="aligned_face_crop"`
   (per image, a map `face_index → JPEG bytes`), keyed by image + mtime + an `align_version`
   tag. Same invalidation machinery as the other feature types (image change → miss;
   align-algorithm change → bump `align_version`).
2. **`align_faces` reuses it**: on a run, load cached crops into `record.aligned_face` /
   `context.aligned_faces`; only decode+warp on a miss, then save. (So `aligned_face` is also
   reliably present at the export step.)
3. **Export reuses it**: `_generate_crops_from_bboxes` saves `record.aligned_face` directly
   instead of re-opening the source photo; source-decode only as a last-resort fallback.
4. **(Phase 2, optional) People & Faces thumbnails** reuse the cached crop instead of cropping
   from the source at display time.

## AC
| # | Criterion | Verified |
|---|---|---|
| 1 | First run caches aligned crops; a second run (same images) hits the cache, no source decode in `align_faces` | unit/integration test (count `Image.open` calls) |
| 2 | Export writes crops from `aligned_face` with **zero** source-photo opens when crops are cached/in-memory | unit test (no `Image.open` on the cached path) |
| 3 | Changing the source image (mtime) invalidates its aligned-crop cache | unit test |
| 4 | Same crop bytes whether from cache or fresh compute | unit test |
| 5 | A pipeline re-run with only clustering-param changes does not re-decode source photos | manual / timing |

## Risks / notes
- **Invalidation coupling:** an aligned crop depends on the detection bbox/landmarks too — if
  detection output changes, alignment must too. Bump `align_version` whenever detection or the
  alignment algorithm changes (documented contract).
- **Storage:** 112×112 JPEGs are ~3–8 KB/face; storing per-image blobs in `universal_cache` is
  modest. Confirm DB growth is acceptable on a large album.
- This fixes the **root** of the status-poll `ReadTimeout` (export no longer re-decodes). A
  separate small mitigation — make the poll tolerate a timeout instead of crashing — is a
  follow-up (file as a sighting), not part of this spec.
- Phase 2 (People thumbnails) crosses into the API/display layer; ship Phase 1 (pipeline +
  export) first.
