# spec-095 — Tasks

> **Regenerated** from spec.md on 2026-06-29. The original `tasks.md` (when this
> was numbered 094) was lost from disk and not recoverable; this breakdown is
> derived from the spec's user stories + acceptance criteria, not the original.

## Slice 1 — Pure engine helpers (no Streamlit)
- [ ] T1.1 `app/geo_vision/engine.py`: `discover_images(folder, limit)` — filter
      by image ext, sort, cap; empty/missing dir → `[]` (AC1).
- [ ] T1.2 `run_models(paths, selected)` — call each `geo_cluster` helper's
      `.calc(Inputs)` only for checked models; unchecked absent from output (AC2).
      Reuse existing JSON disk caches; lazy model load (FR-2, FR-3).
- [ ] T1.3 `accuracy_summary(meta, sclip, gclip)` — hits/total per model,
      ignoring EXIF-less images (AC3).
- [ ] T1.4 Unit tests `ut_GeoVisionEngine` for T1.1–T1.3 (AC1–3, AC6 no-raise).

## Slice 2 — Streamlit app
- [ ] T2.1 `app/geo_vision/main.py`: sidebar (folder text, limit, model
      checkboxes), Run button, progress bar (US1).
- [ ] T2.2 Summary panel: #imgs, EXIF GPS/time coverage, home anchor,
      segmentation outcome.
- [ ] T2.3 Results table — one row/image: thumbnail (HEIC via `pillow_heif`),
      EXIF time/GPS, StreetCLIP top-3 + score bars, GeoCLIP top-3 + prob bars,
      BLIP caption, ✓/✗ vs EXIF (US2, US3, FR-4 honest-confidence label).
- [ ] T2.4 Edge cases: missing/empty folder, EXIF-less, corrupt image, unchecked
      model, model-download notice, limit-cap with skipped-count log (FR-5).

## Slice 3 — Export + map
- [ ] T3.1 CSV download with metadata mandate: source path, capture ts, run ts,
      model names+versions (AC7, FR-6, US4).
- [ ] T3.2 Map: EXIF (green) vs GeoCLIP#1 (orange) pins (US4).

## Close-out
- [ ] T4.1 Manual run + screenshot over Budapest (AC4, AC5).
- [ ] T4.2 README in `app/geo_vision/`; CHANGES_LOG entry.
- [ ] T4.3 Run `/code-review` → `specs/095-geo-vision-studio/REVIEW.md`; resolve
      high-severity findings; flip to Implemented.
