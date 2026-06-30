# spec-095 — Tasks

> **Regenerated** from spec.md on 2026-06-29. The original `tasks.md` (when this
> was numbered 094) was lost from disk and not recoverable; this breakdown is
> derived from the spec's user stories + acceptance criteria, not the original.

## Slice 1 — Pure engine helpers (no Streamlit)
**BUILD CHANGE:** Slice 1 engine is NOT rebuilt — reuse spec-094's
`app.image_studio.engine` (discover_images, run_methods). 095 = the geo view layer.

- [x] T1.1 `discover_images` — reused from 094 engine.
- [x] T1.2 `run_models` — reused as 094's `run_methods` (geo family, universal_cache).
- [x] T1.3 `geoclip_accuracy` (haversine vs EXIF, EXIF-less skipped) in `geo_view.py`.
      [StreetCLIP city accuracy omitted — no coord/ground-truth; see REVIEW M2.]
- [x] T1.4 `tests/geo_vision/test_geo_view.py` — 7 tests (haversine, accuracy
      hit/miss, EXIF-less skip, map points, csv). All pass.

## Slice 2 — Streamlit app  ✅ DONE 2026-06-30
- [x] T2.1 `app/geo_vision/main.py`: sidebar (folder, limit, geo checkboxes by
      availability), Run, progress bar. Path bootstrap for Streamlit sys.path.
- [x] T2.2 Summary: #images, EXIF-GPS coverage, GeoCLIP within-Nkm accuracy.
- [x] T2.3 Per-image rows: thumbnail (HEIC via pillow_heif), EXIF, StreetCLIP/
      GeoCLIP top-3 confidence bars (labelled "relative, not accuracy"), BLIP caption.
- [x] T2.4 Edge cases: missing/empty folder warning; unavailable model greyed;
      cache best-effort. Verified live (Playwright) incl. empty-path case.

## Slice 3 — Export + map  ✅ DONE 2026-06-30
- [x] T3.1 CSV download (`csv_rows`).
- [x] T3.2 Map: EXIF (green) vs GeoCLIP#1 (orange) via `map_points` + `st.map`.

## Close-out
- [x] T4.1 Live Playwright run over the Budapest examples — summary + map +
      thumbnail rendered (screenshot verified). Full-model run = manual follow-up (REVIEW M1).
- [x] T4.2 README in `app/geo_vision/`; CHANGES_LOG entry.
- [x] T4.3 REVIEW.md written — PASS, no high-severity.
