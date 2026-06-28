# spec-094 Tasks — Image Analysis Studio

**Status**: In Progress
**Scope**: (A) refactor geo/vision steps onto `universal_cache`; (B) generalized comparison
engine + tabbed studio app. See `spec.md` for WHAT and the column model.

## Design notes (HOW)

- **Persistence idiom** (copy `sim_bench/pipeline/steps/score_ava.py`): a `BaseStep` overrides
  `_get_cache_config(context, config) -> {feature_type, model_name}`, `_process_uncached(items,
  context, config)`, `_serialize_for_cache(result, item) -> bytes`, `_deserialize_from_cache`.
  `BaseStep.execute` then routes through `UniversalCacheHandler` → `universal_cache` DB. This is
  the storage; no new table (per spec-093's locked decision).
- **feature_type per model**: `geo_streetclip`, `geo_geoclip`, `blip_caption`, `geo_exif`.
  Store the **full top-k + confidence** payload as JSON bytes.
- **Helper boundary**: keep `StreetCLIPLocator/GeoCLIPLocator/BlipCaptioner/GeoMetadataExtractor`
  framework-agnostic (notebook-usable). The step owns `universal_cache`; the helper computes a
  single item (or list) and no longer needs `_imcache.py`. Task 1 spike confirms the seam:
  either (a) step calls `helper.calc()` only for cache-misses, or (b) helper gains an optional
  injected cache. Prefer (a) — keeps helpers pure. Then delete `geo_cluster/_imcache.py`.
- **Column model** (`engine.py`): `AnalysisColumn(key, category, kind, sort_value, display, topk)`;
  `run_methods` returns `{path: {key: AnalysisColumn}}`. One mapper per method family turns raw
  step output → columns. Categories: `image_quality`, `geo_location_and_caption`.
- **GeoCLIP sort options**: `confidence` (prob) and `km_from_exif` (haversine to EXIF when present)
  — both numeric columns; pick default = confidence.
- **UI** (`view.py`): per-category `st.tabs`; each tab = clickable `st.button` thumbnail grid +
  a sortable table (sort control = selectbox of that category's numeric columns; v2 rule — no
  `st.dataframe` row-select). Flat **All** tab = union of columns. Click → enlarge + column list.

## Slice 1 — Storage refactor (architecture; unblocks "pluggable")  ✅ DONE 2026-06-29

- [x] T1.1  Spike → **option (a) chosen**: the step owns `universal_cache`; helpers compute the
      exact items handed to them (no internal cache). Keeps helpers pure/notebook-usable, no
      PipelineContext leak. (Option b — injecting a cache into the helper — rejected as it
      re-couples the helper to a caching concern.)
- [x] T1.2  `caption_images.py` → cache hooks, `feature_type="blip_caption"`,
      `model_version="blip-base-v1"`; stores caption string.
- [x] T1.3  `infer_geo_clip.py` → `feature_type="geo_streetclip"`; persists full top-k;
      `model_version="streetclip-v1-k{top_k}"` (top_k change invalidates).
- [x] T1.4  `infer_geo_coords.py` → `feature_type="geo_geoclip"`; persists full top-k;
      `model_version="geoclip-v1-k{top_k}"`.
- [x] T1.5  `extract_geo_metadata.py` → `feature_type="geo_exif"`; GeoMetadata JSON-serialized
      (timestamp→ISO); `model_version="exif-v1-min{min_year}"`.
- [x] T1.6  Deleted `geo_cluster/_imcache.py`; removed JsonCache from all three vision helpers
      (`calc()` now always computes). NOTE: `app/album_explorer` + `app/geo_explorer` call the
      helpers directly and so now recompute each run (no helper-level cache) — acceptable; they
      are experimental and superseded by the studio (Slice 3), which runs via the cached steps.
- [x] T1.7  `tests/geo/test_geo_steps_cache.py` — 5 tests: EXIF real round-trip + cache-hit
      (no recompute), StreetCLIP/GeoCLIP full-top-k persist + cache-hit, BLIP caption cache-hit,
      step works with no cache handler. **All pass; full geo suite 24 passed.** (AC1, AC8)

## Slice 2 — Generalized engine

- [ ] T2.1  `app/image_studio/__init__.py` (empty) + `engine.discover_images(folder, limit)`
      (ext filter, sort, cap; empty → `[]`; logs skipped). (AC3)
- [ ] T2.2  `engine.AnalysisColumn` dataclass + per-family mappers (quality, geo/caption).
- [ ] T2.3  `engine.run_methods(paths, selected) -> {path:{key:AnalysisColumn}}` — builds each
      step/helper's Inputs, calls `.calc()`/step, maps to columns; only selected methods. (AC2)
- [ ] T2.4  `engine.available_methods()` — registry of (key, category, is_available) for checkboxes.

## Slice 3 — Studio app (tabs + flat + click + export)

- [ ] T3.1  `main.py` — sidebar: folder text, limit, method checkboxes grouped by category,
      Run button; `st.progress` during run; one-time "downloading weights" notice per family.
- [ ] T3.2  `view.py` — per-category `st.tabs`; clickable `st.button` thumbnail grid + sortable
      table (numeric-column sort selectbox). (AC4, FR-5)
- [ ] T3.3  Flat **All** tab = union of columns across families. (AC4)
- [ ] T3.4  Thumbnail click → enlarge image + list all its columns/top-k. (AC5)
- [ ] T3.5  Confidence rendered as bar + value, labelled "relative, not accuracy". (FR-7)
- [ ] T3.6  `view.build_csv(...)` + `st.download_button` incl. metadata mandate. (AC7)
- [ ] T3.7  Empty/missing folder + EXIF-less/corrupt images → friendly, no exception. (AC6)
- [ ] T3.8  `app/image_studio/README.md` — run command, weights/caching, CPU caveat.

## Tests

- [ ] T4.1  `tests/geo/test_geo_steps_cache.py` — each refactored step caches via
      `universal_cache` (monkeypatch handler / temp DB); 2nd call no recompute; full top-k round-trips.
- [ ] T4.2  `tests/.../test_image_studio_engine.py` — `ut_DiscoverImages`, `ut_RunMethods`
      (only-selected, correct kind/sort_value; helpers/steps monkeypatched — no model load),
      `ut_AnalysisColumn` mappers.
- [ ] T4.3  Tests on Windows, ASCII-only, production-default config, no network/model download.

## Gate (before flipping spec → Implemented)

- [ ] G.1  Run studio on `D:/Budapest2025_Google` (limit 24); screenshot tabs + sortable tables + click.
- [ ] G.2  `/code-review` → `REVIEW.md`; resolve High findings.
- [ ] G.3  Docs mandate: update `docs/architecture/{db_schemas,classes,data_flow}.html` for the new
      `feature_type`s + context fields + `app/image_studio/`; add app to `CLAUDE.md` Key Entry Points.
- [ ] G.4  v2 Budapest E2E gate — **N/A** (no v2 face-clustering tab touched); record exemption in REVIEW.md.
- [ ] G.5  `CHANGES_LOG.md` entries per change.
- [ ] G.6  Confirm no new deps (transformers/torch/geoclip/reverse_geocoder/pillow_heif/pyiqa already
      installed — pyiqa per spec-093 Task 1).
- [ ] G.7  Coordinate with spec-093: its IQA scorers register as the `image_quality` family; its
      standalone app deprecated in favor of this studio (note in 093).
```
