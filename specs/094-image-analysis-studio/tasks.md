# spec-094 Tasks — Image Analysis Studio

**Status**: In Progress (Slices 1-4 done; Slice 5 = v2 restructure, not started)
**Scope**: (A) refactor geo/vision steps onto `universal_cache`; (B) generalized comparison
engine + studio app; (C) **v2 restructure** — Configure/Browse navbar, top-level Quality|Geo
toggle, `RunFolder` persistence, consolidate spec-095. See `spec.md`, `ARCHITECTURE.html`,
`MOCK.html`.

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

## Slice 2 — Generalized engine  ✅ DONE 2026-06-29

- [x] T2.1  `app/image_studio/__init__.py` (empty) + `engine.discover_images(folder, limit)`
      (ext filter, sort, cap; missing/empty → `[]`; logs skipped count). (AC3)
- [x] T2.2  `engine.AnalysisColumn` dataclass (key/category/kind/sort_value/display/topk) + mappers
      for both families: geo/caption (exif=coord, streetclip/geoclip=label_conf sorted by
      confidence, blip=text) and image_quality (iqa/sharpness/ava=numeric).
- [x] T2.3  `engine.run_methods(paths, selected, cache_handler=, config=, progress=)` →
      `{path:{key:AnalysisColumn}}`; runs only selected methods, dedups shared backing steps
      (iqa+sharpness → one `score_iqa` run), a failing method is logged+skipped (never fatal). (AC2)
- [x] T2.4  `engine.available_methods()` (key/category/label/available) + `engine.categories()`.
- [x] Tests: `tests/image_studio/test_engine.py` — 12 tests (discover filter/sort/cap/empty;
      all mappers incl. empty-preds; run_methods only-selected, shared-step-once, failing-skip;
      registry). **All pass.** Real-data smoke: engine over 4 Budapest imgs via `universal_cache`
      maps EXIF correctly, 2nd run consistent.

## Slice 3 — Studio app (tabs + flat + click + export)  ✅ DONE 2026-06-30

- [x] T3.1  `main.py` — sidebar: folder text, limit, method checkboxes grouped by category
      (unavailable greyed), Run; `st.progress` callback during run; one-time vision/AVA
      "downloading weights" notice. Repo-root `sys.path` bootstrap (Streamlit runs file directly).
- [x] T3.2  `view.py` — per-category `st.tabs`; clickable `st.button` thumbnail grid + sortable
      table (numeric-column sort selectbox + Desc). (AC4, FR-5)
- [x] T3.3  Flat **All** tab = union of columns across families. (AC4)
- [x] T3.4  Thumbnail click → enlarge image + all columns + top-k expanders. (AC5)
- [x] T3.5  Confidence bar + value, labelled "relative, not accuracy". (FR-7)
- [x] T3.6  `view.build_csv(...)` + `st.download_button` incl. metadata mandate
      (source_path, run_timestamp, spec_version, source_folder). (AC7)
- [x] T3.7  Empty/missing folder → warning; no methods → warning; corrupt image thumb → "(no
      preview)", never raises. (AC6)
- [x] T3.8  `app/image_studio/README.md` — run command, methods, caching, confidence caveat.
- [x] **Verified live (gate G.1)**: Playwright drove the real app over 12 Budapest images with
      EXIF+StreetCLIP+BLIP+IQA → category tabs, sortable table, confidence bars, 36 clickable
      thumbnails, click-to-enlarge with top-k all rendered with real values. Screenshots captured.

## Slice 4 — Reconcile with parallel work (spec-093 + spec-095)  ✅ DONE 2026-07-01

- [x] T4.a  **Wire spec-093 pyiqa metrics** into the engine's `image_quality` family: 6 metrics
      (maniqa/musiq/hyperiqa/brisque/niqe/clipiqa) backed by `ScoreQualityStep`, reading
      `ctx.method_scores[path][metric]`. All share `step_id="score_quality"`; `run_methods` now
      groups by step + merges per-method config (`methods` lists unioned) so maniqa+niqe = ONE
      score_quality run (a distinct step_id per metric would be wrong — `process` replaces
      `method_scores`). Added `_merge_configs`. Availability via `PyIQAModel.is_available`.
      Old `iqa`/`sharpness`/`ava` kept. 3 new tests (merge, single-run, registry); real
      brisque+niqe smoke over Budapest via universal_cache (scored together, 2nd run cached).
- [x] T4.b  **Overlap with spec-095 `app/geo_vision`** (user decision: keep both, document roles):
      cross-linked README "which to use" tables in both `app/image_studio/README.md` and
      `app/geo_vision/README.md`. image_studio = all-families studio; geo_vision = geo map/accuracy
      deep-dive. Both call `run_methods`; no duplicated scoring.

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

---

## Slice 5 — v2 restructure: Configure/Browse + Quality|Geo + RunFolder + consolidate 095  (NEW 2026-07-03)

Design refs: `ARCHITECTURE.html`, `MOCK.html`. User decisions (all confirmed):
navbar **Configure Run | Browse Run**; Browse has a **top-level `Quality | Geo` toggle** (drop the
"All" tab); runs persist as **sub-folders** (`RunFolder`); **merge spec-095** (map + accuracy) into
the Geo view and retire the standalone geo app.

### Design notes (HOW)
- **`RunFolder`** (`app/image_studio/run_folder.py`, PURE — no Streamlit; the ONLY new storage):
  - `run_id = "<YYYY-MM-DD_HHMMSS>_<methods-slug>"`. Timestamp is **passed in**, not generated inside
    (keeps it deterministic/testable).
  - `save(folder, run_id, methods, paths, columns)` → writes `<folder>/.studio_runs/<run_id>/`:
    - `run.json` — manifest `{run_id, source_folder, methods, images:[basename…], created_ts, spec_version, n_images}`
    - `columns.json` — full render payload `{path:{key:{kind,sort_value,display,topk}}}` so Browse
      renders with **no recompute**.
    - `results.csv` — export/human artifact (image × method: display+score) + metadata mandate.
  - `list(folder) -> [manifest]` — read every `.studio_runs/*/run.json`, newest-first; **skip malformed
    dirs, never raise**.
  - `load(folder, run_id) -> {paths, columns, methods, manifest}` — rebuild `AnalysisColumn`s from
    `columns.json`.
  - Filesystem-only. A future DB-backed variant implements the same 3 methods — **do NOT build now**.
- **Navbar** (`main.py`): `st.sidebar.radio(["Configure Run","Browse Run"])`; folder kept in
  `session_state` so both pages share it.
- **Configure Run**: today's sidebar (folder, limit, method checkboxes by family) + Run. On Run:
  `run_methods` → `RunFolder.save(...)` → success note "saved as `<run_id>` → open in Browse Run".
- **Browse Run**: `RunFolder.list(folder)` → run picker (left). On select → `RunFolder.load`. Top-level
  segmented **`Quality | Geo`**:
  - **Quality** → `view.render_quality(...)`: ranking table (sortable, score bars, rank col),
    image_quality columns only.
  - **Geo** → `view.render_geo(...)`: `geo_view.map_points` → `st.map` (EXIF green / GeoCLIP orange)
    + `geo_view.geoclip_accuracy` metrics + label_conf/caption table, geo columns only.
  - **No "All" tab.**
- **view.py**: add `render_quality` + `render_geo` (split of today's `render_results`); keep the
  clickable `st.button` thumbnail grid + click-to-enlarge; delete the All-tab path.
- **Consolidate 095**: import `app.geo_vision.geo_view` into the geo render. **Delete
  `app/geo_vision/main.py`** (keep `geo_view.py` + its tests). Both READMEs: geo_vision is now a helper
  module, not an app.

### Tasks  ✅ DONE 2026-07-03
- [x] T5.1  `app/image_studio/run_folder.py` — `save/list_runs/load` (pure; run.json + columns.json
      + results.csv; malformed dir skipped, never raises; paths re-anchored by basename on load).
- [x] T5.2  `tests/image_studio/test_run_folder.py` — 4 tests: round-trip (columns reconstructed
      incl. topk), newest-first, malformed-dir skipped, empty/missing → `[]`. All pass.
- [x] T5.3  `main.py` — top navbar (Configure Run / Browse Run) via `st.radio`; folder shared via
      `session_state`.
- [x] T5.4  Configure Run — Run → `run_methods` (cache_handler) → `RunFolder.save`; success note +
      "Open in Browse Run" jump.
- [x] T5.5  Browse Run — run picker (`list_runs`) + top-level `Quality | Geo` toggle.
- [x] T5.6  `view.render_quality` — ranking table, defaults to first metric ascending (**worst
      quality first**), score bars, clickable thumbnails + enlarge.
- [x] T5.7  `view.render_geo` — `st.map` (EXIF green / GeoCLIP orange via `geo_view.map_points`) +
      accuracy (`geo_view.geoclip_accuracy`) + label_conf/caption table.
- [x] T5.8  Removed the "All" tab; **deleted `app/geo_vision/main.py`** (kept `geo_view.py` + tests);
      updated geo_vision README (now a helper module).
- [x] T5.9  CHANGES_LOG entry.
- **Verified live (G5.1/G5.2)**: Playwright drove the real app over 8 Budapest images
      (exif+iqa+brisque+niqe) → RunFolder written → Browse Run re-opened it with **no recompute**,
      Quality ranking (worst-first, real thumbnails) + Geo map (EXIF pins) rendered. Screenshots captured.

### Gate (restructure → Implemented)
- [ ] G5.1  **Live Playwright** over `D:/Budapest2025_Google` (limit ~12): Configure a run
      (EXIF+StreetCLIP+GeoCLIP+IQA+MANIQA) → confirm `.studio_runs/<run_id>/` written → Browse Run:
      pick the run, toggle **Quality** (ranking table) then **Geo** (map + accuracy). Screenshots.
- [ ] G5.2  Re-open the saved run → renders with **NO recompute** (no model load; assert `run_methods`
      not called / no cache miss).
- [ ] G5.3  `/code-review` → update `REVIEW.md`; resolve High findings.
- [ ] G5.4  Docs: update `docs/architecture/` + `CLAUDE.md` Key Entry Points (studio app; drop
      `geo_vision` app if listed).
- [ ] G5.5  Flip spec-095 → Implemented-as-merged; confirm `app/geo_vision/main.py` deleted.
