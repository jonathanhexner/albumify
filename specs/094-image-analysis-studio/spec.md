# spec-094 — Image Analysis Studio (generalized per-image multi-method comparison)

**Created**: 2026-06-29 · **Status**: Draft · **Priority**: P2
**Source**: user — wants one place to point at a folder, run many per-image analyses
(image quality, geo-location, caption, …), and inspect every model's output in a
clickable-thumbnail + sortable table, grouped into category tabs.

## Problem
We have several per-image analyses scattered across the pipeline — IQA/AVA quality scorers,
and the spec-022 vision models (StreetCLIP, GeoCLIP, BLIP, EXIF). Two issues:

1. **Inconsistent storage.** Quality steps persist through `universal_cache` (BaseStep cache
   hooks). The geo/vision steps **bypass it** — they write side JSON files via
   `geo_cluster/_imcache.py`, which self-admits it's a shortcut ("Production would use the
   pipeline's universal_cache; this keeps the standalone experiment fast without that wiring").
   So they are pipeline steps in *shape* but not truly pluggable/persisted.
2. **No unified comparison view.** spec-093 proposed an IQA-only comparison app. There is no
   single, generalized tool to run *any* selected analyses over an album and compare outputs
   per image — across families with different output shapes (scalar vs label+confidence vs text).

## Goal
A **generalized image-analysis run + comparison studio**:
- Pick a folder + a set of analysis methods (checkboxes), run them.
- Every method persists through `universal_cache` (full top-k + confidence), like AVA/IQA.
- View results in a clickable-thumbnail, sortable table, **grouped into category tabs**
  (`image_quality`, `geo_location_and_caption`, …), plus a flat **All** tab.

## Relationship to spec-093 (locked)
spec-093 (IQA Method Comparison, Draft) becomes the **`image_quality` family** inside this
engine. This spec **supersedes 093's standalone app** with the generalized studio; 093 keeps
ownership of the quality *scorers* + `ScoreQualityStep`. The engine/UI here consume any
family. (If 093 already shipped its app, we deprecate that app, not its step.)

**Dependency (explicit):** 094's `image_quality` family **DEPENDS ON 093 Slice 1–2** — the
`PyIQAQuality` scorers + the generic `ScoreQualityStep` and its contract
(`context.method_scores: {path: {method: score}}`, `universal_cache` persistence). 094's
`engine.run_methods` calls `ScoreQualityStep` and normalizes each scalar into an
`AnalysisColumn` (`kind=numeric`, `sort_value=score`).

This is a dependency on the **step contract, not on 093's completion**: 094 degrades
gracefully — the geo/caption families and the pre-existing `score_ava` / `score_iqa` /
`musiq` steps work with zero 093 work; only the pyiqa metrics (maniqa/niqe/brisque/
hyperiqa/clipiqa) are absent until 093 lands. **Coherent build order: 093 Slices 1–2 → 094.**
093 Slice 3 (its standalone app) is descoped *because* of this spec — do not build it.

## Core concept — the column model
A run produces, per image, a set of **AnalysisColumns**, each tagged with a category. A column
declares its display type and what to sort by — that single abstraction makes "flat vs tabs"
trivial and lets new families drop in without touching the table.

```
AnalysisColumn:
  key        e.g. "maniqa", "streetclip", "blip"
  category   "image_quality" | "geo_location_and_caption" | ...
  kind       numeric | label_conf | text | coord
  sort_value float | None      # what the column sorts by
  display    str (label / caption / "47.5,19.0") ; for label_conf: top-1 + confidence
  topk       list (full top-k + confidence)  # shown on expand / hover

per image -> {column.key: AnalysisColumn}
```

| Family | Method | kind | sort_value | display | topk stored |
|---|---|---|---|---|---|
| image_quality | maniqa/niqe/brisque/ava/… | numeric | the score | score + rank | n/a |
| geo_location_and_caption | StreetCLIP | label_conf | confidence | "Budapest, HU" 0.87 | 3 cities+scores |
| geo_location_and_caption | GeoCLIP | label_conf / coord | confidence (or km-from-EXIF) | place + (lat,lon) | 3 coords+probs |
| geo_location_and_caption | BLIP | text | — | caption | — |
| geo_location_and_caption | EXIF | coord | — | (lat,lon), time | — |

## What we build
```
app/image_studio/
  main.py     sidebar (folder, limit, method checkboxes by category) + Run
  engine.py   PURE: discover_images; run_methods(paths, selected) -> {path:{key:AnalysisColumn}}
              (builds *Inputs, calls each step/helper .calc(), normalizes to columns)
  view.py     thumbnail st.button grid + sortable table per category; tabs; flat "All"; CSV
  README.md
```
Plus the **architecture fix** — geo/vision steps onto `universal_cache`:
```
caption_images / infer_geo_clip / infer_geo_coords / extract_geo_metadata
  → override _get_cache_config (feature_type per model), _process_uncached,
    _serialize/_deserialize_for_cache   (same idiom as score_ava.py)
  → delete geo_cluster/_imcache.py usage; helpers take an optional cache or
    the step owns caching (decide in Task 1 spike — keep helpers notebook-usable)
```

## User Stories
### US1 — One run, many analyses (P1)
Pick folder + check methods across families, click Run; each method persists to
`universal_cache`; a second run is incremental (cache hits, no recompute).

### US2 — Category tabs + flat view (P1)
Results show as tabs per category (`image_quality`, `geo_location_and_caption`), each a
clickable-thumbnail + sortable table; an **All** tab shows every column flat.

### US3 — Sort + inspect confidence/top-k (P1)
Sort any table by any numeric column (score, confidence, km-from-EXIF). Expanding a row shows
the full top-k with confidence. Confidence labelled "relative, not accuracy".

### US4 — Click a thumbnail (P2)
Clicking a thumbnail (real `st.button` grid, never `st.dataframe` row-select) opens that
image large with all its columns listed.

### US5 — Pluggable into Albumify/FC (P1, architecture)
Because every method is a `universal_cache`-backed step, any of them drops into the Albumify
or FC `default_pipeline` unchanged and its output survives the run / is queryable.

### US6 — Export (P3)
CSV of the current view incl. metadata mandate (source path, capture ts, run ts, model+version).

## Edge Cases
- Missing/empty folder, EXIF-less image, corrupt image → no exception, blank cells.
- Method unchecked / unavailable (`is_available()` False) → omitted/greyed, not run.
- Text/coord columns are non-numeric → excluded from sort, still displayed.
- First run downloads model weights → one-time notice per family.
- HEIC/HEIF thumbnails handled (`pillow_heif`).
- Large folder → `limit` caps; log skipped count (no silent truncation).

## Requirements (brief)
- **FR-1** Every analysis method persists through `universal_cache` with a distinct
  `feature_type`; **full top-k + confidence** stored (not just top-1).
- **FR-2** Geo/vision steps refactored off `_imcache.py` onto BaseStep cache hooks; helpers
  stay notebook-usable (no PipelineContext dependency).
- **FR-3** UI is a thin translator: `engine.py` calls steps/helpers; zero model logic in UI.
- **FR-4** Column model drives both tabbed and flat rendering from one data structure.
- **FR-5** Clickable thumbnails via `st.button` grid (v2 rule), never `st.dataframe` select.
- **FR-6** Never raise on missing/bad metadata or unreadable images.
- **FR-7** Confidence shown with an honest "relative, not accuracy" label.
- **FR-8** Windows paths via `.venv/Scripts/streamlit`; ASCII console output.

## Acceptance Criteria
| # | Criterion | Verified |
|---|---|---|
| 1 | Refactored geo/vision step: 2nd run over Budapest hits `universal_cache` (no recompute), full top-k persisted | unit + manual |
| 2 | `engine.run_methods(paths, selected)` returns only selected methods as `AnalysisColumn`s with correct kind/sort_value | unit |
| 3 | `engine.discover_images` filters/sorts/caps; empty dir → `[]` | unit |
| 4 | App renders category tabs + All tab; each table sortable by numeric columns | manual + screenshot |
| 5 | Clicking a thumbnail opens it large with all columns | manual |
| 6 | EXIF-less / corrupt / empty-folder → no exception | unit + manual |
| 7 | CSV export contains metadata-mandate columns | code/manual |
| 8 | A refactored geo step added to a test `default_pipeline` produces its output in `PipelineContext` end-to-end | integration |

## Out of scope (this spec)
- New scoring models beyond those listed (geo: 022; quality: 093).
- Wiring CLIP into the segmentation axes (spec-022 step A′).
- Semantic axis / LLM theming; DB *results-table* for cross-run comparison (follow-up if needed —
  `universal_cache` is the persistence here, per 093's locked decision).
- Albumify Results-page integration (separate spec once the studio validates the families).

## Notes
- Streamlit has no native folder picker → text input for the path.
- CPU inference; default `limit` small (12); surface a timing hint.
- Reuse, don't reinvent: same `universal_cache` + BaseStep cache idiom as `score_ava.py`;
  same clickable-thumbnail rule as the v2 face tabs.
