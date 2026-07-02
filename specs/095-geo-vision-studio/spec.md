# spec-095 — Geo-Vision Studio (standalone Streamlit app)

**Created**: 2026-06-29 · **Status**: MERGED INTO spec-094 (standalone app retired) · **Priority**: P2

> **Consolidation (2026-07-03, user-approved):** the standalone Geo-Vision Studio is
> **merged into spec-094 (Image Analysis Studio)** — one app, not two. Its unique parts
> (EXIF-vs-GeoCLIP **map**, GeoCLIP-within-Nkm **accuracy**, **confidence bars**) become
> the **Geo & Caption** tab of spec-094's Browse Run page. See
> `specs/094-image-analysis-studio/{ARCHITECTURE.html,MOCK.html}`.
>
> **Kept:** `app/geo_vision/geo_view.py` (pure helpers: haversine, geoclip_accuracy,
> map_points, csv_rows) + its tests — imported by the studio. **Dropped:**
> `app/geo_vision/main.py` (the standalone Streamlit app).
>
> Original build (2026-06-30): a thin UI on the spec-094 engine. Superseded by the merge.
**Source**: user — wants a standalone Streamlit app to point at any photo folder and run the
spec-022 vision/geo models (StreetCLIP, GeoCLIP, BLIP) + EXIF, and inspect what each gives
per image, including the model confidence.

> **History**: originally numbered 094; renumbered to 095 to deconflict with
> `094-image-analysis-studio`. The untracked original directory was lost from disk and this
> file was reconstructed from the captured spec text (its original `tasks.md` was not
> recoverable — the `tasks.md` here is regenerated from this spec).

## Relationship to spec-094 (Image Analysis Studio)
`094-image-analysis-studio` is the **superset** engine: it runs *any* per-image analysis
family (image_quality + geo_location_and_caption) through `universal_cache` with a generalized
column model and category tabs. This spec (095) is the **narrower, geo-focused inspection
app** — a thin UI over the existing `geo_cluster/` helpers and their JSON disk caches, with a
map and confidence bars tuned for judging CLIP geolocation. Decision for the user at review:
keep 095 as the focused geo tool, or fold its geo features into 094's
`geo_location_and_caption` family and retire 095. Default: build/keep both; revisit after 094
validates the generalized engine.

## Problem
The spec-022 vision models (`geo_cluster/streetclip.py`, `geoclip_locator.py`, `captioning.py`)
are real and registered, but the only UIs are the FastAPI `app/album_explorer/` (per-image
table, no confidence visualisation, no thumbnails-on-map) and `app/geo_explorer/` (map, EXIF
only — no CLIP/BLIP). There is no single place to **select a folder, choose which models to
run, and visually compare each model's location guess + confidence against EXIF ground truth.**

The exploration run (`BUDAPEST_EXPLORATION.html`) showed why this matters: CLIP geolocation is
often wrong (6/19 StreetCLIP, 5/19 GeoCLIP top-1) yet returns high confidence — a human needs
to eyeball the image next to the guess to judge it.

## What the models output (drives the UI)
Each helper exposes `calc(inputs) -> result`; per-image outputs:
| Model | Location | Confidence | Top-k |
|---|---|---|---|
| StreetCLIP | city string (from fixed list) | `score` 0–1 (softmax) | yes (3) |
| GeoCLIP | `(lat,lon)` + reverse-geocoded place | `prob` 0–1 | yes (3) |
| BLIP | scene caption (text) | none | no |
| EXIF | `(lat,lon)`, timestamp | n/a (truth) | no |

Confidence is **relative ranking, not accuracy** — the UI must label it honestly.

## What we build
A standalone Streamlit app `app/geo_vision/main.py` (run via `.venv/Scripts/streamlit run
app/geo_vision/main.py`). Thin UI over the existing `geo_cluster/` helpers — **no new domain
logic, no pipeline changes**. Reuses the existing JSON disk caches so re-runs are instant.

```
┌ Sidebar ──────────────┐   ┌ Main ───────────────────────────────────┐
│ folder path (text)    │   │ [Run] progress bar over images           │
│ image limit (number)  │   │ Summary: #imgs, EXIF GPS/time coverage,  │
│ models (checkboxes):  │   │   home anchor, segmentation outcome      │
│  [x] EXIF             │   │ Results table — one row per image:       │
│  [x] StreetCLIP       │   │   thumbnail | EXIF time/GPS | StreetCLIP │
│  [x] GeoCLIP          │   │   top-3+score bars | GeoCLIP top-3+prob  │
│  [x] BLIP             │   │   bars | BLIP caption | ✓/✗ vs EXIF      │
│ [Run]                 │   │ [Download CSV]   Map: EXIF (green) vs     │
└───────────────────────┘   │   GeoCLIP#1 (orange) pins                │
                            └──────────────────────────────────────────┘
```

## User Stories
### US1 — Run models on a folder (P1)
As a user I enter a folder path, pick which models to run, click Run, and see a per-image
table of each model's output. Progress is visible; cached images return instantly.

### US2 — Judge confidence at a glance (P1)
Each geolocation guess shows its top-3 with confidence as a bar, so I can see when the model
was unsure (spread) vs confident (one dominant guess) — and that high confidence ≠ correct.

### US3 — Compare CLIP vs ground truth (P2)
For images with EXIF GPS, a column shows whether StreetCLIP / GeoCLIP top-1 matched the real
city, plus an accuracy summary (e.g. "StreetCLIP 6/19"). EXIF-less images are flagged.

### US4 — Export + map (P3)
Download the full table as CSV (metadata mandate: source path, timestamp, model versions).
A map plots EXIF points vs GeoCLIP#1 points to visualise drift.

## Edge Cases
- Folder missing / empty / no images → friendly message, no crash.
- Image with no EXIF GPS or time → blank cells, never an exception (FR-011 parity).
- Model unchecked → its columns omitted, not run (saves CPU).
- First run downloads models (~600MB + ~1GB) — show a one-time "downloading models" notice.
- Corrupt / unreadable image → caption/preds empty, row still rendered (helpers already swallow).
- Large folder → `limit` caps work; app `log()`s how many were skipped (no silent truncation).
- HEIC/HEIF → handled (helpers register `pillow_heif`); thumbnail must too.

## Requirements (brief)
- **FR-1** UI is a thin translator: call `*.calc(Inputs)`; zero domain logic in the app.
- **FR-2** Reuse existing disk caches; do not re-infer cached images.
- **FR-3** Selected-models-only execution; models load lazily on first use.
- **FR-4** Confidence rendered as a 0–1 bar with the numeric value; labelled "model confidence
  (relative, not accuracy)".
- **FR-5** Never raise on missing/bad metadata or unreadable images.
- **FR-6** CSV export includes source path, capture timestamp, run timestamp, model names+versions.
- **FR-7** Windows-only paths via `.venv/Scripts/streamlit`; ASCII console output.

## Acceptance Criteria
| # | Criterion | Verified |
|---|---|---|
| 1 | `discover_images(folder, limit)` pure helper: filters by ext, sorts, caps; empty dir → `[]` | unit test |
| 2 | `run_models(paths, selected)` returns a per-image dict only for checked models; unchecked absent | unit test |
| 3 | `accuracy_summary(meta, sclip, gclip)` returns hits/total per model, ignoring EXIF-less images | unit test |
| 4 | App starts, folder→Run renders the table with thumbnails + confidence bars | manual + screenshot |
| 5 | Re-run on same folder uses cache (no model reload, instant) | manual |
| 6 | Empty/missing folder and EXIF-less images produce no exception | unit + manual |
| 7 | CSV download contains the metadata-mandate columns | code/manual |

## Out of scope (this spec)
- Any change to `geo_cluster/` helpers or pipeline steps (separate spec if quality fixes needed).
- Wiring CLIP into the segmentation axes (that's spec-022 step A′).
- The semantic axis / LLM theming.
- DB persistence (spec-022 Slice 2).
- Replacing `app/album_explorer/` — see relationship note.

## Relationship to existing apps
`app/album_explorer/` (FastAPI) already renders a per-image table but: no thumbnails, no
confidence bars, no GeoCLIP column, no map, no CSV. This Streamlit app is the **richer,
visual sibling** aimed at model inspection. Decision for the user at review: keep both, or
deprecate `album_explorer` once this lands. Default proposal: **keep both for now**, revisit
after this ships (album_explorer is a lightweight API; this is the human-facing studio).

## Notes
- Streamlit has no native folder picker → text input for the path (same as album_explorer).
- CPU inference; keep default `limit` small (12). Surface a per-image timing hint.
- Confidence honesty (US2 caveat) is a hard requirement, not polish — the Budapest data proves
  a confident-but-wrong UI would mislead.
