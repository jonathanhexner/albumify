# Spec 093 — IQA Method Comparison

**Status:** Draft
**Author:** Jonathan Hexner
**Created:** 2026-06-29

## Relationship to spec-094 (Image Analysis Studio)

**093 is the `image_quality` backend that 094 consumes.** Direction:

- **093 PROVIDES → 094:** the quality scorers (`PyIQAQuality` + existing
  AVA/IQA/MUSIQ) and the generic `ScoreQualityStep`. 094's engine calls this
  step and normalizes each scalar score into an `AnalysisColumn`
  (`kind=numeric`, `sort_value=score`). 093's step contract
  (`context.method_scores: {path: {method: score}}`, `universal_cache`
  persistence) is the dependency 094 relies on — **the contract, not 093's
  completion** (094 degrades gracefully to the pre-existing AVA/IQA steps if
  the pyiqa scorers aren't in yet).
- **094 SUPERSEDES → 093's app:** the standalone comparison UI (Slice 3 below)
  is replaced by 094's generalized studio (`image_quality` tab). **Slice 3 is
  therefore descoped** — do not build it. 093 ships as a backend-only spec
  (Slices 1–2). The pyiqa install spike + opencv cleanup remain valid; they
  feed the scorers regardless of which app renders them.
- **No conflict:** both specs lock the same persistence (`universal_cache`,
  no results table).

Coherent build order: **093 Slices 1–2 → 094**.

## Problem

We want to evaluate which image-quality / aesthetic scorer best flags
low-quality photos (e.g. finger occlusion, blur). Reference examples live in
`examples/finger_occlusion/`. Today scorers are scattered (one pipeline step
each) and there is no way to run several over an album and compare them
side-by-side against our existing AVA and rule-based IQA scores.

## Goal

Given an album path, run a user-selected set of scoring methods over every
image and produce a table: one row per image (with thumbnail), one column per
method (score + within-album rank). Start with the Budapest dataset.

Methods in scope: **AVA** (existing), **IQA / rule_based** (existing),
**MANIQA, MUSIQ, HyperIQA, BRISQUE, NIQE, CLIP-IQA** (via `pyiqa`).

## Non-goals

- Training or fine-tuning any model.
- A "correct" ground-truth quality label / accuracy benchmark (future spec).
- Replacing the existing per-method steps (`score_iqa`, `score_ava`).

## Design

Reuse what exists; add three thin pieces.

```
Album path ─► DiscoverImagesStep ─► ScoreQualityStep(methods=[...]) ─► context.method_scores
                                            │                                  │
                                QualityMethodRegistry.create(name)    {img: {maniqa: .., niqe: ..}}
                                            │                                  │
                ┌────────────────────────────┴───────────────┐                ▼
          PyIQAQuality (NEW)                  existing scorers          Standalone app
          maniqa/hyperiqa/brisque/            rule_based, musiq,        (table + thumbs,
          niqe/clipiqa  — one class,          clip_aesthetic, ...       method checkboxes)
          normalizes via metric.lower_better
```

### 1. `PyIQAQuality` scorer (`sim_bench/quality_assessment/pyiqa_quality.py`)

- Subclass of `QualityAssessor`; one class, registered under each pyiqa metric
  name: `maniqa`, `hyperiqa`, `brisque`, `niqe`, `clipiqa`.
  (`musiq` already has a class wrapping pyiqa — leave it.)
- `__init__(metric_name, device)` lazily calls `pyiqa.create_metric(metric_name)`.
- `assess_image` returns a float; **normalize direction** so higher = better
  by reading `metric.lower_better` (flip BRISQUE / NIQE).
- `is_available()` returns `False` (not raise) when `pyiqa` is not importable,
  so the app/registry can grey the method out.

### 2. `ScoreQualityStep` (`sim_bench/pipeline/steps/score_quality.py`)

- Thin step per spec-053: reads `context.image_paths`, dispatches each method
  via `QualityMethodRegistry.create`, writes `context.method_scores`.
- Config: `{ "methods": ["maniqa", "niqe", ...] }`.
- `requires={"image_paths"}`, `produces={"method_scores"}`,
  `depends_on=["discover_images"]`.
- Cache per `(image, method)` using the `BaseStep` cache hooks
  (`feature_type=f"quality_{method}"`), so re-runs are incremental.
- `context.method_scores: Dict[str, Dict[str, float]]` keyed `path -> {method: score}`.
  Add the field to `PipelineContext`.

#### Persistence decision (locked 2026-06-29)

Two layers, do not conflate:
- **`PipelineContext.method_scores`** = transient in-run hand-off (step → app).
  Lives for one run only; mirrors the existing `iqa_scores` / `ava_scores`
  idiom. Not storage.
- **`universal_cache` DB** = the persistence layer, **reused as-is**. The
  `BaseStep` cache hooks write each `(image, feature_type=quality_<method>,
  model)` through to it, exactly like `score_ava` does today. Re-runs hit
  cache; no recompute.

**No new results table** for v1 (chosen over a dedicated SQL table /
"both"). The app builds its comparison table in memory from cache/context.
If cross-run comparison or CSV export becomes a requirement, add a results
table in a follow-up spec — it does not change the step or context contract.

### 3. Standalone comparison app (`app/iqa_compare/main.py`)

- Streamlit. Inputs: album path text box; checkboxes for each registered
  method (disabled when `is_available()` is False); "Run" button.
- On run: build a `PipelineContext`, execute `DiscoverImagesStep` +
  `ScoreQualityStep(methods=selected)` (NO scoring logic in the UI).
- Output: dataframe-style grid — thumbnail (real `st.button`/`st.image`
  grid per the v2 clickable rule, not `st.dataframe` row-select), file name,
  one column per method (score + rank), sortable by any method.
- Defaults: album = `D:\Budapest2025_Google`; AVA checkpoint =
  `models/album_app/ava_resnet50.pt`.

## Acceptance criteria

1. `pyiqa` installed; `QualityMethodRegistry.list_available()` shows
   maniqa/hyperiqa/brisque/niqe/clipiqa as available.
2. `ScoreQualityStep` runs over `examples/finger_occlusion/` (2 imgs) and over
   Budapest, producing a score per image per selected method; second run hits
   cache (no recompute).
3. App launches, lists methods with unavailable ones disabled, scores the
   selected album, and renders the thumbnail+score table.
4. Finger-occlusion examples score visibly lower than a clean Budapest portrait
   on at least one no-reference metric (sanity, not a hard threshold).
5. Unit tests green on Windows, ASCII-only; production-default config.

## Risks / open questions

- **pyiqa weight downloads**: first call to `create_metric` pulls weights from
  the network. Document in README; cache dir under `~/.sim_bench/` if pyiqa
  allows override.
- **pyiqa dependency weight** (torch/timm versions): verify no conflict with
  pinned `protobuf<4` / existing torch. Spike in Task 1 before committing.
- Score scales differ across metrics; the table shows raw normalized score +
  rank, not a cross-metric-comparable absolute. Acceptable for v1.

## Test plan

- `ut_PyIQAQuality`: `is_available` toggling; direction normalization (mock a
  `lower_better=True` metric → score flipped).
- `test_score_quality_step`: dispatch + cache hit on second run.
- E2E smoke: app run over `examples/finger_occlusion/` via the pipeline path.
