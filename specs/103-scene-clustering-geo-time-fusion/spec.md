# Spec 103: Scene clustering — time-backbone + optional-geo fusion (graceful degradation)

**Status**: In Progress · 2026-07-22 · step wired into the pipeline (gated off); labels (T1) + ARI (T4) still open
**Design update (2026-07-22 — supersedes D3)**: exploration on Budapest/Austria/Germany showed the D3
**additive** blend `w_v·cos + w_t·time + w_g·geo` **over-merges** — its symmetric time term pushes
visually-distinct shots a few minutes apart together. The shipped builder is **Path A** (`SceneDistanceBuilder`):
`d = visual_cos · (1 − boost·exp(−dt_sec/τ))` — a **one-sided short-range time boost** that only ever pulls
near-simultaneous photos together (τ≈60 s), never separates distant ones, so past ~2-3 min looks decide.
Validated to fix both the near-duplicate rescue and the over-merge, with **no chaining** (it feeds the
production HDBSCAN, not a new clusterer). Geo is unused in Path A (visual+time only); the additive
`SceneDistanceFuser` is retained for the sweep. Report: `reports/2026-07-21_scene_clustering_fusion/`.

**Status (original)**: Draft — awaiting approval · 2026-07-20
**Type**: Experiment **that lands in the pipeline** — the winning method replaces/augments the
`cluster_scenes` step. Not a throwaway study: the deliverable is a production pipeline step (spec-053
shape) + docs, gated behind a config flag until the experiment proves it beats today's visual-only.
**Depends on**: `extract_geo_metadata` → `context.geo_metadata` (EXIF GPS + capture time, already in
the pipeline), `geo_cluster.*` (spec-022 segmentation/selector), `cluster_scenes` (today's HDBSCAN),
`extract_scene_embedding` (DINOv2). Related prior art: `geo_temporal_segment` (spec-022) already
segments by a best-scoring axis but is **not wired into scene clustering** — this spec connects them.

## The problem (grounded in the code + real data)

Today `cluster_scenes` is **purely visual**: DINOv2 global embedding → HDBSCAN (cosine,
`min_cluster_size=2`). Capture time and GPS are extracted one step earlier (`extract_geo_metadata`)
and **thrown away**. Vision-only has to rediscover from pixels what a timestamp states for free, and
makes the classic errors: two similar-looking landmarks merge; one moment splits when the light
changes; whatever doesn't group is dumped to the HDBSCAN noise bucket (`-1`).

**But geo is not reliably present.** Probing the real source photos (working set was EXIF-stripped by
downsampling, so this reads originals):

| Trip | GPS present | Capture time (EXIF + `YYYYMMDD_HHMMSS` filename fallback) |
|---|---|---|
| Budapest | 34/77 (**44%**) | ~100% |
| Austria | 46/80 (**58%**) | 100% |
| Germany | **3/80 (4%)** | ~100% |

Two facts drive the whole design:
1. **Geo is wildly inconsistent (44% / 58% / 4%) and mixed *within* a trip** — some photos geotagged,
   some not. Never all-or-nothing.
2. **Time is the reliable backbone** — recoverable ~always from the filename even when EXIF is absent.

## Goal

A scene-clustering method that uses a **hierarchy of priors** and degrades gracefully:

```
TIME    (≈always present)   ──►  backbone: segment the trip into temporal events
 + GEO   (per-photo, optional) ─►  opportunistic refine/split within a segment
 + VISUAL (always present)  ──►  refine within; the floor that never drops out
```

Time is the spine, **geo is a per-photo optional refinement**, visual is the floor. A photo with no
GPS simply skips the geo refinement — no crash, and **no imputation**. When a whole trip is
geo-poor (Germany), the method must **automatically reduce to time+visual and equal-or-beat today's
visual-only**. That degradation is a *binding acceptance gate*, not a hope — Germany is the built-in
stress test.

## Design decisions (defaults — for approval)

- **D1 Priors** = time backbone + optional geo refinement + visual floor (above). NOT a flat feature
  concat: geo enters only where present, per photo.
- **D2 No imputation (binding).** Missing geo = "no geo constraint" for that photo, never a value.
  Imputing to `(0,0)` or the trip-mean would falsely glue all no-geo photos into one cluster — the
  bug this spec exists to avoid. Same for missing time (fall back to filename; if truly absent,
  that photo is time-unconstrained, not time-zero).
- **D3 Distance, not features — and its OWN step (Option B, user-chosen).** The fusion is a dedicated
  pipeline step `build_scene_distance` that emits a **precomputed pairwise distance matrix**
  `context.scene_distance`; `cluster_scenes` then just clusters that matrix. Each concern stays a
  separate, inspectable step: `extract_scene_embedding` (visual) → `build_scene_distance`
  (fuse visual+time+geo) → `cluster_scenes` (cluster). The distance:
  `d(i,j) = w_v·visual_cosine + w_t·time_gap + w_g·geo_haversine`, where the time and geo terms are
  **dropped from the blend (weights renormalized) for any pair missing that signal** — this is what
  makes degradation automatic and per-pair.
- **D4 Two candidate structures**, both evaluated (both consume the D3 step's output):
  - **(a) One-pass** — `cluster_scenes` clusters the fused distance directly.
  - **(b) Two-stage** — time/geo segment first (reuse spec-022 `geo_temporal_segment`) → visual-cluster
    *within* each segment. More faithful to "a scene lives inside a day/place."
- **D5 Reference truth (binding).** Hand-labeled scene grouping on **Budapest + Germany** (the
  geo-rich and the geo-poor extremes). Without it we can only report *differences* — the exact trap
  EXP-1 fell into. A tiny labeling tool (group-the-thumbnails HTML) produces `scene_labels.json`.
- **D6 Metric** = Adjusted Rand Index (ARI) + homogeneity/completeness vs the reference, **stratified
  by geo availability** so a win that only holds on geo-rich photos is visible as such.
- **D7 Ship gated — by step presence, not a flag.** The `build_scene_distance` step is simply **not in
  `default_pipeline`** until the experiment clears the gates. When absent, `cluster_scenes` sees no
  `scene_distance` and clusters the DINOv2 embeddings exactly as today → default behavior is
  byte-identical (Budapest E2E `n_clusters` unchanged), with zero conditional logic to reason about.
  The experiment/opt-in pipeline inserts `build_scene_distance` before `cluster_scenes`.
- **D8 VLM arm = out of scope here.** "VLM proposes/judges scene groups" is a natural follow-up
  (ties to spec-099's overseer idea) but is a separate spec; this one settles the cheap metadata-fusion
  question first.

## Method — the steps (Option B decomposition)

```
extract_scene_embedding ─► scene_embeddings ┐
extract_geo_metadata ────► geo_metadata ─────┼─► build_scene_distance ─► scene_distance ─► cluster_scenes ─► scene_clusters
   (time + geo, upstream)                     ┘        (NEW step)         (precomputed NxN)   (clusters it)
```

**New domain helper** `sim_bench/scene_cluster/geo_time_fusion.py` (spec-053, framework-agnostic):
- `__init__(config)` — weights `w_v/w_t/w_g`, time/geo scales.
- typed `Inputs`: `embeddings` (dict), `geo_metadata` (dict of `GeoMetadata`, some entries
  geo-less/time-less), `image_ids`.
- `calc(inputs) -> SceneDistanceResult` = `{distance_matrix (NxN), image_ids, per_image_signal_used}`.
  Builds the D3 distance with per-pair signal-dropping. `per_image_signal_used` records which priors
  actually applied to each photo — needed for the D6 geo-stratified report. **Returns a distance, not
  clusters** — clustering stays `cluster_scenes`' job.

**New step** `build_scene_distance` (thin translator): reads `context.scene_embeddings` +
`context.geo_metadata`, calls `helper.calc(...)`, writes `context.scene_distance` (new context field)
+ `context.scene_distance_signal` (the stratification map).

**`cluster_scenes` change (minimal):** when `context.scene_distance` is present, cluster it with
`metric="precomputed"`; otherwise cluster the embeddings exactly as today. Output contract
(`scene_clusters`/`scene_cluster_labels`) is **unchanged**, so downstream `select_best`/EXP-2 need no
change.

## Experiment arms (all scored vs the D5 reference)

| Arm | Signals | Note |
|---|---|---|
| A0 baseline | visual only | today's `cluster_scenes` (the bar to beat) |
| A1 | time + visual | no geo at all — the Germany-safe path |
| A2 | time + geo + visual | full fusion (one-pass, D4a) |
| A3 | two-stage | segment (spec-022) → visual within (D4b) |

Report ARI per arm per trip, plus the **geo-absent stratum** of A2/A3 (must not fall below A0/A1).

## Acceptance criteria

| # | Gate | Threshold |
|---|---|---|
| A1 | **Graceful degradation (binding)**: on geo-absent photos, and on the geo-poor trip (Germany), the fused method's ARI **≥ visual-only baseline** | binding — a regression here fails the spec |
| A2 | **No imputation**: unit test proves missing geo/time never becomes a value; a pair missing a signal drops that term (weights renormalize) | pass |
| A3 | Reference labels exist for Budapest + Germany (`scene_labels.json`), produced by the labeling tool | exists |
| A4 | ARI / homogeneity / completeness reported per arm per trip, **stratified by geo availability** | reported |
| A5 | Winner ships as the `build_scene_distance` step; **when it is absent from the pipeline**, `cluster_scenes` is byte-identical to today (Budapest E2E `n_clusters` unchanged) | binding |
| A6 | spec-053 shape: `build_scene_distance` is a thin step file, `calc()` helper returns a distance (typed `Inputs`/`Result`); domain logic in `scene_cluster/`, not the step. `cluster_scenes` change is the minimal precomputed-metric branch only | pass |
| A7 | Docs updated: `docs/architecture/data_flow`, `classes`, `configs/pipeline.yaml` comments, overview if the step contract shifts | pass |
| A8 | `reports/<date>_scene_clustering_fusion/` HTML with inline before/after cluster galleries per trip + the metric tables | exists |
| A9 | REVIEW.md via `/code-review`; no High findings open | pass |

## Pipeline architecture integration (Option B — dedicated step)

- **New step `build_scene_distance`** sits between `extract_scene_embedding` and `cluster_scenes`.
  `requires={"scene_embeddings", "geo_metadata"}`, `produces={"scene_distance", "scene_distance_signal"}`,
  `depends_on=["extract_scene_embedding", "extract_geo_metadata"]`. It tolerates empty geo (degrades
  to visual) — the builder must resolve it even when `geo_metadata` is sparse.
- **New context fields**: `scene_distance` (NxN precomputed distance, aligned to an `image_ids`
  order it also stores) and `scene_distance_signal` (per-image priors-used map for stratification).
- **`cluster_scenes`**: one added branch — if `context.scene_distance` is set, cluster it as
  `metric="precomputed"`; else unchanged. Its `requires` stays as-is (reads `scene_embeddings`); it
  optionally reads `scene_distance`.
- **Gating by presence**: `default_pipeline` does NOT include `build_scene_distance` until proven.
  The experiment runs an opt-in pipeline that inserts it. No flag, no conditional in the default path.
- **Config** (`configs/pipeline.yaml`): a documented `build_scene_distance` block (weights `w_v/w_t/w_g`,
  time scale in minutes, geo scale in metres). `cluster_scenes` unchanged by default.
- **Docs mandate**: update `docs/architecture/data_flow.html` (new node `build_scene_distance` +
  geo→distance→scene edges), `classes.html` (the helper + Inputs/Result + step), and the new config
  block comment. Code-review gate §7 checks this.

## Threats to validity

- **Label subjectivity** — "same scene" is fuzzy (is the walk to the castle one scene or three?).
  Mitigate: written labeling rubric in the tool; solo N=1 labels flagged as such (D6 of spec-102
  lineage); report ARI not accuracy so partial-agreement is graded, not pass/fail.
- **Two trips only** — Budapest (geo-rich) + Germany (geo-poor) are the extremes by design, but n=2
  labeled trips is a pilot; no generalization claim. Austria is an unlabeled sanity check.
- **Filename-time trust** — the `YYYYMMDD_HHMMSS` fallback assumes the camera clock; note that a
  wrong clock would mislead the time backbone (rare; flagged, not corrected here).

## Out of scope

- VLM-proposed / VLM-judged scene grouping (follow-up; ties to spec-099).
- Re-labeling Austria; >2 labeled trips; any generalization claim.
- Changing DINOv2 → other embeddings (a separate axis; hold the embedding fixed to isolate the
  time/geo contribution).
- Touching `select_best` / EXP-2 (the `scene_clusters` output contract is unchanged).

## Deliverables

1. `sim_bench/scene_cluster/geo_time_fusion.py` (helper, `calc()` → distance), tests (incl. the
   no-imputation + degradation units).
2. New `build_scene_distance` step (thin translator) + the minimal `cluster_scenes` precomputed-metric
   branch + two new context fields.
3. Labeling tool → `scene_labels.json` for Budapest + Germany.
4. Scoring script (ARI/homogeneity, geo-stratified) + `reports/<date>_scene_clustering_fusion/`.
5. Docs updates (architecture HTMLs + pipeline.yaml comments).
6. REVIEW.md.
