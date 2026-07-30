# Spec 102: Albumify vs VLM — curation, within-cluster pick, and annotation value-add

**Status**: In Progress — 2026-07-17 (T1 harness done; Budapest pilot prepped, 122 imgs)
**Type**: Experiment (produces `reports/` HTML; **no production code path**). Findings may spawn
feature specs (VLM annotation, VLM reranker) as follow-ups.
**Depends on**: Albumify pipeline (`face_cluster/pipeline.py`, `select_best`, clustering), the three
trip datasets on disk.
**Reviewed at draft time by**: a senior CV researcher and a senior album editor (personas). Their
critiques are folded into the design below and preserved in `EXPERT_REVIEW.md`.

## Goal (the real one)

Not "does Albumify beat a VLM" as a leaderboard. The goal is **to scout where a VLM belongs in this
project** — annotation, album-type classification, coverage reasoning, or selection — using head-to-head
selection as the "is it even competitive" evidence. The honest framing is a **pilot / case study**, not
an inferential benchmark (see Threats to Validity).

## Three sub-experiments

```
EXP-1  Curation      both output an ORDERED K-sequence from the SAME 768px inputs
                     → blind human A/B (ordered, side by side) + structured sub-ratings
                     + objective metrics (coverage P/R, dup-survival, defect rate)
EXP-2  Cluster pick  best-of-cluster on ~50 sampled clusters
                     → 3 blind raters pick best = REFERENCE; report top-1 acc + Kendall tau
EXP-3  Annotation    VLM-only: day/place segmentation, moment-type tags, group captions,
                     album-type + narrative. No Albumify equivalent — this is the scouting arm.
```

## Datasets

| Trip | Path | Role |
|---|---|---|
| Budapest | `D:\Budapest2025_Google` | **Pilot** — known: 15 clusters, 340 faces, labeled. Validate method here first. |
| Austria | `D:\Austria_24` | Expansion (after pilot passes). |
| Germany | `D:\Google_Germany` | Expansion (after pilot passes). |

## Design decisions (defaults, user-approved 2026-07-17)

- **D1 VLM input** = **raw images only** (downsampled 768px). Clean attribution: generalist vs engineered
  pipeline. The "VLM-as-reranker on our metadata" variant is a *follow-up spec*, not this one.
- **D2 Phasing** = **Budapest pilot first** — run all three EXPs end-to-end on Budapest, validate
  prompts + harness + rater sheet, THEN expand to Austria/Germany. De-risks VLM spend.
- **D3 EXP-2 judge** = **human blind pick = ground truth**, but on **~50 sampled clusters** with **3
  raters**, reporting top-1 accuracy + Kendall tau (agreement alone != correctness).
- **D4 Annotation** = **structured JSON** (gradeable, product-mappable) + a free-form `reason` string
  per group.
- **D5 Fairness (resolution)** = Albumify consumes the **same 768px inputs** as the VLM. Resolution is
  held constant so a VLM loss can't be blamed on the harness.
- **D6 Framing** = **PILOT**. With 3 trips (n=3 units) and a small rater pool, no inferential claim is
  made; the writeup says so in the title. **v1 locked as solo N=1** (the owner judges) — explicitly an
  illustrative anecdote, not a measured win-rate. Multi-rater (>=5) is a later upgrade, not v1.
- **D7 VLM** = **Claude Opus 4.8 only** for v1. A second model (GPT) is a follow-up, not this pass.

## Objective — reframed (both experts, independently)

The naive objective "**pick the best K, cover everyone, no duplicates**" is a QC checklist, not an
editorial brief; it produces a technically-correct, soulless album and mis-measures the tools. Reframed:

> **Assemble an ordered sequence that tells the trip's story. QC (sharp / eyes-open / no-occlusion /
> exposed) is a _filter_, not the objective.**

Consequences baked into the prompts and metrics:
- **Ordering matters** — both systems output an ordered sequence; the A/B viewer preserves order.
- **Intentional repetition is allowed** — a burst that shows a moment building is not "duplicates."
- **Protagonists, not egalitarian coverage** — frame count follows who the trip is about.
- **Emotional peak > technical "nice"** — a soft genuine-laughter frame may beat a sharp posed one.
- **Scale variety** — wide / medium / detail intercut; a required opener and closer.
- **K floats** within a range (default 18-24; ~1 per 15 shots), not a hard 20.

## EXP-1 — Curation (detailed)

**Both systems** receive the same 768px JPEGs and must return an **ordered list of K image IDs** plus,
optionally, a one-line rationale per pick.

- **Albumify arm**: normal pipeline (score -> penalties -> select_best -> cluster) on the 768px inputs,
  then an ordering pass (chronological within scene groups is the v1 baseline order).
- **VLM arm**: Claude Opus 4.8, temperature 0. Images batched as **contact-sheet grids** (12-16 thumbs),
  VLM returns picks by grid ID (map phase), then a **reduce pass** across batch-winners produces the
  final ordered K. **Batch composition is logged** (which near-dups landed on different sheets — a known
  systematic handicap that must be reported, not hidden). **>=3 seeds/trip**, pick-stability reported.
  **One trip additionally run full-context (no batching)** to bound the batching penalty.

**Judging** (the headline + the sub-scores):
- **>=5 raters** who have never seen either system's output; **>=1 per trip who was on the trip**
  (only they can judge person/scene coverage). Solo fallback = explicitly labeled N=1 anecdote.
- **Blind**: labels hidden ("Album A / Album B"), left/right randomized per trip, **shown in order**.
- **Forced overall choice** + structured 5-pt sub-ratings: **coverage, photo quality, redundancy,
  narrative**.
- **The "why" prompt** (editor's wording, to capture the real reason not a rationalization):
  *"Which album would you actually want to show someone — and what's the one image or moment that made
  you pick it?"* plus a forced axis: **story vs. slideshow**.
- Report per-rater results + **inter-rater agreement (Krippendorff's alpha)**; alpha < ~0.4 => verdict
  uninterpretable, say so.

**Objective metrics** (no human, computed from our own signals — these are the defensible numbers):
1. **Coverage precision/recall** vs a hand-labeled per-trip roster of persons and distinct scenes/days.
2. **Duplicate-survival rate** — fraction of near-dup pairs (from Albumify clusters) with both members
   surviving in the chosen K.
3. **Defect rate** — blur / occlusion / eyes-closed count in each chosen set, via existing detectors
   (spec-097 occlusion, spec-098 sharpness, eyes if available).

## EXP-2 — Within-cluster best-photo (detailed)

Sample **~50 clusters** across the pilot trip (stratified by cluster size). For each, both systems pick
the single best frame. **3 blind raters** independently pick the best frame per cluster (or rank the
top-3); majority = **reference truth**. Report, per system:
- **Top-1 accuracy** (system pick == reference best).
- **Kendall tau** between system ranking and human ranking where a ranking exists.
This measures *correctness*, not just system-vs-system agreement (the flaw in the first draft).

## EXP-3 — VLM annotation value-add (detailed)

VLM-only; no competition. Structured JSON output per the schema in `PROMPTS.md`:
- **Per group**: `caption`, `scene_type`, `moment_type` (meal/transit/landmark/candid/golden-hour/
  goofing/...), `people`, `day`, `reason` (free-form, why this is the group's best moment).
- **Per album**: `album_type` (trip / kids-growing / family / wedding / milestone-celebration /
  everyday / pet / project / memorial / event / **mixed**), `trip_subtype` (city / road / beach / hike,
  when trip), and a one-line `narrative`.
- **Useful != literal**: day/place segmentation with a human-readable label ("Day 3 — Buda Castle & the
  funicular") and moment-type tags are the target; generic "a bridge over a river" captions are noise
  and are graded down.
- Assessment is qualitative (editor-style rubric in `EXPERT_REVIEW.md`): does this scaffold an album,
  or is it a caption dump?

## Prompts are a versioned deliverable

`PROMPTS.md` holds `curation_v1`, `annotation_v1`, and the exact **rater instruction sheet**. Prompts
are versioned so prompt-sensitivity is inspectable. v1 encodes the reframed objective (story + QC-filter
+ scale variety + opener/closer + protagonist weighting), NOT the naive checklist.

## Deliverables

1. `reports/<date>_albumify_vs_vlm_budapest/report.html` — EXP-1/2/3 for the pilot: ordered A/B galleries,
   verdicts + reasons, sub-rating tables, objective-metric tables, annotation samples inline.
2. Same for Austria and Germany after the pilot validates.
3. `summary.md` per report + a 2-line entry in `reports/EXPERIMENTS.md`.
4. `EXPERT_REVIEW.md` (this dir) — the two expert critiques verbatim + how each point was addressed.

## Acceptance criteria

| # | Gate | Threshold |
|---|---|---|
| A1 | Same-input fairness: both arms consume identical 768px JPEGs | verified by hash of the input set |
| A2 | Ordered output: both arms return an ordered K-sequence; A/B viewer preserves order | pass |
| A3 | Blind protocol: labels hidden, L/R randomized, rater sheet fixed | pass |
| A4 | VLM reproducibility: temp=0, model version logged, >=3 seeds, pick-stability reported | reported |
| A5 | Batching honesty: batch composition + dup-split count logged; 1 full-context ablation run | reported |
| A6 | Objective metrics: coverage P/R, dup-survival, defect rate computed for both arms | reported per trip |
| A7 | EXP-2: top-1 accuracy + Kendall tau vs 3-rater reference on ~50 clusters | reported |
| A8 | Honest framing: report titled "pilot"; inter-rater alpha reported; no over-claim | pass |
| A9 | Report exists with inline ordered galleries + annotation samples | exists |

## Threats to validity (stated up front, per CV review)

- **Self-recognition leakage** — the pipeline's author must not be the sole judge; blind raters required,
  else it's an N=1 anecdote (labeled as such).
- **Batching confound** — VLM windowed low-res thumbnails vs global scoring; mitigated by same-768px
  inputs + a full-context ablation + logged batch composition.
- **Power** — n=3 trips, photos within a trip not independent; no inferential claim. Generality would
  need ~10-15 trips (out of scope; noted as future work).
- **Proxy gap** (editor) — every tool optimizes photographic quality + coverage as a *proxy for personal
  meaning*, loosest exactly at the emotional peaks. Tools are graded on **"how good a first-pass draft,"**
  and a **human-rescue step** (which 3-5 frames would you add back?) is part of the EXP-1 rater sheet.

## Out of scope

- VLM-as-reranker on Albumify metadata (D1 follow-up spec if EXP results warrant).
- Shipping any VLM annotation into the product DB (separate feature spec if EXP-3 is promising).
- Ordering optimization beyond the v1 chronological-within-scene baseline.
- Generalization claims / >3 trips.

## Follow-ups this experiment may spawn

- Feature spec: VLM annotation (day/place segmentation + moment tags + album-type) as a pipeline step.
- Feature spec: VLM reranker arm (the D1 augmented variant).
- Product: album-type classifier + narrative field in the export schema.
