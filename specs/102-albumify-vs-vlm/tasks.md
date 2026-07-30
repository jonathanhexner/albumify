# Tasks — Spec 102: Albumify vs VLM

> Experiment. Output is `reports/` HTML, not production code. Budapest pilot gates the expansion.

## T0 — Decisions [LOCKED 2026-07-17, see spec D1-D6]
- [x] D1 raw-images-only VLM; D2 Budapest pilot first; D3 EXP-2 = 3-rater truth on ~50 clusters;
      D4 structured-JSON annotation + free-form reason; D5 same-768px inputs both arms; D6 framed as pilot.
- [x] Two expert reviews captured -> `EXPERT_REVIEW.md` (verbatim + how addressed).

## T1 — Harness scaffolding  [DONE 2026-07-17]
- [x] T1.1 `scripts/experiment_albumify_vs_vlm.py` (`prep` cmd) + `sim_bench/albumify_vs_vlm/downsample.py`:
      downsample to 768px longest edge (JPEG q80, LANCZOS, EXIF-transposed, HEIC-aware) into a data-drive
      working dir; content hash over decoded pixels -> `input_set_hash` (A1); manifest.json. Budapest prep
      run: 122 imgs (2 .mp4 correctly skipped), hash 878d0c14..., working set on `D:\albumify_vs_vlm`.
- [x] T1.2 `sim_bench/albumify_vs_vlm/roster.py`: day segmentation auto-filled from `YYYYMMDD_HHMMSS`
      filenames (EXIF fallback); persons/scenes stubs for the human. `rosters/budapest.roster.json`
      emitted (Aug 22: 102, Aug 24: 20). **HUMAN STEP PENDING**: hand-label persons + scenes for A6.
- [x] T1.3 `PROMPTS.md`: `curation_v1`, `annotation_v1`, rater sheet, EXP-2 fragment. Encodes the
      reframed objective (story + QC-as-filter + scale variety + opener/closer + protagonist weighting).
- [x] T1.4 11 unit tests (`tests/albumify_vs_vlm/`): max-edge cap + ratio, no-upscale, hash determinism/
      order-independence/content-sensitivity, day segmentation, roster validation. All green.

## T2 — Albumify arm  [DONE 2026-07-17]
- [x] T2.1/2.2 `albumify_arm.py` runs the album pipeline on 768px inputs; `curation.py` enforces exactly
      K (coverage-first over select_best's picks, backfilled) then chronological order. Emits normalized
      `AlbumResult` JSON (schema.py, shared with the VLM arm). Budapest: 20 picks, 14 scenes.
- [x] T2.3 **OOM FINDING**: full 33-step `default_pipeline` OOM-kills this box when occlusion CLIP loads
      atop YOLO+InsightFace+DINOv2+AVA (silent kill, no traceback). Added `faces` variant = default minus
      score_occlusion/score_tilt/straighten + the post-select_best face tail (never influences picks).
      Keeps person-penalty + IQA/AVA + scene-dedup. **Occlusion/tilt penalties disabled in this pilot**
      (SIGHTING filed; release-model-per-step is the real fix).

## T3 — VLM arm  [DONE 2026-07-17]
- [x] T3.1/3.2 `vlm_arm.py`: raw-images-only, map(shortlist per 15-img batch) -> reduce(order exactly K).
      Opus 4.8 (`temperature` deprecated on this model -> removed). IDs validated vs known set; tokens
      logged. Budapest: 50 shortlisted/9 batches, 94k/3.3k tokens (<$2). Sent full images per batch, NOT
      ID-overlay contact sheets (misread risk > token cost at 768px) — deviation from S/EXP-1 noted.
- [ ] T3.3 >=3-seed stability + full-context no-batching ablation (A4/A5) — not yet run (single pass).

## T4 — EXP-1 judging + metrics  [viewer+metrics DONE; human verdict PENDING]
- [x] T4.1 `ab_viewer.py`: blind ordered A/B HTML, labels hidden, L/R seeded-shuffle (A2/A3), images
      base64-embedded, full rater sheet, unblind key written separately. Budapest viewer built (4.5MB).
- [x] T4.2 `metrics.py`: duplicate-survival (Albumify scene clusters as truth) + coverage P/R hook
      (None until roster labelled). Budapest: Albumify redundancy 0.25, VLM 0.20; overlap 6/20; VLM kept
      3 quality-filtered shots. Defect-rate (detectors on the 40 picks) — TODO.
- [ ] T4.3 Krippendorff alpha (A8) — N/A for solo N=1; deferred to multi-rater upgrade.
- [ ] T4.HUMAN **Owner must judge**: open `ab_viewer_budapest.html`, save answers, THEN open unblind key.

## T5 — EXP-2 within-cluster  [HARNESS DONE 2026-07-19; human judging PENDING]
- [x] T5.1 `sim_bench/albumify_vs_vlm/exp2.py`: cluster cases = multi-frame scene clusters (excl.
      singletons + the oversized catch-all >30). Albumify best-of-cluster = argmax composite_score
      (arm now dumps full `composite_scores` in AlbumResult.meta -> no selector re-run). VLM
      best-per-cluster = one call/cluster over raw frames (reuses vlm_arm client). Budapest: 14
      clusters (95 frames). CLI `exp2`. 6 unit tests (no network).
- [x] T5.2a `exp2_viewer.py` + `exp2-viewer` cmd: blind best-frame picker HTML (frames seeded-shuffled
      per cluster, no system tell; copy-answers -> exp2_judging.json). `exp2-metrics` cmd folds
      judging back -> top-1 accuracy per system + system agreement.
- [ ] T5.2b **HUMAN STEP**: open `exp2_picker_budapest.html`, pick best/cluster, save exp2_judging.json,
      run `exp2-metrics`. Solo N=1 (D6); Kendall tau deferred (needs full human ranking, not just best).

## T6 — EXP-3 annotation  [DONE 2026-07-17]
- [x] T6.1 `annotate.py`: per-day moment grouping -> structured JSON (label/scene_type/moment_type/
      people/image_ids/best_id/reason per group; album_type + trip_subtype + narrative per album) +
      text-only album classifier. Budapest: **type=trip/city**, 30 moment groups, 66k/5k tokens.
- [x] T6.2 Qualitative: STRONG. Editorial labels ("The one where they walk with Reagan", "Musical
      fountain lights up at dusk"), caught the finger-occlusion ("...with a sneaky lens finger") and the
      Shoes-on-the-Danube memorial. Rendered as a table in the report EXP-3 section (Playwright-verified).

## T7 — Report + close-out
- [x] T7.1 `report.py` -> `reports/2026-07-17_albumify_vs_vlm_budapest/report.html` (A9) + summary.md +
      EXPERIMENTS.md entry. Titled PILOT. Both albums inline (34 downscaled samples), KPI cards, metric
      table, editorial-reason captions, both OOM + N=1 caveats stated. Playwright-verified render.
- [x] T7.2 EXPANDED to Austria (474 imgs, trip/road) + Germany (797 imgs incl .heic, trip/road).
      Reports `reports/2026-07-18_albumify_vs_vlm_{austria,germany}/` + Artifacts. **Cross-trip finding:
      overlap shrinks with size (Budapest 6/20, Austria 3/20, Germany 0/20)**. Two scale bugs fixed:
      `faces` OOMs >~120 imgs -> big trips ran `minimal` (no person-penalty); VLM reduce 413 -> hierarchical
      narrowing + shortlist cache (test_vlm_reduce.py). **Caveat: Albumify arm not identical across trips**
      (Budapest=faces, Austria/Germany=minimal).
- [ ] T7.3 `/code-review` (experiment-scoped: reproducibility, honest framing, no over-claim) -> REVIEW.md.
- [ ] T7.4 Log learnings (LEARNINGS.md); file any follow-up feature specs the results warrant.

## Open items to confirm with user (non-blocking for drafting)
- [ ] Rater pool: can we get >=5 (or >=3) blind raters, or is this a solo N=1 pilot? Changes A8 headline.
- [ ] Second VLM (GPT) as a data point, or Claude-only for v1?

## Status: IN PROGRESS 2026-07-18 — ALL 3 TRIPS ran end-to-end (EXP-1 + EXP-3). 6 reports/artifacts.
## Remaining: T4.HUMAN (owner judges 3 viewers), T5 (EXP-2 within-cluster), T3.3 (seed-stability/
## ablation), defect-rate metric, roster hand-labeling (A6), T7.3 /code-review, T7.4 learnings.
## Open: SIGHTING-117 (people-aware pipeline OOMs >~120 imgs -> big trips lack person-penalty).
## Decisions locked 2026-07-17: D7 Claude Opus 4.8 only (v1); D6 solo N=1 pilot (v1).
