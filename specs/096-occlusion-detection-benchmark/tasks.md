# spec-096 — Tasks

Parallel tracks after Slice 1 (dataset is the shared foundation).

## Slice 1 — Dataset build (foundation, blocks all tracks)
- [ ] T1.1 `sim_bench/occlusion_bench/dataset.py` — build positives/negatives with
      dataset-prefixed ids, sha1 dedupe, deterministic sha1-based 80/20 split,
      hard-negative seeding, manifest.csv.
- [ ] T1.2 Run → `D:\occlusion_dataset` (49+2 positives; Budapest+Germany+Austria negatives,
      finger duplicates excluded from negatives).
- [ ] T1.3 Unit test (tmp dirs): prefixing, dedupe, split determinism, manifest columns.

## Track A — Haiku (candidate + validator)
- [ ] TA.1 Labeling script: batch Haiku over manifest (occluded? level 0–3 + reason), cached.
- [ ] TA.2 Candidate scores on test split.  TA.3 Disagreement report vs user labels → user review.

## Track B — Encoder probes
- [ ] TB.1 CLIP global probe (control).  TB.2 SigLIP-2/patch-level probe, max-pooled LR.
- [ ] TB.3 CV on train; frozen-test PR-AUC + hard-negative report.

## Track C — Tiny local VLMs
- [ ] TC.1 Moondream/SmolVLM zero-shot prompt harness (CPU), sec/img measured.

## Track D — CNN + synthetic occlusion
- [ ] TD.1 Synthetic occlusion compositor (blurred warm blobs, ordinal opacity, onto train
      negatives; group-split).  TD.2 Fine-tune small ResNet; frozen-test eval.

## Track E — Baseline
- [x] TE.1 Classical 4-cue detector exists (`/tmp` prototype → move into occlusion_bench).

## Close-out
- [ ] T9.1 `eval.py` — one harness, all candidates, RESULTS.md frontier table.
- [ ] T9.2 /code-review → REVIEW.md; CHANGES_LOG; LEARNINGS update with the verdict.

## Progress log
- 2026-07-06 Slice 1 DONE (dataset 51pos/776neg + group-aware split after user caught
  near-dupe leakage: 19 pos scenes / 345 neg groups; 3 tests green).
- 2026-07-06 T9.1 DONE minimal eval.py (StratifiedGroupKFold, scene-level PR-AUC,
  class x 1/group_size weights).
- 2026-07-06 Track F (user-proposed LoG-stats + LR) FIRST RESULT:
  scene PR-AUC mean 0.51 [0.35-0.64], image 0.44, vs random-baseline 0.062 (8x lift).
  First candidate with real learned signal; now the number to beat.
