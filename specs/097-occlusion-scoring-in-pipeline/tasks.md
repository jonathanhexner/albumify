# spec-097 — Tasks

## Slice 0 — research (DONE 2026-07-07)
- [x] T0.1 Saliency ladder research: 3 variants x 122 Budapest, dual-box panels →
      `research_saliency/index.html`. Findings in spec (flat-but-noisy rule;
      overlap-driven penalty makes sky false-boxes benign).

## Slice 1 — Stage 1 (flat penalty)  [DONE 2026-07-09; gate satisfied by adjudicated refit 0.86]
- [x] T1.1 Production artifact `models/occlusion/clip_b32_gmax_v1.npz` (global+tile-max LR,
      832 imgs / 56 pos, adjudicated) via `scripts/train_occlusion_artifact.py`;
      loader `occlusion_bench/scorer.py` (OcclusionScorer.calc, spec-053).
      + user-mandated validations: BLUR SEPARATION (0/60 clean, 0/60 motion-blurred,
      0/60 defocused cross the 0.8 gate; 56/56 real occlusions do — in-sample) and
      EXPLAINABILITY (tile-attention gallery `research_saliency/tile_explain/`; hot-tile
      vs LoG box 13/26 — detection solid, tile localization research-grade → Stage 2).
- [x] T1.2 `score_occlusion` step (universal_cache `occlusion`, artifact-version key;
      context.occlusion_scores/_tiles; spec-034 rows; raw-iteration allow-list).
- [x] T1.3 `occlusion_penalty` (scoring/occlusion_penalty.py): 0 below gate 0.8, then
      weight·P·area_factor(tiles), floor −0.5; wired into select_best composite + yaml.
- [x] T1.4 Tests (14 green): step cache-hit + version invalidation; penalty math, gate
      boundary = bokeh-safety regression, floor, disabled, path tolerance.
- [x] T1.5 Studio `occlusion` column (1−P, higher=clearer, legend) + docs/architecture
      (data_flow.html, classes.html). Verified E2E: real fingers P=0.83 → penalty −0.155
      flips select_best to the clean sibling; studio column 0.171 on finger imgs.

## Slice 2 — Stage 2 (subject-aware)
- [ ] T2.1 Subject-mask ladder helper (faces∪persons → SR×center fallback).
- [ ] T2.2 Overlap penalty (w_subject / w_bg) + severity mapping L1-L3.
- [ ] T2.3 Tests incl. the two real fingers (overlap≈0 → gentle) + synthetic
      subject-covering cases (overlap high → strong).
- [ ] T2.4 UI: show subject box + occluded tiles in image detail.

## Close-out
- [ ] T9.1 /code-review → REVIEW.md; CHANGES_LOG; architecture HTMLs.
