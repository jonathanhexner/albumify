# spec-097 — Tasks

## Slice 0 — research (DONE 2026-07-07)
- [x] T0.1 Saliency ladder research: 3 variants x 122 Budapest, dual-box panels →
      `research_saliency/index.html`. Findings in spec (flat-but-noisy rule;
      overlap-driven penalty makes sky false-boxes benign).

## Slice 1 — Stage 1 (flat penalty)  [GATED on spec-096 adjudicated refit]
- [ ] T1.1 Train + serialize the production probe artifact (winner per refit) into
      `models/occlusion/`; loader with version string.
- [ ] T1.2 `score_occlusion` step (cache hooks; occlusion_scores + occlusion_tiles).
- [ ] T1.3 `occlusion_penalty` in select_best composite (spec-084 pattern) + config.
- [ ] T1.4 Tests: step cache-hit; penalty math; bokeh-safety regression (portrait
      with heavy DOF must NOT be penalized).
- [ ] T1.5 Albumify Results + Studio surfacing; docs/architecture update.

## Slice 2 — Stage 2 (subject-aware)
- [ ] T2.1 Subject-mask ladder helper (faces∪persons → SR×center fallback).
- [ ] T2.2 Overlap penalty (w_subject / w_bg) + severity mapping L1-L3.
- [ ] T2.3 Tests incl. the two real fingers (overlap≈0 → gentle) + synthetic
      subject-covering cases (overlap high → strong).
- [ ] T2.4 UI: show subject box + occluded tiles in image detail.

## Close-out
- [ ] T9.1 /code-review → REVIEW.md; CHANGES_LOG; architecture HTMLs.
