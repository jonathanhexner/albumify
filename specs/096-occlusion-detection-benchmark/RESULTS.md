# spec-096 — Benchmark Results (round 1, pre-adjudication labels)

**Date**: 2026-07-07 · **Dataset**: 51 positives (19 scenes) / 776 negatives (345 groups),
grouped+stratified 5-fold CV, scene-level PR-AUC primary. Random baseline ≈ 0.06.
**Caveat**: labels are PRE-adjudication (82 disputed images pending user review); every
trainable candidate refits in seconds once corrections land.

## Leaderboard (scene PR-AUC)

| # | Candidate | Scene | Image | Cost / latency | Verdict |
|---|---|---|---|---|---|
| 1 | **B: CLIP probe, global+tile-max (OOF)** | **0.83** [0.42–1.0] | 0.81 | free, ~0.5 s/img CPU | **best-verified** |
| 1 | B: CLIP tile-mean probe | 0.83 | 0.82 | same | |
| 3 | B: CLIP tile-max probe | 0.81 | 0.77 | same | |
| 4 | B: CLIP global probe | 0.75 | 0.71 | same | control |
| ? | D2: synth-trained CNN (ResNet18 ft) | 0.87* | 0.51* | free, ~0.1 s/img | *test = 3 scenes only — unproven |
| 5 | D1: frozen ResNet18 probe | 0.64 | 0.57 | free, fast | |
| 6 | F: LoG-stats + LR (user-designed) | 0.51 | 0.44 | free, ~0.2 s/img | best hand-crafted |
| 7 | E: classical 4-cue + ring | 0.43 | 0.33 | free, ~0.13 s/img | baseline |
| 8 | A: Haiku zero-shot | 0.41 | 0.29 | ~$1.3/1k imgs, ~2 s/img API | + validator value |
| — | C: Moondream tiny-VLM | FAILED | — | — | torch 2.3 incompatible (`enable_gqa`); pin old revision to retry |

## Conclusions

1. **Learned boundary ≫ zero-shot, in the same embedding space.** CLIP *prompts* scored
   ~nothing on this task; a linear probe on CLIP *embeddings* scores 0.83. The signal was
   always in the embedding — it could not be phrased as text.
2. **Locality matters (user's patch hypothesis).** Tile variants beat the global probe by
   +0.06–0.08. The defect is a corner phenomenon; global pooling dilutes it.
3. **Contrastive features > classification features.** CLIP embeddings (0.75–0.83) beat
   ImageNet-ResNet18 features (0.64) on identical folds.
4. **Hand-crafted features hit a ceiling ~0.5.** Three feature rounds (IQR, extent counts,
   kurtosis) all landed within noise of 0.51 — mutually redundant physics cues. Kurtosis and
   entropy did rank among the top weights (user's distributional hypothesis confirmed
   directionally), but embeddings encode strictly more.
5. **Synthetic→real transfer is plausible but unproven.** The synth-trained CNN ranks real
   occlusions well in-sample and hits 0.87 on the (3-scene) test split. Needs the
   adjudicated refit + a bigger test set before it can be crowned.
6. **VLM value is labeling, not scoring.** Haiku ranked poorly as a detector (0.41) but its
   disagreement mining produced the 82-image worklist and found the DARK-occluder class
   (backlit fingers/straps) that every warm-assuming approach was structurally blind to.
7. **Tiny local VLMs are fragile infrastructure.** Moondream needs torch ≥2.5; env pins
   (numpy<2, torch 2.3 CPU) made it a no-go without upgrades. Not retried this round.

## Recommendation

Production scorer = **CLIP tile probe** (free, fast, best-verified, gives per-tile
localization for UI + area-scaled penalties). Revisit CNN after adjudication.
Integration design → proposed spec-097 (score_occlusion step + occlusion_penalty in
select_best composite; penalize-don't-delete; bokeh-safe because the driver is the
learned probability, not raw blur).

## Blocking next step

**User adjudication of the 82 disagreements** (review app, Adjudicate page). Then:
refit B/D/F on corrected labels → final leaderboard → close spec-096 (REVIEW.md,
LEARNINGS, /code-review).
