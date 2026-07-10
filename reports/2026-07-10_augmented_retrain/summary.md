# Batch-3 positives + augmentation retrain (2026-07-10)

**Data**: 856 originals / **80 positives** (24 new batch-3), 140
RealBlur negatives, 2988 augmented variants (both classes; hflip/rot/crop/light/noise).
**Eval**: grouped OOF CV x3 seeds, metrics on originals only.

| variant | scene PR-AUC | scene ROC-AUC |
|---|---|---|
| base (856) | 0.9033 +- 0.0078 | 0.9855 +- 0.0014 |
| +RealBlur | **0.9066 +- 0.0061** | 0.9840 +- 0.0015 |
| +RealBlur+aug | 0.8994 +- 0.0054 | 0.9821 +- 0.0006 |

- New positives moved the needle: 0.777 -> ~0.90 scene PR-AUC (same-fold comparison).
- **Augmentation = null result** (CLIP already invariant to these transforms).
- At gate 0.8: 2 FP scenes-worth of images, 32/80 FN
  (user-accepted: mostly minor occlusion; see FN gallery).
- RealBlur holdout: 0/282 over gate with candidate.
- Candidate: `D:\occlusion_dataset\clip_b32_gmax_v3aug.npz` (variant=rb; not promoted).

Details + ROC + FP/FN galleries: [report.html](report.html)
