# Defect scoring #1: Blur (RealBlur-J, 702 pairs)

**Goal**: benchmark pipeline sharpness (Laplacian variance, `rule_based.py`) against a classical and a SOTA
method on real camera-shake blur.

**Method**: RealBlur-J paired sharp/blurred photos (`D:\occlusion_dataset\realblur\j_sample`).
Pair accuracy (sharp must outscore its blurred twin) + ROC-AUC.
Script: `scripts/experiment_defectscore_blur.py`.

| Method | Pair acc | AUC |
|---|---|---|
| **Ours — Laplacian variance** | **99.4%** | 0.871 |
| Classical — Tenengrad | 99.0% | 0.765 |
| SOTA — MUSIQ (pyiqa) | 98.9% | **0.893** |

**Outcome**: our sharpness metric is competitive with SOTA — best at within-scene ranking (best-shot
selection); MUSIQ is better for absolute "is this blurry" thresholds (Laplacian is content-dependent).
The 0.6% failures are dark noisy frames where noise reads as sharpness — the exact failure that becomes
total on the noise benchmark (see `2026-07-10_defect_noise`). Production `/1000` normalization: 0 ties here,
but saturation is a latent risk on bright textured photos.
