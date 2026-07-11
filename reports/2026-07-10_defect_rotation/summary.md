# Defect scoring #4: Rotation (122 album photos x 4 orientations)

**Goal**: pipeline orientation handling is EXIF-only (`exif_transpose`); can any scorer detect *content*
rotation when EXIF is wrong/stripped (screenshots, WhatsApp, scans)?

**Method**: 122 Budapest photos, EXIF-normalized then pixel-rotated 0/90/180/270 and re-encoded (EXIF
stripped). 4-way accuracy = upright variant scores highest of the 4 (chance 25%).
Script: `scripts/experiment_defectscore_rotation.py`.

| Method | 4-way acc | AUC upright-vs-rotated |
|---|---|---|
| Ours — pipeline overall score | 22.1% (chance) | **0.500** (blind) |
| Classical — sky prior (top brighter) | 47.5% | 0.722 |
| SOTA — CLIP ViT-B/32 zero-shot | **82.0%** | 0.822 |

**Outcome**: the pipeline is exactly chance (mean score 0.81056 upright vs 0.81057 rotated) — blind to
rotation by construction. CLIP zero-shot with 3 text prompts reaches 82% with no training and the model is
already in our stack (spec-097). **Proposal (needs spec)**: on EXIF-less images, flag when a rotated variant
beats the current orientation's CLIP upright-score; dedicated orientation CNNs (~98% lit.) are the upgrade path.
