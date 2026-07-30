# ResNet18 fine-tune vs CLIP probe (2026-07-10)

Same protocol both sides (856 originals / 80 pos / +140 RealBlur / grouped OOF CV seed 0).

| model | scene PR-AUC | image PR-AUC | FP≥gate | FN<gate |
|---|---|---|---|---|
| CLIP+LR (+RealBlur) | **0.899** | 0.911 | 2 | 32/80 |
| ResNet18 fine-tuned + aug | 0.768 | 0.736 | 10 | 38/80 |

CNN loses by ~13 points: train loss ~0.01 = memorization,
OOF lags = overfit on 80 positives despite on-the-fly augmentation. CLIP's pretraining is the
advantage a small dataset can't buy back. Details: [report.html](report.html)
