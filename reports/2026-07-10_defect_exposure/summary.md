# Defect scoring #3: Exposure (PIQA23, 1,500 imgs / 50 scenes, human JOD labels)

**Goal**: benchmark pipeline exposure score (`_compute_exposure_quality`: histogram clip + entropy, 30% of
overall quality) against human exposure judgments.

**Method**: PIQA23 Exposure attribute (`D:\DataSets\PIQA23`), JOD scores from pairwise human comparisons;
stratified 30 imgs/scene. Spearman correlation (SROCC) vs JOD.
Script: `scripts/experiment_defectscore_exposure.py`.

| Method | SROCC per-scene mean | global |
|---|---|---|
| Ours — clip + entropy | 0.276 | 0.213 |
| Classical — mid-gray distance (1 line) | 0.327 | 0.239 |
| SOTA — MUSIQ | **0.571** | **0.360** |
| SOTA — CLIPIQA | 0.306 | 0.161 |

**Outcome**: our exposure score barely discriminates and **loses to a one-line brightness baseline**.
Root cause: score compression — 90% of images land in [0.89, 0.99] because the entropy term is ~1 for any
natural photo and the clip penalty counts only pure 0/255 bins, which smartphone pipelines never emit
(a JOD −3.0 under-exposed shot scores 0.965). **Proposed fixes (needs spec)**: soft-clipping penalty
(<5 / >250 mass), add mid-gray term, or swap in MUSIQ (~1s/img CPU). PIQA23 `Details`/`Overall` attributes
kept for later sharpness-vs-human validation.
