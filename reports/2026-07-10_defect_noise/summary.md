# Defect scoring #2: Noise (SIDD-Small, 480 pairs / 160 scenes)

**Goal**: the pipeline has no noise metric (declared in `rule_based.py` docstring, never implemented).
Does any existing score detect real sensor noise?

**Method**: SIDD-Small real smartphone noise (`D:\DataSets\SIDD_Small_sRGB_Only`), clean-GT vs noisy pairs,
3 patches/scene. Pair accuracy (clean must outscore noisy; chance = 50%).
Script: `scripts/experiment_defectscore_noise.py`.

| Method | Pair acc | AUC |
|---|---|---|
| Ours — overall rule-based | **0.0%** | 0.054 |
| Ours — Laplacian sharpness | **0.0%** | 0.013 |
| Classical — wavelet `estimate_sigma` | **100.0%** | 0.993 |
| SOTA — MUSIQ | 61.9% | 0.572 |

**Outcome**: our scoring is **inverted, not just blind** — noise inflates Laplacian variance 39x
(median 12.5 -> 488), and sharpness is 40% of the overall score, so the noisy twin won all 480 pairs
(ISO 3200 example: noisy 0.88 vs clean 0.32). Best-shot selection is biased toward the noisiest frame in
high-ISO bursts. The classical wavelet estimator is near-perfect at ~20 ms/img; MUSIQ is unreliable on
patches. **Proposed fix (needs spec)**: add `estimate_sigma` as a noise metric + noise-correct sharpness
(denoise before Laplacian), then re-run the blur benchmark.
