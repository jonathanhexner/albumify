# Spec 098: Noise-Aware Quality Scoring

**Status**: Implemented (2026-07-12; REVIEW.md: ACCEPT, all gates green)
**Created**: 2026-07-11
**Origin**: defect-scoring benchmarks 2026-07-10 (`reports/2026-07-10_defect_noise/`) — user approved drafting after review of findings.

## Problem

The pipeline has **no noise metric**, and sensor noise (grain from high-ISO / low-light shots) **inverts**
the sharpness score: noise is high-frequency energy, so `cv2.Laplacian(gray).var()` explodes on grainy
images (median 12.5 clean → 487.9 noisy on SIDD, a 39× inflation). Measured consequence: on 480 real
clean/noisy pairs (SIDD-Small), the noisy twin outscored the clean one in **480/480 pairs** on the overall
rule-based score (e.g. ISO 3200: noisy 0.88 vs clean 0.32).

User-visible impact:
1. **Best-shot selection prefers the grainiest frame** of a high-ISO burst (sharpness = 40% of overall score).
2. **Face blur gate is defeated by noise** — a grainy face crop reads as "sharp" and passes.
3. No way to filter/sort by noise in the Image Analysis Studio.

Meanwhile a classical wavelet noise estimator (`skimage.restoration.estimate_sigma`) scored
**100% pair accuracy / AUC 0.993 at ~20 ms/image** on the same data.

## Affected code (all three production Laplacian call sites)

| Site | Role |
|---|---|
| `sim_bench/quality_assessment/rule_based.py:107` (`_compute_sharpness`) | whole-photo quality (Albumify + Image Studio) |
| `face_cluster/quality.py:181` (`compute_blur_scores`) | face-crop blur gate (spec-053 protected) |
| `sim_bench/face_pipeline/quality_scorer.py:91` (`_compute_sharpness`) | legacy face pipeline |

## Solution

### S1. Shared helper module — `sim_bench/quality_assessment/noise_robust.py`
Pure functions (typed, framework-agnostic, notebook-usable per spec-053 conventions):

```python
def noise_robust_laplacian_var(gray, sigma=None) -> float:
    """3x3 median blur, Laplacian variance, then subtract K_SIGMA*sigma^2 (floor 0)."""

def estimate_noise_sigma(img) -> float:
    """Wavelet noise sigma (skimage), green channel, 1024px central crop. Higher = noisier."""

def noise_sigma_to_score(sigma) -> float:
    """1 / (1 + sigma/SIGMA_HALF) -> [0,1], higher = cleaner. SIGMA_HALF = 2.0."""
```

**Formula evolution (recorded 2026-07-11)**: the original plan (median blur only) FAILED gates
A1/A2 on the first validation re-run — 3×3 median cut inflation 39×→11× only, overall pair acc 62%.
Sweeps (`scripts/experiment_spec098_sweep{,2,3}.py`) led to the final formula:
`max(lap(median3(gray)) − 5·σ², 0)` — the σ² term subtracts the noise's own residual Laplacian
contribution. `K_SIGMA = 5` zeroes SIDD inflation with RealBlur pair accuracy unchanged
(0.9929 at k=0 and k=5). σ estimation: green channel + 1024px central crop (6× faster than
3-channel averaging, identical SIDD discrimination, 33 ms on 16 MP → gate A6).

`PyWavelets` + `scikit-image` become declared dependencies (setup.cfg).

### S2. Phase 1 — whole-image scoring (rule_based + Studio)
- `_compute_sharpness` uses `noise_robust_laplacian_var`.
- New `noise` component in `RuleBasedQuality`: `noise_score = 1 / (1 + sigma / SIGMA_HALF)`,
  `SIGMA_HALF = 2.0` (calibrated: clean median 0.76 ≥ 0.7 target; ISO≥1600 noisy 0.14 ≤ 0.2 target).
- Default weights rebalanced (fixed by sweep v3, not hand-tuned):
  `sharpness .30, exposure .25, colorfulness .05, contrast .10, noise .30` → 99.4% SIDD pair acc,
  98.3% on the ISO≥1600 subset. Colorfulness `.20 → .05` deliberately: chroma noise inflates it.
- Studio: register `noise_sigma` as a new method in `model_factory` / engine registry (numeric, lower =
  better → negated per the pyiqa direction convention) so users can sort/inspect noise directly.
- `get_detailed_scores` exposes `noise_sigma` + `noise_score`.

### S3. Phase 2 — face-crop scoring (RISK: shifts blur_score distributions)
- `face_cluster/quality.py::compute_blur_scores` and `face_pipeline/quality_scorer.py::_compute_sharpness`
  switch to `noise_robust_laplacian_var`.
- Denoising **lowers raw Laplacian values across the board**, so any absolute blur threshold
  (profile blur gate) may need re-calibration. This phase is gated on the Budapest E2E baseline
  (`tests/face_clustering/e2e_budapest/`, expected 15 clusters / 340 faces). If the baseline shifts,
  re-calibrate the profile threshold to restore it and document the mapping old→new.

## Acceptance criteria (all measured by re-running the 2026-07-10 benchmark scripts)

| # | Gate | Threshold |
|---|---|---|
| A1 | SIDD noise benchmark (`experiment_defectscore_noise.py`) — overall score pair accuracy | ≥ 95% (was 0%) |
| A2 | SIDD — noise-robust sharpness inflation ratio (noisy/clean median) | < 2× (was 39×) |
| A3 | RealBlur blur benchmark (`experiment_defectscore_blur.py`) — pair accuracy with noise-robust sharpness | ≥ 99% (no regression from 99.4%) |
| A4 | Budapest E2E baseline (Phase 2) | 15 clusters / 340 faces / sizes unchanged (or threshold re-calibrated + justified) |

**A4 resolution (2026-07-12, user-approved "option 1")**: the corrected blur formula scores face
crops at ~0.45× the old scale (rank corr 0.93 over the 340 Budapest faces), so `blur_min` was
re-calibrated **150 → 73.3** in `profile_4.json` and `profile_5.json` (73.3 minimizes gate flips:
19/340 faces vs the old partition; profiles 1–3/smoke have the gate off). Headless same-path
comparison (`scripts/run_profile.py`): baseline code @150 → 7 clusters [26,20,12,7,3,2,2] noise 268;
spec-098 @73.3 → 8 clusters [29,19,13,7,3,2,2,2] noise 263 — within the documented tolerance bands
(`tests/_budapest_baseline.py`: count 6–10, rejected 255–285). The exact 15-cluster/[35,24,…] UI
reference is **unreproducible even with baseline code** (pre-existing app-vs-pipeline divergence,
SIGHTING-113 item 3 — evidence runs `4b6d2fdb`/`46551fdc`/`40f7364d`); scenario B (stored reference
run) still passes. Any profile in the wild with a nonzero `blur_min` tuned for the old scale should
be multiplied by ~0.49 (150→73.3).
| A5 | Unit tests: synthetic noise added to a clean image must lower `noise_score` and NOT raise robust sharpness | pass |
| A6 | Perf: `estimate_noise_sigma` on a 12 MP photo | < 150 ms |

## Out of scope
- Exposure-score fix (separate finding, separate spec if approved).
- Rotation detection (separate finding).
- Denoising the actual images (we only *score* noise).
- pyiqa/deep noise metrics.

## Docs to update (Code Review §7)
- `docs/architecture/` classes HTML (RuleBasedQuality new component + new module).
- `configs/pipeline.yaml` comments if weights surface there.
- `reports/EXPERIMENTS.md` entry for the re-run validation.
