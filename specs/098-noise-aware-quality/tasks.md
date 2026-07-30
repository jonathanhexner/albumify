# Tasks — Spec 098: Noise-Aware Quality Scoring

## Phase 0 — helper + calibration
- [x] T0.1 Create `sim_bench/quality_assessment/noise_robust.py`. Tests landed in
      `tests/quality_assessment/test_noise_robust.py` (13 tests incl. cache-version guard). (A5)
      NOTE: median-blur-only formula failed A1/A2 on real SIDD noise; final formula adds the
      `− K_SIGMA·σ²` correction (see spec §S1 formula evolution + sweep scripts).
- [x] T0.2 `PyWavelets` + `scikit-image` declared in setup.cfg.
- [x] T0.3 `SIGMA_HALF = 2.0` (clean median 0.76 ≥ 0.7; ISO≥1600 noisy 0.14 ≤ 0.2).
- [x] T0.4 Perf: 33 ms on a 16 MP photo (green channel + 1024px central crop). (A6)

## Phase 1 — whole-image scoring
- [x] T1.1 `rule_based.py`: robust sharpness + `noise` component; weights `.30/.25/.05/.10/.30`
      (sweep v3); `get_detailed_scores` extended; `score_iqa` cache bumped `rule_based_v2`.
- [x] T1.2 `noise_iqa` model + Studio "Noise (clean-ness)" column + `context.noise_scores`
      (+ spec-034 contract row; drift test green).
- [x] T1.3 SIDD re-run: overall pair acc **99.4%** (hi-ISO 98.3%), inflation **0×**. (A1, A2)
- [x] T1.4 RealBlur re-run: robust sharpness pair acc **99.29%**. (A3)
- [x] T1.5 `reports/2026-07-11_spec098_validation/` + EXPERIMENTS.md entry.

## Phase 2 — face-crop scoring
- [x] T2.1 `face_cluster/quality.py::compute_blur_scores` → robust helper.
- [x] T2.2 `face_pipeline/quality_scorer.py::_compute_sharpness` → robust helper.
- [x] T2.3 A4 via headless `scripts/run_profile.py` (UI scenario A broken on baseline code too —
      SIGHTING-113). `blur_min` re-calibrated **150 → 73.3** in profile_4/profile_5 (user-approved
      option 1); calibrated run within `_budapest_baseline` tolerance bands. Full resolution in
      spec §A4 resolution. e2e_budapest: 8/11 pass; 3 fails reproduce on baseline (pre-existing).

## Close-out
- [x] T3.1 `classes.html` + `data_flow.html` + spec-034 table updated.
- [x] T3.2 `CHANGES_LOG.md` entry.
- [x] T3.3 REVIEW.md written (ACCEPT, no High findings); spec flipped to Implemented 2026-07-12.
