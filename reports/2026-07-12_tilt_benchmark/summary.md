# spec-099 Phase 0: tilt detection benchmark — NEGATIVE result (2026-07-12)

**Goal**: validate classical tilt estimation (Hough lines, per-family h/v agreement) before wiring
a crooked-photo penalty into `select_best`.

**Method**: 122 Budapest photos, known ±2..10° rotations injected, relative recovery measured;
3 estimator iterations (pooled median -> per-family histogram peak -> support sweep).
Script: `scripts/experiment_tilt_benchmark.py`; estimator `sim_bench/quality_assessment/tilt.py`
(13 unit tests green).

| Metric | Value | Gate |
|---|---|---|
| MAE on confident estimates | **0.32 deg** | <1.5 ✓ |
| Coverage (photos measurable at all) | **4.9%** | FAIL |
| Perf | 49 ms | <100 ✓ |

**Killer finding**: both inspectable confidently-flagged originals are UPRIGHT photos with slanted
scene lines (tunnel perspective; optical-illusion artwork). Pixels can't distinguish "camera
tilted" from "world tilted" — that needs gravity (IMU) or a learned semantic prior. Relaxing
confidence for coverage destroys precision (MAE 0.32 -> 3.3 deg).

**Verdict**: penalty NOT shipped; spec-099 on hold. 0 verified truly-crooked photos in 122 —
low-frequency problem, wrong-cost risk. Estimator + tests + benchmark kept for a future
learned-model attempt. Options documented in report.html.
