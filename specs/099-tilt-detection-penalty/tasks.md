# Tasks — Spec 099: Tilt Detection + Penalty

## Phase 0 — estimator + benchmark (no pipeline wiring yet)
- [ ] T0.1 `sim_bench/quality_assessment/tilt.py`: `estimate_tilt(gray) -> TiltResult`
      (Canny → prob. Hough → fold to deviation space → length-weighted circular median +
      agreement-based confidence; internal downscale to ~1024 px). Unit tests
      `tests/quality_assessment/test_tilt.py`: synthetic grid/edges rotated by known angles
      (±2..10°) recovered within tolerance; blank/noise image → low confidence; sign convention.
- [ ] T0.2 Benchmark script `scripts/experiment_tilt_benchmark.py`: generate known-angle rotations
      of the Budapest album (EXIF-normalized, center-crop to hide corner triangles), measure
      MAE / detection rate / false positives on the untouched originals. Gates A1, A2, A5.
- [ ] T0.3 Sweep GATE_DEG / SLOPE / CAP / CONF_GATE on the benchmark output; record chosen values
      in spec. Experiment report `reports/<date>_tilt_benchmark/` + EXPERIMENTS.md entry. (A6)

## Phase 1 — pipeline + Studio
- [ ] T1.1 `sim_bench/pipeline/steps/score_tilt.py` (≤80 LOC, one step per file): reads
      image_paths, calls helper, writes `context.tilt_angles`; universal_cache
      (`feature_type="tilt"`, `model_name="hough_v1"`).
- [ ] T1.2 `context.tilt_angles` field + spec-034 contract row (drift test green).
- [ ] T1.3 Studio "Tilt (deg)" column (quality family; value −|angle| for higher-is-better sort;
      signed angle + confidence in detail payload).

## Phase 2 — penalty in select_best
- [ ] T2.1 `sim_bench/pipeline/scoring/tilt_penalty.py` mirroring `occlusion_penalty.py`;
      config block in `configs/pipeline.yaml` (gate_deg, slope, cap, conf_gate).
- [ ] T2.2 `select_best`: composite += tilt_penalty; store `context.tilt_penalties` for
      explainability (like occlusion_penalties).
- [ ] T2.3 Penalty unit tests (A3) + composite regression test (A4: no confident tilt ⇒
      composite unchanged).

## Close-out
- [ ] T3.1 Docs: classes.html, data_flow.html, spec-034, pipeline.yaml comments.
- [ ] T3.2 CHANGES_LOG entry; LEARNINGS if a new failure class surfaces.
- [ ] T3.3 `/code-review` → REVIEW.md; resolve High findings; flip to Implemented.
