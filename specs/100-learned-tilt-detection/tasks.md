# Tasks — Spec 100: Learned Tilt Detection (GeoCalib Benchmark)

> Fail-fast validation. No pipeline wiring, no penalty. Reuses spec-099's harness + penalty design.
> A pass unblocks spec-099 Phases 1–2; a fail keeps spec-099 On Hold.

## T0 — Dependency feasibility (BLOCKING — do this first)  [DONE 2026-07-12]
- [x] T0.1 Installed GeoCalib 1.0 into `.venv`. **Resolver hazard caught by dry-run**: naive install
      would upgrade numpy 1.26.4->2.4.6 and pull opencv-python 5.0 (dual-variant conflict vs existing
      opencv-contrib-python 4.11 — SIGHTING-111). Safe install used: `kornia kornia_rs "numpy<2"`
      then `geocalib --no-deps`. Verified post-install: numpy 1.26.4 UNCHANGED, cv2 4.11.0 UNCHANGED,
      pyiqa imports OK, geocalib imports OK. Env intact.
- [x] T0.2 Smoke test on 3 Budapest photos (CPU): returns signed roll via
      `rad2deg(res["gravity"].roll)`. **Bonus**: result dict exposes native `roll_uncertainty`
      (+ `up_confidence`/`latitude_confidence` maps) — a first-class confidence signal, better than
      the residual we planned to synthesize; wire G4 to this. Latency ~2.0 s/img steady-state on CPU
      (3.4 s first w/ warmup) => within G5 <3 s budget.

## T1 — Estimator backend  [DONE 2026-07-12]
- [x] T1.1 `sim_bench/quality_assessment/tilt_geocalib.py`: GeoCalib behind the spec-099
      `TiltResult` contract. `estimate_tilt(image)` (production, confidence from native
      `roll_uncertainty`) + `estimate_tilt_raw(image)` (benchmark, raw uncertainty for the sweep).
      Accepts RGB/gray array or path; downscale to 1024; model loaded once. **Sign flipped**
      (`_SIGN=-1`): GeoCalib roll is opposite spec-099's clockwise-positive convention.
- [x] T1.2 `tests/quality_assessment/test_tilt_geocalib.py` (5 tests, `slow`): contract + range,
      grayscale input, structureless→low-confidence abstention, sign matches spec-099, relative
      recovery within ±2.5°. All green.

## T2 — Benchmark on the existing harness  [IN PROGRESS]
- [x] T2.1 New script `scripts/experiment_geocalib_tilt.py` (kept separate from the Hough benchmark
      so spec-099's committed report stays reproducible). Same relative-recovery protocol, EXIF-norm,
      center-crop, ±{2,4,6,8,10}°. Emits per-image roll + uncertainty + latency to metrics.json.
      Full 122-photo run launched (bg `b8phsg3rq`, ~45 min @ 2 s/img × 11 est/photo).
- [x] T2.2 Named FP set pinned = the 3 originals the classical estimator confidently mis-flagged
      @gate 0.3 (`194451`, `195832`, `170838`) — slanted-scenery cases. Benchmark asserts each stays
      below GeoCalib's flag gate (G3).
- [x] T2.3 Uncertainty-gate sweep (0.5–3.0°). Full 122-photo result: MAE 0.25–0.45° (G1 ✅),
      recall4 ~100% (✅), coverage 28% @unc≤1.5 (G2 ❌ <70%), AUC 0.835 (G4 ⚠️), FP 5.7%@unc≤1.0
      (G3 ⚠️; 2/3 named clean), 1.88 s/img (G5 ✅). Confidence abstains honestly (kaleidoscope
      shots → 10–30° unc, gated out).

## T3 — Report + decision
- [x] T3.1 Reports: `reports/2026-07-12_geocalib_tilt/` (gates) + `reports/2026-07-13_geocalib_budapest_tilt/`
      (album report.html: sorted tilt table w/ low-conf greyed, orig-vs-straightened, up-field
      evidence, 4-panel explainer). EXPERIMENTS.md entry added.
- [>] T3.2 Go/no-go: decision gate NOT cleanly met (G2 fails; G3/G4 marginal). Results block in
      spec.md. **Awaiting user call**: (a) relax G2, ship gated tie-breaker on confident subset,
      (b) hold, or (c) raise coverage. spec-099 stays On Hold until decided.
- [x] T3.3 CHANGES_LOG + LEARNINGS entries written.
- [ ] T3.4 `/code-review` → REVIEW.md — deferred until the go/no-go decision (no ship yet).

## T3 — Report + decision
- [ ] T3.1 `reports/2026-07-12_geocalib_tilt/`: `report.html` (inline before/after samples, the 2
      named FP cases rendered, GeoCalib-vs-classical gate table, confidence-gate sweep, CPU latency)
      + `summary.md` + `metrics.json`. 2-line entry in `reports/EXPERIMENTS.md`. (G6)
- [ ] T3.2 Write the go/no-go: if G1–G4 pass → update spec-099 status to unblock Phases 1–2 with
      `model_name="geocalib_v1"`; else record the failing gate + numbers, spec-099 stays On Hold.
- [ ] T3.3 CHANGES_LOG entry; LEARNINGS line (classical-vs-learned tilt outcome, newest first).
- [ ] T3.4 `/code-review` → REVIEW.md (this is a benchmark spec; §7 doc-drift is light — no new
      pipeline state). Resolve High findings; flip spec-100 to Implemented (result = pass or fail).
