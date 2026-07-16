# Tasks — Spec 101: Auto-Straighten

> Remediation half of the tilt story. Consumes spec-099/100's roll; produces a derived,
> corner-clean straightened asset. No original is ever mutated.

## T0 — Decisions  [LOCKED 2026-07-16, see spec D1-D6]
- [x] Separate terminal `straighten_images` step; preserve_aspect; cubic; orthogonal stages with
      fixability-scaled penalty; gate = subject-preservation (YOLO persons via detect_persons,
      prominent_person_frac config default 0.15) AND area floor (config default 0.70).

## T1 — Core geometry  [DONE 2026-07-16]
- [x] T1.1 `sim_bench/quality_assessment/straighten.py`: `largest_inscribed_rect` (max-area) +
      `aspect_preserving_rect` (same-ratio) + `straighten(rgb, roll, preserve_aspect=True)` +
      `retained_area_fraction`. Framework-agnostic, no border fill ever visible.
- [x] T1.2 `tests/quality_assessment/test_straighten.py` (16 tests): A2 vs brute-force max-area,
      A1 sentinel-border-absent-in-crop, A4 aspect/max-area, identity at 0°, sign symmetry. All green.
      Real GeoCalib round-trip (A3): tilted +16.5° → −0.45° after straighten; level photo untouched.

## T2 — Subject-aware gate (pure, testable)  [DONE 2026-07-17]
- [x] T2.1 `straighten_gate.py`: `decide(roll, conf, w, h, person_bbox_norm, cfg) -> GateResult`
      (conf+angle gate, area floor, YOLO person ≥ prominent_person_frac must be inside the inscribed
      crop; else decline). Pure geometry, no I/O.
- [x] T2.2 8 unit tests: not_tilted, low-conf, landscape straightens, portrait declined_area,
      prominent-person-clipped declined, prominent-person-inside straightens, small-person ignored,
      prominence-is-config. `context.persons` carries normalized bbox (x,y,w,h) — confirmed.

## T3 — Early pipeline step + rebind + provenance  [DONE 2026-07-17]
- [x] T3.1 `straighten_images.py` (EARLY, per D2 reversal): depends_on score_tilt, detect_persons;
      gate each image; on straighten writes leveled derivative to `~/.sim_bench/image_cache/straightened/`
      (key = orig, roll, aspect, version) and rebinds `context.image_paths`/`active_images`. Config
      block in yaml (enabled/gates/prominent_person_frac). Reordered default_pipeline.
- [x] T3.2 `tilt_penalty` — NO code change needed: straightened image's new path misses `tilt_angles`
      → 0; declined image keeps path → penalty. Verified (straightened 0.0, declined −0.1).
- [x] T3.3 `context.straightened_from` field + spec-034 contract row (drift test green) + docs HTML.
- [x] T3.4 4 step tests (straighten+rebind, decline, disabled passthrough, derived-file caching) +
      real portrait declines end-to-end + pipeline dependency resolution (no unmet requires).

## T4 — Report + close-out
- [x] T4.1 Report `reports/2026-07-17_auto_straighten/` (option A): before/after gallery on the 10
      confident tilted Budapest photos w/ real YOLO gate. 7 straightened, 3 declined (area). Median
      retained 79% -> fov penalty 0.086 (validates fov_weight=0.4). summary.md + EXPERIMENTS entry.
- [x] T4.2 Docs: classes.html, data_flow.html, spec-034, CHANGES_LOG updated to option A.
- [x] T4.3 `/code-review` -> REVIEW.md written; §5 blocker raised then CLEARED via option A + real-data
      E2E. Remaining pass-with-followup: provenance wiring (export/UI consume `straightened_from`),
      decide() param grouping, single-source gate config.

## Status: IMPLEMENTED 2026-07-17 (REVIEW ACCEPT, §5 blocker cleared, T4 report + outcome artifact done)

## Follow-ups (non-blocking, filed)
- [ ] Provenance: verify/wire the app export + display to serve the straightened winner via
      `straightened_from` (selected_images now points at cache files for straightened winners).
- [ ] Single source of truth for gate knobs (min_retained_area / prominent_person_frac) shared by
      straighten_images + tilt_penalty (currently duplicated, must-match).
