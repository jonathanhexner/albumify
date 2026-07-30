# REVIEW — spec-099 Phases 1–2 (tilt detection + penalty, GeoCalib backend)

**Reviewed**: 2026-07-13 · **Base**: working tree (branch `specs/093-095-analysis-studios`)
**Scope**: the tilt feature only (spec-099 Phase 1–2 wiring + spec-100 backend). The branch
carries unrelated uncommitted work; that is out of scope here.
**Verdict**: **ACCEPT with follow-ups.** No §1–§7 `fail`. Two items need a user decision
(default-pipeline perf; e2e_budapest applicability) — surfaced below, neither blocks correctness.

---

## Part 1 — How it works

```
score_tilt (new step, default_pipeline after score_occlusion)
  └─ TiltScorer.calc(TiltInputs) ──> GeoCalib roll + confidence   [tilt_geocalib.py, spec-100]
     └─ universal_cache (feature_type="tilt", model_name="geocalib", version="geocalib_v1")
        └─ context.tilt_angles / context.tilt_confidences
             └─ select_best._compute_composite_scores:
                  composite = quality + person_penalty + occlusion_penalty + tilt_penalty
                     └─ TiltPenaltyComputer  [scoring/tilt_penalty.py]
                        0 unless conf >= 0.5 AND |roll| > 3 deg; then -min(0.02*(|roll|-3), 0.15)
                     └─ context.tilt_penalties  (Results breakdown)
   image_studio: METHODS["tilt"] -> _map_tilt column (sort -|roll|, detail=confidence)
```

**Module inventory (new):** `tilt_geocalib.py` (161 LOC), `steps/score_tilt.py` (84),
`scoring/tilt_penalty.py` (80), `tests/pipeline/test_{tilt_penalty,score_tilt}.py`,
`tests/quality_assessment/test_tilt_geocalib.py`. **Edited:** `context.py` (+3 fields),
`select_best.py`, `all_steps.py`, `configs/pipeline.yaml`, `image_studio/{engine,method_info}.py`,
`specs/034/spec.md`, `docs/architecture/{classes,data_flow}.html`.

**Dependency direction:** pipeline → quality_assessment → geocalib (mirrors occlusion →
occlusion_bench). No reverse imports. Every occlusion touch-point was mirrored 1:1.

## Part 2 — Findings by section

**§1 Structure** — `pass`. All modules < 300 LOC, single responsibility. `tilt_penalty.py`
is a closed Config/Computer/Factory family (mirrors occlusion). `score_tilt.py` (84 LOC) is a thin
translator (spec-053): reads context → `TiltInputs` → `calc()` → writes result. No dead code
(`n_lines` kept for TiltResult contract parity, documented).

**§2 Code quality** — `pass-with-followup` (low). `score_tilt._get_scorer` wraps model load in
`except Exception` → logs + disables (returns None → step produces empty → penalty 0). This is
**intentional fail-safe** graceful degradation for an *optional* signal, not a load-bearing swallow
— a missing GeoCalib degrades to today's behavior, never a wrong score. Per-image `except` in
`TiltScorer.calc` mirrors occlusion's skip-on-unreadable. No silent boundary defaults.

**§3 Naming** — `pass`. `score_tilt`/`tilt_penalty`/`tilt_geocalib` parallel the occlusion trio.
`__all__`: `ScoreTiltStep` not added to `all_steps.__all__` — consistent (occlusion isn't either;
registration is via the `@register_step` import).

**§4 Layering & coupling** — `pass-with-followup` (low). **Single writer** confirmed:
`tilt_angles`/`tilt_confidences` written only by `score_tilt`; `tilt_penalties` only by
`select_best`. Follow-up: `_lookup` (path-separator tolerance) is now triplicated across
person/occlusion/tilt penalties — pre-existing pattern; TODO to extract a shared helper.

**§5 Testability** — `pass`. Inventory: **9** unit (penalty math), **2** unit (step cache, mocked
scorer), **3** integration w/ real GeoCalib on real Budapest photos (1 e2e step→cache→penalty +
2 recovery, dataset-gated), **3** contract/abstention. Real E2E present:
`test_real_geocalib_end_to_end_penalizes_tilted` (tilted photo penalized, level one not).
Failure-mode walk-through: "a guessed tilt moves a score" → `test_low_confidence_is_exactly_zero`;
composite regression (A4) → `test_missing_scores_mean_zero_composite_regression`; cache staleness →
`test_model_version_invalidates_cache`; abstention on structureless scenes →
`test_structureless_image_low_confidence`.

**§6 Boundary contracts** — `pass`. New context fields registered in spec-034 §3 →
`test_pipeline_context_contract.py` green (enforced, not just documented). **Config knob → producer
check (SIGHTING-061 class): PASS** — `tilt_penalty` gates on `tilt_angles`/`tilt_confidences`,
produced by `score_tilt`, which is in `default_pipeline`; and when absent the penalty is fail-safe 0.
Dataclasses (frozen) mirror OcclusionInputs/Result — no new Pydantic boundary introduced.

**§7 Documentation** — `pass`. `spec.md` (099 reactivated + 100 created), `tasks.md` (both),
this `REVIEW.md`, `classes.html` + `data_flow.html` updated (drift closed), spec-034 contract,
`CHANGES_LOG`, `LEARNINGS`, `EXPERIMENTS`. No DB column change → `db_schemas.html` N/A.

**§8 Risk register** — `pass-with-followup` (medium). See decisions below. Backwards-compat: runs
without `score_tilt` → penalty 0 → composite bit-identical (unit-proven). Dependency: geocalib +
kornia installed & verified on Windows `.venv` under `numpy<2` (spec-100 T0); recipe documented but
not test-enforced (low follow-up). Pre-existing test failures found (NOT caused by this change,
files unchanged by it): `test_scoring_strategy.py::test_person_penalty_strategy` (stale expected
value) and `test_face_pipeline_e2e.py` ERRORs (data/model fixtures) — see tickets.

## Part 3 — Verdict & follow-ups

| Area | Verdict |
|---|---|
| §1 Structure, §3 Naming, §5 Tests, §6 Contracts, §7 Docs | ACCEPT |
| §2 Code quality, §4 Coupling, §8 Risk | ACCEPT with follow-up |

**Decisions taken (2026-07-13, user away — proceeded on recommended path, confirm on return):**
- (1) Perf: **kept `score_tilt` in `default_pipeline`** (cached; penalty fail-safe).
- (2) e2e_budapest: **waived as not-applicable** — zero `face_cluster/` files changed (git-verified),
  and the real-GeoCalib pipeline e2e passed. If you'd rather run it, say so and I will before flipping.
- Status held at **Code Review** (not yet Implemented) pending your confirmation of the above.

**Decisions for the user (non-blocking):**
1. **Hot-path perf** — `score_tilt` adds GeoCalib ~1.9 s/img to `default_pipeline` *first run*
   (cached after; repeat runs free). Keep in the default pipeline, or make it opt-in like the
   Studio methods? Recommend: keep default (cached, penalty is fail-safe), revisit if albumify
   first-run latency is felt.
2. **e2e_budapest gate** — the binding v2 suite covers the *face_clustering* app; this change's
   blast radius is the *albumify* pipeline + *image_studio* (git diff shows **zero** `face_cluster/`
   files). The relevant e2e (real-GeoCalib pipeline) was run and passed. Recommend: waive
   e2e_budapest as not-applicable, or run it as belt-and-suspenders before merge.

**Follow-up tickets filed:**
- SIGHTING-115 — `test_face_embedding_validation.py` stale import (pre-existing).
- SIGHTING-116 — `test_person_penalty_strategy` stale expected value (pre-existing) — to file.
- TODO — extract shared `_lookup` path-tolerance helper across the 3 penalty computers.
- TODO — architecture test pinning the `numpy<2` / geocalib-`--no-deps` install invariant.
