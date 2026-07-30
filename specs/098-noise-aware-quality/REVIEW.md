# REVIEW — Spec 098: Noise-Aware Quality Scoring

**Reviewed**: 2026-07-12, against `docs/guides/CODE_REVIEW_CHECKLIST.md` (all 8 sections).
**Scope**: working-tree changes on `specs/093-095-analysis-studios` (spec-098 is uncommitted at review time).
**Verdict: ACCEPT** — no unresolved High-severity findings; 3 follow-ups filed (see Part 3).

---

## Part 1 — How it works

### Module inventory
| File | Change |
|---|---|
| `sim_bench/quality_assessment/noise_robust.py` | NEW (~110 LOC): `noise_robust_laplacian_var(gray, sigma=None)` = median3 → Laplacian var → `max(x − 5σ², 0)`; `estimate_noise_sigma(img)` = wavelet σ, green channel, 1024px central crop; `noise_sigma_to_score(σ)` = `1/(1+σ/2)`. Constants `K_SIGMA=5.0`, `SIGMA_HALF=2.0`, both data-calibrated (sweeps in `scripts/experiment_spec098_sweep{,2,3}.py`). |
| `sim_bench/quality_assessment/rule_based.py` | `_compute_sharpness` → robust helper (σ passed in, single estimation); new `_compute_noise_sigma`; new `noise` weight (defaults `.30/.25/.05/.10/.30`); `get_detailed_scores` adds `noise_sigma`/`noise_score`, computes `overall` inline (no double image load). |
| `sim_bench/pipeline/steps/score_iqa.py` | produces `noise_scores`; **cache `model_name` bumped `rule_based → rule_based_v2`** (stale-score protection); `.get("noise")` tolerance for pre-v2 rows. |
| `sim_bench/pipeline/context.py` | new field `noise_scores: dict[str, float]` (+ spec-034 row). |
| `sim_bench/image_quality_models/{iqa_model_wrapper,model_factory}.py` | new `NoiseOnlyIQAModel` / `noise_iqa` registry key. |
| `app/image_studio/engine.py` | "Noise (clean-ness)" column (same `score_iqa` step run). |
| `face_cluster/quality.py` | `compute_blur_scores` → robust helper (spec-053 `calc()` composition unchanged). |
| `sim_bench/face_pipeline/quality_scorer.py` | legacy scorer → robust helper. |
| `setup.cfg` | + `scikit-image>=0.22`, `PyWavelets>=1.4`. |
| Profiles (user dir) | `profile_4.json`, `profile_5.json`: `blur_min` 150 → 73.3 (calibrated mapping, user-approved). |

### Dependency map / data flow
```
noise_robust.py  (cv2, numpy, lazy skimage — imports nothing from face_cluster: no cycle)
   ├── rule_based.py ── ScoreIQAStep ──> context.{iqa,sharpness,noise}_scores ──> Studio columns
   ├── face_cluster/quality.py (QualityGater.compute_blur_scores → blur gate, profile blur_min)
   └── face_pipeline/quality_scorer.py (legacy MediaPipe path)
```
Layering note: `face_cluster` core importing `sim_bench` has precedent (`fc_app_runner.py`); the
imported module is dependency-light and there is no import cycle.

---

## Part 2 — Findings per checklist section

### §1 Structure — pass
Single-responsibility new module (~110 LOC); no dead code; no >4-param functions; dependency direction verified by grep (noise_robust imports only cv2/numpy/skimage).

### §2 Code quality — pass
No new try/except; the one boundary `.get("noise")` in `_store_results` is a documented pre-v2-cache tolerance, not a silent default; comments explain *why* (inflation numbers, calibration provenance).

### §3 Naming / packaging — pass
`noise_robust.py` name matches contents; sibling of `rule_based.py` in the existing `quality_assessment` package; project does not use `__all__` (convention followed).

### §4 Layering / coupling — pass
Single writer for `context.noise_scores` (ScoreIQAStep only). No duplicated logic — the three former inline Laplacian copies now share one helper (net de-duplication). `face_cluster→sim_bench` import: precedented, cycle-free (see Part 1 note).

### §5 Testability — pass
- **Inventory**: 13 unit tests (`tests/quality_assessment/test_noise_robust.py`) — 3 robust-Laplacian behavior, 4 σ-estimation, 2 score mapping, 3 RuleBasedQuality integration on real PNGs on disk, 1 cache-version guard. Real-data E2E: SIDD 480 pairs + RealBlur 702 pairs re-runs (validation report) + 3 headless Budapest album runs.
- **Failure-mode walkthrough**: (a) noise-inversion recurs → `test_noisy_image_scores_lower_overall` + `test_noise_barely_inflates_robust_sharpness` fail; (b) blur detection broken by denoising → `test_blur_still_lowers_robust_sharpness` + A3 benchmark; (c) stale-cache serving (spec-079 class) → `TestScoreIQACacheVersion` fails on revert of the v2 bump; (d) blur-gate self-disable (spec-053 class) → untouched, `QualityGater.calc()` composition preserved.
- Placement: unit tests in `tests/quality_assessment/` ✓.

### §6 Boundary contracts — pass
No new Pydantic/Pandera models. spec-034 context contract updated + enforced by `test_pipeline_context_contract` (was red during review, green after the row was added). Config-knob→producer: `blur_min` gate's producer (`compute_blur_scores`) still wired; `noise` weight's producer is RuleBasedQuality itself; Studio Noise column's producer is ScoreIQAStep ✓.

### §7 Documentation — pass
spec.md (incl. formula evolution + A4 resolution), tasks.md, this REVIEW.md; `classes.html` (QualityGater + new module rows), `data_flow.html` (context state line), spec-034 table; CHANGES_LOG entry; LEARNINGS entry (2026-07-12); validation report + EXPERIMENTS.md index entry. No DB schema change → `db_schemas.html` untouched.

### §8 Risk register — pass-with-followup
- **Backwards compat**: custom profiles with nonzero `blur_min` tuned for the old scale over-reject until scaled by ~0.49 — documented in spec + validation report; profiles 1–3/smoke unaffected (gate off). Old `rule_based` cache rows orphaned (never served) — harmless.
- **Hot path**: +σ-estimation per image ≈ 33 ms (16 MP, measured; gate A6 <150 ms) + median blur inside the Laplacian path; amortized by `universal_cache`.
- **Deferred work**: tickets below.

### Test-suite status at review
`tests/architecture` + `tests/image_quality_models` + spec-098 tests: **151 passed** (after spec-034 row fix). `tests/pipeline`: 149 passed; the 2 fails + 5 errors + 1 collection error are all pre-documented in SIGHTING-113 (items 1, 2, 4, 5) — verified none touch spec-098 paths. e2e_budapest: 8/11 pass; the 3 fails (A/C/F) reproduce identically on baseline code (stash-verified) — pre-existing, evidence filed in SIGHTING-113.

---

## Part 3 — Verdict & tickets

**ACCEPT.** All acceptance gates green (A1 99.4%, A2 0×, A3 99.29%, A4 calibrated + user-approved, A5 13/13, A6 33 ms). No High-severity findings.

Follow-up tickets filed:
1. **TODO.md**: fix `tests/quality_assessment/test_learned_clip.py` + `test_learned_prompts_only.py` (import archived `clip_aesthetic`; aborts package collection) — SIGHTING-113 item 6.
2. **SIGHTING-113 (updated)**: e2e_budapest fresh-run baseline unreproducible on baseline code + 3 UI-level scenario failures — re-anchor `[35,24,14,…]` when item 3 (app-vs-pipeline divergence) is fixed.
3. **TODO.md**: sweep any remaining hard-coded `blur_min`-style thresholds outside profiles (e.g. notebooks, older configs) for the ×0.49 rescale.
