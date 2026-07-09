# spec-097 — Code Review, Slice 1 / Stage 1 (against docs/guides/CODE_REVIEW_CHECKLIST.md)

**Date**: 2026-07-09 · **Scope**: Stage-1 flat occlusion penalty —
`occlusion_bench/scorer.py`, `pipeline/steps/score_occlusion.py`,
`pipeline/scoring/occlusion_penalty.py`, `select_best` wiring, `configs/pipeline.yaml`,
studio surfacing, `models/occlusion/clip_b32_gmax_v1.npz`,
`scripts/train_occlusion_artifact.py`. Stage 2 (subject-aware) NOT in scope — spec stays In Progress.

**Verdict: PASS** (no high-severity findings; follow-ups listed in §8).

## 1 · Structure — pass
- Step file thin (spec-053): translator only; domain logic in `occlusion_bench/scorer.py`
  with `calc(Inputs) → Result`. Penalty math isolated in `scoring/occlusion_penalty.py`
  next to its sibling `person_penalty.py` (same factory/computer shape).
- All new modules ≤ ~130 LOC, single responsibility.

## 2 · Code quality — pass
- Unreadable image ⇒ P=0 (never penalized) + warning with count; no bare excepts.
- No silent defaults on load-bearing paths: gate/weight defaults are named constants in
  one place, mirrored in yaml with comments; artifact load failure raises (no fallback model).

## 3 · Naming — pass
- `occlusion` (trained) vs `clip_occlusion` (experimental prompts, marked superseded in legend).

## 4 · Layering & single-writer — pass
- `pipeline` imports `occlusion_bench` (downward); `occlusion_bench` imports no pipeline code.
- `occlusion_scores/_tiles` written ONLY by `score_occlusion`; `occlusion_penalties` ONLY by
  `select_best` — recorded in the spec-034 contract table.

## 5 · Testability — pass
- Inventory: 8 unit (step cache-hit, artifact-version invalidation, penalty math ×6) +
  2 arch gates updated (raw-iteration allow-list; contract) + 3 real-data E2E checks run
  and recorded: (a) scorer on real fingers P=0.829/0.828 vs clean 0.016; (b) studio
  `run_methods` on `examples/finger_occlusion` → column 0.171; (c) `select_best`
  integration — penalty −0.155 flips the winner to the clean sibling.
- Failure-mode walk-through:
  - *bokeh/night wrongly penalized* → `test_below_gate_is_exactly_zero_bokeh_safety` +
    empirical 0/60 motion-blurred & 0/60 defocused cross the gate (user-mandated check).
  - *stale scores after model upgrade* → `test_artifact_version_invalidates_cache`.
  - *pipelines without the step break select_best* → `test_missing_scores_mean_zero`.

## 6 · Boundary contracts — pass
- Cache key carries `model_version` = artifact version string (upgrade ⇒ recompute).
- Config knob → producer check (SIGHTING-061 class): `occlusion_penalty.enabled` defaults
  true but degrades to 0-penalty when `score_occlusion` didn't run — enforced by code and
  by `test_missing_scores_mean_zero`, documented in the step schema description.

## 7 · Documentation — pass
- spec.md status updated (Stage 1 Implemented; Stage 2 open); tasks.md slice checked off
  with evidence; spec-034 contract rows added; `data_flow.html` + `classes.html` updated;
  CHANGES_LOG entry; studio legend explains 1−P direction and the 0.2 display threshold.

## 8 · Risk register — pass-with-followups
- **Tile localization is research-grade** (hot-tile vs LoG box 13/26): area_factor uses
  tiles only to modulate ±30% of the penalty; Stage 2 owns real localization. Tracked in
  tasks.md Slice 2.
- **In-sample caveat**: 56/56 positives ≥ gate is in-sample; the honest generalization
  number is CV 0.86 [0.79–0.92]. Operating point re-measured at next data growth.
- **Detector retraining loop**: retrain = rerun `scripts/train_occlusion_artifact.py`
  (validations included) + bump `VERSION` → caches self-invalidate.
- **Perf**: +~0.5 s/img CPU on first run (10 CLIP encodes), then cached. Acceptable for
  album-scale (~1 min / 122 imgs); measured during studio E2E.
- Pre-existing suite failures documented in SIGHTING-113 (updated 2026-07-09, items 4–5);
  none touch spec-097 files.
