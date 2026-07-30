# spec-093 Code Review — Slices 1-2 (backend)

**Date**: 2026-06-29 · **Reviewer**: self (against `docs/guides/CODE_REVIEW_CHECKLIST.md`)
**Scope**: `PyIQAModel`, `ScoreQualityStep`, `PipelineContext.method_scores`, tests.
Slice 3 (app) descoped → spec-094.

| # | Section | Verdict | Notes |
|---|---|---|---|
| 1 | Correctness | PASS | Direction normalized (higher=better; raw kept). 10 tests pass incl. real BRISQUE + real brisque/niqe step run on examples. |
| 2 | Tests | PASS | Unit: registry, availability, both direction branches, from_config, cache-hit, unknown-method skip, empty album. Offline + deterministic (fake metric/model); one real offline BRISQUE. |
| 3 | Error handling | PASS | Unknown/unavailable method logged + skipped (run not sunk); per-image failure → score=None, not raised; missing pyiqa → `is_available()` False + clear ImportError. |
| 4 | Naming/style | PASS | Mirrors `ava_model_wrapper` / `iqa_model_wrapper`; full imports; type hints; logging not prints. |
| 5 | No dead/dup code | PASS | One `PyIQAModel` for all 6 metrics; reuses BaseStep cache handler + Serializers. |
| 6 | Security/secrets | N/A | No secrets; pyiqa weights cached under `~/.cache/torch/hub/pyiqa/`. |
| 7 | Docs updated | PASS | `data_flow.html` step + `method_scores` field; spec design corrected to `image_quality_models`; CHANGES_LOG. |
| 8 | Windows/ASCII | PASS | Tests use tmp_path + forward-slash str paths; ASCII output; ran green on win32/py3.11. |

## High-severity findings
None.

## Medium / notes
- **M1 — Step config not Pydantic-validated.** `ScoreQualityStep.process()` is
  overridden (multi-feature cache), so it bypasses `validate_step_config`. Config
  is simple (methods/device/model_configs) and read defensively. Acceptable; a
  typed config model can be added if this step enters a default pipeline.
- **M2 — `score_image` returns negated value for BRISQUE/NIQE** (e.g. -19.7).
  Intentional (higher=better contract). Display layer (spec-094) uses `raw_score`
  for the human-facing number. Documented in code + spec.
- **M3 — AVA via this step needs a checkpoint** passed in
  `model_configs={'ava': {'checkpoint': ...}}`; absent → method skipped (logged),
  not fatal.
- **M4 — Parallel spec-094 engine** currently wires its image_quality family to
  `score_iqa`/`score_ava` (graceful-degradation path). Adopting the new pyiqa
  keys into 094's registry is a follow-up on the 094 side, not a 093 blocker.

## Verdict
**PASS** — no high-severity findings. Backend (Slices 1-2) ready; spec may move to
Implemented once integrated/consumed (Slice 3 owned by spec-094).
