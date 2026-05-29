# Tasks: spec-053 Helper API consolidation + quality-gate step merge

**Status**: Implemented
**Estimated effort**: 4-6 focused hours

7 phases. Phase 1 is the bulk of the work (5 helpers); Phase 2 is the consolidation that fixes the user-visible bug; the rest are migration + gates.

---

## Phase 0 — Helper base types

- [ ] T000 Create `face_cluster/_helper_base.py`:
  ```python
  from typing import Protocol, TypeVar
  C = TypeVar("C"); R = TypeVar("R")
  class PipelineHelper(Protocol[C, R]):
      def calc(self, config: C) -> R: ...
  ```
- [ ] T001 Add 1 import-smoke test in `tests/face_clustering/test_helper_base.py`.

**Check**: protocol importable; runtime_checkable() works.

---

## Phase 1 — Add `calc()` to each helper (additive)

Convention per helper:
- `__init__(config, ...)` stays as it is today (already the pattern).
- New `<Helper>Inputs` dataclass: only per-call data, NOT a PipelineContext.
- New `<Helper>Result` dataclass: only the outputs the step needs.
- New `calc(inputs: Inputs) -> Result` method that composes the existing methods.
- Existing methods stay public for notebook callers.

- [ ] T010 `QualityGater.calc(QualityGateInputs)` — composes `compute_blur_scores → [compute_pose_scores] → select_core_set`. **This is the one that fixes the bug** — `calc()` always calls `compute_blur_scores` first.
- [ ] T011 `KNNGraphBuilder.calc(KNNGraphInputs)` — thin facade over `build_graph()`.
- [ ] T012 `ExemplarSelector.calc(ExemplarInputs)` — facade over `select_exemplars()` (and `get_d10_distances` if used together).
- [ ] T013 `SimplifiedMerger.calc(MergeInputs)` — facade over `merge()`.
- [ ] T014 `RunExporter.calc(RunExportInputs)` — facade over the 14-arg `export()`. `RunExportInputs` has all 14 fields.

Per-helper test (5 tests total, ~20 LOC each): seed minimal inputs, call `helper.calc(inputs)`, call the manual sequence, assert equal outputs.

**Check**: 5 helpers each have `calc()`, each has a green unit test, existing helper tests still pass.

---

## Phase 1.5 — Backfill missing per-helper test files

Two helpers have no dedicated test file today and we're about to add a `calc()` method to each. Land a minimal direct test before the new entry point goes in.

- [ ] T015 NEW `tests/face_clustering/test_knn_graph.py` — 3 cases: empty input → empty graph; 3 core faces with clear separation → expected edges; symmetric K-NN behavior.
- [ ] T016 NEW `tests/face_clustering/test_exemplars.py` — 3 cases: single-face cluster → that face is the exemplar; tight cluster → top-1 near centroid; outlier rejection (d10 threshold).

**Check**: 6 new tests pass; helpers now have a per-class test floor before spec-053 changes them.

---

## Phase 2 — Consolidated `quality_gate` step

- [ ] T020 Create `sim_bench/pipeline/steps/quality_gate.py`:
  - Single `QualityGateStep(BaseStep)` with `name="quality_gate"`.
  - `__init__` builds `QualityGater(fc_config, use_pose_estimation=...)`.
  - `process(context, config)` calls `gater.calc(QualityGateInputs(faces=context.face_records))`.
  - Writes `core_indices`, `holdout_indices`, `face_records` (with blur populated) back to context.
- [ ] T021 Diff the two old step bodies (`filter_quality_gate.py` and `QualityGateFacesStep`) to confirm consolidated behavior covers both. Document any intentional behavior change in spec.md.
- [ ] T022 Add integration test `tests/face_clustering/test_quality_gate_step.py`:
  - Seed faces with known blur scores; assert holdout decisions match `blur_min`.
  - Seed faces with `aligned_face=None` and `blur_min=50`; assert NOT all-holdout (the "self-disable" path triggers correctly).
- [ ] T023 Add register decorator + smoke that `quality_gate` is in the step registry.

**Check**: 3 new tests pass; bug from user report reproducible-and-fixed in T022 second case.

---

## Phase 3 — Migrate both apps to the consolidated step

- [ ] T030 Update `face_cluster/fc_app_runner.py::UNIFIED_CLUSTERING_STEPS`: replace `"quality_gate_faces"` with `"quality_gate"`.
- [ ] T031 Update Albumify entry points: replace any `"filter_quality_gate"` reference with `"quality_gate"`. (Grep for the string.)
- [ ] T032 Delete `sim_bench/pipeline/steps/filter_quality_gate.py`.
- [ ] T033 Delete `QualityGateFacesStep` class from `sim_bench/pipeline/steps/face_clustering_steps.py` (keep the rest of the file).
- [ ] T034 Grep for any remaining references to the old step names in `configs/`, `docs/`, `specs/`, `notebooks/`. Update or document.
- [ ] T035 Run existing Albumify + FC App v2 E2E tests. Both must stay green.

**Check**: grep finds zero references to the old step names outside changelog/historical-spec files; both E2E tests green.

---

## Phase 4 — CLAUDE.md convention

- [ ] T040 Add to CLAUDE.md, after §Implementation gate:
  ```markdown
  ## Pipeline step file convention (mandatory)
  - **One Step class per file.** Files like `face_clustering_steps.py` that bundle multiple steps are anti-pattern; split when touched.
  - **Step files are thin (≤80 LOC).** They read context, build a helper Config, call `helper.calc(cfg)`, write the Result back to context. Domain logic lives in `face_cluster/`, not in the step.
  - **Every domain helper exposes `calc(config: HelperConfig) -> HelperResult`** as its primary entry. Individual methods stay public for non-pipeline callers (notebooks, analyses) but pipeline steps must use `calc()`.
  - **Rationale**: this is what spec-053 codified. The two-quality-gate bug (one step forgot to call `compute_blur_scores`) is impossible under this convention.
  ```

**Check**: section present; manual review.

---

## Phase 5 — Sweep

- [ ] T050 Run `pytest tests/face_clustering tests/architecture -q`. Confirm no regressions vs pre-spec-053 baseline (currently 5 failed, all pre-existing per spec-052).
- [ ] T051 Manual smoke against the user's real album:
  - Run v2 pipeline with `blur_min=50`.
  - Inspect the log for "Blur gate: NOT WIRED" — must NOT appear.
  - Confirm rejected faces appear in holdout (per-gate counters non-zero for blur).

**Check**: full suite no worse; manual smoke shows blur gate ACTIVE.

---

## Phase 6 — Code review gate

- [ ] T060 Run `/code-review` → `REVIEW.md`. Walk §1-§8.
- [ ] T061 Resolve any high-severity findings.
- [ ] T062 Close SIGHTING-068 with cross-reference.
- [ ] T063 Flip spec to Implemented.
- [ ] T064 `CHANGES_LOG.md` entry under `[REFACTOR]`.

**Check**: REVIEW.md exists; zero blockers; spec is Implemented.

---

## Test delta

| Phase | Added | Type |
|---|---|---|
| 0 | 1 | import smoke |
| 1 | 5 | unit (per-helper calc-vs-manual equivalence) |
| 1.5 | 6 | unit (backfill: KNNGraphBuilder + ExemplarSelector, 3 each) |
| 2 | 3 | integration (consolidated step incl. user-bug reproduction) |
| **Total** | **15** | |

Net change: +15 tests, ~0 deleted (old quality-gate tests can be re-aimed at the new step rather than removed).

---

## Sequencing rationale

- Phase 0 before 1 because every `calc()` references the protocol.
- Phase 1 before 2 because the consolidated step uses `QualityGater.calc()`.
- Phase 2 before 3 because we need the new step in the registry before migrating callers.
- Phase 3 before 4 because the convention only makes sense once there's a concrete example.
- Phase 5 before 6 because the review needs a green-or-justified suite.
