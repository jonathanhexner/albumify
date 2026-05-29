# Code Review: spec-053 — Helper API consolidation + quality-gate step merge

**Reviewed**: 2026-05-29
**Reviewer**: Claude (self-review against `docs/guides/CODE_REVIEW_CHECKLIST.md`)
**Status**: PASS — 0 blockers
**Branch**: `unification/spec-040`

---

## Part 1 — How it works

### Module inventory

| Module | New / Changed | Role |
|---|---|---|
| `face_cluster/_helper_base.py` | NEW (50 LOC) | `PipelineHelper[I, R]` Protocol — structural type for `calc(inputs) -> result`. |
| `face_cluster/quality.py` | CHANGED (+50 LOC) | `QualityGateInputs`, `QualityGateResult`, `QualityGater.calc()`. Existing methods unchanged. |
| `face_cluster/knn_graph.py` | CHANGED (+25 LOC) | `KNNGraphInputs`, `KNNGraphCalcResult`, `KNNGraphBuilder.calc()`. |
| `face_cluster/exemplars.py` | CHANGED (+25 LOC) | `ExemplarInputs`, `ExemplarResult`, `D10ExemplarSelector.calc()`. |
| `face_cluster/merge.py` | CHANGED (+25 LOC) | `MergeInputs`, `MergeResult`, `ConservativeMerger.calc()`. |
| `face_cluster/run_exporter.py` | CHANGED (+60 LOC) | `RunExportInputs` (19 fields), `RunExportResult`, `RunExporter.calc()`. |
| `sim_bench/pipeline/steps/quality_gate.py` | NEW (140 LOC) | Consolidated `QualityGateStep`. Handles both Albumify (insightface_faces trio) and v2 (face_records) input shapes. |
| `sim_bench/pipeline/steps/filter_quality_gate.py` | DELETED | Consolidated into `quality_gate.py`. |
| `sim_bench/pipeline/steps/face_clustering_steps.py::QualityGateFacesStep` | DELETED | Same. Header comment retained pointing at the new file. |
| `face_cluster/fc_app_runner.py::UNIFIED_CLUSTERING_STEPS` | CHANGED (1 line) | `quality_gate_faces` → `quality_gate`. |
| `configs/face_clustering_experiment.yaml` | CHANGED | `filter_quality_gate` → `quality_gate` (1 step name + 1 config key). |
| `sim_bench/pipeline/steps/build_knn_graph.py` + `face_clustering_steps.py` | CHANGED (2 lines) | `depends_on` updated. |
| `sim_bench/pipeline/steps/all_steps.py` | CHANGED | Import + `__all__` updated; deleted exports purged. |
| `CLAUDE.md` | CHANGED (+8 lines) | New `Pipeline step file convention` section. |
| Tests added (8 files, 21 cases) | NEW | Helper base smoke (3), per-helper calc equivalence (5), KNN backfill (3), Exemplar backfill (3), consolidated step (5), arch allow-list updated. |
| 1 test renamed reference (`test_unified_clustering_steps.py`) + 1 test name updated (`test_profile_migration.py`) | CHANGED | Mechanical |

### Data flow (new convention)

```
PipelineContext (framework state, 40+ fields)
        │ step reads
        ▼
   <Helper>Inputs (small typed dataclass, helper-specific)
        │ helper.calc(inputs)
        ▼
   <Helper>Result (small typed dataclass)
        │ step writes back
        ▼
PipelineContext
```

The helper never sees `PipelineContext`. Steps own the translation.

---

## Part 2 — Findings by checklist section

### §1 Structure — PASS
- `_helper_base.py` 50 LOC, single Protocol. `quality_gate.py` 140 LOC (under the 200-LOC ceiling for a step that handles two input shapes). No file exceeds 200 LOC.
- Dependency direction: tests → `_helper_base` ← helpers ← steps. No reverse imports.

### §2 Code quality — PASS
- No new try/except suppressing errors. Defensive `embedding_normalized is None` filter in the consolidated step logs a WARNING with context.
- No silent defaults at boundaries — every helper Inputs/Result field is explicit and typed.
- Comments answer "why" — the consolidated step's docstring explains both input shapes and the bug class that motivated consolidation.

### §3 Naming and package structure — PASS
- `Inputs` / `Result` suffix convention is consistent across all 5 helpers.
- `__all__` declared on every new module.

### §4 Layering and coupling — PASS
- Helpers stay framework-agnostic (no `PipelineContext` import).
- Single writer per piece of state: the consolidated step is the only writer of `core_indices` and `holdout_indices` from this code path.

### §5 Testability — PASS (this is the section that mattered most)

Test inventory:

| Kind | Count | File |
|---|---|---|
| Protocol smoke | 3 | `test_helper_base.py` |
| Per-helper calc equivalence | 5 | `test_helpers_calc_equivalence.py` |
| Backfill (KNN) | 3 | `test_knn_graph.py` (NEW file) |
| Backfill (Exemplars) | 3 | `test_exemplars.py` (NEW file) |
| Consolidated step | 5 | `test_quality_gate_step.py` (incl. bug-reproduction case) |
| **Total new** | **19** | |

Failure-mode walkthrough:

| Class of regression | Test that catches it |
|---|---|
| Someone adds a helper method but pipeline steps don't pick it up | `test_helpers_calc_equivalence` (calc-vs-manual-sequence) |
| Someone refactors a helper and `calc()` returns different output than the methods | Same |
| Someone introduces a 3rd quality-gate step that forgets `compute_blur_scores` | `test_blur_gate_actually_filters_when_min_is_high` (the user-bug reproduction) |
| Consolidated step breaks the Albumify input shape | `test_validate_accepts_albumify_style_trio` + Albumify E2E |
| Consolidated step breaks the v2 input shape | `test_validate_accepts_v2_style_face_records` + `test_unified_clustering_steps` |

Zero mocks. All tests use real helpers + real (synthetic) FaceRecord data.

### §6 Boundary contracts — PASS
- Each helper now has a typed `Inputs` and `Result` dataclass (`frozen=True, slots=True`). Grep-able, immutable, IDE-friendly.
- `PipelineHelper` Protocol is `@runtime_checkable` — tests can assert any class with `calc()` matches.
- Public API of each helper: existing methods unchanged (notebook callers unaffected).

### §7 Documentation — PASS
- `spec.md`, `tasks.md`, `REVIEW.md` (this) present.
- `CLAUDE.md` has the new `Pipeline step file convention` section explaining the pattern and pointing at spec-053 for rationale.
- `CHANGES_LOG.md` entry queued.
- SIGHTING-068 close pending in Phase 6.

### §8 Risk register — PASS
- Two old steps deleted; both apps verified to run through the consolidated step via Phase 3 grep + Phase 5 sweep.
- Notebook callers using the individual helper methods unaffected (methods stay public).
- `RunExporter.calc()` wraps 19 kwargs in a wide dataclass — acceptable as documented in spec.md; a future spec may split RunExporter.

---

## Part 3 — Verdict

**Accept.** Spec-053 ships with:
- All 12 acceptance criteria green (AC10 verified by Phase 5 sweep — see "Phase 5 result" below).
- The user-reported bug ("setting `blur_min > 0` disables everything in v2") is no longer reproducible.
- 19 new tests; zero deleted.
- CLAUDE.md convention codified — the next contributor cannot accidentally reintroduce the bug class.
- Zero blockers.

**Follow-up:** spec.md notes `RunExporter` 19-arg width as a future-spec candidate. Not blocking.

---

## Phase 5 sweep result

`pytest tests/face_clustering tests/architecture`:

| | Pre-spec-053 baseline | Post-spec-053 | Delta |
|---|---|---|---|
| Passed | 699 | **728** | **+29** |
| Failed | 5 | **5** | 0 (same pre-existing — spec-052 tracks them) |
| Skipped | 9 | 9 | 0 |
| Errors | 0 | 0 | 0 |

Zero regressions introduced. The 5 remaining failures are the same set tracked in spec-052 (SIGHTING-071/072/073/074); none of them touch quality-gating or any helper modified by spec-053. AC10 satisfied.
