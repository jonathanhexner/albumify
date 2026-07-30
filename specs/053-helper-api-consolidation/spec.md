# spec-053 — Helper API consolidation + quality-gate step merge

**Created**: 2026-05-29
**Status**: Implemented
**Predecessors**: spec-040 (unification), spec-048 (data layer); reveals the bug that motivated this work
**Related sightings**: SIGHTING-068 (blur gate not wired in v2 chain)

---

## Problem

Two real symptoms, one root cause.

**Symptom 1 — duplicated step.** Two pipeline steps do nearly the same job with subtly different behavior:

| Step file | Step name | Calls `compute_blur_scores`? | Used by |
|---|---|---|---|
| `sim_bench/pipeline/steps/filter_quality_gate.py` | `filter_quality_gate` | yes | Albumify |
| `sim_bench/pipeline/steps/face_clustering_steps.py::QualityGateFacesStep` | `quality_gate_faces` | **no** | FC App v2 |

The v2 step never calls `compute_blur_scores` before `select_core_set`. Every face arrives with `blur_score = 0.0`, the gate self-disables (logs `"Blur gate: NOT WIRED"`), and any production run with `blur_min > 0` silently has no blur filtering. This matches the user's empirical experience.

**Symptom 2 — helpers don't have a uniform shape.** Each helper class (`QualityGater`, `KNNGraphBuilder`, `ExemplarSelector`, `SimplifiedMerger`, `RunExporter`) has its own multi-stage API. Steps that wrap them have to know the right calling order, the right kwargs, the right intermediate state. Forgetting one stage (Symptom 1) is the failure mode that just bit us.

**Root cause.** Steps are thin wrappers, but the helpers they wrap expose multi-method APIs with implicit ordering constraints. Nothing forces "call X before Y." The two quality-gate steps drifted because the constraint lived in the step author's head.

---

## What we build

A small standard for helpers + apply it.

| Module | Role | New / changed |
|---|---|---|
| `face_cluster/_helper_base.py` | `PipelineHelper[C, R]` protocol: one public entry `calc(config: C) -> R`. Plus generic `HelperConfig` / `HelperResult` base dataclasses. | **NEW** (~40 LOC) |
| `face_cluster/quality.py` | Add `QualityGater.calc(QualityGateConfig) -> QualityGateResult` that composes `compute_blur_scores → compute_pose_scores → select_core_set`. Existing methods kept public. | **CHANGED** (~50 LOC added) |
| `face_cluster/knn_graph.py` (or wherever) | Add `KNNGraphBuilder.calc(KNNGraphConfig) -> KNNGraphResult` (likely a thin rename of `build_graph`). | **CHANGED** (~20 LOC) |
| `face_cluster/exemplars.py` | Same pattern for `ExemplarSelector.calc(...)`. | **CHANGED** (~20 LOC) |
| `face_cluster/merge.py` | Same pattern for `SimplifiedMerger.calc(...)`. | **CHANGED** (~20 LOC) |
| `face_cluster/run_exporter.py` | `RunExporter.calc(RunExporterConfig) -> RunExportResult` wrapping the 14-arg `export()`. | **CHANGED** (~30 LOC) |
| `sim_bench/pipeline/steps/quality_gate.py` | **NEW** consolidated step `quality_gate` (replaces both `filter_quality_gate` and `quality_gate_faces`). Calls `QualityGater.calc(...)`. | **NEW** (~80 LOC) |
| `sim_bench/pipeline/steps/filter_quality_gate.py` | **DELETED** (consolidated). | delete |
| `sim_bench/pipeline/steps/face_clustering_steps.py::QualityGateFacesStep` | **DELETED** (consolidated). | delete inside file |
| `app/face_clustering_v2/...` + Albumify entry points | Update step lists to reference `quality_gate` instead of the two old names. | **CHANGED** (3-5 line edits) |
| `face_cluster/fc_app_runner.py::UNIFIED_CLUSTERING_STEPS` | Replace `quality_gate_faces` with `quality_gate`. | **CHANGED** (1 line) |
| `CLAUDE.md` | Add a "Pipeline step file convention" section that codifies: one step per file, thin wrapper, every domain helper exposes `calc(config) -> result`. | **CHANGED** |

---

## What we don't build

- New helpers. Only existing ones get the `calc()` facade.
- Removal of the existing multi-stage methods. They stay public — some callers (notebooks, cluster analysis) use them directly.
- Caching changes. Cache invalidation rework is a separate concern.
- Pydantic / Pandera validation of `HelperConfig` / `HelperResult` shapes. Plain `@dataclass(frozen=True, slots=True)` for now; typed validation can come later if needed.
- New "helper discovery" registry. The existing `PipelineStep` registry is the discovery surface; helpers are just plain classes.

---

## Locked decisions

1. **Config in `__init__`; per-call inputs to `calc()`; both additive.** Existing helpers already take config in `__init__`; we keep that. `calc(inputs: Inputs) -> Result` is the new public entry point. Existing methods stay public for non-pipeline callers (notebooks, analyses).
2. **`Inputs` and `Result` are small typed dataclasses, NOT `PipelineContext`.** A helper must not depend on the pipeline framework. The step is the translator: context → Inputs, Result → context. Reasons:
   - Notebooks / ad-hoc analyses use helpers without constructing a context.
   - The helper's contract is explicit (Inputs lists exactly what it needs; Result lists exactly what it produces).
   - Framework concerns (context plumbing) live in steps; domain concerns live in helpers; the boundary is the dataclass.
3. **`Inputs` / `Result` are `@dataclass(frozen=True, slots=True)`.** Typed, immutable, grep-able. No kwargs through `calc()`.
4. **One step per file** in `sim_bench/pipeline/steps/`. Files with multiple Step classes get split when touched. (Out of scope beyond the quality-gate split; documented as convention.)
5. **Consolidated step is named `quality_gate`.** Old names `filter_quality_gate` and `quality_gate_faces` are removed and any references updated.
6. **Albumify migration is in-scope.** Both apps must run the consolidated step before this spec is Implemented. No half-migration.
7. **`PipelineHelper` is a Protocol, not an ABC.** Helpers don't need to inherit — structural typing is enough.

---

## Data contracts

```python
# face_cluster/_helper_base.py

from typing import Protocol, TypeVar

I = TypeVar("I")  # Inputs type (per-call data)
R = TypeVar("R")  # Result type

class PipelineHelper(Protocol[I, R]):
    """A domain helper that performs one bounded calculation.

    Convention:
      - Config goes in __init__ (helper instances are bound to a config).
      - Per-call data goes in Inputs (a small typed dataclass).
      - Output is a typed Result dataclass.

    Pipeline steps in sim_bench/pipeline/steps/ are thin wrappers:
        helper = MyHelper(config)            # built once per step run
        result = helper.calc(Inputs(...))    # one entry point
        # write result.* to context

    Steps must NOT call individual helper methods directly — use calc().
    Non-pipeline callers (notebooks, analyses) may still call the
    individual methods for finer control.
    """
    def calc(self, inputs: I) -> R: ...
```

Example for QualityGater:

```python
# face_cluster/quality.py (additions only)

@dataclass(frozen=True, slots=True)
class QualityGateInputs:
    faces: list[FaceRecord]

@dataclass(frozen=True, slots=True)
class QualityGateResult:
    core_indices: list[int]
    holdout_indices: list[int]
    verdicts: list[QualityVerdict]
    faces: list[FaceRecord]  # with blur_score / pose populated

class QualityGater:
    # __init__(config, use_pose_estimation, device) — unchanged
    # compute_blur_scores, compute_pose_scores, select_core_set — unchanged

    def calc(self, inputs: QualityGateInputs) -> QualityGateResult:
        """Single entry point. Composes blur → pose → core-set selection.

        Pipeline steps call this. The individual compute_*/select_core_set
        methods stay public for notebook callers.
        """
        faces = self.compute_blur_scores(inputs.faces)
        if self.use_pose_estimation:
            faces = self.compute_pose_scores(faces)
        core, holdout, verdicts = self.select_core_set(faces)
        return QualityGateResult(
            core_indices=core, holdout_indices=holdout,
            verdicts=verdicts, faces=faces,
        )
```

The step then becomes purely a translator:

```python
# sim_bench/pipeline/steps/quality_gate.py
class QualityGateStep(BaseStep):
    def process(self, context, config):
        fc_cfg = FaceClusterConfig(**config)
        gater = QualityGater(fc_cfg, use_pose_estimation=config.get('use_pose_estimation', False))
        result = gater.calc(QualityGateInputs(faces=context.face_records))
        context.core_indices = result.core_indices
        context.holdout_indices = result.holdout_indices
        context.face_records = result.faces  # blur scores now populated
```

Note the helper never sees `PipelineContext` — it only sees `QualityGateInputs`.

---

## Acceptance criteria

| # | Criterion | How verified |
|---|---|---|
| AC1 | `PipelineHelper` protocol + `HelperConfig`/`HelperResult` base types exist in `face_cluster/_helper_base.py` | Phase 0 — import test |
| AC2 | Each of 5 helpers has a `calc(config) -> result` method that composes the existing pipeline | Phase 1 — per-helper unit test asserting `calc()` output matches the manually-composed sequence |
| AC3 | New `quality_gate` step exists, replaces `filter_quality_gate` AND `quality_gate_faces` | Phase 2 — step registry test + grep guard (no remaining references) |
| AC4 | With `blur_min=50` and real images that have computable blur, faces below threshold ARE held out (the bug the user reported is gone) | Phase 2 — integration test on real fixture |
| AC5 | Albumify pipeline still runs end-to-end through the consolidated step | Phase 3 — existing Albumify E2E green |
| AC6 | FC App v2 chain still runs end-to-end through the consolidated step | Phase 3 — existing fc_app_v2_e2e green |
| AC7 | `UNIFIED_CLUSTERING_STEPS` references `quality_gate` (not the old name) | Phase 3 — grep |
| AC8 | CLAUDE.md has a "Pipeline step file convention" section | Phase 4 — manual review |
| AC9 | Existing 5 helpers' multi-method APIs unchanged (no break in notebook / cluster-analysis callers) | Phase 1 — existing tests for each helper green |
| AC10 | Full `pytest tests/face_clustering tests/architecture` — 0 new failures vs pre-spec-053 baseline | Phase 5 |
| AC11 | `/code-review` produces REVIEW.md with zero high-severity findings | Phase 6 |
| AC12 | Per-helper unit tests exist for all 5 helpers (KNNGraphBuilder + ExemplarSelector get newly-added files) | Phase 1.5 |

---

## Risks

| Risk | Mitigation |
|---|---|
| `RunExporter.export()` has 14 args — the Config dataclass is wide and ugly | Accept it for this spec. The ugliness is in the export contract itself; pinning it in one typed dataclass is still an improvement over scattered kwargs. A separate spec could split RunExporter later. |
| Notebook callers using `QualityGater.compute_blur_scores` directly break | They don't — the existing methods stay public. Only the **steps** are migrated to call `calc()`. |
| Albumify migration breaks something subtle (different `blur_min` semantics, different return shape) | Phase 3 keeps the existing Albumify integration test as the gate. Migration only lands when that test stays green. |
| `quality_gate_faces` rename breaks references in YAML configs, docs, etc. | Grep for `quality_gate_faces` and `filter_quality_gate` in `configs/`, `docs/`, `specs/` — update or document. Phase 2 task. |
| Consolidation reveals a 3rd subtle behavioral difference between the two old steps | Phase 2 explicitly diffs the two step bodies and locks the consolidated behavior with both legacy callers' integration tests. |

---

## Definition of Done

- All 11 acceptance criteria green.
- SIGHTING-068 closed with cross-reference to spec-053.
- Both Albumify and FC App v2 confirmed running the consolidated step on a real-image fixture.
- The user's empirical bug ("setting `blur_min > 0` disables everything in v2") cannot be reproduced.
- CHANGES_LOG entry, REVIEW.md filed, spec status flipped to Implemented.
