# Feature Specification: Unified Pipeline Framework

**Created**: 2026-05-16
**Status**: In Progress — Phases 0 through 6 complete; Phase 7 (legacy retirement) pending 2-week equivalence-test burn-in; Phase 8 (final doc collapse) deferred until Phase 7 lands.
**Resolves**: the structural root cause behind SIGHTING-058 / -059 / -060 / -061 / -062 / -064 / -065 / -066 and the spec-033 follow-up backlog
**Branch**: `unification/spec-040` (active — created 2026-05-16 off main `740403d`)

## Status table (2026-05-18)

| Phase | What landed | Status |
|---|---|---|
| 0 | `tests/face_clustering/test_albumify_e2e.py` (FR-033-1) on main | ✅ Done (2026-05-16) |
| 1 | `face_cluster_legacy/` re-export package (virtual rename) | ✅ Done |
| 2 | 16 Pydantic step configs registered in `STEP_CONFIG_MODELS` | ✅ Done |
| 3 | 8 unified clustering steps on `context.face_records: List[FaceRecord]` | ✅ Done |
| 4 | Schema v5: `images`, `scene_clusters`, `scene_cluster_assignments` tables + `area_ratio` / `bbox_*_ratio` columns + Pandera schemas | ✅ Done |
| 5 | `face_cluster/fc_app_runner.py` — thin runner over the unified framework | ✅ Done |
| 6 | `tests/face_clustering/test_legacy_vs_v2_equivalence.py` — ≥95% pairwise agreement gate | ✅ Done (passes on synthetic fixture) |
| 7 | Delete `face_cluster_legacy/`, `face_cluster_bridge.py`, `app/face_clustering_legacy/` | ⏳ Pending 2-week burn-in |
| 8 | Final doc collapse (drop legacy vs v2 comparison columns from `db_schemas.html` / `classes.html`) | ⏳ After Phase 7 |

**Companion docs in this dir**:
- `spec.md` (this file) — what & why
- `tasks.md` — phase-by-phase checklist
- `CONCRETE_PLAN.md` — file-level migration (target architecture, per-phase file changes, test gates, rollback notes, risk register, open questions)
- `COVERAGE.md` — cross-reference: which other PRDs / sightings / open items this spec absorbs or makes moot

## Problem Statement

sim-bench has **two pipeline frameworks** that produce face-clustering output:

1. **Albumify** — `sim_bench/pipeline/` with `PipelineContext` (a ~30-field dataclass holding nested dicts) and steps that read/write that context.
2. **FC App standalone** — `face_cluster/pipeline.py` with `_RunContext` (a 17-field dataclass) and a hand-written stage runner.

They share the **clustering engine** (`face_cluster/{knn_graph, clustering, exemplars, merge, quality}.py`) but reach it through completely different paths. `face_cluster_bridge.py` is the adapter that converts Albumify's nested context dicts into face_cluster's `FaceRecord` and `PipelineConfig`.

Every problem the spec-033 refactor surfaced traces back to this dual-framework structure:

- **SIGHTING-058** — JSON+SQLite drift because two writers existed.
- **SIGHTING-059** — bridge dropped 5 fields; gates force-disabled.
- **SIGHTING-060** — same column (`face.area`) in different units depending on which producer wrote it.
- **SIGHTING-061** — bridge unblocked a gate against data the Albumify path doesn't compute.
- **SIGHTING-062** — two write paths producing `filter_decisions` rows on Albumify because the bridge connects them.
- **SIGHTING-064 / -065** — DB columns whose origin differs between the two pipelines.
- **spec-033 P-F** — config_diff CLI exists *because* the two pipelines have different config shapes.

Spec-033 added boundary contracts (Pydantic, Pandera). They catch shape mismatches but not graph mismatches. As long as two frameworks exist, every contract needs to be enforced on both sides, every signal needs a producer on both sides, and the bridge keeps being the silent gap.

**This spec collapses the two pipelines into one.** The output is one framework, one context type, one config schema, one step abstraction. The bridge is deleted. FC App standalone becomes a thin wrapper that drives the same framework with a different list of steps.

## User Stories

### US1 — One context type (Priority: P1)
As a step author, I want one place to read from and one place to write to, so that "where does field X come from" has one answer regardless of which app invokes the pipeline.

**Acceptance Criteria**:
- A single `PipelineContext` dataclass replaces both today's `PipelineContext` and `_RunContext`. spec-034's field contract extends to the unified type.
- No code path constructs both types; the legacy `_RunContext` is removed.
- Architecture test `test_one_context_type.py` asserts the union of context types in the codebase is `{PipelineContext}`.

### US2 — One config schema (Priority: P1)
As a UI widget, I want my value to flow to exactly one config field that exactly one step reads, so that profile load/save, config diff, and validation all work the same way on both apps.

**Acceptance Criteria**:
- Every face-clustering step (filter_*, cluster_people, score_*, align_*, insightface_*, extract_*) has a Pydantic `BaseModel` config with `extra="forbid"` (extending spec-033 P-G to every step, closing FR-033-6).
- The FC App's `face_cluster/config.py::PipelineConfig` dataclass is removed; FC App invocations construct the same per-step models the Albumify path uses.
- `config_diff` CLI loses its `effective_config_from_albumify` / `effective_config_from_fc_config` split — there's one canonical shape.

### US3 — One step abstraction (Priority: P1)
As an engineer adding a step, I want one base class, one registry, one set of conventions, so that a step works identically when invoked by either app.

**Acceptance Criteria**:
- Both apps invoke steps via `sim_bench.pipeline.registry`. The FC App's hand-written stage runner in `face_cluster/pipeline.py::FaceClusteringPipeline` is removed; FC App becomes a wrapper that calls `PipelineService` (or equivalent) with a face-clustering-only step list.
- `face_cluster/pipeline.py` shrinks dramatically — clustering logic stays, the framework code goes.

### US4 — Bridge deleted (Priority: P1)
As a debugger, I want no adapter between the two halves of the system, because the adapter is where bugs live.

**Acceptance Criteria**:
- `sim_bench/pipeline/steps/face_cluster_bridge.py` is deleted.
- The functions inside it (`faces_to_face_records`, `build_fc_config`, `run_face_cluster_knn`) are absorbed into proper pipeline steps that operate directly on the unified context.
- Architecture test ensures the file does not return.

### US5 — Single FaceRecord origin (Priority: P1)
As `db_schemas.html` reader, I want each DB column to have one "Origin" entry, not "Albumify origin" vs "FC App origin".

**Acceptance Criteria**:
- All FaceRecord fields are populated by the same producer step regardless of which app invokes the pipeline (e.g., `blur_score` comes from `insightface_score_blur` step on both paths).
- `db_schemas.html` collapses its two-origin column back to one.
- SIGHTING-060 / -064 / -065 resolved as part of this work: `area_ratio`, dedicated `images` table, single unit convention.

### US6 — Both apps still work end-to-end (Priority: P1, regression-only)
Albumify's `default_pipeline` and the FC App's standalone run both produce the same artifacts they do today (or strictly more, never less), with the same observable quality.

**Acceptance Criteria**:
- spec-035's E2E test (Albumify on a 5-image fixture) passes before AND after the refactor.
- A new FC App E2E test asserts standalone run produces identical output for the same input (modulo run_id / timestamps).
- `RunStore.image_detail(path)` on a unified-pipeline run is non-empty for any face-bearing image.

## Edge Cases

- **Steps unique to one app** — e.g., `select_best` is Albumify-only; `manual_merge` is FC-App-only. Both apps share the framework but assemble different step lists; that's expected.
- **Identity refinement** — currently invoked only on Albumify path. After unification, FC App can opt in if desired.
- **Caching layer** — `universal_cache` is keyed by `(image_path, feature_type, model_name)`. Unification keeps the same cache; cache keys don't change.
- **Legacy v4 DBs** — schema gets a v5 bump (collapsing two-origin columns, adding `area_ratio`, splitting `images` out). One-shot migration script for existing runs.
- **In-flight runs** — refactor doesn't preserve in-flight runs across the migration; document that scheduled runs must complete or be re-started.

## Requirements (brief)

- FR-001: Single `PipelineContext` definition; `_RunContext` removed.
- FR-002: Every face-clustering step has a Pydantic config model registered in `STEP_CONFIG_MODELS`. Closes FR-033-6.
- FR-003: `face_cluster_bridge.py` deleted; its responsibilities absorbed into named steps.
- FR-004: FC App standalone is a thin wrapper over the unified framework (≤200 LOC of glue + a step list).
- FR-005: `face_cluster/config.py::PipelineConfig` dataclass removed; callers migrated to per-step Pydantic models.
- FR-006: `face_cluster/pipeline.py` shrinks; framework code removed, clustering logic kept.
- FR-007: DB schema v5: collapse two-origin columns to one origin; add `images` table (SIGHTING-065); add `*_ratio` columns (SIGHTING-064); one-shot migration script.
- FR-008: `db_schemas.html` collapses to single-origin column.
- FR-009: spec-035's Albumify E2E test passes throughout. New FC App E2E test added; both green at the end.
- FR-010: spec-036's config-to-producer contract still passes (with the waiver list empty — every gate has a producer in the unified pipeline, including blur via specs/037).
- FR-011: Branch-merge gate: `unification/spec-040` does not merge to main until both E2E tests green and a 2-week burn-in on at least one real album.

## Non-Goals

- **Not rewriting the clustering algorithm.** `face_cluster/{knn_graph, clustering, exemplars, merge, quality}.py` stay as-is. This is a framework refactor, not an algorithm refactor.
- **Not changing pipeline-yaml format.** Step list ordering, config keys, and defaults stay compatible.
- **Not migrating non-face-clustering steps** (e.g., `select_best`, `cluster_by_identity`) — they continue to use whatever pattern they had; only the face-clustering subset is unified.
- **Not changing the UI surface.** Tooltips, sliders, profile load/save stay where they are; their backing store consolidates.

## Locked architectural constraints

- **NO bridge / adapter / translator classes in the final architecture.** Producers write Pydantic objects directly onto context. Consumers read those same Pydantic objects. The only translation in the entire pipeline is the Pydantic-object → DataFrame → SQL row chain at write time (and its reverse at read time), with Pandera as the single validation hop. Specifically:
  - **`face_cluster_bridge.py` is deleted** (Phase 7 — already in the plan).
  - **No `assemble_face_records` step.** The plan previously had this as a "convert insightface_faces dict to FaceRecord list" translator step. Removed. Instead, `insightface_detect_faces` writes `context.face_records: List[FaceRecord]` directly.
  - **No `context.insightface_faces` dict-of-dicts.** Replaced by `context.face_records: List[FaceRecord]`. Scoring steps mutate Pydantic attributes, not dict keys.
  - **No `context.face_embeddings` dict.** Replaced by `face.embedding` attribute on each `FaceRecord` (set by `extract_face_embeddings` directly on the existing records).
  - **Same rule for the scene side**: `cluster_scenes` writes `context.scene_clusters: List[SceneClusterRecord]` directly. No `context.scene_clusters` dict-of-lists or separate `scene_cluster_labels` dict.
- **One canonical Python representation per concept.** `FaceRecord`, `ImageRecord`, `SceneClusterRecord` are the only face/image/scene shapes that exist mid-pipeline. The DB row is a direct serialization of those shapes (Pandera-validated).
- **Cross-package imports of shared types are OK.** `sim_bench/pipeline/steps/insightface_detect_faces.py` may import from `face_cluster.types` — `face_cluster.types` is the shared contract module, not a layer to be hidden.

## Risks

- **Blast radius is large.** Every face-clustering step is touched. Mitigation: lockstep with the FR-033-1 E2E test (must land on main first; serves as regression net throughout).
- **Schema v5 migration is one-shot.** Mitigation: write the migration script + a "before/after" assertion test on a snapshot DB.
- **FC App users with saved profiles** — profile shape changes. Mitigation: a profile-migration utility that re-shapes existing JSONs at first load.
- **Hidden coupling** discovered mid-refactor. Mitigation: 2-week timebox per phase; if a phase blows scope, file a sighting and reduce scope.

## Sequencing (high-level — full plan in tasks.md)

1. **Phase 0** — Land FR-033-1 E2E test on main. Block on it.
2. **Phase 1** — Branch `unification/spec-040`. Define unified `PipelineContext` + add to spec-034. No code migrations yet — just the type.
3. **Phase 2** — Migrate every face-clustering step to Pydantic config (extends spec-033 P-G to the long tail). Closes FR-033-6 in passing.
4. **Phase 3** — Replace `face_cluster_bridge.faces_to_face_records` with proper named pipeline steps. Bridge file shrinks.
5. **Phase 4** — Schema v5 migration: collapse origins, add `images` table, add `*_ratio` columns. One-shot script + Pandera updates.
6. **Phase 5** — Remove `face_cluster/pipeline.py::FaceClusteringPipeline` and `face_cluster/config.py::PipelineConfig`. FC App becomes a wrapper.
7. **Phase 6** — Delete the bridge file. Architecture test prevents its return.
8. **Phase 7** — `db_schemas.html` and `classes.html` collapse two-origin columns. `config_diff` simplified. Docs reviewed.
9. **Phase 8** — 2-week burn-in on a real album. Then merge to main.

## Open Questions

- [NEEDS CLARIFICATION] Does the FC App keep its own Streamlit UI, or does it become a "face-clustering mode" toggle inside the Albumify Streamlit app? Recommendation: keep separate UIs (different audiences), share backend.
- [NEEDS CLARIFICATION] Schema v5 migration: do we re-run old DBs through a migrator, or treat them as read-only (with new runs going to v5)? Recommendation: read-only legacy; new runs to v5. Cheaper, safer.
- [NEEDS CLARIFICATION] What's the burn-in success criterion? Recommendation: zero unhandled errors + no quality regression on a manually-labeled 100-image album, comparing pre- and post-merge `RunStore.image_detail()` outputs for ≥95% agreement on cluster assignments.

## References

- spec-033 master plan and follow-ups: `specs/033-data-integrity/MASTER_PLAN.md`, `REVIEW.md`, `FOLLOW_UPS_ROADMAP.html`
- spec-034 pipeline context contract: `specs/034-pipeline-context-contract/spec.md`
- spec-035 E2E acceptance test (Phase 0 prerequisite): `specs/035-albumify-e2e-acceptance/spec.md`
- spec-036 config-to-producer contract: `specs/036-config-producer-contract/spec.md`
- spec-037 InsightFace blur step (resolves SIGHTING-061): `specs/037-insightface-blur-step/spec.md`
- spec-038 ExportRequest Pydantic: `specs/038-export-request-pydantic/spec.md`
- spec-039 step config registry guard: `specs/039-step-config-registry-guard/spec.md`
