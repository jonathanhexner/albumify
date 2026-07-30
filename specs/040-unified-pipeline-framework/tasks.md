# Tasks: Unified Pipeline Framework (040)

> **⚠️ This checklist is the original 2026-05-16 plan; checkboxes were never updated as work shipped. For current phase status see `spec.md`'s Status table and `REVIEW.md`'s findings ledger. Both are authoritative; this file is kept for the original `T0xx` task ids and design notes.**

## Design Notes

- **D1**: Sequential phases. Each phase ends with a green test suite + a checkpoint diff. No phase merges until the prior is stable.
- **D2**: The unified `PipelineContext` is the *Albumify* one, extended — not a new third type. FC App migrates to it.
- **D3**: `face_cluster_bridge` is replaced by real pipeline steps, not by another adapter. Each formerly-bridge concern becomes a named, registered step.
- **D4**: Schema v5 is a one-shot migration. Old DBs read-only; new runs to v5. Don't try to in-place migrate millions of v4 rows.
- **D5**: 2-week burn-in is non-negotiable. The branch lives until burn-in passes.

## Phase 0 (on main — prerequisite)

- [ ] T000 Land spec-035 (FR-033-1 E2E test) on main. Branching for spec-040 only after this is green.
- [ ] T001 Tag main at the pre-unification baseline: `git tag pre-spec-040 main`.
- [ ] T002 Create branch: `git checkout -b unification/spec-040 main`.

**Checkpoint**: branch created from a green-tested main. spec-035 E2E test runs locally in <60s.

## Phase 1 — Unified PipelineContext

- [ ] T010 In `sim_bench/pipeline/context.py`, add fields that exist on `_RunContext` but not `PipelineContext`: `core_indices`, `holdout_indices`, `cluster_result`, `graph_result`, `merged_cluster_result`, `merge_log`, `merge_metadata`, `crop_manifest`, `cap_decisions`, `cap_summary`.
- [ ] T011 Update `specs/034-pipeline-context-contract/spec.md` to include the new fields. Architecture test `test_pipeline_context_contract.py` stays green.
- [ ] T012 Leave `_RunContext` untouched for now (no callers migrated yet); add a deprecation comment.

**Checkpoint**: `PipelineContext` is a superset of both today's types. CI green.

## Phase 2 — Pydantic config for every face-clustering step

- [ ] T020 Inventory face-clustering steps not yet in `STEP_CONFIG_MODELS`: `align_faces`, `detect_face_orientation`, `extract_scene_embedding`, `score_face_*` (every existing scorer that doesn't yet have a typed model), `cluster_scenes`, `cluster_by_identity`, `identity_refinement`, `score_iqa`, `score_ava`, `validate_alignment`, `filter_quality_gate`, `filter_portraits`, `filter_best_faces`.
- [ ] T021 For each, add a Pydantic model under `sim_bench/pipeline/steps/configs/<step_name>.py`. Use `Field(description=...)` per field.
- [ ] T022 Register each in `STEP_CONFIG_MODELS`.
- [ ] T023 Wire `validate_step_config` at the top of each step's `process()` (or rely on the `BaseStep.process` shim for template-method steps).
- [ ] T024 Run `validate_step_config` against every step in `configs/pipeline.yaml` — none should raise.
- [ ] T025 Close FR-033-6 (specs/039) by enabling its CI guard; allowlist should be empty.

**Checkpoint**: every face-clustering step has a Pydantic config; spec-039's `test_registry_covers_face_clustering_steps` passes with no allowlist entries.

## Phase 3 — Replace bridge functions with real steps

- [ ] T030 New step `assemble_face_records` (in `sim_bench/pipeline/steps/`) — reads `context.insightface_faces`, `context.face_embeddings`, and other context fields; writes `context.face_records: List[FaceRecord]`. Replaces `faces_to_face_records()`.
- [ ] T031 New step `quality_gate_faces` — replaces the inline `QualityGater.select_core_set()` call inside `run_face_cluster_knn`; writes `context.core_indices`, `context.holdout_indices`.
- [ ] T032 New step `build_face_knn_graph` — replaces inline `KNNGraphBuilder.build_graph()`; writes `context.graph_result`.
- [ ] T033 New step `cluster_face_components` — replaces inline `ConnectedComponentsClusterer.cluster()`; writes `context.cluster_result`.
- [ ] T034 New step `select_face_exemplars` — replaces inline exemplar selection.
- [ ] T035 New step `merge_face_clusters` (already exists as `cluster_diameter_cap`-adjacent? consolidate) — replaces inline `ConservativeMerger.merge_clusters_with_logging`.
- [ ] T036 Update `default_pipeline` in `configs/pipeline.yaml` to invoke the new steps in order, in place of the monolithic `cluster_people` step.

**Checkpoint**: spec-035 E2E test still passes; `cluster_people` is now a thin sequencer (or removed entirely).

## Phase 4 — Schema v5 migration

- [ ] T040 In `face_cluster/db/schema.py`: bump `SCHEMA_VERSION = 5`.
- [ ] T041 New `images` table DDL: `(image_path PK, image_id, iqa_score, ava_score, sharpness_score, scene_cluster_id, n_faces, created_at)`. Move the 4 image-level columns off `faces`.
- [ ] T042 Add `area_ratio REAL NOT NULL` and `bbox_*_ratio REAL` columns to `faces`. Deprecate (don't remove yet) the unit-mixed `area / bbox_*` columns.
- [ ] T043 Update Pandera schemas: `FACES_SCHEMA` adds `area_ratio` with `Check.in_range(0.0, 1.0)`; new `IMAGES_SCHEMA`.
- [ ] T044 Migration script `scripts/migrate_v4_to_v5.py`: reads a v4 DB, writes a v5 DB. Idempotent. Doesn't delete the v4.
- [ ] T045 `RunExporter` writes both `faces` (with `area_ratio` populated) and `images`. Pandera-validates both.
- [ ] T046 `RunStore.image_detail()` reads from the new `images` table; collapses denormalized fields off `faces`.
- [ ] T047 spec-035 E2E test still green on v5 output.

**Checkpoint**: a fresh run produces v5; legacy v4 runs still readable (read-only); SIGHTING-064/-065 resolved.

## Phase 5 — Retire FaceClusteringPipeline + PipelineConfig

- [ ] T050 New `face_cluster/fc_app_runner.py` (≤200 LOC) — invokes the unified pipeline framework with a face-clustering-only step list. Replaces `face_cluster/pipeline.py::FaceClusteringPipeline`.
- [ ] T051 Migrate every caller of `FaceClusteringPipeline().run(config)` to call the new wrapper.
- [ ] T052 Delete `face_cluster/pipeline.py::FaceClusteringPipeline`, `_RunContext`, and the supporting stage runner.
- [ ] T053 Delete `face_cluster/config.py::PipelineConfig` — callers use per-step Pydantic models.
- [ ] T054 Migrate FC App's profile shape: utility script at `scripts/migrate_fc_profiles.py` re-shapes existing `~/.sim_bench/profiles/*.json` to the unified format.

**Checkpoint**: FC App standalone runs through the unified framework; both E2E tests green.

## Phase 6 — Delete the bridge

- [ ] T060 Delete `sim_bench/pipeline/steps/face_cluster_bridge.py`.
- [ ] T061 New architecture test `test_no_bridge.py` — asserts the file doesn't exist; CI catches any attempted re-introduction.
- [ ] T062 Remove the SIGHTING-061 pin (was already lifted by specs/037; this just verifies).
- [ ] T063 spec-036's `GATE_WAIVERS` should be empty at this point.

**Checkpoint**: no bridge; no waivers; all gates have producers; CI guards prevent regression.

## Phase 7 — Documentation collapse

- [ ] T070 `docs/architecture/db_schemas.html`: collapse the two "Origin" columns into one. Add v5 schema notes; remove obsolete unit-mixed warnings.
- [ ] T071 `docs/architecture/classes.html`: remove `_RunContext`, `FaceClusteringPipeline`, the dataclass `PipelineConfig`. Add the new pipeline steps from Phase 3.
- [ ] T072 `docs/architecture/data_flow.html`: collapse before/after section; the "after" becomes the only flow.
- [ ] T073 `docs/architecture/db_global.html`: update `pipeline_results.fc_export_dir` and `action_log` cross-references; both apps now produce the same artifact layout via the same framework.
- [ ] T074 `docs/architecture/index.html`: archive the spec-033 docs to a "historical" section now that the duplication is gone.
- [ ] T075 spec-033 status updated: `Implemented + superseded by spec-040`.
- [ ] T076 `config_diff` CLI simplified: drop `effective_config_from_albumify` / `effective_config_from_fc_config` — there's one shape.

**Checkpoint**: every architecture HTML reflects the unified state; no "Albumify origin vs FC App origin" columns remain.

## Phase 8 — Burn-in + merge

- [ ] T080 Run unified pipeline on a labeled 100-image album. Save baseline `RunStore.image_detail()` outputs for every image.
- [ ] T081 Pre-merge: compare cluster assignments to pre-spec-040 baseline (the `pre-spec-040` tag). ≥95% agreement is the success bar; lower = file a sighting and don't merge.
- [ ] T082 2-week burn-in: branch runs nightly on the labeled album; alert on any change in cluster count or assignment-agreement metric.
- [ ] T083 Code review (`/code-review` slash command) on the full branch. High-severity findings block.
- [ ] T084 Merge `unification/spec-040` → `main`. Tag `post-spec-040 main`. Delete the branch.

**Checkpoint**: main is on the unified framework; the spec-033 follow-up backlog (FR-033-2/3/4/5/6/7) is either resolved or made moot by the unification.

## Open follow-ups after spec-040

- Remove deprecated `area` / `bbox_*` columns (post-burn-in, one release later).
- Migrate non-face-clustering steps (select_best, cluster_by_identity, etc.) to Pydantic configs — optional, lower priority.
- Re-evaluate whether the global DB's `action_log` and SQLAlchemy `pipeline_runs` should merge (no longer two pipelines to track separately).
