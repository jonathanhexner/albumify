# Tasks: Config-Driven Pipeline + Iterative Manual Merge

**Feature**: 007-config-driven-pipeline-iterative-merge
**Date**: 2026-04-17
**Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md)

---

## Phase 1: Setup

*(No project initialization needed — all modules exist. Proceed to Foundational.)*

---

## Phase 2: Foundational

Prerequisites shared by all user stories. Must complete before any US phase begins.

- [ ] T001 Read `face_cluster/types.py` to confirm `GraphResult` fields (especially `distance_matrix`, `edges`, `adjacency`) before implementing source loaders in `face_cluster/pipeline.py`
- [ ] T002 Grep all callers of `FaceClusteringPipeline.run(image_dir, output_dir)` and `pipeline.recluster(` across `app/` and `tests/` to produce a complete list before changing the signature in `face_cluster/pipeline.py`

---

## Phase 3: User Story 1 — Config-Driven Pipeline (P1)

**Goal**: Single `pipeline.run(config)` method driven by `PipelineConfig`. Presets cover full run, recluster, remerge.

**Independent test**: `pipeline.run(PipelineConfig.full_run(...))`, `pipeline.run(PipelineConfig.recluster(...))`, and `pipeline.run(PipelineConfig.remerge(...))` each produce a valid output directory. All existing tests pass.

- [ ] T003 [US1] Add `stages`, `source_dir`, `output_dir`, `on_progress` fields to `PipelineConfig` dataclass in `face_cluster/config.py`
- [ ] T004 [US1] Add `full_run()`, `recluster()`, `remerge()` classmethod preset factories to `PipelineConfig` in `face_cluster/config.py`
- [ ] T005 [US1] Update `PipelineConfig.__post_init__` validation to skip new fields and fix config serialization in `_init_context` to exclude non-serializable fields (`on_progress`) in `face_cluster/pipeline.py`
- [ ] T006 [US1] Add `_load_source_full_run()` (no-op), `_load_source_recluster()` (current `recluster()` body), and `_load_source_remerge()` (loads faces + embeddings + cluster_result + reconstructs GraphResult) private methods to `FaceClusteringPipeline` in `face_cluster/pipeline.py`
- [ ] T007 [US1] Refactor `FaceClusteringPipeline.run()` to accept a single `PipelineConfig` argument: resolve stage list from `config.stages`, dispatch to the correct source loader based on first stage, derive DB `mode` string from stage list, then execute stages in order in `face_cluster/pipeline.py`
- [ ] T008 [US1] Delete `FaceClusteringPipeline.recluster()` method and `RECLUSTER_STAGES` class attribute in `face_cluster/pipeline.py`
- [ ] T009 [P] [US1] Update `app/face_clustering.py` `_run_pipeline()` to call `pipeline.run(PipelineConfig.full_run(image_dir, output_dir, K=K, ...))` in `app/face_clustering.py`
- [ ] T010 [P] [US1] Update `app/face_clustering.py` `_run_recluster()` to call `pipeline.run(PipelineConfig.recluster(source, output, K=K, ...))` in `app/face_clustering.py`
- [ ] T011 [P] [US1] Update all `pipeline.recluster(...)` calls in `tests/face_clustering/test_merge_stage.py` to use `pipeline.run(PipelineConfig.recluster(...))` in `tests/face_clustering/test_merge_stage.py`
- [ ] T012 [P] [US1] Update `pipeline.recluster(...)` call in `tests/face_clustering/test_pipeline_history_hook.py` to use `pipeline.run(PipelineConfig.recluster(...))` in `tests/face_clustering/test_pipeline_history_hook.py`
- [ ] T013 [US1] Run `.venv/Scripts/python -m pytest tests/face_clustering/ -v` and fix any failures before proceeding

---

## Phase 4: User Story 2 — Iterative Manual Merge (P2)

**Goal**: Clicking Apply saves a snapshot, runs remerge, Merge Analysis resets to fresh candidates.

**Independent test**: After clicking Apply with approved pairs, a new run appears in History and the Merge Analysis tab shows candidates from the new run only.

**Depends on**: Phase 3 complete (remerge preset must exist).

- [ ] T014 [US2] Create `face_cluster/manual_merge_snapshot.py` with `save_manual_merge_snapshot(parent_result, merged_cluster_result, approved_pairs, rejected_pairs, config, output_dir)` that writes `faces.csv`, `clusters.csv`, `embeddings.npy`, `embedding_face_ids.npy`, `crop_manifest.json`, `pipeline_run.json` and logs a `manual_merge` DB row in `face_cluster/manual_merge_snapshot.py`
- [ ] T015 [US2] Write `tests/face_clustering/test_manual_merge_snapshot.py`: run `save_manual_merge_snapshot()` in `tmp_path`, then call `load_pipeline_result()` on the output, assert field types and values match (writer-reader contract test) in `tests/face_clustering/test_manual_merge_snapshot.py`
- [ ] T016 [US2] Implement `_load_source_remerge()` in `FaceClusteringPipeline`: load `PipelineResult` via `load_pipeline_result`, derive `core_indices` from `face.is_core`, load crop manifest as absolute paths, build `GraphResult` with pairwise cosine distance matrix from embeddings, load merged cluster_result (or base if no merged) in `face_cluster/pipeline.py`
- [ ] T017 [US2] Write `tests/face_clustering/test_pipeline_remerge.py`: full run → save snapshot → `pipeline.run(PipelineConfig.remerge(...))` → assert `faces_merged.csv` exists and cluster count is valid in `tests/face_clustering/test_pipeline_remerge.py`
- [ ] T018 [US2] Add `save_manual_merge_snapshot` to `face_cluster/__init__.py` exports in `face_cluster/__init__.py`
- [ ] T019 [US2] Replace the Apply Approved Merges handler body in `app/face_clustering.py`: call `save_manual_merge_snapshot()`, then start an `_AsyncState` worker that calls `pipeline.run(PipelineConfig.remerge(...))`, then on worker completion update `st.session_state.pipeline_result` and call `_invalidate_run_caches()` in `app/face_clustering.py`
- [ ] T020 [US2] Add "Re-run exemplar selection before merge" checkbox (key `remerge_with_exemplars`) to the Apply section in `app/face_clustering.py`; pass `with_exemplars=st.session_state.remerge_with_exemplars` to `PipelineConfig.remerge()` in `app/face_clustering.py`
- [ ] T021 [US2] On remerge completion, clear merge-related session state keys (pair decisions, pending candidates, approved/rejected maps) before re-rendering Merge Analysis in `app/face_clustering.py`
- [ ] T022 [US2] Add "No further merges possible" message to Merge Analysis tab when remerge result has zero merge candidates in `app/face_clustering.py`
- [ ] T023 [US2] Remove `_render_next_round_section` function and all call sites (superseded by snapshot+remerge flow) from `app/face_clustering.py`

---

## Phase 5: User Story 3 — History Includes All Run Types (P3)

**Goal**: History tab shows manual_merge and remerge runs alongside full runs and reclusters.

**Independent test**: After an iterative merge cycle, the History tab lists all four run types with distinct labels.

**Depends on**: Phase 4 complete (manual_merge and remerge rows must exist in DB).

- [ ] T024 [US3] Extend `_list_available_runs()` to include `"manual_merge"` and `"remerge"` in the `types=` filter in `app/face_clustering.py`
- [ ] T025 [US3] Add a `type` column to the History tab table displaying the `action_type` value with a short human-readable label (e.g. "Manual Merge", "Remerge") in `app/face_clustering.py`

---

## Phase 6: Polish

- [ ] T026 Run `.venv/Scripts/python -m py_compile` on all modified files and fix any syntax errors
- [ ] T027 Run `.venv/Scripts/python -m pytest tests/face_clustering/ -v` on Windows and confirm no `charmap` codec errors (ASCII-only output in all new code paths)
- [ ] T028 Update `docs/FEATURE_REQUESTS.md`: mark "Iterative Manual Merge Re-evaluation" as Done
- [ ] T029 Append `[FEATURE]` entry to `CHANGES_LOG.md` covering all changes made

---

## Dependencies

```
T001, T002 (Foundational)
    |
T003 -> T004 -> T005 -> T006 -> T007 -> T008 (pipeline refactor, sequential)
                                    |
                        T009, T010, T011, T012 [P] (caller updates, parallel)
                                    |
                                  T013 (test gate)
                                    |
                T014 -> T015 [P]  (snapshot writer + test, parallel after T014)
                T016 -> T017 [P]  (remerge loader + test, parallel after T016)
                T018              (exports)
                    |
                T019 -> T020 -> T021 -> T022 -> T023 (app Apply flow, sequential)
                    |
                T024 -> T025 (History tab, sequential)
                    |
            T026, T027, T028, T029 [P] (Polish, parallel)
```

## Parallel Opportunities

- T009, T010, T011, T012 — app and test caller updates are all independent file changes
- T015 and T017 — snapshot test and remerge test are independent once T014 and T016 exist
- T026, T027, T028, T029 — Polish tasks are all independent

## Implementation Strategy

**MVP (US1 only)**: Complete T001-T013. Delivers a clean, unified pipeline API. Existing functionality preserved.

**Full delivery**: Complete all phases. Each phase is independently testable and adds visible user value.
