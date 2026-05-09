# Tasks: Interactive Merge Approval + ML Merge Classifier

**Plan**: plan.md
**Created**: 2026-04-12

## Phase 1 — Simplify Merger

- [ ] Remove 5 adaptive-threshold fields from `PipelineConfig` in `config.py`
- [ ] Remove `_compute_cluster_thresholds()`, `_compute_global_threshold()`, adaptive branch from `merge.py`
- [ ] Simplify `_evaluate_merge_evidence()` to use fixed `merge_exemplar_threshold`
- [ ] Update `merge_metadata` output (drop `cluster_thresholds`/`global_threshold`)

## Phase 2 — apply_manual_merges()

- [ ] Implement `apply_manual_merges()` in `merge.py` with union-find transitivity
- [ ] Export `apply_manual_merges` from `face_cluster/__init__.py`

## Phase 3 — MergeDecision Persistence

- [ ] Add `merge_decisions` field to `PipelineResult` in `pipeline.py`
- [ ] Add `save_merge_decisions()` and `load_merge_decisions()` to `export.py`
- [ ] Update `load_pipeline_result()` in `loader.py` to load `merge_decisions.json`

## Phase 4 — Merge Approval UI

- [ ] Pre-populate `merge_approval_decisions` session state from heuristic on run load
- [ ] Add "Show contested only" toggle + sort by ambiguity to `_render_all_decisions_table()`
- [ ] Replace static `action` column with Approve/Reject buttons per row
- [ ] Add `_render_approval_controls()` (bulk buttons, tally, Apply, Save)
- [ ] Wire "Apply" to `apply_manual_merges()`, show inline cluster count + gallery
- [ ] Wire "Save" to `save_merge_decisions()`
- [ ] Remove hard guard on `merged_cluster_result is None`

## Phase 5 — Tests

- [ ] Write `tests/face_clustering/test_merge.py` with `ut_ApplyManualMerges` tests
- [ ] Add `ut_MergeDecisions` tests to `tests/face_clustering/test_export.py`
- [ ] Add `ut_SimplifiedMerger` tests to `tests/face_clustering/test_merge_stage.py`
- [ ] Run full test suite, verify all pass
