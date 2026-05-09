# Iterative Manual Merge -- Design Discussion Summary

**Date**: 2026-04-17
**Status**: Architecture revised — pending spec-kit and implementation approval

## Starting problem

1. The user did a pipeline run in `D:\sim-bench\results\Austria_24` and it did not appear in the History tab.
2. After clicking "Apply Approved Merges" in Merge Analysis, nothing shows in History and there is no visible "Round N" flow to reconsider new merges on the merged clusters.

## Findings

### Austria_24 missing from History

- `run_history_db.list_actions(types=["pipeline_run", "recluster"])` returned no row for it (verified by `list_actions` inspection).
- Root cause: likely a stale Streamlit process that predated the DB-hook feature in `face_cluster/pipeline.py::_init_context` (which already calls `run_history_db.start_action()` -- see `face_cluster/pipeline.py` L322).
- Backfill: `run_history_db.upsert_run()` inserted the Austria_24 row manually.
- Mistake I made: I also added duplicate `start_action` / `complete_action` / `fail_action` calls in `app/face_clustering.py` around `pipeline_worker` and `recluster_worker`. These duplicate the existing pipeline hooks and must be reverted.

### Manual merge history and re-merge flow

**Current state:**

- `apply_manual_merges()` in `face_cluster/merge.py` L695 is in-memory only; it returns a new `ClusterResult` but never writes to disk.
- The Merge Analysis Apply handlers (`app/face_clustering.py` L2447, L2980) do write a `merge_apply` action row to the DB, but its payload only contains `round`, `n_approved`, `n_rejected`, `clusters_before` -- no thresholds, no approved/rejected pair lists.
- An in-memory iterative loop exists: after Apply, `propose_merge_candidates(manual_result, dm, threshold)` populates `st.session_state.pending_candidates` and `_render_next_round_section` (L2905) renders a "Round N Candidates" block. This is invisible when the pending list is empty and sits at the bottom of a long tab when non-empty.
- `pipeline.recluster()` runs `cluster -> exemplars -> merge -> export`; kNN re-clustering discards any manual merges.

## Requirement (final, from user)

Clicking "Apply Approved Merges" must behave like a new clustering step:

1. **Persist** the manually merged state as a new run on disk (new folder under `results/`) and as a new row in the History Pipeline Runs table.
2. **Re-run the merge stage** (not clustering) on top of the new snapshot, using current thresholds, so new candidate pairs among the merged clusters are proposed automatically.
3. **Open Merge Analysis on the new snapshot**: previously approved pairs are no longer relevant, because the new snapshot has fresh cluster IDs and only the new pair decisions matter.
4. The user can repeat Apply -> new snapshot -> new Merge Analysis indefinitely, stopping when the merge stage finds no more candidates.

This is symmetric with what happens after base clustering today:

```
pipeline.run() -> base clusters -> auto-merge -> merged clusters
   <same structure>
apply_approved -> snapshot (new folder + history row)
              -> auto-merge on snapshot -> new merged clusters
              -> user reviews in Merge Analysis again
```

## Architectural revision (2026-04-17)

During review, the original approach was revised. The discussion exposed a deeper design issue: `recluster()` was an ad-hoc second public method doing what should be a config variation of `run()`. Adding `remerge()` as a third method would compound the problem.

### Agreed architectural pattern

`FaceClusteringPipeline` follows the classic config-driven pipeline pattern:

```python
pipeline = FaceClusteringPipeline()
result = pipeline.run(config)  # single public method
```

`PipelineConfig` is extended to carry `stages` (which stages to run), `source_dir`, and `output_dir` in addition to algorithm parameters. Named **preset factories** cover standard use cases:

```python
PipelineConfig.full_run(source, output, **kwargs)
# stages: discover → embed → quality → crops → cluster → exemplars → merge → export

PipelineConfig.recluster(source, output, **kwargs)
# stages: cluster → exemplars → merge → export
# source loader: reads crops + embeddings from previous run

PipelineConfig.remerge(source, output, **kwargs)
# stages: merge → export
# source loader: reads cluster snapshot (post manual-merge)
```

Each preset has a **source loader** that populates `_RunContext` before stage execution. The loader is determined by which stages are present, not by a mode flag.

`recluster()` as a standalone method is removed and replaced by `PipelineConfig.recluster(...)`.

### Remaining agreed work

1. ✅ **Revert duplicate DB hooks** in `app/face_clustering.py` — done.

2. **Refactor `FaceClusteringPipeline`**: add `stages`, `source_dir`, `output_dir` to `PipelineConfig`; implement preset factories; reduce `run()` to a single generic executor; remove `recluster()` method.

3. **`face_cluster/manual_merge_snapshot.py`**: `save_manual_merge_snapshot()` writes a snapshot dir from an in-memory `ClusterResult` after manual merges are applied. Writes `faces.csv`, `clusters.csv`, `embeddings.npy`, `crop_manifest.json` (absolute paths), `pipeline_run.json` with approved/rejected pair lists. Logs a `manual_merge` row to `run_history_db`.

4. **Merge Analysis "Apply" buttons**: call `save_manual_merge_snapshot()`, then `pipeline.run(PipelineConfig.remerge(snapshot_dir, remerge_dir))`, then load the result into `st.session_state.pipeline_result` with cache invalidation. Merge Analysis tab resets to fresh candidates.

5. **History table**: extend `_list_available_runs` to include `manual_merge` and `remerge` action types with `source_type` sub-label.

## Out of scope

- Changing `propose_merge_candidates` / `apply_manual_merges` internals.
- The in-memory `_render_next_round_section` loop -- superseded by the snapshot + remerge flow.
- Any ML work.

## Tests

- `tests/face_clustering/test_manual_merge_snapshot.py` -- snapshot file schema and DB row.
- `tests/face_clustering/test_pipeline_remerge.py` -- remerge skips clustering and produces `faces_merged.csv`.
- `tests/face_clustering/test_history_includes_manual_merge.py` -- `_list_available_runs` returns the new types.
- Integration: pipeline.run -> manual apply -> snapshot -> remerge -> new candidates differ from old, previously approved pairs absent.
