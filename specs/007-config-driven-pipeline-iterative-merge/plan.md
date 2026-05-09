# Implementation Plan: Config-Driven Pipeline + Iterative Manual Merge

**Branch**: `007-config-driven-pipeline-iterative-merge` | **Date**: 2026-04-17 | **Spec**: [spec.md](spec.md)

## Summary

Refactor `FaceClusteringPipeline` from a class with multiple ad-hoc methods (`run`, `recluster`) into a single generic stage executor driven entirely by `PipelineConfig`. Then use the new architecture to implement iterative manual merge: Apply saves a snapshot, pipeline runs the remerge preset, and the Merge Analysis tab resets to fresh candidates.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: Streamlit (UI), NumPy, pandas (data), InsightFace (embeddings)
**Storage**: SQLite (`run_history_db`), filesystem (output directories with CSV/NPY/JSON)
**Testing**: pytest (existing test suite under `tests/face_clustering/`)
**Target Platform**: Windows 10 (primary), should also work on Linux/macOS
**Project Type**: Desktop application (Streamlit) + library (`face_cluster/`)
**Constraints**: Non-blocking UI via `_AsyncState` pattern; pipeline runs in background thread

## Constitution Check

No constitution file exists (`.specify/memory/constitution.md` not found). Skipped.

## Project Structure

### Source Code (affected files)

```text
face_cluster/
  config.py              # Add stages, source_dir, output_dir to PipelineConfig; add preset factories
  pipeline.py            # Unify run() + recluster() into single run(config); add source loaders
  manual_merge_snapshot.py  # NEW: save_manual_merge_snapshot()
  loader.py              # Minor: ensure it can load snapshot dirs
  merge.py               # No changes (apply_manual_merges stays as-is)

app/
  face_clustering.py     # Update callers: run() and recluster() -> run(config); Apply -> snapshot + remerge

tests/face_clustering/
  test_merge_stage.py    # Update recluster tests to use preset config
  test_pipeline_history_hook.py  # Update recluster test to use preset config
  test_manual_merge_snapshot.py  # NEW: snapshot writer-reader contract test
  test_pipeline_remerge.py       # NEW: remerge preset E2E test
```

## Design

### Phase 1: Config-driven pipeline refactor

#### 1a. Extend `PipelineConfig`

Add three new fields and preset factory classmethods:

```python
@dataclass
class PipelineConfig:
    # ... existing algorithm fields ...

    # Pipeline execution fields
    stages: Optional[list[str]] = None       # e.g. ["discover", "embed", "quality", ...]
    source_dir: Optional[str] = None         # input directory (images or previous run)
    output_dir: Optional[str] = None         # output directory
    on_progress: Optional[Callable] = None   # progress callback (not serialized)

    # Preset factories
    @classmethod
    def full_run(cls, source_dir, output_dir, **kwargs) -> "PipelineConfig":
        return cls(
            stages=["discover", "embed", "quality", "crops",
                    "cluster", "exemplars", "merge", "export"],
            source_dir=str(source_dir),
            output_dir=str(output_dir),
            **kwargs,
        )

    @classmethod
    def recluster(cls, source_dir, output_dir, **kwargs) -> "PipelineConfig":
        return cls(
            stages=["cluster", "exemplars", "merge", "export"],
            source_dir=str(source_dir),
            output_dir=str(output_dir),
            **kwargs,
        )

    @classmethod
    def remerge(cls, source_dir, output_dir, *, with_exemplars=False, **kwargs) -> "PipelineConfig":
        stages = (["exemplars", "merge", "export"] if with_exemplars
                  else ["merge", "export"])
        return cls(
            stages=stages,
            source_dir=str(source_dir),
            output_dir=str(output_dir),
            **kwargs,
        )
```

`stages` defaults to `None`, which means "full run" (all stages). This preserves backward compatibility for callers that construct `PipelineConfig()` without specifying stages.

#### 1b. Unify `FaceClusteringPipeline`

The class exposes a single public method:

```python
def run(self, config: PipelineConfig) -> PipelineResult:
```

Internally:

1. Resolve effective stage list from `config.stages` (or default `FULL_STAGES`).
2. Determine which **source loader** to call based on the first stage in the list:
   - First stage is `discover` -> no loading; `config.source_dir` is a raw image directory.
   - First stage is `cluster` -> load faces + embeddings from source dir (current recluster logic).
   - First stage is `merge` or `exemplars` -> load faces + embeddings + cluster_result from source dir.
3. Run stages in order via the existing `_execute_stage` loop.
4. Build result, finalize, return `PipelineResult`.

The existing `recluster()` method is deleted. The old `run(image_dir, output_dir, on_progress)` signature is replaced by `run(config)`. All callers are updated in the same change.

`_init_context` takes the config directly instead of separate args. The `mode` string for DB history is derived from the stage list (full stages -> `"pipeline_run"`, recluster stages -> `"recluster"`, merge-only -> `"remerge"`).

#### 1c. Source loaders

Three private methods, selected automatically by a lookup dict keyed on first stage:

```python
_SOURCE_LOADERS = {
    "discover": "_load_source_full_run",
    "cluster":  "_load_source_recluster",
    "exemplars": "_load_source_remerge",
    "merge":    "_load_source_remerge",
}
```

- `_load_source_full_run(ctx)` — no-op (discover populates `ctx.image_paths`).
- `_load_source_recluster(ctx)` — current `recluster()` body: load faces, core/holdout indices, crop manifest, embeddings from source dir.
- `_load_source_remerge(ctx)` — everything recluster loads + loads `cluster_result` (merged if available, else base) and `graph_result` (pairwise distances from embeddings).

Each loader validates that required inputs exist and raises a clear error if not.

#### 1d. Update callers

**`app/face_clustering.py`**:
```python
# Before: pipeline = FaceClusteringPipeline(config); pipeline.run(image_dir, output_dir)
# After:  pipeline = FaceClusteringPipeline(); pipeline.run(PipelineConfig.full_run(image_dir, output_dir, K=K, ...))

# Before: pipeline.recluster(source, output)
# After:  pipeline.run(PipelineConfig.recluster(source, output, K=K, ...))
```

**Tests** (`test_merge_stage.py`, `test_pipeline_history_hook.py`):
```python
# Before: FaceClusteringPipeline(config).recluster(src, out)
# After:  FaceClusteringPipeline().run(PipelineConfig.recluster(src, out, **config_kwargs))
```

### Phase 2: Manual merge snapshot

#### 2a. `face_cluster/manual_merge_snapshot.py`

New module with a single public function:

```python
def save_manual_merge_snapshot(
    parent_result: PipelineResult,
    merged_cluster_result: ClusterResult,
    approved_pairs: list[tuple[int, int]],
    rejected_pairs: list[tuple[int, int]],
    config: PipelineConfig,
    output_dir: Path,
) -> Path:
```

Writes to `output_dir`:
- `faces.csv` — same schema as `export.py`, with `cluster_id` updated from `merged_cluster_result`
- `clusters.csv` — from `merged_cluster_result`
- `embeddings.npy` + `embedding_face_ids.npy` — copied from parent
- `crop_manifest.json` — absolute paths to parent's crops
- `pipeline_run.json` — metadata: `source_type: "manual_merge"`, parent `run_id`, approved/rejected pair lists, thresholds, timestamp

Logs a `manual_merge` row to `run_history_db`. Returns `output_dir`.

#### 2b. Remerge source loader

`_load_source_remerge(ctx)`:
1. Calls `load_pipeline_result(source_dir)` to get faces + embeddings.
2. Loads `cluster_result` (merged if present, else base) from the source — this is the post-manual-merge state.
3. Derives `core_indices` and `holdout_indices` from `face.is_core`.
4. Loads `crop_manifest` with absolute paths.
5. Builds `graph_result` from embeddings (pairwise distance matrix) — needed by merge stage.

If `exemplars` is in the stage list, exemplar selection re-runs before merge. Otherwise, merge uses the exemplars already in the loaded `cluster_result`.

### Phase 3: Apply flow in the app

#### 3a. Wire up Apply button

In `app/face_clustering.py`, the "Apply Approved Merges" handler becomes:

```python
# 1. Apply merges in memory (existing call)
manual_result = apply_manual_merges(cr, approved_pairs, dm)

# 2. Save snapshot to disk
snapshot_dir = results_root / f"{run_id}_merge_{round_num}"
save_manual_merge_snapshot(result, manual_result, approved, rejected, config, snapshot_dir)

# 3. Run remerge pipeline (async via _AsyncState)
remerge_config = PipelineConfig.remerge(
    snapshot_dir,
    results_root / f"{run_id}_remerge_{round_num}",
    with_exemplars=st.session_state.get("remerge_with_exemplars", False),
    merge_enabled=True,
    # inherit other params from current config
)
worker = _AsyncState()
worker.start(lambda: FaceClusteringPipeline().run(remerge_config))
```

When the worker completes, `st.session_state.pipeline_result` is updated, `_invalidate_run_caches()` is called, and merge-related session state keys are cleared. The Merge Analysis tab renders from the new result with fresh candidates.

#### 3b. Checkbox for exemplar re-selection

A single checkbox in the Apply section:

```python
st.checkbox("Re-run exemplar selection before merge", key="remerge_with_exemplars")
```

When checked, `PipelineConfig.remerge(..., with_exemplars=True)` is used, adding `exemplars` to the stage list.

### Phase 4: History tab

`_list_available_runs()` currently filters on `types=["pipeline_run", "recluster"]`. Extend to include `"manual_merge"` and `"remerge"`. Add a `type` column to the displayed table showing the run type.

No other changes needed — `load_pipeline_result` already handles any directory with `faces.csv` + `pipeline_run.json`.

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Remap bugs when loading snapshot for remerge | Snapshot stores face-index-level data (not graph-local), so no remap needed. Contract test verifies. |
| Apply button blocks UI | Uses existing `_AsyncState` pattern — runs in background thread, UI polls via `st.rerun()`. |
| Missing embeddings or graph_result in snapshot | Source loader validates required files before any stage executes. |
| Old tests break when `recluster()` is removed | All callers updated in same change. Grep for `recluster(` confirms no remaining uses. |
| Stale Streamlit widget keys after Apply | Apply handler calls `_invalidate_run_caches()` and clears merge-related session state keys. |
| `on_progress` callback leaks into serialized config | Exclude from `pipeline_run.json` serialization (skip callable fields). |

## Complexity Tracking

No constitution violations to justify.
