# Data Model: Config-Driven Pipeline + Iterative Manual Merge

**Date**: 2026-04-17

## Entities

### PipelineConfig (extended)

Existing fields unchanged. New fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `stages` | `Optional[list[str]]` | `None` | Ordered stage names to execute. `None` = all stages (full run). |
| `source_dir` | `Optional[str]` | `None` | Input directory (raw images, previous run, or snapshot). |
| `output_dir` | `Optional[str]` | `None` | Output directory for this run. |
| `on_progress` | `Optional[Callable]` | `None` | Progress callback. Not serialized. |

Factory classmethods: `full_run()`, `recluster()`, `remerge()`.

### ManualMergeSnapshot (directory on disk)

Not a Python class — a directory written by `save_manual_merge_snapshot()`.

| File | Schema | Source |
|------|--------|--------|
| `faces.csv` | Same as `export.py` output — `cluster_id` reflects post-merge state | Updated from `merged_cluster_result` |
| `clusters.csv` | Same as `export.py` output | From `merged_cluster_result` |
| `embeddings.npy` | float32 (n_faces x 512) | Copied from parent |
| `embedding_face_ids.npy` | int32 (n_faces,) | Copied from parent |
| `crop_manifest.json` | `{face_id: absolute_path}` | Absolute paths to parent's crops |
| `pipeline_run.json` | See below | Written fresh |

#### `pipeline_run.json` for snapshot

```json
{
  "run_id": "20260417_143022_merge_1",
  "source_type": "manual_merge",
  "parent_run_id": "20260417_140000",
  "parent_output_dir": "/path/to/parent",
  "approved_pairs": [[1, 3], [5, 7]],
  "rejected_pairs": [[2, 4]],
  "merge_round": 1,
  "config": { ... },
  "started_at": "2026-04-17T14:30:22",
  "status": "complete",
  "stages": {}
}
```

### Run history DB row

Existing `run_history_db` schema unchanged. New `action_type` values:

| action_type | When created |
|---|---|
| `"pipeline_run"` | Full pipeline run (existing) |
| `"recluster"` | Recluster run (existing) |
| `"manual_merge"` | Snapshot saved after Apply (new) |
| `"remerge"` | Pipeline remerge run (new) |

## Relationships

```
full_run (pipeline_run)
    |
    +-- recluster (different params, same images)
    |       |
    |       +-- manual_merge snapshot
    |               |
    |               +-- remerge
    |                       |
    |                       +-- manual_merge snapshot (round 2)
    |                               |
    |                               +-- remerge (round 2)
    |                                       ...
    +-- manual_merge snapshot (from full run directly)
            |
            +-- remerge
                    ...
```

Each node produces an output directory. Parent-child relationship tracked via `parent_output_dir` in `pipeline_run.json`.
