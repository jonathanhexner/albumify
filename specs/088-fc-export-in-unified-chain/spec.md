# spec-088 — Wire "Export for analysis" into the unified clustering chain

**Created**: 2026-06-25 · **Status**: Implemented · **Priority**: P1
**Source**: SIGHTING-107 — the Albumify "Export for analysis" toggle is dead; Albumify runs
can never be opened in the Face Clustering app for diagnosis.

## Problem
The FC-analysis export only ever ran inside the deprecated monolithic `cluster_people` step
(`export_for_analysis()` called only at `cluster_people.py:291`; `context.fc_export_dir` set
only at `face_cluster_export.py:159`). spec-079 replaced `cluster_people` with the unified
8-step chain, which has no export. So the toggle is sent but ignored, and every Albumify run
has `fc_export_dir = NULL` (verified on `Budapest2025_Google_run15`).

## What we build
A thin export step at the end of the unified chain that reuses the existing
`export_for_analysis()` helper. All its inputs are already in `context` after the chain:
`face_records`, `core_indices`, `cluster_result`, `merged_cluster_result`, `merge_log`.

1. **New step `face_cluster_analysis_export`** (`sim_bench/pipeline/steps/`, ≤80 LOC per
   spec-053): reads context, builds the config object the helper needs, calls
   `export_for_analysis(...)` only when its `export_for_analysis` config flag is true.
   Sets `context.fc_export_dir`.
2. **Route the flag in `_broadcast_clustering_config`** (NOT via `FCParams` — it's an IO
   concern, and `FCParams` is `extra="forbid"` + parity-tested against `FCConfig`). The
   existing UI already sends `cluster_people.export_for_analysis`; `_broadcast` now copies it
   (plus the `FCParams` dump, so the step can serialize the config) into the
   `face_cluster_analysis_export` step config. **No UI change, no FCParams change.**
3. **Add the step to `configs/pipeline.yaml` `default_pipeline`** after `assign_people_clusters`.

## AC
| # | Criterion | Verified |
|---|---|---|
| 1 | With toggle on, a run writes `results/<album>/face_clustering_<ts>/` incl. `_v4/face_clustering.db` | integration test / manual |
| 2 | `context.fc_export_dir` (→ `pipeline_results.fc_export_dir`) is set (not NULL) | unit/integration test |
| 3 | The export opens in the FC app loader (`face_cluster.loader.load_pipeline_result`) | test loads the written dir |
| 4 | Toggle off → no export, `fc_export_dir` stays None (no perf cost) | unit test |
| 5 | Adding the step does NOT change clustering output (export is read-only) | budapest e2e / cluster-count unchanged |
| 6 | `_broadcast` routes the flag (+params) to the export step when on; not when off | unit test ✓ |

## Notes / risks
- `export_for_analysis()` serializes the config (`pipeline_run.json`); `FCParams` is Pydantic,
  the helper expected a dataclass-ish `fc_cfg` — the step must pass a compatible object
  (small adapter or `model_dump`). Confirm during impl.
- **Album-name fallback** (separate gotcha): exports land under `results/album/...` because
  `context.album_name` isn't set in Albumify → fix as part of this spec (set `album_name`) or
  file follow-up.
- Out of scope: the over-merge quality issue (6 vs ~15 people) — separate investigation that
  this export unblocks.
