# Tasks: Pipeline Run Observability (012)

## Status: COMPLETE

All tasks implemented and tested. 50 spec-012-related tests pass.

## Design Notes

- **D1**: Quality verdicts computed in `quality.py`, stored on `FaceRecord.quality_verdict`, flattened to `faces.csv` columns by `export.py`. Return type of `select_core_set()` extended to 3-tuple `(core, holdout, verdicts)`.
- **D2**: `quality_config.json` written by `pipeline.py:_write_quality_config()` at the quality gate stage. Contains exact thresholds used.
- **D3**: `quality_summary` block (rejected counts, near-threshold counts) merged into `pipeline_run.json` by `pipeline.py:_update_quality_summary()`.
- **D4**: `det_score` captured from InsightFace in `embedding.py:123` and stored on `FaceRecord`. `d10_score` assigned in `pipeline.py:_select_exemplars()` from `D10ExemplarSelector` output.
- **D5**: Cluster provenance tracked via `origin` + `parent_cluster_ids` columns in `clusters.csv`. Base clusters get `origin=base`, merged clusters get `origin=auto_merge` with parent IDs derived from union-find over merge_log. Manual merge snapshot writes `origin=manual_merge`.
- **D6**: `clusters_stage_base.csv` created by renaming `clusters.csv` before merge overwrite in `export_merged_results()`.
- **D7**: `merge_metadata.json` includes `merge_candidate_threshold`, `merge_exemplar_threshold`, `n_candidates_proposed` for downstream UIs.
- **D8**: Loader backward compat — `loader.py` reads new columns with `pd.notna()` guards and `None` defaults.
- **D9**: UI panels — `_render_quality_report()` and `_render_cluster_provenance()` in `app/face_clustering/shared.py`, wired into Face Analysis and Cluster Analysis tabs respectively.

## Completed Tasks

### Phase 1: Schema
- [x] T001 Baseline tests pass
- [x] T002 `GateResult` + `QualityVerdict` dataclasses in `face_cluster/types.py`
- [x] T003 `ClusterOrigin` enum + `ClusterMetadata` dataclass in `face_cluster/types.py`
- [x] T004 `FaceRecord` extended with `quality_verdict`, `rejection_reason`, `det_score`, `d10_score`
- [x] T005 Unit tests in `tests/face_clustering/test_types.py`

### Phase 2: Quality verdicts + config snapshot
- [x] T006 `select_core_set()` returns `(core, holdout, verdicts)` — `face_cluster/quality.py`
- [x] T007 `rejection_reason` populated as first failing gate name
- [x] T008 `_quality_gate()` assigns `face.quality_verdict` and `face.rejection_reason`
- [x] T009 `quality_config.json` written by `_write_quality_config()` — `face_cluster/pipeline.py`
- [x] T010 `quality_summary` in `pipeline_run.json` via `_update_quality_summary()`
- [x] T011 Tests in `tests/face_clustering/test_quality.py`

### Phase 3: det_score + d10_score
- [x] T012 `det_score` captured from InsightFace `Face` object — `face_cluster/embedding.py`
- [x] T013 d10 values exposed by `D10ExemplarSelector`
- [x] T014 `d10_score` assigned to core faces in `_select_exemplars()` — `face_cluster/pipeline.py`

### Phase 4: Export writers + loader
- [x] T017 `faces.csv` includes quality columns, det_score, d10_score — `face_cluster/export.py`
- [x] T018 `clusters.csv` includes `origin=base` and `parent_cluster_ids=[]` — `face_cluster/export.py`
- [x] T019 `export_merged_results()`: renames to `clusters_stage_base.csv`, writes merged provenance
- [x] T020 `merge_metadata` includes threshold keys — `face_cluster/merge.py`
- [x] T021 `save_manual_merge_snapshot()` writes `origin=manual_merge` — `face_cluster/manual_merge_snapshot.py`
- [x] T023 `load_pipeline_result()` reads new columns with graceful fallback — `face_cluster/loader.py`
- [x] T024-T028 Tests in `test_export.py`, `test_manual_merge_snapshot.py`

### Phase 5-6: UI
- [x] T029 `_render_quality_report()` in `app/face_clustering/shared.py`
- [x] T031 Wired into Face Analysis tab
- [x] T032 `_render_cluster_provenance()` in `app/face_clustering/shared.py`
- [x] T033 Wired into Cluster Analysis tab

### Phase 7: Polish
- [x] T035-T039 Trackers updated, tests pass on Windows
