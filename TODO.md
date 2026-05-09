# TODO - Open Tasks

**Format:** `[STATUS] Task description | Date | Owner`

**Status values:**
- `[ ]` TODO - Not started
- `[>]` IN_PROGRESS - Currently working
- `[x]` DONE - Completed
- `[!]` BLOCKED - Waiting on something

---

## ML Merge Interface (spec 010)
Full plan: `specs/010-ml-merge-interface/tasks.md`

### Phase 1: Setup
- [x] T001 Verify test suite passes clean before changes | 2026-04-21 | Claude

### Phase 2: Foundational (backend)
- [x] T002 Add ml_prob/ml_pred fields to MergeDecisionRow in face_cluster/analysis_views.py | 2026-04-21 | Claude
- [x] T003 Add ml_threshold field to MergeAnalysisView in face_cluster/analysis_views.py | 2026-04-21 | Claude
- [x] T004 Implement compute_ml_merge_view() in face_cluster/analysis_views.py | 2026-04-21 | Claude
- [x] T005 Implement compute_pair_feature_contributions() in face_cluster/analysis_views.py | 2026-04-21 | Claude
- [x] T006 Write ut_MLMergeView tests in tests/face_clustering/test_merge_analysis.py | 2026-04-21 | Claude
- [x] T007 Write ut_FeatureContributions tests in tests/face_clustering/test_merge_analysis.py | 2026-04-21 | Claude

### Phase 3: US1 — Three-State Decision Refactor (P1)
- [x] T008-T012 Write US1 app tests in tests/face_clustering/test_streamlit_app.py | 2026-04-21 | Claude
- [x] T013 Remove auto-pre-fill from render_merge_analysis_tab() in app/face_clustering.py | 2026-04-21 | Claude
- [x] T014-T015 Add merge_decision_sources session state + update all decision writes | 2026-04-21 | Claude
- [x] T016 Split Smart Approve into Smart Approve + Smart Reject buttons | 2026-04-21 | Claude
- [x] T017 Add three-state summary bar (Approved/Rejected/Undecided counts) | 2026-04-21 | Claude
- [x] T018 Gate Apply+Remerge on n_approved >= 1 | 2026-04-21 | Claude
- [x] T019-T021 Update _collect_all_merge_labels + session step metadata + ML badge on cards | 2026-04-21 | Claude

### Phase 4: US2 — ML Model as Merge Proposer (P1)
- [x] T022-T023 Write US2 app tests | 2026-04-21 | Claude
- [x] T024-T030 Implement ML mode: mode selector, controls, async predict, pre-fill, card badges, detail section, ML Training tab redirect | 2026-04-21 | Claude

### Phase 5-8: US3-US6 (P2-P3)
- [x] T031-T033 ML Probability Overview Panel + low-separation warning | 2026-04-21 | Claude
- [x] T034-T035 Per-pair feature contributions in ML card detail + store pair_features | 2026-04-21 | Claude
- [ ] T036-T038 Heuristic vs ML comparison mode, agreement indicator, Disagreements filter | 2026-04-21 | Claude
- [ ] T039-T040 Wider candidates badge for extended-range pairs | 2026-04-21 | Claude

### Polish
- [x] T041 Update CHANGES_LOG.md | 2026-04-21 | Claude
- [x] T042 Update docs/FEATURE_REQUESTS.md | 2026-04-21 | Claude
- [x] T043 Full test run | 2026-04-21 | Claude
- [x] T044 Append learning to docs/LEARNINGS.md | 2026-04-21 | Claude

---

## Face Clustering — Cohesive Sub-Package (SIGHTING-008) [DONE - PENDING PLAYWRIGHT E2E]
Full plan: `FACE_CLUSTERING_PLAN.md`

### Phase 1: Fix Quality Gating
- [x] Add `require_pose: bool = False` to `face_cluster/config.py` | 2026-04-01 | Claude
- [x] Fix `face_cluster/quality.py` `select_core_set()`: skip pose check when pose=None and require_pose=False | 2026-04-01 | Claude
- [x] Add `tests/face_clustering/test_quality_gating.py` (4 tests) | 2026-04-01 | Claude

### Phase 2: Missing Pipeline Stages
- [x] Create `face_cluster/crops.py`: `save_crops()` → crops + crop_manifest.json | 2026-04-01 | Claude
- [x] Add `tests/face_clustering/test_crops.py` (3 tests) | 2026-04-01 | Claude
- [x] Create `face_cluster/export.py`: `export_results()` → faces.csv, clusters.csv, export_summary.json | 2026-04-01 | Claude
- [x] Add `tests/face_clustering/test_export.py` (4 tests) | 2026-04-01 | Claude

### Phase 3: Pipeline API
- [x] Create `face_cluster/pipeline.py`: `FaceClusteringPipeline` + `PipelineResult` with progress callback | 2026-04-01 | Claude
- [x] Update `face_cluster/__init__.py` to export new classes | 2026-04-01 | Claude

### Phase 4: Script Cleanup
- [x] Update `scripts/run_face_clustering.py` to call `FaceClusteringPipeline` (≤60 lines) | 2026-04-01 | Claude
- [ ] Archive `scripts/export_clustering_data.py` → `archive/scripts/` | 2026-04-01 | Claude

### Phase 5: Unified Streamlit App
- [x] Create `app/face_clustering.py`: 3-tab app (Run Pipeline, Browse Clusters, Debug) with st.progress() | 2026-04-01 | Claude

### Phase 6: Tests
- [x] Create `tests/face_clustering/test_pipeline_e2e.py` using `test_data/face_clustering/source_images/` — 3 PASSED | 2026-04-01 | Claude
- [x] Create `tests/face_clustering/test_streamlit_app.py` using `streamlit.testing.v1.AppTest` — 3 PASSED | 2026-04-01 | Claude
- [x] Install Playwright: `pip install playwright pytest-playwright && playwright install chromium` | 2026-04-01 | Claude
- [x] Create `tests/face_clustering/test_streamlit_e2e.py` using Playwright on `D:\Google_Germany` | 2026-04-01 | Claude
- [ ] Run Playwright E2E tests against live app on `D:\Google_Germany` — requires `streamlit run app/face_clustering.py` first | 2026-04-01 | User+Claude

---

## Face Clustering Refactoring

### Phase 1: Initial Clustering (Current)

- [x] Create pipeline steps using sim_bench/pipeline framework | 2026-03-28 | Claude
  - [x] filter_quality_gate.py
  - [x] build_knn_graph.py
  - [x] cluster_connected_components.py
  - [x] select_exemplars.py
  - [x] compute_debug_distances.py
  - [x] Updated export_for_labeling.py

- [x] Create YAML config for experiments | 2026-03-28 | Claude
  - configs/face_clustering_experiment.yaml

- [ ] Update workbench app to use pipeline | 2026-03-27 | Claude
  - Use PipelineExecutor instead of manual orchestration
  - Load config from YAML

- [ ] Add debug UI with distance visualization | 2026-03-27 | Claude
  - 5 closest within cluster
  - 5 furthest within cluster
  - 5 closest outside cluster
  - Exemplar distance matrix

- [ ] Test on test_data/face_clustering | 2026-03-27 | Claude
  - Verify 3 clusters (3 people)
  - Verify no embedding/crop mismatches
  - Verify distances computed correctly

### Phase 2: Iterative Split/Merge (Later)

- [ ] Design bridge detection algorithm | TBD | Research
- [ ] Implement iterative_split_merge step | TBD | Claude
- [ ] Add stage comparison UI | TBD | Claude
- [ ] Tune thresholds using labeled data | TBD | Research

### Documentation

- [x] Organize face clustering documentation | 2026-03-28 | Claude
  - [x] Consolidate 34+ docs into face_cluster/docs/
  - [x] Create logical structure (design, algorithms, pipeline, ui, workflows, archive)
  - [x] Create entry point README with navigation
  - [x] Archive outdated docs with explanations
  - [x] Create redirect from docs/face_clustering.md

- [ ] Create missing documentation | 2026-03-28 | Claude
  - [ ] algorithms/quality_gating.md
  - [ ] algorithms/exemplar_selection.md
  - [ ] algorithms/ml_merging.md
  - [ ] pipeline/steps.md
  - [ ] pipeline/configuration.md
  - [ ] ui/labeling_app.md
  - [ ] workflows/experimentation.md

### Testing

- [ ] Implement embedding validation tests | 2026-03-28 | Claude
  - [ ] Create tests/data/face_embedding_validation/ with test images (5 synthetic + 2 real)
  - [ ] Implement test_embeddings_match_after_gating (CRITICAL)
  - [ ] Implement test_face_ids_sequential
  - [ ] Implement test_saved_crops_match_embeddings (CRITICAL)
  - [ ] Implement test_gated_faces_not_in_output
  - [ ] Implement test_no_offset_after_gating (requires synthetic data)
  - [ ] Implement test_pipeline_performance_baseline

### Cleanup

- [ ] Archive old experimental scripts | 2026-03-27 | Claude
  - scripts/debug_*.py
  - scripts/compare_*.py
  - scripts/analyze_*.py
  - scripts/trace_*.py
  - Move to archive/scripts/

- [ ] Archive old notebooks | 2026-03-27 | Claude
  - Keep: debug_knn_graph_clustering.ipynb
  - Archive rest to archive/notebooks/

- [ ] Remove duplicate/obsolete code | 2026-03-27 | Claude
  - Review face_cluster/ vs sim_bench/clustering/
  - Document which is for what

- [ ] Update README.md | 2026-03-27 | Claude
  - Add face clustering workbench instructions
  - Point to face_cluster/docs/

---

## General

- [ ] Review and update LEARNINGS.md | Ongoing | Claude
- [ ] Update CHANGES_LOG.md for major changes | Ongoing | Claude

---

## Pipeline Run Observability (spec 012)

- [x] Implement spec 014: Force Merge widget in Cluster Analysis tab | 2026-04-22 | Claude
- [x] Implement spec 012 phases 1-6: QualityVerdict types, quality gating verdicts, det_score + d10_score capture, export writers + provenance, Quality Report UI panel, Cluster Provenance UI panel | 2026-04-22 | Claude
## Run History & Annotations (spec 013)
Full plan: `specs/013-run-history-annotations/tasks.md`

### Phase 2: DB Migration (foundational)
- [x] T003 Extend action_log with source_album, run_name, parent_run_id, run_kind, comment, config_json, n_core columns | 2026-04-23 | Claude
- [x] T004 Update _HOT_FIELDS + start_action/complete_action signatures in face_cluster/run_history_db.py | 2026-04-23 | Claude
- [x] T005 Add update_comment() helper with 2048-char limit in face_cluster/run_history_db.py | 2026-04-23 | Claude
- [x] T007 Write test_history_migration.py | 2026-04-23 | Claude

### Phase 3: Overwrite Protection (US2)
- [x] T008 Create face_cluster/run_naming.py with allocate_run_dir() | 2026-04-23 | Claude
- [x] T009 Write test_run_naming.py | 2026-04-23 | Claude

### Phase 4-5: Helpers (US5, US1)
- [x] T010 Create face_cluster/config_diff.py | 2026-04-23 | Claude
- [x] T012 Create face_cluster/run_history.py with search() + distinct_albums() | 2026-04-23 | Claude

### Phase 6-10: App Changes (US1, US2, US3, US4, US5)
- [x] T014 Rewrite render_history_tab() with filter bar + st.dataframe | 2026-04-23 | Claude
- [x] T016-T018 Source-album propagation through pipeline and app dispatch paths | 2026-04-23 | Claude
- [x] T019 Replace local output-path computations with allocate_run_dir() | 2026-04-23 | Claude
- [x] T020-T021 Add comment field (History table + run header) | 2026-04-23 | Claude
- [x] T022-T024 render_run_header() with config diff expander | 2026-04-23 | Claude

### Phase 11: Run Summary (US6 — requires spec 012)
- [x] T025-T027 render_run_summary() panel — implemented in history_tab.py; renders gracefully for pre-012 runs | 2026-04-23 | Claude

### Phase 12: Polish
- [x] T028 Export new types from face_cluster/__init__.py | 2026-04-23 | Claude
- [x] T030 Append CHANGES_LOG.md entry | 2026-04-23 | Claude
- [x] T031 Close SIGHTING-025 in docs/SIGHTINGS.md | 2026-04-23 | Claude

---

**Last updated:** 2026-04-23
