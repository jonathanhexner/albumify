# TODO - Open Tasks

**Format:** `[STATUS] Task description | Date | Owner`

**Status values:**
- `[ ]` TODO - Not started
- `[>]` IN_PROGRESS - Currently working
- `[x]` DONE - Completed
- `[!]` BLOCKED - Waiting on something

## spec-058 follow-ups (from REVIEW.md, low severity)

- [ ] spec-058 F-1: add `test_foreign_keys_match_per_table` to `tests/face_clustering/db/test_orm_matches_schema.py` using `PRAGMA foreign_key_list` so FK drift is caught by the drift-guard. ~20 min.
- [ ] spec-058 F-2: add "ORM models (per-run DB)" subsection to `docs/architecture/classes.html` listing the 10 new model classes + note that the per-run `Base` is separate from `face_cluster/repositories/_orm_base.py`. ~30 min.

## spec-059 follow-ups (from REVIEW.md, low severity)

- [ ] spec-059 F-1: add `test_engine_disposed_on_repository_gc` to `tests/run_db/test_session.py` — assert the cached engine releases its connection when the owning RunStore / Repository is garbage collected. ~20 min.
- [ ] spec-059 F-2: update `docs/architecture/classes.html` — RunStore + ClusterAnalysisRepository rows mention SQLAlchemy backing; add `sim_bench/run_db/_session.py` factory entry. ~20 min.
- [ ] spec-059 F-3: update `docs/architecture/data_flow.html` — read-path nodes show ORM models instead of raw SQL. ~30 min.

---

## spec-044 follow-ups (from REVIEW.md, 2026-05-25)
- [x] F-1 docs/architecture/classes.html: add RunHistoryRepository section (spec-043 oversight) + ColumnDef row (spec-044 addition). Cross-link to architecture_standards.md §B0 / §B0.1. | 2026-05-25 | Claude
- [x] F-2 docs/architecture/db_global.html action_log section: replace "module-level functions (no class wrapper)" with RunHistoryRepository (lines ~283, 285); add producer column row to the column table (~line 297). | 2026-05-25 | Claude
- [~] F-3 OBSOLETE — superseded by spec-046. The entire ColumnDef / _COLUMNS pattern is being retired (replaced by SQLAlchemy ORM models + Alembic). | 2026-05-28 | spec-046

---

## spec-046 — SQLAlchemy + Alembic data layer  ✅ IMPLEMENTED 2026-05-28
Full plan: `specs/046-sqlalchemy-data-layer/tasks.md` — see REVIEW.md and CODE_AUDIT.html.

## spec-048 — Data layer cleanup (spec-046 follow-ups)  ✅ IMPLEMENTED 2026-05-28
Full plan: `specs/048-data-layer-cleanup/tasks.md` — see REVIEW.md.
Resolved spec-046 SMELL-1/2/3/4/5/8 and added centralized `face_cluster/_paths.py`. 6 new permanent drift-guard tests.

- [ ] F-2 follow-up (P3, ~15 min): Annotate `docs/architecture/classes.html` + `db_global.html` for the spec-046 ORM-based design (carry-over from spec-046 F-1). | 2026-05-28 | unassigned

## spec-047 (planned, depends on spec-046) — Full-system E2E gate
- [ ] Spec to be drafted. One Playwright test that runs the real pipeline against `D:\sim-bench\test_data\face_clustering`, verifies DB writes via Repository, verifies UI render. Universal ship gate for future architectural specs. | 2026-05-28 | unassigned

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

## spec-032: filter context (2026-05-12)
- [x] P0: FilterContext primitive + tests | 2026-05-12 | Claude
- [x] P1: Migrate filter_quality + QualityGater + _save_crops (closes SIGHTING-059 Issue 1 at primitive level) | 2026-05-12 | Claude
- [x] P2: RunExporter writes filter_decisions table + RunStore reader | 2026-05-12 | Claude
- [x] P4: UI ↔ filter alignment static check (flagged 4 real phantoms, all documented) | 2026-05-12 | Claude
- [x] P5: no-raw-collection-iteration static check (grandfathered allow-list) | 2026-05-12 | Claude
- [ ] P3: FC App Run Summary tab surfacing filters.summary() | Claude
- [ ] P6: SIGHTING-060 area-unit fix (bbox_area_pixels canonical helper, migrate 3 producers, promote config_min_face_size to real rejector) | Claude
- [ ] P7: Remove legacy advisory fields (quality_passed, face["filter_passed"], quality_*_pass CSV columns) | Claude

## spec-031: max_diameter cap step (2026-05-11)
- [x] Phase 1: implement cap step (merged-only scope, full + exemplar thresholds, split action) | 2026-05-11 | Claude
- [ ] Phase 2 (deferred): option to expand scope to all clusters (catches base-clustering chains too) | Claude
- [ ] Phase 3 (deferred): FC App "Cluster Cap" tab to surface cap_decisions.json visually | Claude

## SIGHTING-059 — face clustering data integrity (2026-05-10)
- [x] Item 4: profile load/save in Run tab | 2026-05-10 | Claude
- [ ] Item 1: face_46/47 (cluster 6) have no crops despite passing quality | Claude
- [ ] Item 2: area column unit mismatch (fraction vs pixels label) | Claude
- [ ] Item 3a: blur_score=0.0 for ALL faces — blur step broken or stripped during export | Claude
- [ ] Item 3b: det_score=NaN for ALL faces — never persisted | Claude
- [ ] Item 3c: merge_log.json missing `*_pass` boolean fields | Claude
- [ ] HTML diagnostic report visualizing tracing of all of the above | Claude

## spec-033 — data integrity cleanup (2026-05-15)
Master plan: `specs/033-data-integrity/MASTER_PLAN.md`
- [x] P-A: UI label honesty + surface hidden min_bbox_ratio | 2026-05-15 | Claude
- [x] P-B: write spec-034 context contract | 2026-05-15 | Claude
- [x] P-G: typed step configs (Pydantic) | 2026-05-15 | Claude
- [x] P-C: storage plumbing + Pydantic FaceRecord (resolves SIGHTING-059 Items 1, 3a, 3b) | 2026-05-15 | Claude
- [x] P-H: Pandera DB I/O schemas | 2026-05-15 | Claude
- [x] P-D: RunStore.image_detail (extends spec-023) | 2026-05-15 | Claude
- [x] P-F: config parity (FC App ↔ Albumify) | 2026-05-15 | Claude
- [x] P-E: drift-prevention architecture tests (landed alongside their phases) | 2026-05-15 | Claude

## spec-040 — Unified Pipeline Framework (2026-05-16)
Branch: `unification/spec-040` (separate, depends on FR-033-1 on main first)
Spec: `specs/040-unified-pipeline-framework/`
- [x] Phase 0 (blocks branch): land FR-033-1 (E2E test) on main | 2026-05-16 | Claude
- [ ] Phase 1: unified PipelineContext (extend spec-034) | Claude
- [ ] Phase 2: Pydantic config for every face-clustering step (closes FR-033-6) | Claude
- [ ] Phase 3: replace bridge functions with real pipeline steps | Claude
- [ ] Phase 4: schema v5 (images table per SIGHTING-065; area_ratio per SIGHTING-064) | Claude
- [ ] Phase 5: retire FaceClusteringPipeline + face_cluster/config.py PipelineConfig | Claude
- [ ] Phase 6: delete face_cluster_bridge.py; architecture test prevents return | Claude
- [ ] Phase 7: collapse doc HTMLs to single-origin | Claude
- [ ] Phase 8: 2-week burn-in on labeled album; then merge | Claude

## spec-033 follow-ups (2026-05-15)
Review: `specs/033-data-integrity/REVIEW.md` | Roadmap: `specs/033-data-integrity/FOLLOW_UPS_ROADMAP.html`
- [x] FR-033-1: E2E acceptance test → `specs/035-albumify-e2e-acceptance/` | 2026-05-16 | Claude
- [ ] FR-033-2: Config-to-producer contract → `specs/036-config-producer-contract/`
- [ ] FR-033-3: InsightFace blur scoring step + lift SIGHTING-061 pin → `specs/037-insightface-blur-step/`
- [ ] FR-033-4: ExportRequest Pydantic bundle → `specs/038-export-request-pydantic/`
- [ ] FR-033-5: Verify no duplicate filter_decisions rows → SIGHTING-062
- [ ] FR-033-6: STEP_CONFIG_MODELS registry guard → `specs/039-step-config-registry-guard/`
- [ ] FR-033-7: Fix bridge pose-lookup operator precedence → SIGHTING-063
- [ ] FR-033-8: Delete or relocate `notebook_diagnostic.py` (this entry) | Claude

---

**Last updated:** 2026-05-15
