# TODO - Open Tasks

**Format:** `[STATUS] Task description | Date | Owner`

**Status values:**
- `[ ]` TODO - Not started
- `[>]` IN_PROGRESS - Currently working
- `[x]` DONE - Completed
- `[!]` BLOCKED - Waiting on something

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

**Last updated:** 2026-03-27
