# Feature Requests

This file tracks feature requests from users. Claude should scan this on init to check for open requests.

---

<!-- Add new entries at the top, newest first -->

### 2026-05-09: Storage Ownership Refactor — RunStore / RunExporter (spec-030)
**Status**: SPEC READY
**Requested by**: User, 2026-05-09 (during diagnosis of SIGHTING-058)
**Spec**: `specs/030-storage-ownership-refactor/spec.md`
**Resolves**: SIGHTING-058
**Architecture audit**: `specs/030-storage-ownership-refactor/architecture_audit.html`
**Description**:
Eliminate face-clustering storage duplication. Today 6 logical information types are written to 21 storage locations across 14 files per run; the DB `merge_decisions` table has 12 columns while `merge_log.json` has 17 fields, causing data loss for Albumify-produced runs. Replace four scattered modules (`loader.py`, `export.py`, `result_db.py`, `face_cluster_export.py`) with two: `RunExporter` (single writer used by both apps, byte-identical layouts) and `RunStore` (single reader, no fallback chains, fail-loud on missing artifacts). Run directory becomes exactly 5 artifacts: `face_clustering.db`, `embeddings.npy`, `embedding_face_ids.npy`, `pipeline_run.json`, `crops/`. DB schema becomes the contract — full 17-field `merge_decisions`, `embeddings` table removed (lives in npy only), `run_metadata` absorbs the deleted JSON files. UI gains three-state outcome label (MERGED/PASSED/REJECTED) and "disabled" margin badge. 5-phase rollout: ship writer alongside legacy → ship reader alongside legacy → cut UI over → stop writing legacy → migrate old runs and delete dead code.

### 2026-05-01: Trip Detection — Geographic & Temporal Event Clustering
**Status**: SPEC READY
**Requested by**: User, 2026-05-01
**Spec**: `specs/022-trip-detection/spec.md`
**Description**:
Extract EXIF GPS coordinates and timestamps from photos, cluster them into trips/events
(using time gaps + location jumps), reverse-geocode to city/country names, and present
as "Your trip to Dubai, May 24-28 2022" in the UI. Eight integration areas identified:
metadata extraction, persistence, trip detection algorithm, trip naming, album organization
UI, people cross-reference (V2), selection strategy (V2), export (V2).

### 2026-04-30: Pipeline Protective Layer — Step Error Capture + Degenerate Output Alerts
**Status**: OPEN
**Requested by**: User, 2026-04-30
**Description**:
Two complementary failure-mode protections for the pipeline:
1. **Step exception capture**: Wrap each pipeline step in try/except; store `{step, error, traceback}` as `step_errors` JSON in `pipeline_results` DB. Surface in Results tab as ✗ error entries.
2. **Degenerate output alerts**: Post-step sanity checks (0 clusters, 100% quality rejection, 0 selected images); store as `step_warnings` JSON in `pipeline_results`. Surface in Results tab as ⚠ warning entries.
Both mechanisms together ensure no silent or hard failures go undiagnosed from the UI.
**Triggered by**: (a) `select_best` crashing with `RuntimeError: Numpy is not available` — job showed FAILED with no detail in UI; (b) `cluster_people` returning all-noise silently with `blur_score=0.0` bug.

### 2026-04-30: Unified Face Clustering in Main App (spec-020)
**Status**: DONE (core implementation; deep-link T5 pending)
**Requested by**: User, 2026-04-30
**Spec**: `specs/020-main-app-face-clustering/spec.html`
**Description**:
Replace the main album app's HDBSCAN-based face clustering with the mature face_cluster/ algorithm (mutual-kNN + iterative merge). Added as `face_cluster_knn` method in existing `cluster_people` step. Exports artifacts for standalone Face Clustering App analysis. Connected-apps approach: main app runs clustering, standalone app provides deep analysis/recluster/ML training.

### 2026-04-30: Merge Gallery "By Iteration" Default View
**Status**: DONE
**Requested by**: User, 2026-04-30
**Description**:
Replace the flat list + iteration filter with an iteration-grouped default view. Each iteration is a collapsible section showing the merged pair first, then rejections sorted by exemplar distance. 3-way radio: "By iteration" (default) | "Flat list" | "Transitive groups".

### 2026-04-30: p25 Cross-Distance OR Gate for Merge Algorithm
**Status**: DONE
**Requested by**: User, 2026-04-30
**Spec**: `specs/019-merge-cross-dist-or-gate/spec.md`
**Description**:
Add an OR path to Gate A: `p25_exemplar_dist <= T1 OR p25_cross_dist <= T2` where p25_cross_dist is the 25th percentile of ALL cross-cluster node-pair distances. Also: unique-pair support count (greedy bipartite matching, each node used at most once) as an alternative to raw support count in Gate B.

### 2026-04-25: Merge Label Verification Tab
**Status**: SPEC READY
**Requested by**: User, 2026-04-25
**Spec**: `specs/017-merge-label-verification/spec.md`
**Description**:
Manual labeling UI for merge decisions. Shows cluster pairs ordered by distance with exemplar crops side-by-side. User clicks Merge/Reject/Skip. Labels saved to training_db with source='human'. Needed because heuristic labels are unreliable (only 8 positive labels, Germany_10 rejects visually incorrect). Canonical runs: one per source dataset with crop fallback.

---

### 2026-04-25: Merge EDA Data Validation & Feature Engineering
**Status**: DONE
**Requested by**: User, 2026-04-25
**Spec**: `specs/016-merge-eda-validation/spec.md`
**Description**:
Investigate and fix data quality issues in merge EDA notebooks (`notebooks/face_clustering/eda_merge_explore.ipynb`, `eda_merge_ml.ipynb`). Key issues:
- Deduplicate runs with identical merge decisions
- Fix transitivity labeling (if A+B merged and B+C merged, A-C should be positive)
- Align candidate threshold with pipeline default (0.45 not 0.9)
- Flag t_global leakage (constant per run, acts as run identifier)
- Add derived ratio features (dist_over_t_local, dist_over_t_global, etc.)
- Harvest non-candidate pairs as strong negative labels
- Fix broken crop display paths

---

### 2026-04-24: Face Detail Popup — click any face anywhere in the app
**Status**: IMPLEMENTED
**Requested by**: User, 2026-04-24
**Spec**: `specs/015-face-detail-popup/spec.md`
**Description**:
Clicking any face crop (in any tab: cluster gallery, Face Analysis, Merge Analysis, etc.) should open a modal popup showing the full face detail without switching tabs. The popup should include:
- Large face crop image
- Gate status per criterion with value vs threshold (blur: 82.3 >= 50 PASS, pose_yaw: 38.1 > 30 FAIL, etc.)
- det_score, blur, area, pose (yaw/pitch/roll)
- Cluster assignment and source image
- Free-text comment field (persisted per face per run, stored as `face_comments.json` in the run output_dir)
- Button to navigate to Face Analysis tab for full detail (nearest neighbors, co-image faces, etc.)

**Implementation notes**:
- Use `st.dialog` (Streamlit ≥1.28)
- Shared helper `show_face_detail_popup(face_id, output_dir)` callable from any tab
- Comments stored in `{output_dir}/face_comments.json` as `{face_id: "comment text"}`; writer-reader contract must be defined before implementation

---

### 2026-04-24: App-wide UI improvements (History, Face Analysis, Cluster gallery)
**Status**: IMPLEMENTED
**Requested by**: User, 2026-04-24
**Description**:
Several UI improvement requests bundled:
1. **History — output column**: Show `album\run_name` (e.g., `Noa2_5_2\base_1`) instead of just `base_1` so the run's location is immediately clear
2. **Face Analysis — filterable table**: Replace the one-at-a-time selectbox with a sortable/searchable dataframe showing all faces: face_id, gate status, blur, area, yaw, pitch, roll, det_score, cluster, source image, per-gate pass/fail columns. Row click opens face detail (or popup).
3. **Clusters (Base) / Clusters (Merged) — sortable table**: Replace the card gallery with an `st.dataframe` with columns: cluster_id, size, diameter, avg_intra_dist, n_exemplars, nearest_cluster, nearest_dist. Sortable by any column. Row click still expands cluster detail.

---

### 2026-04-23: Run History & Run Annotations
**Status**: IMPLEMENTED
**Requested by**: User, 2026-04-23
**Spec**: `specs/013-run-history-annotations/spec.md`
**Resolves**: SIGHTING-025 (run overwrite / output folder collision)
**Description**:
P0 problem: derived runs from the same source album appear as different albums in the History tab because the output folder name bleeds into the album column. Fix: `source_album` anchored to the original input directory, inherited by all derived runs. Plus: unique output-path allocation (overwrite protection), per-run free-text comments, searchable/filterable History table, run header with parent pointer + config diff, and Run Summary panel (depends on spec 012).

---

### 2026-04-22: Pipeline Run Observability
**Status**: IMPLEMENTED (pending spec 013 for cross-run UI dashboard)
**Requested by**: User, 2026-04-22
**Spec**: `specs/012-pipeline-observability/spec.md`
**Resolves**: SIGHTING-022 (opaque quality gating), SIGHTING-024 (no cluster provenance)
**Description**:
Pipeline stages compute rich per-face and per-cluster decisions but discard them — the user cannot answer "why did this face pass?" or "how did this cluster form?" from the run artifacts. Four user stories: (1) per-face quality verdicts with gate values/thresholds persisted in `faces.csv` and surfaced in the Face Analysis tab; (2) cluster provenance (`origin`, `parent_cluster_ids`) on `clusters.csv` plus a `clusters_stage_base.csv` snapshot before merge; (3) `det_score` (InsightFace) and `d10_score` (exemplar centrality) added to `faces.csv` — both already computed today; (4) run-level observability dashboard in the History tab showing quality funnel, merge summary, and "0 candidates at threshold X" for empty merge_log runs. Backward compatible — legacy runs load without new columns.

---

### 2026-04-22: Force Merge (User-Driven Cluster Merge)
**Status**: IMPLEMENTED
**Requested by**: User, 2026-04-22
**Spec**: `specs/014-force-merge/spec.md`
**Description**:
Select any two clusters and force-merge them, even if their exemplar distance exceeds `merge_candidate_threshold`. Preview shows full merge evidence (same as a Merge Analysis card). Extracted from spec 011 US3 — no ML dependency.

---

### 2026-04-22: ML Model Expansion & Merge Interpretability
**Status**: Open (Spec phase — US3 extracted to spec 014)
**Requested by**: User, 2026-04-22
**Spec**: `specs/011-ml-model-expansion/spec.md`
**Description**:
Three-part feature: (1) Add decision tree, random forest, and CatBoost to the ML model zoo (currently LR, XGBoost, MLP). (2) Feature importance dashboard after training — ranked bar chart mapping top features to heuristic config defaults so user can tune thresholds without ML. (3) Harvest non-candidate cluster pairs as negative training samples to improve class balance.

---

### 2026-04-21: ML Merge Application Interface
**Status**: Done (T001-T035 implemented; T036-T044 remaining polish + P3 features)
**Requested by**: User, 2026-04-21
**Spec**: `specs/010-ml-merge-interface/spec.md`
**Description**:
Replace the flat-table "Apply to Current Run" in the ML Training tab with a proper visual interface for applying trained ML merge models. The ML model becomes an alternative merge proposer (alongside the heuristic 4-gate ConservativeMerger), shown through the same grouped gallery with face crops, confidence tiers, and Apply + Remerge workflow. Includes three-state decision model (Approve/Reject/Undecided) and ML-as-first-guess pre-filling. Key elements: mode selector in Merge Analysis tab (Heuristic vs ML Model), probability threshold slider, ML-driven confidence grouping, per-pair feature contributions, heuristic agreement indicator.

---

### 2026-04-21: Session Operation Pipeline
**Status**: Open
**Requested by**: User, 2026-04-21
**Spec**: `specs/009-session-op-pipeline/spec.md`
**Description**:
Replace the per-action folder pattern in the face clustering app with a session-based operation pipeline. Every action (recluster, merge, split) gets appended to an ordered op list shown in the UI. Folder mechanics are hidden from the user. When the user loads an earlier op and runs a new action, subsequent ops are superseded (renamed with `_` prefix on disk) but remain accessible via a "show superseded" toggle. All ops live under a single session root folder.

---

### 2026-04-20: Smart Merge Grouping (Transitive Reduction + Auto-Approve)
**Status**: Done
**Requested by**: User, 2026-04-19
**Spec**: `specs/008-smart-merge-grouping/spec.md`
**Sighting**: SIGHTING-021
**Description**:
Merge Analysis tab shows ALL pairwise candidates individually (226 pairs, 23 pages on Austria24_2). User wants:
1. Transitive grouping — connected components via union-find, one decision per group instead of per pair
2. Component cohesion scoring — if most pairs in a group pass 4/4 gates, auto-approve the group
3. Smart Approve button — one click to auto-resolve obvious groups, show only borderline ones for review
4. Group-level UI with per-pair override capability

---

### 2026-04-17: Persistent Run History in DB (face_cluster runs)
**Status**: Open
**Requested by**: User, 2026-04-17
**Description**:
Every recluster and full-pipeline run should be recorded in the SQLite DB (`~/.sim_bench/sim_bench.db`) so run history is available across app restarts. Currently run metadata lives only in `pipeline_run.json` and logs in `<output_dir>/logs/*.log`. If the output directory is deleted or moved, the history is lost.

**Desired behaviour**:
- A new `face_cluster_runs` table stores: `run_id`, `mode` (full/recluster), `source_dir`, `output_dir`, `started_at`, `ended_at`, `status`, `n_faces`, `n_clusters`, `config_json`, `log_file_path`.
- The History tab reads from DB first (falling back to filesystem scan) so runs appear even after output dirs are renamed.
- A "View Log" button in the History tab opens the log file inline (last N lines) without leaving the app.

**What already exists**:
- `pipeline_run.json` in each output dir (with `log_file` path)
- `<output_dir>/logs/*.log` with full DEBUG log
- `_list_available_runs()` does a filesystem scan already

### 2026-04-17: Iterative Manual Merge Re-evaluation
**Status**: Done
**Description**:
After clicking "Apply Approved Merges" in the Merge Analysis tab, the app applies the user's decisions once and stops. No second round of candidates is ever evaluated on the newly merged clusters. This means transitive merges (where cluster AB, formed by merging A+B, is now close enough to merge with C) are silently missed.

The `ConservativeMerger` already handles this correctly via an internal iterative loop (`merge.py:109`). The gap is in the manual approval workflow: `apply_manual_merges` is one-shot, and the Merge Analysis tab never re-computes candidates on `st.session_state.merge_approval_result`.

**Desired behaviour**:
After "Apply Approved Merges", the tab automatically runs a second (and further) rounds of candidate evaluation on the post-merge result, presenting a fresh set of merge candidates until no new ones are found. Each round is clearly labelled ("Round 2 of 3") and the user can stop at any round.

**Also tracked as**: SIGHTING-019 (filed in error — this is a feature gap, not a bug)

---

### 2026-04-17: Merge Safety Parameters — Always Visible and Accessible
**Status**: Open
**Description**:
Four merge safety parameters exist in the code (`merge_support_frac`, `merge_support_min`, `merge_margin`, `merge_diameter_expansion_factor`) but are only rendered inside the "Merge Parameters" expander, which is itself hidden behind a `merge_enabled` checkbox in the Recluster tab. A user examining why a pair was rejected in the Merge Analysis tab has no way to find or adjust these parameters without first knowing to enable merge in a separate tab.

**Desired behaviour**:
1. Merge safety parameters should be visible (optionally greyed out) even when `merge_enabled` is unchecked, so users can preview and pre-configure them.
2. The Merge Analysis tab should show the parameters that were used to generate the current merge log (read-only summary), with a direct link/button to adjust them in Recluster.
3. Alternatively: expose a "Merge config" panel in the Merge Analysis tab that pre-populates Recluster sliders when the user clicks "Re-run with these settings".

---

### 2026-04-17: Persist Parameter Defaults in App
**Status**: Open
**Description**:
All clustering, merge, and quality parameters currently reset to hardcoded defaults every time the app is opened. Users who always work with a particular album need to re-enter the same `merge_candidate_threshold`, `distance_threshold`, `blur_min`, etc. on every session. There is no way to save a working configuration from the UI.

**Desired behaviour**:
- A "Save as default" button on the Recluster tab (and optionally Quality/Clustering sections) that writes the current slider values to a config profile stored in `~/.sim_bench/profiles/<name>.json`.
- On next app start, a "Load profile" selector (defaulting to "last used") pre-populates all sliders.
- Multiple named profiles supported (e.g., "Germany album", "Portrait studio").
- The existing `PipelineConfig` dataclass already has all the required fields; the feature is purely persistence + UI wiring.

---

### 2026-04-17: Labeling Review + ML Training Dashboard
**Status**: In Progress (Phase 1 implemented — Phase 2 pending)
**Description**:
Two new sections in the face clustering app:
1. **Labeling Review** — rename/extend current Training Data tab with label audit (disagreement detection), inline label flip, and active labeling suggestions
2. **ML Training** — configure, train, and evaluate merge classifiers (Logistic Regression, XGBoost, MLP) from the UI with dataset config, feature group selection, results visualization (metrics, confusion matrix, ROC, feature importance), model save/load/compare, and apply-to-current-run prediction
**Spec**: `specs/006-ml-training-dashboard/spec.md`
**Plan**: `specs/006-ml-training-dashboard/plan.md`

---

### 2026-04-14: ML-Based Cluster Merging
**Status**: Open (Spec phase)
**Description**:
Replace/augment the rule-based ConservativeMerger with a trained binary classifier. Collect rich feature vectors (~50 features) for each candidate cluster pair during human labeling in the Merge Approval UI, then train a lightweight model (XGBoost / logistic regression) to predict merge/reject.
**Spec**: `docs/ML_CLUSTER_MERGING.md`
**Feature Plan**: `docs/ML_MERGE_FEATURES_PLAN.md`

---

### 2026-04-14: Cluster Gallery UX
**Status**: Done
**Description**:
Replace the select-then-open cluster workflow in Clusters (Base) and Clusters (Merged) tabs with an expandable gallery. Each cluster shows thumbnails (3 exemplar face crops) and key stats (size, diameter) when collapsed. Expanding reveals the full existing detail view. Sorted by size descending by default with sort controls.
**Spec**: `specs/004-cluster-gallery-ux/spec.md`

---

### 2026-04-14: Unified Merge Decision Gallery
**Status**: Done
**Description**:
Replace the four separate merge analysis sections (All Merge Decisions table, Near Misses, Merged Pairs, Rejected Candidates) with a single unified filterable gallery. Each row shows exemplar face crops for both clusters inline with gate metrics and approve/reject buttons. Adds filter (All/Merged/Rejected/Near Misses/Contested), sort (exemplar distance, gates passed), and pagination.
**Spec**: `specs/003-unified-merge-gallery/spec.md`

---

### 2026-04-11: Interactive Merge Approval UI + ML Merge Classifier
**Status**: Open
**Description**:
Combined feature: interactive merge approval in the face clustering app that doubles as the labeling interface for training an ML merge classifier.

**Scope**:
1. **Interactive merge approval UI** — show proposed merges with exemplar crops, distances, gate pass/fail; user accepts/rejects; clusters update live
2. **Simplify heuristic merger** — drop adaptive per-cluster thresholds (alpha/beta/percentile), use fixed distance threshold; reduce knob count
3. **ML merge classifier** (phase 2) — train on accept/reject decisions from the UI; features from `ClusterPairFeatures` + future additions (image count, group sizes, pose distribution)

**Key insight**: The approval UI *is* the labeling workflow — no separate labeling step needed. Every accept/reject becomes a training sample.

**Spec**: `specs/002-interactive-merge-approval/spec.md`

---

### 2026-04-10: spec-kit workflow integration
**Status**: Done
**Request**: Integrate github/spec-kit `templates/` and `scripts/` folders; produce WORKFLOW.md for structured feature development
**Details**: Created `WORKFLOW.md` documenting the 7-step spec-kit workflow and its integration with existing sim-bench conventions (TODO.md, FEATURE_REQUESTS.md, CHANGES_LOG.md, etc.)

### 2026-04-07: History tab, Run tab stage plan, Nearest Clusters thumbnails
**Status**: Done
**Description**: (1) output_folder in History table, (2) live stage execution plan in Run tab, (3) exemplar thumbnails + Go button in Nearest Clusters
**Notes**: Germany_7 confirmed as first fully clean run — all 7 stages tracked, status=complete, all output files present. Main bugs cleared.

### 2026-04-07: Clustering Debug — Mapping Tables, Validation, Worked Example
**Status**: Done
**Description**:
1. Run clustering validation test on Germany_run_5 — check cross-cluster proximity
2. Add clear face-to-cluster and face-to-source-image mapping tables in the app
3. Add worked algorithm example showing concrete edge construction from actual run data

**Implementation**:
- Validation: 975 cross-cluster pairs under dist 0.2 identified, two merge groups found (9 clusters / 5 clusters). Not an index bug — merge stage was disabled.
- Mapping tables: Three expandable tables in Run Overview tab (Face->Cluster, Face->Source Image, Cluster->Faces)
- Worked example: Interactive section in Run Overview — user picks a cluster, sees step-by-step pairwise distances, kNN, mutual edges, and component formation with face crops.

---

### 2026-04-01: Face Clustering — Cohesive Tested Sub-Package with Streamlit Integration
**Status**: Open
**Description**:
Replace the disconnected collection of face-clustering scripts with a single, tested `face_cluster` sub-package that owns the entire cycle from raw images to labeled clusters. Triggered by SIGHTING-008.

**Requirements**:
1. `face_cluster/pipeline.py` — `FaceClusteringPipeline` class with `run(image_dir, output_dir, config, on_progress=None)` as the single public entrypoint
2. `face_cluster/crops.py` — save aligned crops + `crop_manifest.json` (currently missing, referenced in RECOVERY_PLAN.md)
3. `face_cluster/export.py` — produce `faces.csv`, `clusters.csv`, `export_summary.json` (currently missing)
4. Fix quality gating: graceful degradation when SixDRepNet unavailable (pose filter optional, not fatal)
5. Single unified Streamlit app with pipeline runner tab (progress bars, per-stage status), cluster browser tab (labeling), debug tab — replacing the two separate apps
6. Full A-to-Z test suite in `tests/face_clustering/`:
   - Per-stage unit tests (quality gate, kNN graph, clustering, crops, export)
   - E2E test on `test_data/face_clustering/source_images/` (15 images, expect ≥3 clusters)
   - Streamlit app tests using `streamlit.testing.v1.AppTest`
7. Archive `scripts/export_clustering_data.py` — replace with thin `scripts/run_face_clustering.py` that calls `FaceClusteringPipeline`

**Non-negotiable constraints**:
- `face_cluster/` = pure algorithms only, no file I/O except `crops.py` and `export.py`
- No test uses real album paths — synthetic fixtures only (except the designated E2E test)
- Every stage writes output before next stage starts (resumable)
- No feature considered complete without passing test

---

### 2026-03-24: Face Clustering — Clean Architecture & Full Traceability
**Status**: Open
**Description**:
Establish a clean, tested face clustering pipeline with clear component boundaries and full lineage
(image_path → bbox → embedding → crop → cluster_id). Each stage is independently testable.

**Components**:
1. `face_cluster/crops.py` — save aligned crops + crop_manifest.json
2. `face_cluster/export.py` — produce faces.csv, clusters.csv, export_summary.json
3. `scripts/run_face_clustering.py` — single orchestration script (replaces benchmark + export scripts)
4. `tests/face_clustering/` — per-stage tests + E2E test (synthetic fixtures only)
5. `face_cluster/types.py` — make `image_path` non-optional on FaceRecord
6. Archive all one-off debug scripts and notebooks

**Spec**: `RECOVERY_PLAN.md`

### 2026-02-27: ML-Based Cluster Merging Pipeline
**Status**: Open
**Description**:
Replace heuristic-based cluster merging with a logistic regression model trained on manually labeled data.

**Components**:
1. **Export Pipeline**: Run clustering (mutual kNN + connected components) and export 3 CSVs:
   - `faces.csv` - one row per face with metadata
   - `clusters.csv` - one row per cluster with stats
   - `candidate_pairs.csv` - cluster pair features for merge candidates
2. **Labeling Interface**: Streamlit app to visualize clusters and manually assign corrected_identity labels
3. **Training Pipeline**: Generate training data from corrected labels, train logistic regression, evaluate model
4. **Deployment**: Keep both heuristic and ML mergers for comparison

**Features for candidate pairs**:
- min_exemplar_dist, p10_cross_dist, p50_cross_dist, support_fraction
- diameter_ratio, cluster_size_min, cluster_size_ratio
- T_A, T_B, T_local, T_global (adaptive thresholds)

**Files to create**:
- `scripts/export_clustering_data.py` - CLI script for batch export
- `notebooks/export_clustering_data.ipynb` - Interactive version for exploration
- `app/face_clustering_labeling.py` - Streamlit labeling interface
- `scripts/train_merge_classifier.py` - Train and evaluate logistic regression
- `face_cluster/ml_merge.py` - MLMerger class using trained model

**Output directory**: `results/face_clustering_training/`

### 2026-02-24: KNN Graph Clustering Notebook with Quality Gating
**Status**: In Progress
**Description**:
Implement a comprehensive face clustering pipeline for small batches (10-20 faces) with:
- Quality gating using pose (yaw/pitch/roll) + blur + top-N largest faces per image
- Mutual kNN graph + distance threshold + connected components clustering
- d10-based exemplar selection (dense core points)
- Optional cluster splitting safeguard (diameter-based)
- Optional holdout attachment (vote+margin)
- Step-by-step Jupyter notebook for debugging/tuning

**Architecture**:
- Core library: `face_cluster/` module with config, embedding, quality, knn_graph, clustering, exemplars, attach, viz
- Dataclasses: PipelineConfig, FaceRecord, GraphResult, ClusterResult
- Notebook: Similar to `notebooks/debug_hybrid_simple.ipynb`, one cell per stage
- Brute force distance matrix (no FAISS), full interpretability

**Priorities**:
- High precision (avoid false merges)
- Interpretable step-by-step debugging
- Manual intervention points (override core set, edit edge list)

### 2026-02-21: Mutual KNN Two-Stage Clustering
**Status**: Done
**Description**:
Implement `mutual_knn_two_stage` clustering algorithm:
1. **Stage 1**: Build mutual kNN graph (no/loose threshold) → connected components
2. **Stage 2**: Iterative prune & reassign loop with pluggable pruning strategy

Pruning strategy (RedundantSupportStrategy):
- Base condition: closest_dist ≤ α·X (relaxed threshold)
- AND one of:
  - Redundant support: ≥m neighbors within β·X
  - Separation: δ margin to next-best cluster
- Unassigned samples try other clusters; if none fit → noise (-1)

**Parameters**:
- X=0.45 (base threshold), α=1.1 (relaxation), β=1.05 (support radius)
- m=2 (min support), δ=0.15 (separation margin)

**Files Created**:
- `sim_bench/clustering/pruning_strategies.py` - PruningStrategy base + RedundantSupportStrategy
- `sim_bench/clustering/mutual_knn_two_stage.py` - Main clusterer
- `tests/clustering/test_mutual_knn_two_stage.py` - Unit tests (18 tests)

**Files Modified**:
- `sim_bench/clustering/distance_utils.py` - Added cluster debug utilities
- `sim_bench/clustering/base.py` - Registered new algorithm
- `configs/clustering_benchmark.yaml` - Added 3 config variants

**Benchmark Results** (274 faces):
- mutual_knn_two_stage: 36 clusters, 27 noise (balanced)
- mutual_knn_two_stage_strict: 33 clusters, 59 noise (more conservative)
- mutual_knn_two_stage_loose: 23 clusters, 9 noise (larger clusters)
- Compare: hdbscan=8 clusters (over-merged), mutual_knn=139 clusters (fragmented)

---

### 2026-02-20: Clustering Diagnostics - UMAP & Distance Analysis
**Status**: In Progress
**Description**:
Add diagnostic visualizations to face clustering debug app:

1. **Per-cluster UMAP**: Plot UMAP of faces in each cluster, colored by:
   - Membership probability
   - Detection confidence
   - Face size
   - Yaw angle (if available)
   - Goal: Identify if two clear subgroups exist (HDBSCAN tunable) vs smooth transition (embedding issue)

2. **Global UMAP**: Run UMAP on all faces, color by cluster assignment

3. **Distance distribution overlap**:
   - Given known same-person pairs and different-person pairs
   - Compute embedding distances and show histogram
   - If distributions overlap heavily → embeddings are the bottleneck, not clustering

4. **Condensed tree merge scale**:
   - Visualize HDBSCAN condensed tree
   - Show where mis-merged identities became one branch
   - If merge happens late (high distance) → fixable with `cluster_selection_epsilon`
   - If merge happens early → no density valley, parameter tweaks won't help

5. **Membership probability analysis** per face

**Notes**:
- User reports HDBSCAN variants are over-merging compared to before
- Specific test case: Person 1 (0000, 0081, 0079, 0082, 0091, 0135) vs Person 2 (0002, 0046, 0183, 0249)

**Implemented**:
- [x] `scripts/face_distance_report.py` - HTML report for distance analysis between two groups
- [x] `app/face_clustering_debug/pages/embedding_analysis.py` - New tab with:
  - Global UMAP (colored by cluster/confidence/frontal score)
  - Per-cluster UMAP
  - Distance comparison tool
  - HDBSCAN condensed tree visualization
- [ ] UMAP color by membership probability (needs HDBSCAN soft clustering)
- [ ] UMAP color by yaw angle (needs pose data in FaceInfo)

---

### 2026-02-20: Pipeline Cache Layer (LMDB + Parquet)
**Status**: Open
**Description**:
Create unified cache layer for pipeline artifacts:
1. LMDB for images (original, raw crops, aligned crops)
2. Parquet for features (embeddings, scores, metadata)
3. Generic getter/setter API for all steps
4. Cache invalidation based on file mtime
5. CLI commands for cache inspection

**Notes**:
- Proposal in `docs/PROPOSAL_PIPELINE_CACHE_LAYER.md`
- Replaces current SQLite `UniversalCache` approach
- Enables fast repeat runs without recomputation

---
<!-- Format:
### YYYY-MM-DD: Feature title
**Status**: Open / In Progress / Done / Verified
**Description**: What the user requested
**Notes**: Implementation notes, feedback, etc.
-->

### 2026-02-19: ML Developer Skills & Benchmark Comparison Tool
**Status**: Done
**Description**:
1. Document ML developer best practices from recent experience
2. Create benchmark comparison tool for comparing runs
3. Emphasize reproducibility, debug panels, and always testing

**Notes**:
- Created `docs/ML_DEVELOPER_SKILLS.md` with comprehensive guidelines
- Created `scripts/compare_benchmarks.py` for comparing benchmark runs
- Added learnings about coordinate systems and debug panels to LEARNINGS.md

---

### 2026-02-19: Three-Version Face Debug Panel
**Status**: Done
**Description**:
For every face, provide access to:
1. Original full image (with bbox and landmarks drawn)
2. Raw cropped face (bbox only, no alignment)
3. Transformed/aligned face (5-point affine aligned)

Purpose: Allow easy debugging of face detection/alignment pipeline without extra effort.

**Notes**:
- Added `get_face_crop_raw()` and `get_original_image_with_bbox()` to loaders
- New `render_face_debug_panel()` shows all three side-by-side
- Click 🔍 on any face in gallery to see debug panel

---

### 2026-02-19: Clustering Algorithm Documentation & Face Debug Improvements
**Status**: Done
**Description**:
1. Add `doc_explanation` (5-6 lines) and `decision_parameters` dict to every clustering algorithm
2. Show relevant parameters, thresholds, and actual values in debug UI
3. Fix face alignment (use 5-point landmarks, not just 2 eyes)
4. Add image filename to face gallery captions
5. Add detailed face view on click (landmarks overlay, filename, alignment info)

**Notes**:
- Sprint plans in `docs/SPRINT_PLANS_CLUSTERING_DEBUG.md` - all 10 sprints complete
- All clustering methods documented: HDBSCAN, hybrid_hdbscan_knn, hybrid_closest_face, Tcore2all, merge_twotier, attach_strong1, mutual_knn, dbscan, kmeans, hierarchical
- 5-point alignment implemented in `face_alignment.py` using ArcFace reference template
- Dynamic algorithm explanation loads doc_explanation from clustering method class
- Fixed landmark coordinate normalization for display
