# Feature Requests

This file tracks feature requests from users. Claude should scan this on init to check for open requests.

---

<!-- Add new entries at the top, newest first -->

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
