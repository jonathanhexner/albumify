# Change Log

**Purpose**: Track all code modifications with timestamps for debugging and history.

**Format**: Each entry includes date/time (ISO 8601), files modified, change description, and reason.

---

### 2026-02-27 11:30:00
**Files**: `notebooks/export_clustering_data.ipynb`
**Change**: Created interactive notebook version of clustering export
**Reason**: Provide both CLI script (batch processing) and notebook (exploration) for data export

**Features**:
- Step-by-step pipeline with visualizations (blur distribution, T_A distribution, diameter distribution, feature correlations)
- Config cell for easy parameter editing
- Quality gating with pose estimation (optional)
- Cluster statistics computation and visualization
- Candidate pair generation with distance distribution plots
- Feature computation with correlation matrix
- CSV export with summary
- Same output as CLI script, but interactive for exploration

### 2026-02-27 11:26:00
**Files**: `scripts/export_clustering_data.py`, `docs/PLAN_ML_CLUSTER_MERGING.md`, `docs/FEATURE_REQUESTS.md`
**Change**: Created Phase 1 of ML-based cluster merging pipeline - export script
**Reason**: Replace heuristic cluster merging with logistic regression model trained on labeled data

**Implementation**:
- Created `scripts/export_clustering_data.py` with CLI interface
- Runs clustering pipeline: quality gating → kNN graph → connected components → exemplar selection
- Exports 3 CSVs:
  - `faces.csv` - one row per face with metadata (face_id, image_path, cluster_id, bbox, blur_score, pose, is_core)
  - `clusters.csv` - one row per cluster with stats (cluster_id, size, exemplar_ids, diameter, T_A, mean_blur, face_ids)
  - `candidate_pairs.csv` - cluster pair features (12 features: min_exemplar_dist, p10/p50_cross_dist, support_fraction, diameter_ratio, cluster sizes, T_A/T_B/T_local/T_global)
- Tested on sample dataset (254 faces → 39 clusters → 39 candidate pairs)
- All features computed correctly, no NaN values

**Next Steps**: Phase 2 (Streamlit labeling interface), Phase 3 (training script), Phase 4 (deployment)

---

### 2026-02-26 06:00:00
**Files**: `docs/KNN_CLUSTERING_PIPELINE.md`, `docs/MERGE_CRITERIA_EXPLAINED.md`
**Change**: Added tables and comprehensive failure analysis to documentation
**Reason**: User questions about failure patterns and Min_Dist vs Exemplar_Dist. Enhanced docs with:

**1. Summary Tables** - Quick reference for all criteria:
- Criterion name, what it checks, parameters, defaults, recommendations
- Failure patterns and what they mean
- Min_Dist vs Exemplar_Dist comparison

**2. Understanding Failures** - Detailed analysis:
- Table showing failure patterns (Exemplar only, Support only, Multiple, etc.)
- When each pattern is a problem vs working correctly
- Example: "Exemplar + Support + Diameter" = genuinely different clusters (DON'T merge)

**3. Why Changing Alpha Won't Always Help**:
- Table showing T_merge at different alpha values
- Example: clusters (3, 6) with exemplar_dist=0.413
  - Even with alpha=0 (100% global): T_merge=0.307 < 0.413 → still fails
  - Conclusion: These ARE different clusters (tight internally, far apart)

**4. Min_Dist vs Exemplar_Dist**:
| Metric | Definition | Used In | Purpose |
|--------|------------|---------|---------|
| Min_Dist | ANY two faces | Close Clusters | Geometric proximity |
| Exemplar_Dist | Exemplars only | Merge Decisions | Robust merge decisions |

**Example**: Pair (4, 23) has Min_Dist=0.278 but Exemplar_Dist=0.520
- NOT in Merge Decisions (exemplar_dist > 0.45)
- Shows outliers close, but cores far apart (correct behavior)

**5. Troubleshooting Decision Tree** - Step-by-step debugging guide

Documentation now answers: "Why aren't these merging?" with clear, tabular explanations.

### 2026-02-26 05:30:00
**Files**: `face_cluster/merge.py`
**Change**: Fixed merge_margin=0 to actually disable margin check
**Reason**: User correctly identified that even with merge_margin=0, the margin check was still enforcing "B must be THE CLOSEST cluster to all of A's exemplars".

**The Problem**:
```python
if dist_to_b + 0.0 > dist:  # Still requires B to be closest!
    return False
```

Even with margin=0, the check required B to be the absolute closest cluster for every exemplar in A. This is too strict - we already have exemplar distance, support, and diameter checks.

**The Fix**: Added early return when margin=0:
```python
def _check_margin(...):
    if self.config.merge_margin == 0.0:
        return True  # Disable check entirely
    # ... rest of check
```

Now `MERGE_MARGIN=0` truly disables the margin criterion, relying on the other 3 criteria.

### 2026-02-26 05:00:00
**Files**: `docs/KNN_CLUSTERING_PIPELINE.md` (new), `docs/MERGE_CRITERIA_EXPLAINED.md` (updated), `notebooks/debug_knn_graph_clustering.ipynb`
**Change**: Complete pipeline documentation for production use
**Reason**: User requested full pipeline documentation (not just merge criteria) to translate notebook to production code. Created comprehensive guide:

**docs/KNN_CLUSTERING_PIPELINE.md** - Complete pipeline documentation:
- Overview with pipeline diagram
- Each stage explained in detail (A through F4)
- All parameters documented with defaults
- Complete production code example
- ClusterSnapshot analysis/debugging guide
- Troubleshooting section

**Margin Criterion Clarified** - Added concrete example:
```
Exemplar X from cluster A:
  Distance to B: 0.233
  Distance to C: 0.240

Check (merge_margin=0.05):
  0.233 + 0.05 = 0.283
  Is 0.283 < 0.240? NO → FAIL

Reason: C is within margin (0.007 gap)
```

**Margin is distance to SECOND-CLOSEST cluster** - Must be ≥ merge_margin larger than distance to proposed merge partner.

**Notebook updated**: Added `MERGE_MARGIN=0.0` parameter with comment pointing to docs

**Purpose**: Team can now implement production pipeline from docs without notebook.

### 2026-02-26 04:00:00
**Files**: `face_cluster/analysis.py`, `notebooks/debug_knn_graph_clustering.ipynb`
**Change**: Simplified merge analysis - clean DataFrames, explanation in notebook
**Reason**: User feedback - too many prints, code bloated and hard to maintain. Refactored to:

**1. Clean Code** - Removed verbose prints, simplified logic:
- New method: `get_merge_decisions_df()` returns DataFrame (no side effects)
- New method: `get_close_clusters_df()` returns DataFrame
- `plot_decision_boundaries()` calls `get_merge_decisions_df()` and displays
- `plot_close_clusters()` calls `get_close_clusters_df()` and displays
- Fewer if statements, cleaner structure

**2. Explanation in Notebook** - Not in code:
- Added markdown cell before Stage F2 with:
  - 4 merge criteria
  - Adaptive threshold formula
  - DataFrame column explanations
- Code stays simple and maintainable

**3. Merge Decisions DataFrame**:
```
C1 C2 Size1 Size2 Exemplar_Dist  T_A  T_B T_local T_global T_merge  Gap Merged Failed
0  1   3     2     0.35          0.25 0.30 0.30   0.27     0.291   0.059 False  Exemplar
2  3   2     2     0.32          0.28 0.26 0.28   0.27     0.277   0.043 False  Exemplar, Support
```
Shows exactly why each pair didn't merge.

**4. Close Clusters DataFrame**:
```
C1 C2 Size1 Size2 Min_Dist T_merge   Gap
1  3   2     2     0.026   0.291    -0.265  (would merge)
0  1   3     2     0.048   0.291    -0.243  (would merge)
```
Negative Gap = would merge if criteria passed.

**Result**: Clean, maintainable code + clear DataFrames + explanation in notebook where it belongs!

### 2026-02-26 03:00:00
**Files**: `face_cluster/types.py`, `face_cluster/analysis.py` (new), `face_cluster/merge.py`, `face_cluster/__init__.py`, `notebooks/debug_knn_graph_clustering.ipynb`, `test_clustersnapshot_workflow.py` (new)
**Change**: Added unified ClusterSnapshot for analysis and filename traceability
**Reason**: User correctly identified that passing around multiple variables (faces, core_indices, cluster_result, graph_result) is messy and makes analysis difficult. Implemented and **tested** comprehensive solution:

**1. Filename Traceability** - Added to FaceRecord:
- `image_path: Optional[str]` - Full path to source image
- `face_index: Optional[int]` - Which face in that image (0, 1, 2...)
- Notebook now loads metadata JSON and populates with actual filenames (e.g., "20250822_112331.jpg")
- 274 faces from 104 unique images properly tracked

**2. ClusterSnapshot Class** (face_cluster/analysis.py):
Unified data structure for cluster analysis at any stage:
- Contains: faces, core_indices, labels, distance_matrix, clusters, stats, exemplars
- Decision metadata: cluster_thresholds, merge_candidates (for sensitivity analysis)
- Standard analyses built-in:
  - `plot_overview()` - Top K clusters with N faces each, shows source images
  - `plot_close_clusters()` - Clusters nearly merged (sensitivity to threshold)
  - `plot_widest_clusters()` - Distance distribution for widest clusters (quality check)
  - `plot_decision_boundaries()` - Per-cluster thresholds vs merge candidates
  - `compare_with()` - Before/after comparison (e.g., pre/post merge)
  - `get_source_images()` - Unique source images per cluster
  - `print_cluster_sources()` - Image breakdown (helps spot bad merges)
- Factory method: `ClusterSnapshot.from_result()` creates from ClusterResult
- Properties: n_clusters, n_noise, n_core, n_total

**3. Merge Decision Metadata** - Updated ConservativeMerger:
- Stores `last_thresholds` (per-cluster adaptive thresholds)
- Stores `last_candidates` (all proposed merges with evidence)
- Enables sensitivity analysis: "How close were we to merging clusters X and Y?"

**4. Updated Notebook Workflow** - Now clean and consistent:
```python
# After initial clustering
snapshot_initial = ClusterSnapshot.from_result(
    cluster_result, faces, core_indices, distance_matrix,
    stage="initial_clustering", config=config
)
snapshot_initial.print_summary()
snapshot_initial.plot_overview()
snapshot_initial.plot_widest_clusters(top_k=3)

# After merge
snapshot_merged = ClusterSnapshot.from_result(
    ..., cluster_thresholds=merger.last_thresholds,
    merge_candidates=merger.last_candidates
)
snapshot_merged.plot_decision_boundaries()  # Why these merges happened
snapshot_merged.plot_close_clusters(top_k=5)  # Sensitivity analysis
snapshot_merged.compare_with(snapshot_initial)  # What changed
```

**Benefits**:
- Single data structure replaces passing around 4+ variables
- Consistent analysis at any stage (initial, merge, split, final)
- Full traceability: cluster → faces → source images → filenames
- Sensitivity analysis: decision boundaries, close clusters, widest clusters
- Easy before/after comparison
- Standard analyses work identically across all stages

This makes the library much more usable for exploratory analysis and the notebook much cleaner!

**Testing** - Created `test_clustersnapshot_workflow.py` to verify:
- ✓ Metadata JSON loading with actual filenames (274 faces from 104 images)
- ✓ FaceRecord creation with image_path and face_index fields
- ✓ ClusterSnapshot.from_result() factory method
- ✓ Properties: n_clusters, n_noise, n_core, n_total
- ✓ get_source_images() and print_cluster_sources() methods
- ✓ Full clustering workflow (20 faces → 5 clusters + 4 noise)
- ✓ Merge workflow with decision metadata (last_thresholds, last_candidates)
- ✓ compare_with() before/after comparison
- All tests passed successfully!

### 2026-02-26 02:15:00
**Files**: `test_notebook_execution.py` (new)
**Change**: Created comprehensive test script to verify notebook execution
**Reason**: User asked "does the notebook run without errore?" - Created test_notebook_execution.py to verify all pipeline components work:
- ✓ All imports (face_cluster module + viz functions)
- ✓ Config creation with merge parameters
- ✓ FaceRecord creation with embeddings and aligned faces
- ✓ Quality gating (blur scores + core set selection)
- ✓ Full pipeline flow: distance matrix → mutual kNN graph → clustering → exemplar selection → conservative merge
- ✓ Existing embeddings loading (found 274 embeddings + face crops)
- All tests passed successfully!
**Note**: SixDRepNet pose estimation is optional (requires `pip install sixdrepnet`). Without it, notebook runs with blur-only quality filtering.

### 2026-02-24 22:45:00
**Files**: `face_cluster/config.py`, `face_cluster/merge.py`
**Change**: Implemented hybrid global/local adaptive thresholds for merging
**Reason**: User correctly pointed out hard thresholds don't adapt to data. Implemented elegant solution:
- **Per-cluster thresholds**: T_i = P90 of exemplar pairwise distances (captures cluster-specific scale)
- **Global threshold**: T_global = median([T_1, ..., T_n]) (dataset-wide context)
- **Hybrid formula**: T_merge = α × max(T_A, T_B) + (1-α) × T_global
  - Uses MAX not MIN (allows merging across different density regions)
  - α=0.7 default (70% local, 30% global)
  - Prevents both over-fragmentation (local) and over-merging (global)
- **Adaptive diameter**: max_allowed = max(diam_A, diam_B) × expansion_factor (default 1.5)
- **Config params**: merge_use_adaptive_threshold, merge_exemplar_percentile, merge_threshold_alpha, merge_diameter_expansion_factor
- Thresholds recomputed after each merge (stays adaptive throughout iterations)
Much more robust across different datasets, lighting conditions, and embedding spaces!

### 2026-02-24 22:30:00
**Files**: `face_cluster/config.py`, `face_cluster/merge.py`, `face_cluster/__init__.py`, `notebooks/debug_knn_graph_clustering.ipynb`
**Change**: Added Stage F2 - Conservative Merge with multi-evidence approach
**Reason**: User requested merge functionality to reduce over-fragmentation from connected components. Implemented:
- `ConservativeMerger` class in face_cluster/merge.py with multi-evidence criteria:
  - (A) Exemplar agreement: min exemplar distance ≤ threshold
  - (B) Support count: sufficient cross-cluster pairs below threshold
  - (C) Margin vs next best: prevent ambiguous chain merges
  - (D) Post-merge diameter: safety valve against over-wide clusters
- New config parameters: merge_enabled, merge_candidate_threshold, merge_exemplar_threshold, merge_pair_threshold, merge_support_min, merge_support_frac, merge_margin, post_merge_diameter_max
- Stage F2 cell in notebook (runs after exemplar selection, before splitting)
- Iterative merging: proposes candidates → evaluates all evidence → merges best pair → repeat
Now pipeline is: Detect → Quality gate → Graph → Cluster → Exemplars → Merge → Split → Attach

### 2026-02-24 22:20:00
**Files**: `notebooks/debug_knn_graph_clustering.ipynb`
**Change**: Added autoreload for development to imports cell
**Reason**: User got TypeError because Jupyter cached old module version. Added `%load_ext autoreload` and `%autoreload 2` to imports cell to automatically reload changed modules without kernel restart.

### 2026-02-24 22:15:00
**Files**: `face_cluster/quality.py`, `notebooks/debug_knn_graph_clustering.ipynb`
**Change**: Added SixDRepNet pose estimation for existing embeddings
**Reason**: User correctly pointed out we should use facial orientation filtering even with existing embeddings. Integrated:
- `PoseEstimator` class in quality.py using SixDRepNet (from sim_bench.face_pipeline.pose_estimator)
- `compute_pose_scores()` method in QualityGater to estimate yaw/pitch/roll from face crops
- Config toggle: `ESTIMATE_POSE_FROM_CROPS = True/False` in notebook
- Stage A0 now:
  - Computes blur scores from face crops (always)
  - Optionally computes pose from face crops using SixDRepNet (if enabled)
  - Filters by blur AND pose (if available): abs(yaw) <= yaw_max, abs(pitch) <= pitch_max, abs(roll) <= roll_max
  - Shows distribution of yaw/pitch/roll angles
Now properly uses ALL quality assessment methods (blur + pose) even with existing embeddings.

### 2026-02-24 22:00:00
**Files**: `notebooks/debug_knn_graph_clustering.ipynb`
**Change**: Fixed quality gating for existing embeddings - now uses blur filtering
**Reason**: User correctly pointed out we should use quality assessment even with existing embeddings. Changed Stage A0 to:
- Compute blur scores from aligned face crops (using QualityGater.compute_blur_scores())
- Filter faces: core if blur >= BLUR_MIN, holdout if below
- Show blur score distribution (min/median/max)
- For new images: use full quality gating (blur + pose + per-image filtering)
- For existing embeddings: use blur-only filtering (no pose data available)
Now properly leverages the framework's quality assessment instead of skipping it.

### 2026-02-24 21:45:00
**Files**: `notebooks/debug_knn_graph_clustering.ipynb`
**Change**: Added dual-mode support - load existing embeddings OR detect from new images
**Reason**: User pointed out notebook couldn't load face crops from benchmark results. Added:
- Config toggle: `USE_EXISTING_EMBEDDINGS = True/False`
- `get_face_crop()` helper function to load from results/face_clustering_benchmark/face_crops/
- Stage A now supports both paths:
  - Option A: Load pre-computed embeddings + face crops (fast, no InsightFace)
  - Option B: Run InsightFace on new images (requires installation)
- All 274 benchmark faces treated as core set when loading existing
- Handles large datasets (>100 faces) by sampling for visualizations

### 2026-02-24 21:30:00
**Files**: `test_face_cluster.py`, `test_face_cluster_clustering.py`, `notebooks/test_knn_with_existing_embeddings.ipynb`, `check_benchmark_data.py`
**Change**: Added verification tests for face_cluster library
**Reason**: User requested verification that the library actually runs. Created:
- `test_face_cluster.py`: Basic functionality test with mock data
- `test_face_cluster_clustering.py`: Clustering verification with synthetic 3-cluster data (passes)
- `notebooks/test_knn_with_existing_embeddings.ipynb`: Test notebook using pre-computed embeddings from benchmark results (274 faces)
- `check_benchmark_data.py`: Utility to inspect benchmark data
All tests pass successfully. Library is verified working.

### 2026-02-24 21:00:00
**Files**: `face_cluster/__init__.py`, `face_cluster/types.py`, `face_cluster/config.py`, `face_cluster/embedding.py`, `face_cluster/quality.py`, `face_cluster/knn_graph.py`, `face_cluster/clustering.py`, `face_cluster/exemplars.py`, `face_cluster/attach.py`, `face_cluster/viz.py`, `notebooks/debug_knn_graph_clustering.ipynb`, `docs/FEATURE_REQUESTS.md`
**Change**: Created comprehensive KNN graph clustering library and debug notebook
**Reason**: User requested face clustering pipeline for small batches (10-20 faces) with high precision and interpretability. Implemented:
- **face_cluster/** module with 10 files:
  - types.py: Dataclasses (FaceRecord, GraphResult, ClusterResult)
  - config.py: PipelineConfig with all hyperparameters
  - embedding.py: InsightFace wrapper for detection + embeddings
  - quality.py: QualityGater for pose/blur filtering + core/holdout split
  - knn_graph.py: KNNGraphBuilder for mutual kNN graph construction
  - clustering.py: ConnectedComponentsClusterer with optional splitting
  - exemplars.py: D10ExemplarSelector using d10 density metric
  - attach.py: HoldoutAttacher with vote+margin strategy
  - viz.py: Visualization helpers (heatmaps, graphs, face grids)
- **notebooks/debug_knn_graph_clustering.ipynb**: Step-by-step notebook with:
  - Config cell for easy hyperparameter tuning
  - 8 stages (A-G) with visualizations
  - Manual intervention points (override core set, edit edge list)
  - Export results to JSON
- Logged feature request in docs/FEATURE_REQUESTS.md

---

### 2026-02-23 18:15:00
**Files**: `notebooks/debug_hybrid_simple.ipynb`
**Change**: Added UMAP-based clustering exploration section
**Reason**: User wants to cluster in UMAP 2D space (where clusters are visible) using KMeans, then analyze with existing tools. Added:
- `cluster_in_umap_space()` - Computes UMAP, runs KMeans, visualizes with centroids
- 4 new cells for UMAP cluster analysis:
  - Statistics table (with exemplars and thresholds)
  - Face galleries
  - Cluster pair comparison
  - Threshold analysis
Now can compare UMAP-based clustering vs HDBSCAN in original space.

### 2026-02-23 18:10:00
**Files**: `notebooks/debug_hybrid_simple.ipynb`
**Change**: Added PCA dimensionality reduction option (128/256 dims)
**Reason**: User observed UMAP shows clear clusters but HDBSCAN over-merges in 512-dim space. Added PCA preprocessing options to configs:
- PCA-128: 94% variance, 22 clusters (prevents over-merging!)
- PCA-256: 99.93% variance, 18 clusters (middle ground)
- No PCA: 8 clusters with massive 161-face cluster (over-merged)
PCA helps by reducing noise dimensions and making distances more meaningful.

### 2026-02-23 18:05:00
**Files**: `notebooks/debug_hybrid_simple.ipynb`
**Change**: Added exemplar statistics to main table
**Reason**: User requested exemplar distances and thresholds in the table. Updated `compute_all_cluster_stats()` to include:
- `n_ex` - number of exemplars
- `ex_min`, `ex_med`, `ex_p90`, `ex_max` - exemplar pairwise distances
- `t_med_iqr` - threshold using median + 1.5×IQR (original idea)
- `t_d3_p90` - threshold using all-faces d3 P90
- `t_ex_p90` - threshold using exemplar pairwise P90 (hybrid algorithm)
Now table shows all three threshold methods side-by-side for comparison.

### 2026-02-23 18:00:00
**Files**: `notebooks/debug_hybrid_simple.ipynb`
**Change**: Added exemplar-based threshold analysis
**Reason**: User asked about exemplar distances and merge/split criteria from hybrid algorithm. Added `analyze_exemplar_threshold()` function that:
- Selects top-10 exemplars (smallest d3 values)
- Computes exemplar pairwise distances
- Compares 3 threshold methods: exemplar P90 (hybrid algo), all-faces P90, median+1.5×IQR
- Shows exemplar faces
This reveals what the hybrid algorithm actually uses for merge decisions.

### 2026-02-23 17:50:00
**Files**: `notebooks/debug_hybrid_simple.ipynb`
**Change**: Restored all missing functions and visualization cells
**Reason**: User correctly pointed out I removed important sections. Added back:
- `visualize_umap()` - UMAP visualization of clusters
- `show_clusters()` - Face galleries for largest clusters
- `show_cluster_pair()` - Compare two clusters with faces and distance distributions
- `show_cluster_threshold_analysis()` - Detailed threshold analysis with plots
Now has 9 cells total with all functionality preserved.

### 2026-02-23 17:45:00
**Files**: `notebooks/debug_hybrid_simple.ipynb`
**Change**: Rewrote notebook from scratch with correct cell types, executed and verified
**Reason**: Cell 0 was incorrectly marked as markdown causing NameError. Completely rewrote, tested with jupyter nbconvert --execute, confirmed all 5 cells run successfully and produce correct table output.

### 2026-02-23 17:40:00
**Files**: `notebooks/debug_hybrid_simple.ipynb`
**Change**: Fixed dtype mismatch for HDBSCAN (added .astype(np.float64))
**Reason**: HDBSCAN requires float64 but distance matrix was float32, causing ValueError. Verified working - produces clean table with 8 clusters from 274 faces.

### 2026-02-23 17:35:00
**Files**: `notebooks/debug_hybrid_simple.ipynb`
**Change**: Fixed data paths to use correct results directory
**Reason**: Notebook was loading from wrong path. Fixed to use `../results/face_clustering_benchmark/` with glob to load most recent embeddings file. Face crops also load from same directory.

### 2026-02-23 17:30:00
**Files**: `notebooks/debug_hybrid_simple.ipynb`
**Change**: Complete rewrite to clean, minimal notebook (5 cells)
**Reason**: User requested cleanup - too messy and unmanageable with excessive prints. New version:
- Minimal prints, all output via pandas DataFrames
- Single function `compute_all_cluster_stats()` returns table for all clusters
- Table columns: cluster, n, d3_min/p10/med/p90/max, pw_min/p10/med/p90/max, t_med_iqr, t_p90
- 5 cells total: load → functions → test configs → show table → optional faces
- Removed duplicate cells and verbose output

### 2026-02-23 17:15:00
**Files**: `notebooks/debug_hybrid_simple.ipynb`
**Change**: Added `show_all_clusters_table()` function and demonstration cell
**Reason**: User requested table view with statistics for all cluster IDs. Table shows:
- d3 statistics (min, P10, median, P90, max, IQR)
- Pairwise distance statistics (min, P10, median, P90, max, IQR)
- Both threshold methods (median+1.5×IQR vs P90)
Added markdown cell explaining column names.

### 2026-02-23 17:00:00
**Files**: `notebooks/debug_hybrid_simple.ipynb`
**Change**: Added comprehensive inner cluster distance statistics
**Reason**: User requested detailed statistics. Now shows for both d3 and pairwise distances:
- Min, Max, P10, P90, Median, IQR
- Plots with P10/P90 lines on pairwise histogram

### 2026-02-23 16:50:00
**Files**: `notebooks/debug_hybrid_simple.ipynb`
**Change**: Added threshold calculation functions
**Reason**: User asked how thresholds are calculated. Added `compute_cluster_stats()` and `show_cluster_threshold_analysis()`

### 2026-02-23 16:40:00
**Files**: `notebooks/debug_hybrid_simple.ipynb`
**Change**: Complete reorganization with utility functions and epsilon explanation
**Reason**: User requested simpler organization and epsilon clarification. New structure:
- Utility functions: `visualize_umap()`, `show_clusters()`, `show_cluster_pair()`
- Test 4 HDBSCAN configs (explained epsilon: higher = fewer clusters)
- Simple interface: change `selected_idx` to switch configs
- Optional hybrid algorithm comparison at end

---

### 2026-02-23 16:20:00
**Files**: `notebooks/debug_hybrid_simple.ipynb`
**Change**: Added UMAP visualization and interactive 2-cluster debugger
**Reason**: User requested visual cluster inspection. Added:
- UMAP plot colored by cluster labels
- Interactive 2-cluster debug: shows faces, distances, histograms, and heatmap side-by-side
- Set CLUSTER_A and CLUSTER_B to compare any two clusters

### 2026-02-23 16:10:00
**Files**: `notebooks/debug_hybrid_simple.ipynb`
**Change**: Added explanatory markdown cells throughout notebook
**Reason**: User needed clarification on results. Added 5 markdown sections explaining phases, thresholds (t_original vs t_current), distance plots, and decision logic.

### 2026-02-23 16:00:00
**Files**: `notebooks/debug_hybrid_simple.ipynb`
**Change**: Rewrote notebook from scratch with face visualization
**Reason**: File became corrupted. Clean rewrite with 7 core cells + explanations + visualizations.

### 2026-02-23 15:45:00
**Files**: `notebooks/debug_hybrid_simple.ipynb`
**Change**: Fixed IndexError when looking up singleton clusters
**Reason**: `find_close_pairs` was trying to look up thresholds for singleton clusters (size=1) that were skipped in `compute_stats`. Now filters to only include clusters with size >= 2.

### 2026-02-23 15:30:00
**Files**: `notebooks/debug_hybrid_simple.ipynb` (NEW)
**Change**: Created concise debugging notebook for hybrid clustering algorithms
**Reason**: User needs simple tool (max 150 lines) to understand why hybrid HDBSCAN algorithms produce poor results despite clear UMAP clusters. Notebook helps debug threshold computation and merge decisions.
**Features**:
- Compare threshold methods: median + k×IQR (original idea) vs percentile (current)
- Analyze specific cluster pairs (why didn't they merge?)
- Find closest cluster pairs with merge predictions
- Test simplified merge algorithm

---

## 2026-02-21 (Feature: Mutual KNN Two-Stage Clustering)

**Files**:
- `sim_bench/clustering/pruning_strategies.py` (NEW)
- `sim_bench/clustering/mutual_knn_two_stage.py` (NEW)
- `sim_bench/clustering/distance_utils.py` (modified - added cluster debug utilities)
- `sim_bench/clustering/base.py` (modified - registered new algorithm)
- `configs/clustering_benchmark.yaml` (modified - added 3 config variants)
- `tests/clustering/test_mutual_knn_two_stage.py` (NEW)

**Change**: Implemented two-stage mutual kNN clustering with pluggable pruning strategy

**Details**:
Two-stage algorithm separates graph construction from membership validation:

**Stage 1**: Build mutual kNN graph (no/loose threshold) → connected components = initial clusters

**Stage 2**: Iterative refinement loop:
1. Prune: For each sample, validate membership using pruning strategy
2. Reassign: Unassigned samples try to join valid clusters
3. Repeat until convergence or max iterations

**Pruning Strategy (RedundantSupportStrategy)**:
Sample stays in cluster if:
- Base condition: closest_dist ≤ α·X (X=0.45, α=1.1 → max 0.495)
- AND one of:
  - Redundant support: ≥m neighbors within β·X (m=2, β=1.05)
  - Separation: margin to next-best cluster ≥ δ (δ=0.15)

**New modules**:
- `pruning_strategies.py`: PruningStrategy ABC + RedundantSupportStrategy
- `mutual_knn_two_stage.py`: MutualKNNTwoStageClusterer
- `distance_utils.py`: Added closest_distance_to_cluster(), all_cluster_distances(), support_count(), separation_margin()

**Debug data stored**: Raw distance_matrix + cluster_members for frontend analysis

**Benchmark results (274 faces)**:
| Method | Clusters | Noise | Top sizes |
|--------|----------|-------|-----------|
| mutual_knn_two_stage | 36 | 27 | 75, 52, 25, 18, 16 |
| mutual_knn_two_stage_strict | 33 | 59 | 61, 50, 19, 16, 16 |
| mutual_knn_two_stage_loose | 23 | 9 | 156, 35, 27, 19, 6 |
| hdbscan | 8 | 23 | 161, 40, 27, 7, 6 |
| mutual_knn (original) | 139 | 0 | 46, 22, 15, 14, 9 |

**Reason**: User requested two-stage clustering that first builds kNN graph clusters, then prunes weak connections with controllable strategy allowing larger distances if multiple neighbors support membership or clear separation from other clusters.

---

## 2026-02-20 (Feature: kNN Split for HDBSCAN Variants)

**Files**:
- `sim_bench/clustering/hybrid_hdbscan_knn.py` (modified)
- `sim_bench/clustering/hybrid_closest_face.py` (modified)
- `scripts/cluster_knn_components.py` (NEW)
- `scripts/benchmark_hdbscan_split.py` (NEW)

**Change**: Added post-clustering split functionality using kNN connected components

**Details**:
The HDBSCAN variants were over-merging faces into large clusters. Added a new Stage 3 (split) phase:

1. For each cluster >= `split_min_cluster_size` (default: 10):
   - Build kNN graph (k = `split_k` neighbors per face)
   - Prune edges where cosine_similarity < `split_threshold`
   - Find connected components in pruned graph
   - If multiple components exist, split into separate clusters

2. New parameters added to both hybrid methods:
   - `split_enabled`: Enable/disable split phase (default: True)
   - `split_threshold`: Cosine similarity threshold (default: 0.65)
   - `split_min_cluster_size`: Min size to consider splitting (default: 10)
   - `split_k`: K neighbors for kNN graph (default: 20)

3. New scripts:
   - `cluster_knn_components.py`: Standalone kNN + connected components clustering
   - `benchmark_hdbscan_split.py`: Compare HDBSCAN variants with different split thresholds

**Reason**: User reported HDBSCAN variants over-merging ~161 faces into one cluster. The kNN connected components approach ensures faces only stay clustered if connected by strong similarity paths.

---

## 2026-02-20 (Feature: Embedding Analysis Tools)

**Files**:
- `scripts/face_distance_report.py` (NEW)
- `scripts/debug_face_distances.py` (NEW)
- `app/face_clustering_debug/pages/embedding_analysis.py` (NEW)
- `app/face_clustering_debug/main.py` (modified)
- `app/face_clustering_debug/components/face_grid.py` (modified - key_prefix param)
- `app/face_clustering_debug/pages/parameter_tuning.py` (modified)
- `app/face_clustering_debug/pages/overview.py` (modified)

**Change**: Added embedding analysis tools for diagnosing clustering issues

**Details**:
1. **face_distance_report.py**: Generates HTML diagnostic report comparing two groups of faces
   - Shows face thumbnails, distance histograms, overlap analysis
   - Identifies problematic pairs with high intra-group distances
2. **Embedding Analysis tab**: New tab in face clustering debug app with:
   - Global UMAP visualization colored by cluster/confidence/frontal score
   - Per-cluster UMAP to identify sub-groups
   - Distance comparison tool for same/different person analysis
   - HDBSCAN condensed tree visualization
3. **Fixed duplicate key error**: Added `key_prefix` parameter to `render_face_grid`

---

## 2026-02-20 (Fix: Face Alignment Margin, Landmark Swap Bug, and Debug App)

**Files**:
- `sim_bench/pipeline/steps/align_faces.py` (modified)
- `scripts/benchmark_face_clustering.py` (modified)
- `app/face_clustering_debug/services/db_loader.py` (modified)
- `app/face_clustering_debug/services/file_loader.py` (modified)
- `tests/pipeline/test_face_orientation_detection.py` (modified)

**Change**: Fixed generous crop margin calculation, removed incorrect landmark swapping, fixed debug app

**Reason**: User reported pixel smearing artifacts and incorrect alignment in aligned faces.

**Root Causes Identified**:
1. `crop_face_generous` was computing margin from landmark span (~114px) instead of bbox (~302px)
2. **Critical Bug**: After rotation, landmarks were being swapped (`[1,0,2,4,3]`), but this was WRONG. Landmark labels (L_eye, R_eye) refer to the PERSON's left/right eye, not image position. The affine transform handles this correctly without swapping.

**Details**:
1. **crop_face_generous**: Added `bbox` parameter, handles both `x_px/y_px` and `x/y` formats
2. **rotate_image_and_landmarks**: REMOVED incorrect landmark swap after rotation
3. **align_face_with_orientation**: Updated to accept and pass bbox parameter
4. **benchmark script**: Now passes bbox dict to alignment functions
5. **db_loader.py**: Uses `align_face_with_orientation` with orientation detection
6. **file_loader.py**: Looks for `face_XXXX_aligned.jpg` (new format)

---

## 2026-02-19 (Refactor: Face Alignment Pipeline - SIGHTING-001)

**Files**:
- `sim_bench/pipeline/steps/detect_face_orientation.py` (NEW)
- `sim_bench/pipeline/steps/align_faces.py` (NEW)
- `sim_bench/pipeline/steps/validate_alignment.py` (NEW)
- `sim_bench/pipeline/steps/crop_faces.py` (NEW)
- `sim_bench/pipeline/steps/extract_face_embeddings.py` (REWRITTEN - single responsibility)
- `sim_bench/pipeline/steps/all_steps.py` (modified)
- `configs/pipeline.yaml` (modified)
- `tests/pipeline/test_face_orientation_detection.py` (NEW)
- `tests/pipeline/test_face_alignment.py` (NEW)
- `tests/pipeline/steps/test_extract_face_embeddings.py` (REWRITTEN)
- `docs/SIGHTINGS.md` (modified)

**Change**: Implemented single-responsibility face alignment architecture to fix upside-down face detection

**Reason**: SIGHTING-001 - Face #118 was upside-down but only rotated 6° instead of 180°. Root cause: `compute_roll_angle()` only measures eye-line tilt, not face orientation.

**Details**:
1. **detect_face_orientation step**: Analyzes 5-point landmarks to detect 0°/90°/180°/270° rotation needed
   - Checks vertical relationships (eyes above nose, nose above mouth)
   - Stores `orientation_angle` in face_info

2. **align_faces step**: Applies orientation correction + 5-point affine alignment
   - Pre-rotates image by detected orientation
   - Transforms landmarks to rotated coordinates
   - Stores aligned crops in `context.aligned_faces`

3. **validate_alignment step**: Verifies alignment worked correctly
   - Runs face detection on aligned crops
   - Checks landmarks are near expected positions

4. **crop_faces step**: Simple bbox cropping without alignment (for debug)

5. **extract_face_embeddings**: REWRITTEN with true single responsibility
   - Requires `aligned_faces` from `align_faces` step
   - No fallbacks, no "backward compatibility" - deterministic flow
   - Just reads aligned faces and extracts embeddings

6. **Unit tests**: 40 tests total
   - 15 tests for orientation detection (including Face #118 regression)
   - 16 tests for alignment
   - 9 tests for extract_face_embeddings (updated for new architecture)

---

## 2026-02-19 (Feature: Three-Version Face Debug Panel)

**Files**:
- `app/face_clustering_debug/services/protocols.py` (modified)
- `app/face_clustering_debug/services/file_loader.py` (modified)
- `app/face_clustering_debug/services/db_loader.py` (modified)
- `app/face_clustering_debug/components/face_detail.py` (modified)
- `app/face_clustering_debug/pages/overview.py` (modified)

**Change**: Added comprehensive debug panel showing all three versions of each face

**Reason**: User requested ability to easily debug face detection/alignment pipeline without extra effort

**Details**:
- Added new protocol methods:
  - `get_face_crop()` - 5-point aligned face (existing)
  - `get_face_crop_raw()` - bbox crop only, no alignment (NEW)
  - `get_original_image_with_bbox()` - full image with bbox/landmarks drawn (NEW)
- New `render_face_debug_panel()` component shows all three side-by-side:
  1. **Original + BBox**: Source image with green bbox and colored landmark dots
  2. **Raw Crop**: Bbox-only crop with no rotation/alignment
  3. **Aligned Crop**: 5-point affine aligned to ArcFace template
- Updated gallery to show debug panel when clicking 🔍 on any face
- Each version shows appropriate landmarks for that stage of the pipeline

---

## 2026-02-19 (Fix: Landmark-Face Alignment Mismatch)

**Files**:
- `app/face_clustering_debug/components/face_detail.py` (modified)
- `app/face_clustering_debug/pages/overview.py` (modified)

**Change**: Fixed landmark positions not matching aligned face crops

**Reason**: When 5-point alignment is used, the face is transformed to ArcFace reference template positions. The original landmarks (pre-alignment) don't match the aligned crop.

**Details**:
- Added `_ARCFACE_REF_LANDMARKS_NORMALIZED` constant (the target positions for 5-point alignment)
- `render_face_detail()` now uses reference landmarks by default (since crops are 5-point aligned)
- Added `use_aligned_landmarks` parameter to optionally use original landmarks
- Updated Explorer tab to use reference landmarks for its face display
- Landmarks should now overlay correctly on aligned face crops

**Note**: If you have old cached face crops (pre-5-point alignment), they may still show misalignment. Re-run the benchmark to regenerate crops with proper alignment.

---

## 2026-02-19 (Fix: Face Detail Integration)

**Files**:
- `app/face_clustering_debug/pages/overview.py` (modified)
- `app/face_clustering_debug/services/file_loader.py` (modified)
- `app/face_clustering_debug/services/db_loader.py` (modified)

**Change**: Fixed face detail view integration and landmark display

**Reason**: User couldn't see filename, landmarks in the gallery view

**Details**:
- Updated overview gallery to use `render_face_grid` component (with 🔍 inspect button)
- Added session state for persistent face selection
- Click 🔍 on any face to see detail panel with landmarks, filename, metrics
- Fixed landmark coordinate normalization:
  - Landmarks from InsightFace are in pixel coords
  - Now normalized to 0-1 range relative to face bbox for display
- Updated db_loader.get_face_crop() to use 5-point alignment when landmarks available

---

## 2026-02-19 (Sprint 10 Complete)

**Files**:
- `app/face_clustering_debug/components/algorithm_explanation.py` (rewritten)
- `app/face_clustering_debug/pages/merge_decisions.py` (modified)
- `app/face_clustering_debug/pages/attach_decisions.py` (modified)
- `app/face_clustering_debug/pages/overview.py` (modified)

**Change**: Dynamic algorithm explanation using clustering method metadata

**Reason**: Sprint 10 - Show actual doc_explanation and decision_parameters from clustering methods

**Details**:
- `render_algorithm_explanation()` now accepts algorithm and params arguments
- Dynamically loads clustering method using `load_clustering_method()`
- Displays `doc_explanation` from the actual clustering class
- Shows `decision_parameters` table with current vs default values
- Added `render_decision_summary()` for compact per-decision displays
- Updated merge_decisions, attach_decisions, and overview pages to pass algorithm/params
- Falls back to general guide if algorithm not found

---

## 2026-02-19 (Sprint 9 Complete)

**Files**:
- `sim_bench/pipeline/utils/face_alignment.py` (modified)
- `sim_bench/pipeline/steps/extract_face_embeddings.py` (modified)

**Change**: Implemented 5-point face alignment using ArcFace reference template

**Reason**: Sprint 9 - Replace 2-point (eye-only) roll rotation with proper 5-point affine transform

**Details**:
- Added `align_face_5point()` function using cv2.estimateAffinePartial2D
- ArcFace reference template (112x112) scaled to target size (256)
- Uses all 5 landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
- Falls back to roll-angle alignment if 5-point fails or landmarks unavailable
- Added `compute_alignment_quality()` for debugging transform quality
- Similarity transform normalizes position, scale, and rotation in one step

**Impact**: Face crops will be properly aligned regardless of head tilt. Cached embeddings may need clearing if alignment-sensitive.

---

## 2026-02-19 (Sprint 8 Complete)

**Files**:
- `app/face_clustering_debug/components/face_detail.py` (modified)

**Change**: Enhanced face detail view with full metadata

**Reason**: Sprint 8 - Show landmarks with labels, filename, path, all metrics

**Details**:
- Landmarks now labeled: LE (left eye), RE (right eye), N (nose), LM/RM (mouth)
- File info section: filename, full path, copyable code block
- Face metrics: confidence, frontal_score, eye_bbox_ratio, pose angles
- Bbox coordinates displayed
- Landmark legend in expandable section

---

## 2026-02-19 (Sprint 7 Complete)

**Files**:
- `app/face_clustering_debug/components/face_grid.py` (modified)

**Change**: Added image filename to face grid captions

**Reason**: Sprint 7 - Show filename for easier debugging of specific faces

**Details**:
- Caption now shows `⭐#42 IMG_1234` format (star for exemplars, index, truncated filename)
- Added `_truncate_filename()` helper to keep captions readable
- Extracts filename from `face.image_path` using `Path.stem`

---

## 2026-02-19 (Sprints 5-6 Complete)

**Files**:
- `sim_bench/clustering/hybrid_hdbscan_knn_Tcore2all.py` (modified)
- `sim_bench/clustering/hybrid_hdbscan_knn_merge_twotier.py` (modified)
- `sim_bench/clustering/hybrid_hdbscan_knn_attach_strong1.py` (modified)
- `sim_bench/clustering/mutual_knn.py` (modified)
- `sim_bench/clustering/dbscan.py` (modified)
- `sim_bench/clustering/kmeans.py` (modified)
- `sim_bench/clustering/hierarchical.py` (modified)

**Change**: Added documentation attributes to all remaining clustering methods

**Reason**: Sprints 5-6 - Complete clustering algorithm documentation

**Details**:
- Sprint 5: Tcore2all, merge_twotier, attach_strong1 variants documented
- Sprint 6: mutual_knn, dbscan, kmeans, hierarchical documented
- All 10 clustering methods now have doc_explanation and decision_parameters

---

## 2026-02-19 (Sprint 4 Complete)

**Files**:
- `sim_bench/clustering/hybrid_closest_face.py` (modified)

**Change**: Added documentation attributes to HybridHDBSCANClosestFace

**Reason**: Sprint 4 - Document hybrid_closest_face decision parameters (d3_cross, merge_min_faces, NOT min_dist)

**Details**:
- Added `doc_explanation`: 6-line explanation of face-based (not exemplar) merge decisions
- Added `decision_parameters`: 7 parameters (merge_min_faces, merge_threshold_multiplier, d3_cross role)
- Updated `_compute_stats()` to populate `last_run_info`

---

## 2026-02-19 (Sprint 3 Complete)

**Files**:
- `sim_bench/clustering/hybrid_hdbscan_knn.py` (modified)

**Change**: Added documentation attributes to HybridHDBSCANKNN

**Reason**: Sprint 3 - Document hybrid_hdbscan_knn decision parameters

**Details**:
- Added `doc_explanation`: 6-line explanation of exemplar-based merge/attach
- Added `decision_parameters`: 7 parameters (threshold_floor/ceiling, merge_min_pairs, attach_min_exemplars, etc.)
- Updated `_compute_final_stats()` to populate `last_run_info`

---

## 2026-02-19 (Sprint 2 Complete)

**Files**:
- `sim_bench/clustering/hdbscan.py` (modified)

**Change**: Added documentation attributes to HDBSCANClusterer

**Reason**: Sprint 2 - Document HDBSCAN decision parameters

**Details**:
- Added `doc_explanation`: 6-line explanation of density-based clustering
- Added `decision_parameters`: min_cluster_size, min_samples, cluster_selection_epsilon, cluster_selection_method
- Updated `cluster()` to populate `last_run_info` with runtime values

---

## 2026-02-19 (Sprint 1 Complete)

**Files**:
- `sim_bench/clustering/base.py` (modified)

**Change**: Added documentation attributes to ClusteringMethod base class

**Reason**: Sprint 1 - Foundation for clustering algorithm documentation

**Details**:
- Added `doc_explanation` class attribute (5-6 line algorithm explanation)
- Added `decision_parameters` class attribute (dict of param metadata)
- Added `last_run_info` instance attribute (stores thresholds from last run)
- Added `get_decision_info()` method (returns structured info for UI)

---

## 2026-02-19 (Sprint Plans)

**Files**:
- `docs/SPRINT_PLANS_CLUSTERING_DEBUG.md` (created)
- `docs/FEATURE_REQUESTS.md` (modified)

**Change**: Created sprint plans for clustering algorithm documentation and face debug improvements

**Reason**: User requested structured documentation for clustering algorithms with decision parameters, plus face gallery improvements

**Details**:
- 10 sprints covering: base class, all clustering algorithms, face grid filename, face detail view, 5-point alignment, decision UI
- Each clustering method will have `doc_explanation` and `decision_parameters` attributes
- Face alignment to use proper 5-point affine transform instead of 2-point rotation

---

## 2026-02-19 (HEIC Support)

**Files**:
- `sim_bench/pipeline/utils/image_cache.py` (modified)
- `sim_bench/image_quality_models/siamese_model_wrapper.py` (modified)
- `requirements.txt` (modified)

**Change**: Added HEIC/HEIF image format support

**Reason**: Pipeline failed with `PIL.UnidentifiedImageError` on .heic files from iPhone

**Details**:
1. Added `pillow-heif>=0.16` to requirements.txt
2. Registered HEIC opener in image_cache.py (central image loading)
3. Updated siamese_model_wrapper.py to use ImageCache instead of direct Image.open()

---

## 2026-02-19

**Files**:
- `README.md` (modified)
- `CLAUDE.md` (modified)
- `docs/FEATURE_REQUESTS.md` (created)

**Change**: Improved documentation organization

**Reason**: User requested moving app documentation to README.md and improving CLAUDE.md

**Details**:
1. **README.md**: Expanded "Streamlit Apps" section to "Applications" with all 5 apps (album, photo_organization, photo_analysis, face_clustering_debug, face_clustering_comparison)
2. **CLAUDE.md**: Simplified app commands to reference README.md, added clustering test commands
3. **Created docs/FEATURE_REQUESTS.md**: Missing file referenced in CLAUDE.md General section

---

## 2026-02-19 11:00:00

**Files**:
- `configs/clustering_benchmark.yaml` (modified)
- `scripts/benchmark_face_clustering.py` (modified)

**Change**: Added new clustering methods to benchmark config and made benchmark script dynamic

**Reason**: User requested ability to benchmark the new hdbscan_pca and mutual_knn methods

**Details**:

1. **Updated benchmark config** (`clustering_benchmark.yaml`):
   - Added `hdbscan_pca_128`: HDBSCAN with 128-dim PCA
   - Added `hdbscan_pca_256`: HDBSCAN with 256-dim PCA
   - Added `mutual_knn_k10_t70`: Mutual KNN with k=10, threshold=0.70
   - Added `mutual_knn_k10_t65`: Mutual KNN with k=10, threshold=0.65
   - Added `mutual_knn_k5_t70`: Mutual KNN with k=5, threshold=0.70
   - Total: 8 clustering methods now available for benchmarking

2. **Made benchmark script dynamic** (`benchmark_face_clustering.py`):
   - Added `get_clustering_methods_from_config()` to auto-discover methods from YAML
   - Updated `run_clustering_methods()` to accept dict of method configs
   - Added `run_clustering_method()` for running a single method
   - Summary now dynamically prints stats for all methods
   - No more hardcoded method names

---

## 2026-02-19 10:00:00

**Files**:
- `sim_bench/clustering/hdbscan_pca.py` (created)
- `sim_bench/clustering/mutual_knn.py` (created)
- `sim_bench/clustering/base.py` (modified)
- `sim_bench/pipeline/steps/cluster_people.py` (modified)
- `app/streamlit/components/pipeline_runner.py` (modified)
- `tests/clustering/test_mutual_knn.py` (created)

**Change**: Added two new face clustering algorithms: HDBSCAN+PCA and Mutual KNN

**Reason**: User requested additional clustering methods to improve face clustering quality

**Details**:

1. **HDBSCAN+PCA** (`hdbscan_pca.py`):
   - Applies PCA dimensionality reduction before HDBSCAN clustering
   - Configurable PCA dimensions: 64, 128 (default), 256
   - Reduces noise in high-dimensional embeddings
   - All standard HDBSCAN parameters supported

2. **Mutual KNN** (`mutual_knn.py`):
   - L2-normalizes embeddings
   - Computes cosine similarity matrix: S = E @ E.T
   - Finds top-k neighbors for each embedding (default k=10)
   - Builds mutual-KNN graph: edge (i,j) iff j in top-k(i) AND i in top-k(j) AND S[i,j] >= threshold
   - Runs connected components (scipy.sparse.csgraph)
   - Default similarity_threshold=0.70
   - No FAISS, no PCA, no HDBSCAN - pure numpy + scipy

3. **Factory Registration** (`base.py`):
   - Added `hdbscan_pca` and `mutual_knn` to clustering registry

4. **Pipeline Integration** (`cluster_people.py`):
   - Added support for both new methods in cluster_people step
   - Proper config parameter passing to clustering factory

5. **UI Controls** (`pipeline_runner.py`):
   - Added method selector with all 4 options: hdbscan, hdbscan_pca, mutual_knn, agglomerative
   - Added PCA dimensions dropdown (64/128/256) for hdbscan_pca
   - Added KNN k slider (3-20) and similarity threshold slider (0.50-0.90) for mutual_knn

6. **Tests** (`test_mutual_knn.py`):
   - Unit tests for both new clustering methods
   - Edge cases: empty input, single sample, high threshold

---

## 2026-02-17 11:00:00

**Files**:
- `app/face_clustering_debug/__init__.py` (created)
- `app/face_clustering_debug/models/__init__.py` (created)
- `app/face_clustering_debug/models/schemas.py` (created)
- `app/face_clustering_debug/services/__init__.py` (created)
- `app/face_clustering_debug/services/protocols.py` (created)
- `app/face_clustering_debug/components/__init__.py` (created)
- `app/face_clustering_debug/pages/__init__.py` (created)
- `docs/face_clustering_debug_app/REQUIREMENTS.md` (created)
- `docs/face_clustering_debug_app/ARCHITECTURE.md` (created)
- `docs/face_clustering_debug_app/TASKS.md` (created)

**Change**: Phase 1 of Face Clustering Debug App - Setup & Models

**Reason**: Complete rewrite of face clustering debug app with proper modularity and SOLID principles

**Details**:
1. Created folder structure: `app/face_clustering_debug/` with pages/, components/, services/, models/ subdirs
2. Implemented data models (schemas.py - 83 lines):
   - FaceInfo: face metadata including landmarks and pose
   - ClusterInfo: cluster with threshold stats
   - MergeDecision: merge decision record
   - AttachDecision: attachment decision record
   - ClusteringResult: complete result container
3. Defined DataLoaderProtocol interface (protocols.py - 65 lines)
4. Created requirements, architecture, and task breakdown documentation

---

## 2026-02-17 10:00:00

**Files**:
- `docs/FACIAL_CLUSTERING_DEBUG.md`
- `sim_bench/clustering/hybrid_hdbscan_knn.py`

**Change**: Fixed documentation inaccuracies in face clustering debug guide

**Reason**: Review found discrepancies between documentation and actual code implementation

**Details**:
1. Fixed threshold formula for `hybrid_hdbscan_knn`: was incorrectly documented as `Q3(d3) + 1.5×IQR`, actual is `median(exemplar_pairwise) + 2.0×IQR`
2. Fixed parameter defaults: `iqr_multiplier` is 2.0 (not 1.5), `threshold_ceiling` for hybrid_closest_face is 0.90 (not 1.50)
3. Added Quick Reference table at top of document
4. Added missing parameters: `attach_min_neighbors`, `max_iterations`, `iqr_multiplier`
5. Updated "Units Mismatch" section to "Design Note: Consistent Units" (both algorithms use consistent units)
6. Fixed algorithm comparison table to reflect actual differences
7. Also fixed docstring in `hybrid_hdbscan_knn.py` to match implementation

---

## 2026-02-16 (Code Review Fixes)

**Files**:
- `sim_bench/clustering/base.py` - Added collect_debug_data to base class signature
- `sim_bench/clustering/dbscan.py` - Updated signature
- `sim_bench/clustering/hdbscan.py` - Updated signature
- `sim_bench/clustering/kmeans.py` - Updated signature
- `sim_bench/clustering/hierarchical.py` - Updated signature
- `sim_bench/clustering/hybrid_closest_face.py` - Updated signature
- `sim_bench/clustering/hybrid_hdbscan_knn.py` - Added input validation
- `configs/clustering_benchmark.yaml` - Fixed parameter names
- `scripts/benchmark_face_clustering.py` - Removed unused import
- `app/face_clustering_comparison.py` - Refactored large function, consolidated imports
- `tests/clustering/test_hybrid_hdbscan_knn.py` - NEW: Unit tests

**Changes**:
1. Fixed API consistency - added collect_debug_data parameter to base class and all implementations
2. Fixed config parameter name mismatches (merge_min_links → merge_min_pairs, etc.)
3. Added input validation for NaN/Inf/zero vectors in embeddings
4. Added comprehensive unit tests for hybrid clustering
5. Refactored render_debug_merge_decisions into smaller helper functions
6. Consolidated matplotlib imports at module level
7. Removed unused shutil import

**Reason**: Expert SW architect review identified these issues

---

## 2026-02-16

**Files**:
- `sim_bench/clustering/hybrid_hdbscan_knn.py`
- `scripts/benchmark_face_clustering.py`
- `app/face_clustering_comparison.py`

**Change**: Added debug page for Hybrid kNN clustering analysis

**Details**:
1. Modified `HybridHDBSCANKNN` to return detailed decision data:
   - Added `MergeDecision` and `AttachDecision` dataclasses
   - Extended `ClusterState` with d3 stats (q1, q3, iqr, raw_threshold)
   - `_merge_clusters()` now collects merge decision logs with cross-distance matrices
   - `_attach_noise()` now collects attachment decision logs with candidate info
   - New `collect_debug_data` parameter to enable debug data collection
   - `_compute_final_stats()` includes debug section with all decision data

2. Updated benchmark script to collect debug data for hybrid_knn method

3. Added "Debug: Hybrid kNN" page to face_clustering_comparison.py with 6 sections:
   - Cluster Overview: Full face grid (no truncation), d3 stats table, exemplar marking
   - Inter-Cluster Distances: Heatmap of min distances, threshold comparison table
   - Merge Decisions: Explorer showing why clusters did/didn't merge, cross-distance matrices
   - Attachment Decisions: Explorer for noise point attachment decisions
   - Parameter Tuning: Interactive sliders to re-run clustering with new parameters
   - Face Distance Lookup: Tool to check embedding distance between any two faces

**Reason**: User requested debug capabilities to understand why clusters didn't merge and to tune parameters interactively.

---

## 2026-02-15 23:55:00

**Files**:
- `scripts/benchmark_face_clustering.py`

**Change**: CRITICAL BUG FIX #2 - Missing EXIF rotation when saving face crops

**Reason**: User reported Face 83 crop showed non-face content, but bbox visualization showed correct face. Investigation revealed crops were being taken from un-rotated images while bbox coordinates were relative to EXIF-rotated images.

**The Bug**:
- InsightFace detects faces on correctly-oriented images (respects EXIF rotation)
- Bbox coordinates stored relative to rotated dimensions  
- `save_single_face_crop()` opened images WITHOUT `ImageOps.exif_transpose()`
- Cropped from wrong location in portrait/rotated photos
- Result: Random image regions saved as "face crops"

**The Fix**:
```python
# Before (WRONG):
img = Image.open(image_path)

# After (CORRECT):
img = ImageOps.exif_transpose(Image.open(image_path))
```

**Impact**: ALL crops from portrait/rotated images were wrong. Explains why:
- High-confidence "faces" showed non-face content
- Embeddings were confused (trained on actual faces, got random textures)
- Clustering was grouping random image regions

**Testing**: MUST re-run benchmark to regenerate all face crops.

---

## 2026-02-15 23:45:00

**Files**:
- `scripts/benchmark_face_clustering.py`

**Change**: CRITICAL BUG FIX - Face crop index misalignment

**Reason**: User reported images showing no faces despite high confidence and low distances to other faces. Investigation revealed a fatal index mapping bug.

**The Bug**:
When saving face crops, some faces were skipped due to invalid bboxes (e.g., 236/254 saved). However:
1. Crop filenames used the original metadata index (with gaps: face_0000, face_0002, face_0003, ...)
2. Clustering and metadata still referenced all 254 faces
3. Streamlit loaded `face_0002.jpg` thinking it was metadata[2], but it was actually metadata[3]'s crop!

**The Fix**:
1. `save_face_crops()` now uses a sequential counter for saved crops (face_0000, face_0001, face_0002, ...)
2. Returns list of successfully saved indices
3. Metadata and embeddings are filtered to match saved crops before clustering
4. Result: Perfect 1:1 alignment between metadata index, crop filename, and clustering labels

**Testing**: Must re-run benchmark to regenerate properly aligned data.

---

## 2026-02-15 23:10:00

**Files**:
- `scripts/benchmark_face_clustering.py` (major update)
- `app/face_clustering_comparison.py` (complete rewrite)

**Change**: Enhanced clustering benchmark and comparison UI with detailed exploration

**Reason**: User feedback identified several issues:
1. Face crops weren't aligned (despite embeddings being aligned)
2. Difficult to compare methods due to misaligned cluster numbers
3. Need detailed cluster exploration with metrics, landmarks, and distance analysis

**Details**:

**Benchmark Script (`scripts/benchmark_face_clustering.py`)**:
1. **Face crop alignment**: Added `align_crop_by_roll()` to rotate saved face crops by roll angle (same alignment as embeddings)
2. **Enhanced metadata collection**: Now includes `roll_angle`, `pitch_angle`, `yaw_angle`, `frontal_score`, `eye_bbox_ratio`, `asymmetry_ratio`
3. **Cluster statistics**: Added `calculate_cluster_statistics()` to compute:
   - Intra-cluster distances (min/max/mean/std using cosine distance)
   - Nearest external face distance for each cluster
4. **Embeddings export**: Save embeddings to `.npy` file for use in Streamlit distance matrix calculations
5. Added `cv2` import for image rotation

**Streamlit App (`app/face_clustering_comparison.py`)**:
1. **Overview page** with side-by-side:
   - Cluster statistics tables showing intra-cluster distances and nearest external distances
   - Cluster galleries with filtering (min size, max clusters shown)
2. **Detailed cluster explorer** for each method with:
   - Face thumbnails with 5-point landmarks overlaid
   - Face quality metrics table (confidence, frontal score, pose angles, eye/width ratio, asymmetry)
   - Intra-cluster distance matrix heatmap (requires embeddings)
   - 5 nearest faces outside the cluster with distances
3. **Better UI organization**: Clear page navigation, method separation, expandable clusters
4. **Embeddings loading**: Loads `.npy` embeddings file for distance calculations

**Testing**:
- Both files pass linting
- Ready for re-run of benchmark to test all new features

---

## 2026-02-15 18:00:00

**Files**:
- `scripts/benchmark_face_clustering.py`

**Change**: Added file-based logging and fixed numpy.bool_ serialization

**Reason**: Logs were only going to stdout, making post-execution debugging difficult. JSON serialization was failing on numpy boolean types.

**Details**:
1. Added `setup_logging()` function to write logs to both console and `results/face_clustering_benchmark/logs/benchmark_TIMESTAMP.log`
2. Extended `NumpyTypeConverter` to handle `np.bool_` types (in addition to integers, floats, and arrays)
3. Logging now captures all pipeline and clustering operations for debugging

---

## 2026-02-15 01:20:00

**Files**:
- `sim_bench/clustering/hybrid_hdbscan_knn.py` (created)
- `sim_bench/clustering/base.py` (modified)
- `scripts/benchmark_face_clustering.py` (created)
- `app/face_clustering_comparison.py` (created)
- `configs/clustering_benchmark.yaml` (created)

**Change**: Implemented Hybrid HDBSCAN+kNN Face Clustering with Benchmark and Comparison Tools

**Reason**: Address face clustering issues where correct faces were not being clustered together

**Details**:
1. **Hybrid Clustering Algorithm** (`hybrid_hdbscan_knn.py`):
   - Stage 1: HDBSCAN for dense identity cores
   - Stage 2: Build cluster-level kNN graph between centroids
   - Stage 3: Merge clusters with mutual kNN links, ≥2 cross-links, and distance checks
   - Stage 4: Attach singletons to nearest clusters or create singleton clusters
   - Configurable parameters: knn_k, merge_min_links, merge_distance_ceiling, singleton_attach_threshold

2. **Clustering Factory Update** (`base.py`):
   - Added `hybrid_hdbscan_knn` to clustering method registry
   - Enables loading via `load_clustering_method({'algorithm': 'hybrid_hdbscan_knn', ...})`

3. **Benchmark Script** (`benchmark_face_clustering.py`):
   - Runs full pipeline on album to extract face embeddings (with filtering)
   - Executes both HDBSCAN and Hybrid methods on same face data
   - Saves face crops (112x112) for visualization
   - Outputs JSON results with labels, statistics, and merge details
   - Generates `results/face_clustering_benchmark/` directory structure

4. **Streamlit Comparison App** (`face_clustering_comparison.py`):
   - Side-by-side visual comparison of clustering methods
   - Metrics table: clusters, noise/singletons, avg/min/max sizes
   - Cluster gallery with face crops in grid layout
   - Merge details view showing hybrid algorithm decisions
   - Filtering: min cluster size, show/hide singletons
   - Sorting: by size or ID

5. **Configuration** (`clustering_benchmark.yaml`):
   - HDBSCAN config matching current pipeline defaults
   - Hybrid kNN config with recommended parameter values
   - Pipeline steps for face extraction with filtering
   - Output directory and face crop settings

**Usage**:
```bash
# Run benchmark
python scripts/benchmark_face_clustering.py --album-path D:\Budapest2025_Google

# View results
streamlit run app/face_clustering_comparison.py
```

---

## 2026-02-14 15:30:00

**Files**:
- `app/streamlit/pages/face_management.py` (created)
- `app/streamlit/main.py` (modified)
- `app/streamlit/components/sidebar.py` (modified)

**Change**: Implemented Phase 2 Frontend Foundation for Face Management

**Reason**: Second phase of Face Management UI implementation - creating the main page and navigation

**Details**:
1. **Face Management Page** (`pages/face_management.py`):
   - Full page with 4 tabs: "Needs Help", "All Faces", "People", "Pending Changes"
   - **Needs Help Tab**: Shows borderline faces needing user decision with confirm/reassign/skip buttons
   - **All Faces Tab**: Grid/list view of all faces with status filtering, selection checkboxes
   - **People Tab**: Shows all detected people with expandable details, exemplar counts, rename option
   - **Pending Changes Tab**: Batch mode control, change list with undo/reorder, apply all button
   - Batch/live mode toggle with automatic change application in live mode
   - Session state management for pending changes, selections, and mode
   - Helper functions for action descriptions, change management

2. **Navigation Updates** (`sidebar.py`):
   - Added "Faces" page to navigation (icon: 🎭)
   - Positioned between "People" and "Debug"

3. **Main App Updates** (`main.py`):
   - Added import for `render_face_management_page`
   - Added "faces" route to pages dictionary

---

## 2026-02-14 14:00:00

**Files**:
- `sim_bench/api/database/models.py` (modified)
- `sim_bench/api/schemas/face.py` (created)
- `sim_bench/api/services/face_service.py` (created)
- `sim_bench/api/routers/faces.py` (created)
- `sim_bench/api/main.py` (modified)

**Change**: Implemented Phase 1 Backend Foundation for Face Management

**Reason**: First phase of Face Management UI implementation

**Details**:
1. **FaceOverride Model** (models.py):
   - New model to persist user face corrections
   - Fields: face_key, status, person_id, embedding (for "not_a_face" learning)
   - Relationships to Album, PipelineRun, Person
   - Indexes for fast lookup

2. **Face Schemas** (schemas/face.py):
   - FaceInfo: Complete face information with status, assignment, quality metrics
   - PersonDistance: Distance from face to person with exemplar matches
   - BorderlineFace: Face in uncertainty zone for "Needs Help" wizard
   - PersonSummary: Person overview for listing
   - FaceAction: Single action request (assign/unassign/untag/not_a_face)
   - BatchChangeRequest/Response: Batch operations

3. **FaceService** (services/face_service.py):
   - get_all_faces(): List faces with status and assignments
   - get_face_distances(): Compute distances to all people
   - get_borderline_faces(): Find faces needing user decision
   - get_people_summary(): List people with counts
   - apply_batch_changes(): Apply multiple changes with optional recluster
   - create_person(): Create new person from faces
   - Helper methods for embeddings, thumbnails, overrides

4. **Faces Router** (routers/faces.py):
   - GET /faces: List faces with optional status filter
   - GET /faces/needs-help: Get borderline faces
   - GET /faces/people: Get people summary
   - GET /faces/{face_key}: Get single face
   - GET /faces/{face_key}/distances: Get distances to people
   - POST /faces/{face_key}/action: Apply single action
   - POST /faces/batch: Apply batch changes
   - POST /faces/person: Create new person

---

## 2026-02-14 12:00:00

**Files**:
- `docs/FACE_MANAGEMENT_MODULES.md` (created)

**Change**: Created detailed module specifications for Face Management feature

**Reason**: User requested clear plans for each individual module

**Details**:
- 14 modules documented with complete specifications
- Each module includes: Purpose, File location, Dependencies, Interface, Implementation steps, Test cases
- Backend modules: FaceOverride Model, Face Schemas, FaceService, Faces Router
- Frontend modules: Page, FaceCard, FaceGrid, ActionMenu, NeedsHelpWizard, PendingChangesPanel, PersonDetail, FaceDetailSheet, Toasts
- API Client extensions documented
- Dependency graph showing implementation order
- Effort estimates for each module (S/M/L)

---

## 2026-02-14 11:00:00

**Files**:
- `docs/FACE_MANAGEMENT_UI_PLAN.md` (created)

**Change**: Created comprehensive UI/UX implementation plan for Face Management page

**Reason**: User requested a detailed plan before implementing the Face Management UI

**Details**:
- 7 implementation phases with 25+ tasks
- Backend: New FaceOverride model, FaceService, faces router
- Frontend: 10 new components (FaceCard, ActionMenu, NeedsHelpWizard, etc.)
- State management design with batch/live modes
- Testing strategy (unit, integration, manual)
- Accessibility requirements
- Open questions documented

---

## 2026-02-14 10:00:00

**Files**:
- `sim_bench/api/database/models.py`
- `sim_bench/pipeline/steps/attachment_strategies.py` (created)
- `sim_bench/pipeline/steps/identity_refinement.py` (created)
- `sim_bench/pipeline/context.py`
- `sim_bench/pipeline/steps/cluster_by_identity.py`
- `sim_bench/api/services/people_service.py`
- `sim_bench/api/services/event_service.py` (created)
- `sim_bench/api/routers/events.py` (created)
- `sim_bench/pipeline/steps/all_steps.py`
- `configs/pipeline.yaml`
- `tests/pipeline/steps/test_attachment_strategies.py` (created)
- `tests/pipeline/steps/test_identity_refinement.py` (created)

**Change**: Implemented Identity Refinement system for improved face clustering quality

**Reason**: HDBSCAN assigns noise faces (cluster_id=-1) which were incorrectly grouped as a single Person. Same-person faces sometimes have borderline distances (~0.45) due to hairstyle/expression variations. Need post-processing to refine cluster assignments.

**Details**:

1. **UserEvent Model** (`models.py`):
   - Generic event tracking table for user actions, feedback, AI requests
   - Fields: event_type, event_data (JSON), status, result, source, is_undone, undone_by_id
   - Supports undo capability via is_undone flag and undone_by_id reference

2. **Attachment Strategies** (`attachment_strategies.py`):
   - Factory pattern with 3 strategies: CentroidStrategy, ExemplarStrategy, HybridStrategy
   - ClusterInfo dataclass holds centroid and exemplar embeddings
   - AttachmentResult dataclass with attached, cluster_id, confidence, distances
   - Thresholds: centroid_threshold=0.38, exemplar_threshold=0.40, reject_threshold=0.45
   - Multi-exemplar matching: `>= max(2, ceil(0.3*K))` with small cluster special case

3. **Identity Refinement Step** (`identity_refinement.py`):
   - Separates noise cluster (-1) from core clusters
   - Selects K exemplars per cluster using quality_diverse method
   - Computes normalized centroids from embeddings
   - Attempts attachment using configurable strategy (hybrid default)
   - Applies stored user overrides (attach, split, reassign, create)
   - Outputs: refined_people_clusters, unassigned_faces, cluster_exemplars, cluster_centroids, attachment_decisions

4. **Pipeline Context Updates** (`context.py`):
   - Added fields: refined_people_clusters, unassigned_faces, cluster_exemplars, cluster_centroids, attachment_decisions, user_overrides

5. **Downstream Integration**:
   - `cluster_by_identity.py`: Uses refined_people_clusters if available
   - `people_service.py`: Tracks assignment_method (core, auto_attached, user_assigned) and assignment_confidence

6. **Event Service** (`event_service.py`):
   - record_event(): Persists user actions to database
   - undo_event(): Marks event as undone, optionally replays inverse
   - apply_face_override(): Creates face_assign events

7. **Events API** (`events.py`):
   - POST /events: Record new event
   - POST /events/{id}/undo: Undo specific event
   - GET /events: List events with filtering
   - POST /events/face-assign: Shortcut for face assignment

8. **Configuration** (`pipeline.yaml`):
   - Added identity_refinement to default_pipeline after cluster_people
   - Full config section with all thresholds and options

9. **Tests**:
   - `test_attachment_strategies.py`: Tests for all 3 strategies, cosine distance, factory
   - `test_identity_refinement.py`: Tests for noise separation, face key generation, centroid computation, exemplar selection, disabled pass-through, integration tests

---

## 2026-02-13 16:30:00

**Files**:
- `sim_bench/pipeline/steps/extract_face_embeddings.py`

**Change**: Fixed critical bug - extract_face_embeddings now depends on score_face_frontal

**Details**:
- Changed `depends_on=["insightface_detect_faces"]` to `depends_on=["score_face_frontal"]`
- Without this fix, the topological sort could run embedding extraction BEFORE filtering steps
- This caused all faces to get embeddings (filter_passed and is_clusterable fields didn't exist yet)
- Now the dependency chain is: `insightface_detect_faces` → `filter_faces` → `score_face_frontal` → `extract_face_embeddings` → `cluster_people`

**Reason**: Root cause of why face filtering wasn't being applied to clustering

---

## 2026-02-13 16:00:00

**Files**:
- `app/streamlit/pages/debug.py` (NEW)
- `app/streamlit/main.py`
- `app/streamlit/components/sidebar.py`
- `app/streamlit/pages/people.py`
- `sim_bench/pipeline/steps/filter_faces.py`
- `sim_bench/pipeline/steps/score_face_frontal.py`
- `sim_bench/pipeline/steps/extract_face_embeddings.py`

**Change**: Added Debug page and verbose logging for face filtering

**Details**:
1. **New Debug Page** (`/debug`):
   - Face scores table showing all filter/frontal metrics per face
   - Filtering explanation with thresholds
   - Image detail view with per-face metrics
   - Config knobs (read-only for now)
   - Accessible from sidebar navigation

2. **Verbose Logging**:
   - `filter_faces`: Logs total/passed/failed counts, failures by criterion, sample scores
   - `score_face_frontal`: Logs frontal score distribution, clusterable counts, samples
   - `extract_face_embeddings`: Logs total faces, selected for embedding, skipped counts

3. **People Page Enhancement**:
   - Added "Face Filtering Summary" expander explaining the pipeline
   - Added button to open Debug page
   - Added Debug button in no-people state

**Reason**: User requested debug tools to verify face filtering is applied and understand clustering results

---

## 2026-02-13 14:30:00

**Files**:
- `sim_bench/pipeline/steps/filter_faces.py` (NEW)
- `sim_bench/pipeline/steps/score_face_frontal.py` (NEW)
- `sim_bench/pipeline/steps/extract_face_embeddings.py`
- `sim_bench/pipeline/steps/all_steps.py`
- `sim_bench/pipeline/scoring/person_penalty.py`
- `sim_bench/api/services/pipeline_service.py`
- `app/streamlit/models.py`
- `app/streamlit/api_client.py`
- `app/streamlit/components/metrics.py`
- `configs/pipeline.yaml`
- `docs/FACE_FILTERING_PLAN.md`

**Change**: Implemented face filtering and frontal scoring pipeline

**Details**:
1. **filter_faces step**: Removes small/low-confidence faces
   - Filters by: min_confidence (0.5), min_bbox_ratio (0.02), min_relative_size (0.3), min_eye_ratio (0.01)
   - Marks faces with `filter_passed`, `filter_scores`, `filter_reason`
   - Keeps all faces but marks which passed (for debugging)

2. **score_face_frontal step**: Computes frontal score and marks clusterable faces
   - Frontal score from: eye_bbox_ratio + asymmetry_score
   - Computes roll_angle from eye landmarks
   - Computes centrality (distance from image center)
   - Marks `is_clusterable` based on frontal_score threshold (0.4)

3. **extract_face_embeddings**: Modified for roll alignment and filtering
   - Skips non-clusterable faces (`filter_passed=False` or `is_clusterable=False`)
   - Applies roll alignment to face crops before embedding extraction

4. **person_penalty**: Added frontal penalty
   - New `frontal_penalty_weight` config (0.3)
   - Penalty = (1 - best_frontal_score) * weight * centrality
   - Only applies if best_frontal_score < frontal_threshold (0.6)

5. **UI updates**: Display new scores in metrics table
   - Faces column shows "passed/total"
   - New columns: Frontal, Central, Roll, Clusterable

6. **pipeline.yaml**: Added new steps to default_pipeline
   - filter_faces runs after insightface_detect_faces
   - score_face_frontal runs after filter_faces

**Reason**: Improve face clustering quality by filtering small/unreliable faces and excluding non-frontal faces from clustering

**Note**: Delete database (`~/.sim_bench/sim_bench.db`) before testing to clear stale cache

---

## 2026-02-12 21:20:00

**Files**:
- `notebooks/debug_face_analysis.ipynb`

**Change**: Added filtered high-quality face visualizations section

**Details**:
- New section "Filtered High-Quality Faces" placed before distance matrix
- Shows only faces passing all quality checks: Size OK (≥70px), Frontal (eye/width ≥0.20), Symmetric (asymmetry <1.8)
- Two visualizations:
  1. Cropped faces with landmarks (filtered)
  2. Embedding distance matrix (filtered)
- Faces labeled with "(HQ)" to indicate high-quality
- Helps focus analysis on reliable face detections

**Reason**: User requested filtered visualizations to analyze only high-quality faces that meet all criteria

---

## 2026-02-12 21:15:00

**Files**:
- `notebooks/debug_face_analysis.ipynb`

**Change**: Added EXIF rotation handling for proper image orientation

**Details**:
- Added `ImageOps` to PIL imports
- Applied `ImageOps.exif_transpose()` after all `Image.open()` calls
- Ensures smartphone photos with EXIF rotation metadata display correctly
- Updated 3 image loading locations: full visualization, cropped faces, metrics calculation

**Reason**: User requested proper image rotation handling to display images in correct orientation

---

## 2026-02-12 21:10:00

**Files**:
- `notebooks/debug_face_analysis.ipynb`

**Change**: Added face_width/image_width ratio to Face Metrics Summary

**Details**:
- Added "Face/Img" column showing face_width / image_width ratio
- Shows what portion of image width the face occupies (0.0-1.0)
- Helps identify close-up vs distant faces
- Image dimensions loaded once per image for efficiency

**Reason**: User requested face width to image width ratio for face scale analysis

---

## 2026-02-12 21:00:00

**Files**:
- `notebooks/debug_face_analysis.ipynb`

**Change**: Enhanced Face Metrics Summary with derived quality metrics

**Details**:
Added InsightFace-based quality heuristics to metrics table:
- Inter-eye distance (pixels)
- Eye/Width ratio (inter_eye / bbox_width) - frontal face indicator (>0.20 = frontal)
- Asymmetry ratio (max nose-to-eye / min nose-to-eye) - symmetry indicator (<1.8 = symmetric)
- Quality flags: Frontal?, Symmetric?, Size OK? (✓/✗)
- Thresholds: eye/width >= 0.20 (frontal), asymmetry < 1.8 (symmetric), width >= 70px (size)

**Reason**: User requested additional derived metrics from landmarks to assess face quality and profile detection

---

## 2026-02-12 20:45:00

**Files**:
- `notebooks/debug_face_analysis.ipynb`

**Change**: Added cropped face visualization section with landmarks

**Details**:
- Shows cropped faces in rows (one row per image, up to 4 faces per row)
- Landmarks overlaid on cropped faces
- 20% padding around face bounding boxes
- Each face labeled with image name and face number
- Placed before Face Metrics Summary section

**Reason**: User requested separate cropped face visualization to better see landmark positions on individual faces

---

## 2026-02-12 20:30:00

**Files**:
- `notebooks/debug_face_analysis.ipynb` (created)

**Change**: Created face analysis debug notebook for visualizing face detection results from database

**Details**:
- Queries `universal_cache` table for face detections, landmarks, and embeddings
- Queries `pipeline_results` table for face scores (pose, eyes, smile)
- Visualizes faces with bounding boxes and 5-point landmarks overlay
- Displays face metrics summary table
- Calculates and visualizes embedding distance matrix
- Database path: `Path.home() / '.sim_bench' / 'sim_bench.db'`
- Uses numpy deserialization for embeddings (not pickle)

**Reason**: User requested debugging notebook to analyze face detection results for specific images, with no try/except blocks and minimal logging

---

## 2026-02-11 01:00:00

**Files**:
- `sim_bench/pipeline/face_embedding/insightface_native.py`

**Change**: Fix InsightFace native extractor to use recognition model directly instead of re-detecting faces

**Critical Fix**:
The extractor was calling `app.get()` (face detector) on pre-cropped face images. This failed because:
- Face detectors expect full images with context
- Tight crops of faces are hard to detect (face too close to edges)
- Caused failures and zero-vector fallbacks

**New Approach**:
1. Extract recognition model directly from FaceAnalysis app
2. Use `rec_model.get_feat()` directly on cropped faces (bypasses detection)
3. Resize crops to 112x112 (expected input size for InsightFace recognition)
4. Normalize embeddings manually
5. Fallback to full detection if recognition model not found

**Additional Fixes**:
- Added dimension validation before accessing shape[2]
- Added graceful handling for grayscale images (convert to BGR)
- Added validation for None/empty images
- Added better logging with face index for debugging

**Reason**: User reported 500 errors. Root cause: we were passing already-cropped faces to a face detector, which failed. Now we use the recognition model directly on crops, which is what it's designed for.

---

## 2026-02-11 00:30:00

**Files**:
- `sim_bench/pipeline/face_embedding/insightface_native.py`

**Change**: [SUPERSEDED BY 01:00:00] Initial bug fix attempt

**Reason**: First attempt at fixing dimension checks, but missed the core issue of re-running detection on crops.

---

## 2026-02-11 00:00:00

**Files**:
- `sim_bench/pipeline/face_embedding/` (new package)
  - `base.py` - Abstract base class for face embedding extractors
  - `custom_arcface.py` - Custom ArcFace model extractor
  - `insightface_native.py` - InsightFace native w600k_r50 extractor
  - `factory.py` - Factory for creating extractors
  - `__init__.py` - Package init
- `sim_bench/pipeline/steps/extract_face_embeddings.py`
- `configs/pipeline.yaml`
- `configs/pipeline_custom_arcface.yaml` (backup)
- `app/streamlit/components/pipeline_runner.py`

**Change**: Add face embedding backend strategy pattern with InsightFace native support

1. **New Strategy Pattern Architecture**:
   - Created `face_embedding/` package with pluggable extractors
   - `BaseFaceEmbeddingExtractor`: Abstract interface with `extract_batch()`, `extract_single()`, `embedding_dim`, `model_name`
   - `CustomArcFaceExtractor`: Uses existing trained ArcFace model (arcface_resnet50.pt)
   - `InsightFaceNativeExtractor`: Uses InsightFace's built-in w600k_r50 model
   - `FaceEmbeddingExtractorFactory`: Creates extractors based on config backend

2. **Updated extract_face_embeddings Step**:
   - Refactored to use factory pattern instead of direct service calls
   - `_get_extractor()`: Lazy loads extractor using factory
   - `_get_cache_config()`: Uses `extractor.model_name` for cache key (enables separate caches per backend)
   - `_process_uncached()`: Uses `extractor.extract_batch()` instead of direct service call
   - Config schema supports `backend`, `checkpoint_path`, `device`, `model_name`

3. **Configuration Updates**:
   - `pipeline.yaml`: Added `backend: insightface` as default with inline documentation
   - Backed up original config to `pipeline_custom_arcface.yaml`
   - Default backend is `insightface` for better rotation invariance

4. **UI Updates**:
   - Added "Face Embedding" section in Advanced Configuration
   - Backend selector: "insightface" (default) or "custom"
   - Info display shows which model is active (InsightFace w600k_r50 or arcface_resnet50.pt)
   - Config passed to pipeline includes full embedding configuration

**Reason**: Custom ArcFace model lacks rotation invariance, causing poor clustering on rotated faces. InsightFace's built-in w600k_r50 is trained on 600K+ identities with extensive augmentation including rotation. Strategy pattern allows runtime selection between backends and easy future extensions (e.g., CLIP face embeddings).

**Technical Details**:
- Different backends use different cache keys (`arcface_custom` vs `arcface_insightface`)
- Switching backends will recompute embeddings (cache miss by design)
- InsightFace backend re-runs face detection on crops to extract embeddings
- Both backends produce 512-dim normalized embeddings

---

## 2026-02-10 03:00:00

**Files**:
- `sim_bench/api/services/people_service.py`
- `app/streamlit/components/gallery.py`

**Change**: Improved People feature error handling and debugging

1. **people_service.py `get_person_images()`**:
   - Added debug logging for face_instances count and thumbnail path
   - Added validation to skip faces with empty image_path
   - Added fallback: if no valid face_instances, use thumbnail_image_path

2. **people_service.py `create_from_clusters()`**:
   - Added validation to skip faces with invalid paths (empty, '.', 'None')
   - Added warning logs when skipping invalid faces

3. **gallery.py `render_image_card()`**:
   - Improved error message: now shows filename when path is missing

**Reason**: User reported "No image path" for all images when viewing a person's photos. The root cause is likely stale Person records created before path handling fixes were applied. Added validation and fallbacks to handle edge cases, plus logging to help diagnose issues.

**Action Required**: Re-run the pipeline to regenerate Person records with correct face_instances data.

---

## 2026-02-10 02:00:00

**Files**:
- `sim_bench/pipeline/steps/cluster_people.py`
- `app/streamlit/components/pipeline_runner.py`
- `app/streamlit/components/people_browser.py`
- `configs/pipeline.yaml`

**Change**: Added HDBSCAN cluster merge epsilon to reduce over-segmentation

1. **cluster_people.py**: Added `cluster_selection_epsilon` parameter
   - Merges clusters within this distance of each other
   - Higher value = more merging = fewer clusters
   - Default: 0.3

2. **pipeline_runner.py**: Added "Cluster Merge Distance" slider (0.0-0.8)
   - Only shown when HDBSCAN method selected
   - Also lowered min_cluster_size minimum to 1

3. **people_browser.py**: Fixed deprecation warning
   - Changed `use_column_width=True` to `use_container_width=True`

**Reason**: User reported too many clusters (over-segmentation). The `cluster_selection_epsilon` parameter tells HDBSCAN to merge clusters that are close together, reducing fragmentation of the same person into multiple clusters.

---

## 2026-02-10 01:30:00

**Files**:
- `app/streamlit/components/pipeline_runner.py`
- `app/streamlit/components/metrics.py`

**Change**: Added min face size config and improved metrics table

1. **pipeline_runner.py**: Added "Min Face Size (px)" slider (20-100, default 50)
   - Controls minimum face size in pixels to be considered
   - Applied to insightface_detect_faces, insightface_score_expression/eyes/pose

2. **metrics.py**: Enhanced per-image metrics table with clearer body/face columns
   - "Body" column: ✓ if body detected
   - "Face" column: Face count
   - "BodyPose": Body facing camera score
   - "FacePose": Face frontal score
   - Renamed "Sharpness" to "Sharp" for column width

**Reason**: User requested min face size threshold control and clearer display of body vs face detection with their respective pose scores.

---

## 2026-02-10 01:00:00

**Files**:
- `sim_bench/api/services/pipeline_service.py`
- `sim_bench/pipeline/scoring/person_penalty.py`
- `sim_bench/pipeline/steps/cluster_by_identity.py`

**Change**: Fixed face scores showing as None and improved multi-face handling

1. **pipeline_service.py `_build_image_metrics()`**:
   - Normalized paths for cache key lookups (forward slashes)
   - Fixed InsightFace face score retrieval using correct face_index

2. **person_penalty.py**:
   - Normalized all paths for cache key lookups
   - Changed from only looking at `face_0` to checking ALL faces
   - Now uses WORST score across all faces (as user requested)
   - Added `_get_face_count()` helper for both MediaPipe and InsightFace

3. **cluster_by_identity.py**:
   - Fixed to work with InsightFace faces (was only using MediaPipe `context.faces`)
   - Normalized paths in face-to-person lookup
   - Now checks both `context.faces` and `context.insightface_faces`

**Reason**: User reported face Pose/Eyes/Smile scores all showing as None. Root cause was path format mismatch (backslashes vs forward slashes on Windows). Also fixed penalty computation to use worst score from all faces, not just first face.

---

## 2026-02-10 00:30:00

**Files**:
- `app/streamlit/api_client.py`
- `app/streamlit/components/gallery.py`

**Change**: Fixed People image viewing errors

1. **api_client.py**: `_parse_image()` now handles both `path` and `image_path` keys
   - The `get_person_images` API returns `image_path` but `_parse_image` was looking for `path`
   - Now checks both keys: `data.get("path") or data.get("image_path", "")`

2. **gallery.py**: Added error handling to thumbnail loading
   - `_load_thumbnail_cached()` now returns None on error instead of crashing
   - `_load_thumbnail()` handles None bytes gracefully
   - `render_image_card()` checks for empty path before trying to load

**Reason**: User got error when clicking on a photo in the People tab. Root cause: API endpoint returns `image_path` but parser expected `path`, resulting in empty path and file-not-found errors.

---

## 2026-02-10 00:15:00

**Files**:
- `sim_bench/pipeline/steps/cluster_people.py`
- `app/streamlit/components/pipeline_runner.py`
- `configs/pipeline.yaml`

**Change**: Added HDBSCAN support for people clustering (now the default)

1. **cluster_people.py**: Added HDBSCAN method alongside agglomerative
   - HDBSCAN auto-determines optimal clusters based on density
   - Handles noise (outlier faces not forced into clusters)
   - Uses normalized embeddings (euclidean on normalized ≈ cosine distance)
   - Logs noise point count

2. **pipeline_runner.py**: Updated UI with method selector
   - Dropdown to choose: "hdbscan" (default) or "agglomerative"
   - HDBSCAN shows "Min Faces per Person" slider (2-5)
   - Agglomerative shows "Identity Distance Threshold" slider (0.3-0.9)

3. **pipeline.yaml**: Updated cluster_people config
   - method: hdbscan (default)
   - min_cluster_size: 2
   - min_samples: 2

**Reason**: User asked about intelligent threshold selection. HDBSCAN automatically finds natural clusters without requiring manual threshold tuning - it only needs `min_cluster_size` (minimum faces to form a "person").

---

## 2026-02-09 12:00:00

**Files**:
- `sim_bench/api/services/people_service.py`
- `sim_bench/pipeline/steps/cluster_people.py`
- `sim_bench/pipeline/steps/extract_face_embeddings.py`
- `sim_bench/pipeline/steps/insightface_detect_faces.py`
- `sim_bench/api/services/pipeline_service.py`

**Change**: Fixed People feature data flow with multiple fixes:

1. **BBox format handling in PeopleService**: Both `create_from_clusters()` and `_get_thumbnail_info()` now handle both dict-style and object-style bbox (InsightFace stores as dict, MediaPipe as objects)

2. **Path normalization for cache keys**: Normalized paths to forward slashes across all steps to ensure consistent cache key lookup:
   - `extract_face_embeddings._generate_cache_key()`: uses forward slashes
   - `insightface_detect_faces._get_cache_config()`: normalizes paths
   - `cluster_people._collect_faces_with_embeddings()`: normalizes paths when looking up embeddings

3. **Enhanced logging**: Added detailed debug logging to trace face embedding storage and people cluster creation:
   - `cluster_people`: Logs counts of faces found from each source (MediaPipe vs InsightFace), matched vs unmatched embeddings
   - `extract_face_embeddings`: Logs number of embeddings stored with sample keys
   - `pipeline_service`: Logs people cluster count and Person record creation

**Reason**: User reported People tab is empty, Person column is empty. Investigation revealed:
- Path format mismatch on Windows (backslash vs forward slash) caused face embeddings to not be found when looked up in `cluster_people` step
- BBox stored as dict by InsightFace but `PeopleService` expected object with `.x`, `.y` attributes
- No logging made it difficult to trace where the data flow broke

---

## 2026-02-08 10:30:00

**Files**:
- `sim_bench/pipeline/steps/detect_faces.py` (removed debug code)
- Deleted `yolov8s-pose.pt` files (version mismatch)

**Change**: Removed debug traceback logging; deleted old YOLO model files causing version mismatch

**Reason**: MediaPipe error is FIXED (pipeline correctly uses InsightFace). YOLO error `'Conv' object has no attribute 'bn'` was caused by model files saved with different ultralytics version. Deleting them allows ultralytics to download fresh compatible versions.

---

## 2026-02-08 10:15:00

**Files**: `sim_bench/pipeline/steps/detect_faces.py`

**Change**: Added traceback logging to `_get_crop_service()` to debug why MediaPipe is being loaded

**Reason**: Pipeline steps list does NOT include `detect_faces`, yet MediaPipe is still loading. Added stack trace logging to identify exactly which code path is calling `_get_crop_service()`. (Now removed after confirming MediaPipe is no longer called)

---

## 2026-02-08 10:00:00

**Files**: `sim_bench/pipeline/executor.py`

**Change**: Added logging to show resolved pipeline steps after dependency resolution

**Reason**: Debugging MediaPipe error - need to verify which steps are actually being executed after `PipelineBuilder.build()` resolves dependencies. This will reveal if `detect_faces` step is being incorrectly added by dependency resolution.

---

## 2026-02-03 16:30:00 ✅ COMPLETE

**Files**: 
- `.gitattributes`
- `models/album_app/arcface_resnet50.pt`
- `models/album_app/ava_resnet50.pt`
- `models/album_app/siamese_comparison_model.pt`
- Git history (rewritten)

**Change**: Migrated all PyTorch model files (.pt) to Git LFS and rewrote repository history

**Reason**: User requested moving .pt files to Git LFS to reduce repository size and improve clone/push/pull performance for large binary files.

**Details**:
- Ran `git lfs install` to initialize Git LFS
- Ran `git lfs track "*.pt"` to configure LFS tracking
- Updated `.gitattributes` with `*.pt filter=lfs diff=lfs merge=lfs -text`
- Committed all 3 model files as LFS objects (99% rewrite)
- Rewrote entire Git history using `git lfs migrate import --everything`
- Uploaded 309 MB LFS objects to remote storage
- Force-pushed rewritten history (commit 851bbe4) to GitHub
- Verified: local and remote in sync, all 3 model files tracked by LFS
- Repository size reduced by ~300 MB

**Commits**: 
- `9df805c` - "chore: migrate model files (.pt) to Git LFS"
- `6f4f1ee` - History rewrite commit
- `851bbe4` - Final documentation commit

**Verification**: `git lfs ls-files` shows 3 files, `git status` shows "up to date with origin/main"

---

## 2026-02-03 16:00:00

**Files**: 
- `CLAUDE.md`
- `CHANGES_LOG.md` (created)

**Change**: Updated CLAUDE.md with current architecture and added change tracking requirement

**Reason**: User requested CLAUDE.md be updated to reflect current state (Streamlit + FastAPI, recent bug fixes, etc.) and instructed to always maintain a change log after every modification.

**Details**:
- Corrected architecture section: Now correctly states Streamlit + FastAPI (not NiceGUI)
- Added "Recent Updates (Feb 2026)" section documenting bug fixes and features
- Updated app launch commands to show backend + frontend startup
- Added pipeline steps information (18-step engine)
- Added API endpoint development instructions
- Added debugging tips section with common issues and solutions
- Created CHANGES_LOG.md with template and initial entries
- Added prominent instruction at top: "After EVERY code change, append to CHANGES_LOG.md"

---

## 2026-02-03 15:45:00

**Files**: 
- `sim_bench/api/schemas/result.py`
- `sim_bench/api/services/result_service.py`
- `app/streamlit/components/gallery.py`

**Change**: Added "Final Score" (composite_score) column to cluster debug spreadsheet

**Reason**: User requested visibility of the final selection score used for ranking images. This helps debug why certain images were selected over others by showing the weighted combination of IQA, AVA, sharpness, and face scores.

**Details**:
- Added `composite_score` field to `ImageMetrics` schema
- Updated `_build_image_dict()` to include composite_score from stored metrics
- Added "Final Score" column to cluster debug table (4th column after Selected)
- Score displayed with 3 decimal places (e.g., 0.856)

---

## 2026-02-03 15:30:00

**Files**: `app/streamlit/pages/results.py`

**Change**: Simplified cluster view by removing redundant API call and manual is_selected marking loop

**Reason**: Completed Task 6 of cluster debug view implementation. The enriched `get_clusters()` API now returns images with `is_selected` already set, making the separate `get_selected_images()` call and manual marking loop unnecessary.

**Details**:
- Removed `client.get_selected_images(job_id)` call in "By Cluster" mode
- Removed 7 lines of manual is_selected marking code
- Reduced code from 10 lines to 3 lines
- Improved performance by eliminating redundant API call

---

## 2026-02-03 (Earlier - Claude Code CLI Session)

**Files**: Multiple (11 files total)

**Changes**: Bug fixes and feature enhancements for Streamlit + FastAPI album app

**Bug Fixes**:
1. **Siamese model config loading** (`sim_bench/pipeline/steps/select_best.py`)
   - Fixed config key mismatch: now reads from nested `config["siamese"]` dict
   - Properly extracts checkpoint_path, tiebreaker_range, duplicate_threshold

2. **HDBSCAN clustering parameters** (`sim_bench/pipeline/steps/cluster_scenes.py`)
   - Fixed parameter pass-through: now passes min_samples, metric, cluster_selection_epsilon, cluster_selection_method
   - Updated config: min_cluster_size changed from 3 to 2

3. **Image EXIF rotation** (`app/streamlit/components/gallery.py`)
   - Added `_load_image_for_display()` helper using `ImageOps.exif_transpose()`
   - Applied to all st.image() call sites

**Feature Enhancements**:
1. **Per-image score display** (result_service.py, schemas, api_client, models)
   - Backend now returns: face_pose_scores, face_eyes_scores, face_smile_scores, is_selected, sharpness
   - Extracted `_build_image_dict()` helper to avoid code duplication

2. **Portrait indicators** (gallery.py)
   - Gallery shows sharpness in score line
   - Portrait indicators: Eyes (Open/Closed), Expression (Smiling/Neutral)

3. **Metrics table with CSV export** (metrics.py, results.py)
   - Added "Metrics Table" tab with DataFrame showing all image scores
   - CSV download button for export
   - Fixed PyArrow mixed-type error by converting Cluster column to string

4. **Cluster debug view** (Multiple files - Tasks 1-5)
   - Expanded ClusterInfo schema with selected_count, has_faces, face_count, person_labels
   - Enriched get_clusters() to return full ImageMetrics objects
   - Added persona sub-grouping: groups images by people appearing in them
   - Added thumbnail spreadsheet with base64-encoded 80px image previews
   - Shows all scores in table format for debugging selection decisions

---

## 2026-02-04 10:30:00

**Files**: 
- `SCORE_PERSISTENCE_DEBUG_PLAN.md` (created and updated)

**Change**: Created comprehensive debug and fix plan for missing scores + People Management feature

**Reason**: User reported that Pose, Eyes, Smile, and Final Score columns are showing None values, and People column is empty. User also clarified they want full people management (identify, name, filter).

**Details**:
- Documented 4 main problems: face scores None, composite_score None, people empty, face detection threshold
- Root cause hypothesis: scores computed but not persisted to database image_metrics JSON
- **Split into 3 sprints**:
  - **Sprint 1** (1.5 hrs): Fix score persistence to database
  - **Sprint 2** (2 hrs): Full People Management UI (gallery, naming, filtering)
  - **Sprint 3** (30 min): Tune face detection (DECREASE threshold 0.5→0.3 to catch more faces)
- Clarified face detection issue is FALSE NEGATIVES (missing real faces), not false positives
- Expanded People feature to include:
  - New People Gallery page with thumbnails and naming interface
  - Person filtering in Results page
  - Backend API endpoints for get_people() and update_person_name()
- Test strategy with 10-image test album
- Clear success criteria per sprint
- Identified 11 files (4 new, 7 modified)
- Total estimated fix time: 4-5 hours (can split across sessions)

**Next Steps**: User to choose Option A (Sprint 1 first) or Option B (all sprints together)

---

## Historical Changes (Pre-Log)

For changes before this log was created, see:
- `FEBRUARY_2026_UPDATES.md` - Recent session details
- `CLUSTER_DEBUG_VIEW_COMPLETE.md` - Complete cluster view implementation
- `FINAL_SCORE_COLUMN_ADDED.md` - Final score column addition
- `DOCUMENTATION_UPDATE_SUMMARY.md` - Documentation updates
- `MILESTONES.md` - Major project milestones

---

## Instructions for Claude Code

**After EVERY code modification**:
1. Append a new entry to this file
2. Use ISO 8601 timestamp format (YYYY-MM-DD HH:MM:SS)
3. List ALL files modified
4. Describe WHAT changed (be specific)
5. Explain WHY (user request, bug fix, refactor, etc.)
6. Include relevant details (line numbers, function names, key values)

**Example Template**:
```markdown
## YYYY-MM-DD HH:MM:SS

**Files**: 
- `path/to/file1.py`
- `path/to/file2.py`

**Change**: [One-line summary]

**Reason**: [Why this change was needed]

**Details**:
- [Specific change 1]
- [Specific change 2]
```

This log helps:
- Debug issues by tracking when changes were made
- Understand evolution of codebase
- Coordinate between different AI sessions
- Provide context for future development

---

### 2026-02-04 12:00:00
**Files**: `sim_bench/pipeline/context.py`
**Change**: Added `composite_scores: dict[str, float]` field to PipelineContext dataclass
**Reason**: Composite scores were computed transiently in select_best but never persisted; needed a field to store them for database persistence

### 2026-02-04 12:01:00
**Files**: `sim_bench/api/services/pipeline_service.py`
**Change**: Replaced inline image_metrics dict comprehension with `_build_image_metrics()` helper method that correctly aggregates per-face scores by iterating over detected faces using cache keys (`"path:face_N"`), includes composite_score, and calls `PeopleService.create_from_clusters()` to persist Person records after pipeline completion
**Reason**: Face scores (pose/eyes/smile) were always None because `pipeline_service.py` looked up scores by image path, but face scoring steps store scores keyed by cache key format (`"path:face_0"`). Also, Person records were never created because `create_from_clusters()` was never called from the pipeline execution flow.

### 2026-02-04 12:02:00
**Files**: `sim_bench/pipeline/steps/select_best.py`
**Change**: Added `context.composite_scores[path] = score` loop after scoring images in `_select_from_cluster()` to persist computed composite scores back into the pipeline context
**Reason**: Composite scores were computed for ranking but discarded after selection; they need to be stored in context so `pipeline_service.py` can persist them to the database

### 2026-02-04 12:03:00
**Files**: `sim_bench/api/services/people_service.py`
**Change**: Fixed BoundingBox serialization in `create_from_clusters()` - replaced `list(face.bbox)` with explicit `[face.bbox.x, face.bbox.y, face.bbox.w, face.bbox.h]` for both face_instances and thumbnail_bbox
**Reason**: BoundingBox is a dataclass and not iterable; `list(bbox)` would raise TypeError at runtime

### 2026-02-04 12:04:00
**Files**: `sim_bench/api/services/result_service.py`
**Change**: Changed person display name from `f"Person {person.person_index}"` to `f"Person {person.person_index + 1}"` (1-based indexing)
**Reason**: person_index is 0-based (cluster ID), but user-facing display should be 1-based for readability

### 2026-02-04 12:05:00
**Files**: `sim_bench/pipeline/steps/detect_faces.py`, `sim_bench/face_pipeline/crop_service.py`, `configs/pipeline.yaml`, `configs/global_config.yaml`
**Change**: Lowered face detection confidence threshold from 0.5 to 0.3 across all config defaults and code defaults
**Reason**: Threshold of 0.5 was causing false negatives (missing real faces); lowering to 0.3 catches more real faces while the existing min_face_ratio (2%) filter still rejects tiny artifacts

### 2026-02-05 10:00:00
**Files**: `sim_bench/api/services/pipeline_service.py`
**Change**: Added `cluster_people` step to DEFAULT_PIPELINE (after `extract_face_embeddings`, before `cluster_by_identity`)
**Reason**: People tab was empty because faces were extracted but never globally clustered by identity. Without `cluster_people`, `people_clusters` dict is empty, so no Person records were created.

### 2026-02-05 10:01:00
**Files**: `app/streamlit/pages/results.py`, `app/streamlit/components/gallery.py`
**Change**: Changed `render_cluster_gallery(clusters, show_all_images=True)` and increased column count to 6 when showing all images
**Reason**: Results view was only showing 4 images per cluster due to `show_all_images=False` and `max_preview=4` limit

### 2026-02-05 10:02:00
**Files**: `configs/pipeline.yaml`
**Change**: Lowered `detection_confidence` from 0.3 to 0.2, lowered `min_face_ratio` from 0.02 to 0.01
**Reason**: Still missing faces in many cases; more aggressive detection thresholds to catch smaller and less confident faces

### 2026-02-05 11:00:00
**Files**: `app/streamlit/components/pipeline_runner.py`
**Change**: Added comprehensive UI controls for pipeline configuration:
- Face Detection: detection_confidence slider, min_face_ratio slider
- Selection: max_score_gap slider, duplicate_threshold slider, siamese_enabled checkbox
- Added `cluster_people` to DEFAULT_PIPELINE and STEP_DISPLAY_NAMES
**Reason**: Most pipeline parameters were only editable via YAML; now controllable from UI

### 2026-02-05 11:01:00
**Files**: `sim_bench/pipeline/context.py`, `sim_bench/pipeline/steps/select_best.py`
**Change**: Added `siamese_comparisons` list field to PipelineContext; updated `_apply_siamese_tiebreaker` and `_check_near_duplicate` to log each comparison with type, images, winner, confidence, method
**Reason**: Siamese comparisons were invisible; now stored for debugging and display

### 2026-02-05 11:02:00
**Files**: `sim_bench/api/database/models.py`, `sim_bench/api/services/pipeline_service.py`, `sim_bench/api/services/result_service.py`, `sim_bench/api/routers/results.py`
**Change**: Added `siamese_comparisons` JSON column to PipelineResult model, persist comparisons to DB, added `get_comparisons()` service method and `/comparisons` API endpoint
**Reason**: Comparison log needs to be persisted and accessible via API

### 2026-02-05 11:03:00
**Files**: `app/streamlit/api_client.py`, `app/streamlit/pages/results.py`
**Change**: Added `get_comparisons()` API client method; added "Comparisons" tab showing tiebreaker results (which image won) and duplicate checks (accepted/rejected)
**Reason**: Users can now see exactly which Siamese comparisons were made and their outcomes

### 2026-02-05 12:00:00
**Files**: `sim_bench/pipeline/steps/cluster_people.py`
**Change**: Rewrote `process()` to collect faces from `context.faces` and `context.face_embeddings` instead of requiring `context.all_faces` (which was never populated)
**Reason**: People tab was empty because `cluster_people` required `all_faces` from `filter_best_faces` step which wasn't in the pipeline

### 2026-02-05 12:01:00
**Files**: `app/streamlit/pages/results.py`
**Change**: Added thumbnails to Comparisons tab - both tiebreaker and duplicate check sections now show image thumbnails side by side
**Reason**: User requested visual comparison of images in the comparisons view

### 2026-02-05 12:02:00
**Files**: `app/streamlit/components/metrics.py`
**Change**: Added thumbnails to per-image metrics table, added "Final" (composite_score) column, uses `st.column_config.ImageColumn` for thumbnail display
**Reason**: User requested thumbnails in metrics table for easier identification

### 2026-02-05 12:03:00
**Files**: `configs/pipeline.yaml`
**Change**: Lowered `min_face_ratio` from 0.01 to 0.005 (0.5%), lowered `detection_confidence` from 0.2 to 0.15
**Reason**: Still missing faces; allowing very small faces to be detected

### 2026-02-05 14:00:00
**Files**: `app/streamlit/api_client.py`
**Change**: Fixed `_parse_person()` to map `thumbnail_image_path` to `representative_face` field; added `get_subclusters()` method
**Reason**: API returns `thumbnail_image_path` but client model expected `representative_face`; also need API method for fetching face sub-clusters

### 2026-02-05 14:01:00
**Files**: `sim_bench/api/database/models.py`
**Change**: Added `face_subclusters = Column(JSON)` to PipelineResult model
**Reason**: Need to persist face-based sub-clusters (images grouped by face identity within each scene cluster)

### 2026-02-05 14:02:00
**Files**: `sim_bench/api/services/pipeline_service.py`
**Change**: Added serialization of `context.face_clusters` to `face_subclusters` JSON in PipelineResult when saving completed pipeline
**Reason**: Sub-clusters computed by `cluster_by_identity` step were not being persisted to database

### 2026-02-05 14:03:00
**Files**: `sim_bench/api/services/result_service.py`, `sim_bench/api/routers/results.py`
**Change**: Added `get_subclusters(job_id)` service method and `GET /{job_id}/subclusters` API endpoint
**Reason**: Need to expose face sub-clusters via REST API for frontend display

### 2026-02-06 10:00:00
**Files**: `app/streamlit/pages/results.py`
**Change**: Added "Sub-Clusters" tab to results page showing face-based sub-clusters within each scene cluster
**Reason**: User requested sub-clusters to be displayed - shows images grouped by unique face combinations (e.g., A+B, A-only, B-only, no faces)

**Details**:
- Added `_render_subclusters_tab()` function
- Uses expandable sections for each scene cluster
- Sub-clusters sorted by face count (descending)
- Shows face count, identity signature, and thumbnail grid (up to 6 images)
- Uses emoji indicators: 👥 for faces, 📷 for no-face clusters

### 2026-02-06 11:00:00
**Files**:
- `app/streamlit/components/people_browser.py`
- `app/streamlit/models.py`
- `app/streamlit/api_client.py`

**Change**: Fixed People tab to show cropped face thumbnails and added inline rename

**Reason**: Person thumbnails were showing full image instead of just the face; user requested ability to edit person name from grid view

**Details**:
- Added `thumbnail_bbox` field to Person model
- Updated `_parse_person()` to include thumbnail_bbox from API
- Updated `_render_person_thumbnail()` to crop face from image using bbox with 30% padding
- Added inline rename functionality to `render_person_card()` - click pencil icon to rename
- Pass album_id to render_person_card for rename API calls

### 2026-02-06 12:00:00
**Files**: `sim_bench/pipeline/steps/cluster_by_identity.py`

**Change**: Fixed critical bug - sub-clustering now uses global person IDs from cluster_people instead of independent embedding quantization

**Reason**: Selection logic was inconsistent with People tab. "Person 3" in People tab was computed by global clustering, but sub-clustering used a different quantized embedding hash. This caused images with the same person to be placed in different sub-clusters and not compete properly.

**Details**:
- Added `_build_face_to_person_lookup()` to map (image_path, face_index) → person_id
- Modified `process()` to look up person IDs from `context.people_clusters`
- Removed old `_compute_identity_signature()` that used embedding quantization
- Updated dependency: now depends on `cluster_people` instead of `extract_face_embeddings`
- Sub-cluster identity now shows "Person_0+Person_1" format for clarity
- Added `person_ids` list to sub-cluster metadata for downstream use

### 2026-02-06 12:30:00
**Files**:
- `sim_bench/pipeline/steps/detect_faces.py`
- `sim_bench/face_pipeline/types.py`
- `sim_bench/api/services/people_service.py`

**Change**: Store cropped face images to disk for faster thumbnail loading

**Reason**: Previously cropped faces in memory but discarded them. For People tab thumbnails, had to re-crop from full image every time. Now save to `.faces/` directory.

**Details**:
- Added `_get_faces_dir()`, `_get_face_crop_path()`, `_save_face_crop()` helpers
- Modified `_serialize_faces()` to save crops to `{album}/.faces/{image}_face_{n}.jpg`
- Added `crop_path` field to `CroppedFace` dataclass
- Modified `_deserialize_faces()` to load from saved crop if available
- Updated `people_service.create_from_clusters()` to use `crop_path` for thumbnail if available
- Thumbnail stored as direct path to cropped face (no bbox needed when pre-cropped)

### 2026-02-06 12:31:00
**Files**: `app/streamlit/components/gallery.py`

**Change**: Make gallery images display with consistent square aspect ratio

**Reason**: Portrait and landscape images had different heights in grid, causing inconsistent visual layout

**Details**:
- Updated `_load_image_for_display()` to crop to center square by default
- Added `make_square` parameter (default True) for control
- Gallery now shows uniform thumbnail grid

### 2026-02-06 12:32:00
**Files**: `app/streamlit/components/people_browser.py`

**Change**: Fixed bbox coordinate conversion in person thumbnail cropping

**Reason**: Bbox values are stored in relative coordinates (0-1 range) but code was using them as pixel values, causing incorrect crop regions

**Details**:
- Added conversion: `x = x_rel * img_w`, etc.
- Padding calculation now works correctly with pixel values

### 2026-02-06 13:00:00
**Files**:
- `sim_bench/pipeline/steps/select_best.py`
- `app/streamlit/components/people_browser.py`
- `configs/pipeline.yaml`

**Change**: Fixed duplicate detection logic and added missing People page features

**Reason**: Multiple issues reported:
1. Duplicate detection was incorrectly using Siamese confidence instead of embedding similarity
2. People page missing `enable_selection` parameter and `render_merge_dialog` function
3. Name editing didn't show proper error messages
4. Near-identical images being selected due to too-strict threshold

**Details**:
- Rewrote `_check_near_duplicate()` to use embedding similarity only (Siamese CNN compares quality, not similarity)
- Added `_get_embedding_similarity()` helper method
- Lowered duplicate threshold from 0.95 to 0.85 (more aggressive filtering)
- Added `enable_selection` parameter to `render_people_grid()`
- Added `render_merge_dialog()` function for merging people
- Added `_add_to_merge_selection()` and `_remove_from_merge_selection()` helpers
- Added error handling and messages to inline rename functionality

### 2026-02-06 13:15:00
**Files**:
- `app/streamlit/components/gallery.py`
- `app/streamlit/components/metrics.py`
- `app/streamlit/pages/results.py`

**Change**: Fixed ALL thumbnail functions to produce consistent square images

**Reason**: Images were displaying at different sizes because `thumbnail()` only shrinks and maintains aspect ratio

**Details**:
- Fixed `_load_thumbnail` in gallery.py - crop to center square, resize to exact 300x300
- Fixed `_image_to_base64_thumbnail` in gallery.py - crop to center square, resize to exact size
- Fixed `_image_to_base64_thumbnail` in metrics.py - crop to center square, resize to exact size
- Fixed `_load_thumbnail` in results.py - crop to center square, resize to exact size
- All functions now: 1) crop to center square, 2) resize to exact requested size with LANCZOS

### 2026-02-06 13:45:00
**Files**:
- `sim_bench/api/services/pipeline_service.py`
- `configs/pipeline.yaml`

**Change**: Switched default pipeline from MediaPipe to InsightFace

**Reason**: User requested InsightFace as the default backend

**Details**:
- Updated `DEFAULT_PIPELINE` in pipeline_service.py to use InsightFace steps:
  - `detect_persons` (YOLOv8-Pose)
  - `insightface_detect_faces` (InsightFace SCRFD)
  - `insightface_score_expression/eyes/pose`
- Added `cluster_people` step (missing from original InsightFace config)
- Updated `select_best` config with InsightFace scoring:
  - `scoring_backend: insightface`
  - `scoring_strategy: insightface_penalty`
  - Penalty weights for body/face/eyes/smile/pose

### 2026-02-06 14:00:00
**Files**: `app/streamlit/components/gallery.py`

**Change**: Fixed gallery image sizes with caching and fixed pixel width

**Reason**: Images were still displaying at inconsistent sizes despite previous fixes

**Details**:
- Changed from `use_column_width=True` to `width=THUMBNAIL_SIZE` (200px)
- Added `@st.cache_data` decorator for thumbnail caching (faster loading)
- Thumbnails now stored as JPEG bytes in Streamlit cache
- All images display at exactly 200x200 pixels (fits 4 per row)
- Truncated long filenames to prevent layout issues

### 2026-02-06 14:30:00
**Files**:
- `app/streamlit/components/gallery.py`
- `app/streamlit/components/metrics.py`
- `app/streamlit/models.py`
- `app/streamlit/api_client.py`
- `sim_bench/api/services/pipeline_service.py`

**Change**: Added InsightFace metrics to Results view

**Reason**: User requested new InsightFace metrics (person detection, body facing score) to be displayed

**Details**:
- Added `person_detected`, `body_facing_score`, `person_confidence` to ImageInfo model
- Updated `_build_image_metrics()` to include InsightFace person detection data
- Updated `_parse_image()` in API client to parse new fields
- Updated `_render_face_info()` to show body facing score
- Updated cluster score table to include Person and Body columns
- Updated per-image metrics table to include InsightFace metrics

### 2026-02-06 14:35:00
**Files**: `app/streamlit/components/gallery.py`

**Change**: Changed thumbnail resize to preserve aspect ratio

**Reason**: User requested images not be cropped, just resized to fixed width

**Details**:
- Changed from square crop + resize to width-only resize
- Thumbnails now 200px wide with proportional height
- Full image content preserved (no cropping)

### 2026-02-07 10:00:00
**Files**: `notebooks/eda_yolo_insightface.ipynb` (created)

**Change**: Created EDA notebook for YOLOv8 person detection and InsightFace face analysis

**Reason**: User requested notebook to explore model outputs and experiment with detection results

**Details**:
- **Part 1: YOLOv8 Person Detection**
  - Loads YOLOv8-Pose model (configurable size: n/s/m/l/x)
  - Shows model outputs: bounding boxes, confidence, 17 COCO keypoints, body facing score
  - Top 5 highest/lowest confidence detections
  - Front-facing vs side-facing analysis
  - Cell to run on specific user-selected image
- **Part 2: InsightFace Face Analysis**
  - Loads InsightFace buffalo_l model
  - Shows outputs: bbox, confidence, 5-point landmarks, age, gender, pose angles
  - Heuristic smile score from mouth/eye ratio
  - 5 images with faces / 5 without
  - 5 smiling / 5 not smiling
  - Age/gender distribution charts
  - Cell to run on specific image
- **Part 3: Combined Analysis**
  - Merges YOLOv8 + InsightFace results
  - Categories: person+face, person-only, face-only, neither
  - 5 person+face images, 5 person-no-face (back turned)
  - 5 smiling persons, 5 not smiling
  - Combined visualization on specific image
- Dataset: `D:\Budapest2025_Google`
- Helper functions for visualization with bounding boxes and keypoints

---

### 2026-02-07 09:00:00
**Files**:
- `sim_bench/pipeline/context.py`
- `sim_bench/pipeline/steps/insightface_score_expression.py`
- `sim_bench/pipeline/steps/insightface_score_eyes.py`
- `sim_bench/pipeline/steps/insightface_score_pose.py`
- `sim_bench/pipeline/scoring/strategy.py`
- `sim_bench/pipeline/steps/extract_face_embeddings.py`

**Change**: Fixed 4 bugs in InsightFace pipeline

**Reason**: Loop logic bug caused face lookup to always return first face or None; missing context attributes caused AttributeError; scoring strategy assumed attributes existed without defensive checks; extract_face_embeddings had incomplete dependency list

**Details**:
1. **Bug 1 (CRITICAL) - Loop logic error**: Fixed `_find_face()` in 3 files (insightface_score_expression.py:91-93, insightface_score_eyes.py:91-93, insightface_score_pose.py:91-93). Changed from `return face if face_matches else None` (returns on first iteration!) to `if face.get('face_index') == face_index: return face` followed by `return None` outside loop.

2. **Bug 2 (HIGH) - Missing context attributes**: Added `persons: dict[str, dict]` and `insightface_faces: dict[str, dict]` fields to PipelineContext dataclass in context.py (after line 32).

3. **Bug 3 (HIGH) - Missing defensive checks**: Changed `context.persons.get()` and `context.insightface_faces.get()` to use `getattr(context, 'persons', {})` pattern in strategy.py at lines 77, 94, 100, 122. This prevents AttributeError when context doesn't have these attributes.

4. **Bug 4 (MEDIUM) - Wrong dependency metadata**: Updated `depends_on` in extract_face_embeddings.py from `["detect_faces"]` to `["detect_faces", "insightface_detect_faces"]` so step runs after either MediaPipe or InsightFace face detection.

---

### 2026-02-07 11:30:00
**Files**: `CLAUDE.md`

**Change**: Improved CLAUDE.md for better clarity and reduced verbosity

**Reason**: User ran `/init` command to improve the Claude Code guidance file

**Details**:
- Condensed project overview to bullet points
- Consolidated common commands into single code block
- Added concrete code example for creating new pipeline steps
- Added table format for key entry points
- Documented both pipelines (default + insightface)
- Removed "Recent Updates" section (transient info that becomes stale)
- Removed verbose module structure list (easily discoverable)
- Removed redundant "Adding New Components" section (replaced with code example)
- Added model weights location section
- Streamlined debugging tips
- Reduced overall length by ~40% while preserving essential information

---

### 2026-02-07 12:00:00
**Files**: `docs/architecture/PIPELINE_CALL_CHAIN.md` (created)

**Change**: Created comprehensive documentation explaining why MediaPipe is being called

**Reason**: User encountered protobuf/MediaPipe compatibility error and wanted to understand the full call chain

**Details**:
- Traced complete call chain from user click → frontend → API → executor → step → MediaPipe
- **Root cause identified**: Frontend `pipeline_runner.py:12-27` has outdated `DEFAULT_PIPELINE` using MediaPipe steps (`score_face_eyes`), while backend `pipeline_service.py:22-38` has updated InsightFace pipeline
- Frontend sends its step list to backend, overriding backend's default
- Documented both pipelines (MediaPipe vs InsightFace) with step comparisons
- Explained dependency resolution and why `detect_faces` gets auto-added
- Included ASCII diagrams showing the problem flow
- Provided 3 fix options:
  1. Update frontend DEFAULT_PIPELINE to use InsightFace steps (recommended)
  2. Don't pass steps from frontend, let backend use its default
  3. Downgrade protobuf (not recommended)

---

### 2026-02-08 12:30:00
**Files**: `docs/architecture/CONFIG_SINGLE_SOURCE_OF_TRUTH_PLAN.md` (created)

**Change**: Created comprehensive plan to make YAML config the single source of truth

**Reason**: User identified that there are 3 conflicting sources for pipeline definition (YAML, frontend, backend) and wants a clean architecture

**Details**:
- **Problem**: Frontend hardcodes `DEFAULT_PIPELINE` (MediaPipe), backend has different `DEFAULT_PIPELINE` (InsightFace), YAML has yet another version
- **Solution**: YAML → DB (on startup sync) → Frontend (fetches from API)
- **5 Phases**:
  1. Update YAML to current InsightFace pipeline
  2. Improve config sync (sync YAML to DB on every startup, not just first run)
  3. Remove hardcoded pipelines from frontend and backend
  4. Add user settings persistence (save/load user preferences to DB)
  5. Migration script for existing installations
- **New DB columns**: `is_system`, `user_id`, `parent_profile_id` for ConfigProfile
- **New API endpoints**: `GET/POST /config/user/{user_id}` for user settings
- **Key principle**: User profiles store only OVERRIDES, not full config - they inherit from default and automatically get updates when YAML changes

---

### 2026-02-08 13:00:00
**Files**:
- `configs/pipeline.yaml`
- `sim_bench/api/database/models.py`
- `sim_bench/api/services/config_service.py`
- `sim_bench/api/services/pipeline_service.py`
- `sim_bench/api/routers/config.py`
- `app/streamlit/api_client.py`
- `app/streamlit/components/pipeline_runner.py`

**Change**: Implemented "YAML as Single Source of Truth" for pipeline configuration

**Reason**: Resolve the 3-source-of-truth problem where YAML, frontend, and backend all had different pipeline definitions

**Details**:
- **Phase 1**: Updated `pipeline.yaml` with `minimal_pipeline` option (InsightFace already set as default)
- **Phase 2**:
  - Added `is_system`, `user_id`, `parent_profile_id` columns to ConfigProfile model
  - Updated `config_service.py` with `sync_default_profile()` that syncs YAML→DB on every startup
  - Added `get_available_pipelines()` helper function
  - Added user profile methods: `get_user_profile()`, `save_user_profile()`, `get_user_config()`, `delete_user_profile()`
- **Phase 3**:
  - Removed hardcoded `DEFAULT_PIPELINE` from `pipeline_service.py`
  - Updated `start_pipeline()` to load steps from config service when not provided
  - Removed hardcoded pipelines from `pipeline_runner.py`
  - Frontend now fetches pipelines from API via `get_available_pipelines()`
- **Phase 4**:
  - Added API endpoints: `GET/POST/DELETE /config/user/{user_id}` and `GET /config/pipelines`
  - Added API client methods: `get_available_pipelines()`, `get_user_config()`, `save_user_config()`
  - Frontend loads saved user settings on page load
  - Added "Save Settings" button to persist user preferences
  - Config slider values now restore from saved settings

**To apply**: Delete `sim_bench.db` and restart the API to create fresh database from YAML

---

### 2026-02-08 14:00:00
**Files**:
- `sim_bench/pipeline/steps/score_ava.py`
- `sim_bench/pipeline/context.py`
- `sim_bench/pipeline/scoring/strategy.py`
- `sim_bench/pipeline/steps/insightface_score_pose.py`
- `sim_bench/pipeline/steps/insightface_score_eyes.py`
- `sim_bench/pipeline/steps/insightface_score_expression.py`
- `sim_bench/pipeline/insightface_pipeline/face_cropper.py` (created)
- `sim_bench/pipeline/insightface_pipeline/__init__.py`
- `configs/pipeline.yaml`

**Change**: Fixed face scoring (AVA, Pose, Eyes, Smile) to produce meaningful quality metrics

**Reason**: Scoring steps were returning hardcoded 0.5 values or using inconsistent scales, making quality-based selection ineffective

**Details**:

1. **AVA Score Normalization**:
   - Modified `score_ava.py:_store_results()` to divide scores by 10 before storing
   - AVA model returns 1-10 scale, now normalized to 0-1 at storage time
   - Removed redundant normalization from `context.py:get_image_score()` (line 104)
   - Removed redundant normalization from `strategy.py:InsightFacePenaltyScoring.compute_score()` (line 60)
   - Changed default fallback from `5.0 / 10.0` to `0.5` in strategy.py

2. **Pose Scoring from InsightFace Landmarks**:
   - Replaced stub `FacePoseScorer.compute_score()` in `insightface_score_pose.py`
   - New algorithm computes frontal score from 5-point landmarks (left_eye, right_eye, nose)
   - Calculates eye center, eye vector, and nose deviation from eye line
   - Normalizes yaw by eye distance to get frontal score (1 = frontal, 0 = profile)
   - Added `import numpy as np` for calculations

3. **Face Cropping Utility**:
   - Created `face_cropper.py` with `InsightFaceCropper` class
   - Takes InsightFace bbox, applies configurable margin (default 30%), resizes to 256x256
   - Handles EXIF rotation with `ImageOps.exif_transpose()`
   - Exported from `__init__.py`

4. **Eye Scoring via MediaPipe on Cropped Faces**:
   - Replaced stub `EyeStateScorer.compute_score()` in `insightface_score_eyes.py`
   - Uses `InsightFaceCropper` to get 256x256 face crop
   - Runs MediaPipe Face Mesh on crop
   - Calls existing `detect_eye_state()` from `portrait_analysis/eye_state.py`
   - Normalizes EAR (Eye Aspect Ratio) to 0-1 score
   - Added config parameters: `crop_margin`, `target_size`, `ear_threshold`
   - Updated `_find_face()` to enrich face_data with `original_path`
   - Removed unused `NeutralScorer` class

5. **Smile Scoring via MediaPipe on Cropped Faces**:
   - Replaced stub `ExpressionScorer.compute_score()` in `insightface_score_expression.py`
   - Uses same `InsightFaceCropper` approach as eye scoring
   - Runs MediaPipe Face Mesh on crop
   - Calls existing `detect_smile()` from `portrait_analysis/smile_detection.py`
   - Returns normalized smile score (already 0-1 from utility)
   - Added config parameters: `crop_margin`, `target_size`, `width_threshold`
   - Updated `_find_face()` to enrich face_data with `original_path`
   - Removed unused `NeutralScorer` class

6. **Pipeline Config Updates**:
   - Updated `pipeline.yaml` with new config parameters for InsightFace scoring steps
   - `insightface_score_expression`: crop_margin, target_size, width_threshold
   - `insightface_score_eyes`: crop_margin, target_size, ear_threshold
   - `insightface_score_pose`: simplified config (uses 5-point landmarks, no external model needed)

---

### 2026-02-09 15:00:00
**Files**:
- `requirements.txt`
- `sim_bench/pipeline/utils/__init__.py` (created)
- `sim_bench/pipeline/utils/image_cache.py` (created)
- `sim_bench/pipeline/insightface_pipeline/face_analyzer.py`
- `sim_bench/pipeline/insightface_pipeline/face_cropper.py`
- `sim_bench/pipeline/steps/extract_face_embeddings.py`
- `sim_bench/portrait_analysis/analyzer.py`
- `sim_bench/face_pipeline/crop_service.py`

**Change**: Fixed protobuf compatibility and created global image cache with EXIF normalization

**Reason**: Two issues: (1) protobuf 6.x incompatible with MediaPipe causing `'MessageFactory' object has no attribute 'GetPrototype'` error, (2) EXIF transpose happening inconsistently causing bbox coordinate mismatches ("Coordinate 'right' is less than 'left'" warnings)

**Details**:

1. **Protobuf Version Fix**:
   - Added `protobuf>=3.20,<4` to requirements.txt
   - This version works with both MediaPipe and Streamlit

2. **Global Image Cache** (`sim_bench/pipeline/utils/image_cache.py`):
   - Created `ImageCache` singleton class with persistent disk cache
   - Cache location: `~/.sim_bench/image_cache/`
   - EXIF-first cache key strategy:
     - If image has EXIF DateTimeOriginal: `SHA256(datetime + make + model + size)`
     - Fallback: `SHA256(first_64KB + last_64KB + size)`
   - Images normalized once (EXIF transposed, RGB converted) and cached as JPEG
   - SQLite index for fast lookups
   - API: `get()`, `get_pil()`, `get_dimensions()`, `clear()`, `evict()`, `get_stats()`

3. **Updated Image Consumers**:
   - `face_analyzer.py`: Use `get_image_cache().get()` instead of `Image.open()`
   - `face_cropper.py`: Use `get_image_cache().get_pil()`, added bbox validation
   - `extract_face_embeddings.py`: Use cache, added crop coordinate validation
   - `portrait_analysis/analyzer.py`: Use cache in `_load_image()`
   - `face_pipeline/crop_service.py`: Use cache in `_load_image()`

4. **Benefits**:
   - Consistent EXIF handling across all pipeline steps
   - Bbox coordinates always match image orientation
   - Performance: images normalized once, cached for reuse
   - Shared across albums (same image = one cached copy)

---

### 2026-02-09 16:00:00
**Files**:
- `sim_bench/pipeline/steps/cluster_people.py`
- `sim_bench/pipeline/scoring/quality_strategy.py`
- `app/streamlit/pages/results.py`

**Change**: Fixed People tab, Siamese comparisons logging, and UI clarity

**Reason**: Multiple issues reported: People tab empty, no Siamese comparisons displayed, "All Filtered" confusing

**Details**:

1. **Fix cluster_people for InsightFace pipeline** (`cluster_people.py`):
   - Root cause: Step only looked at `context.faces` (MediaPipe), not `context.insightface_faces` (InsightFace)
   - Added `_collect_faces_with_embeddings()` method that works with both pipelines
   - Created `FaceForClustering` dataclass for lightweight face representation
   - Properly looks up embeddings using cache key format (`"path:face_N"`)
   - Now `context.people_clusters` gets populated, Person records get created

2. **Log Siamese comparisons from quality strategies** (`quality_strategy.py`):
   - Root cause: `_apply_siamese_refinement()` and `_run_tournament()` didn't have access to `context`
   - Updated `SiameseRefinementQuality._apply_siamese_refinement()` to accept `context` and log comparisons
   - Updated `SiameseTournamentQuality._run_tournament()` to accept `context` and log comparisons
   - Comparisons now logged to `context.siamese_comparisons` with type='refinement' or type='tournament'

3. **Rename "All Filtered" to "All Processed"** (`results.py`):
   - Changed view mode option from "All Filtered" to "All Processed"
   - Updated description from "passed quality filter" to "processed by pipeline"
   - Updated metric label to "All Processed"

**Cascading effects**:
- People tab will now show detected people
- Person column in cluster view will be populated
- Sub-clustering by identity will work properly
- Comparisons tab will show Siamese refinement/tournament comparisons

---

### 2026-02-12 12:00:00
**Files**:
- `tests/pipeline/test_face_recognition_benchmark.py` (created)
- `tests/pipeline/test_face_pipeline_e2e.py` (created)
- `scripts/clear_face_embedding_cache.py` (created)
- `scripts/check_bbox_format.py` (created)
- `scripts/check_zero_vectors.py` (created)
- `docs/FACE_RECOGNITION_FIX_PLAN.md` (created)

**Change**: Diagnosed and fixed People tab showing all faces as one person

**Root Cause**: 93.8% of cached face embeddings were zero vectors from a previous buggy code version. Zero vectors are identical, so HDBSCAN clustered them all into one person.

**Investigation**:
1. Database analysis showed 410/437 embeddings were zero vectors
2. Created benchmark test with CASIA WebFace data - embedding model works correctly
3. Created E2E test with Budapest 2025 data - actual pipeline steps work correctly
4. Conclusion: Stale cached zero vectors were the problem, not current code

**Fix**: Cleared face_embedding cache (437 entries) and people table (3 entries)

**Tests Created**:
- `test_face_recognition_benchmark.py`: Tests InsightFace embedding extraction on pre-cropped faces
- `test_face_pipeline_e2e.py`: Tests actual pipeline steps (detection → embedding → clustering)

**Action Required**: Re-run the pipeline on albums to regenerate embeddings and people clusters

---

### 2026-02-16 01:35:00
**Files**:
- `sim_bench/clustering/hybrid_hdbscan_knn.py` (complete rewrite)
- `sim_bench/clustering/hybrid_closest_face.py` (bug fix)

**Change**: Simplified face clustering algorithm to use median + 2×IQR threshold

**Reason**: Previous complex algorithm with exemplars, minimum pairs, and distinct exemplar requirements was too restrictive (0 merges, 0 attachments). User requested simpler, statistically grounded approach.

**New Algorithm**:
1. HDBSCAN → initial clusters
2. For each cluster, compute T = median(K-NN distances) + 2×IQR
3. Iteratively:
   - Merge: if closest inter-cluster pair ≤ min(T_a, T_b)
   - Attach: noise point → cluster if closest face ≤ T
4. Repeat until no changes

**Parameters**:
- `knn_k`: 3 (neighbors for local cohesion)
- `iqr_multiplier`: 2.0
- `threshold_floor`: 0.3 (minimum threshold)
- `max_iterations`: 10

**Benchmark Results** (Budapest2025_Google, 254 faces):
- HDBSCAN baseline: 24 clusters, 83 noise
- **New algorithm: 5 clusters, 0 noise** (14 merges, 93 attached)
- Old hybrid_closest: 112 clusters, 0 noise (0 merges, no attachment)

**Bug Fix** (`hybrid_closest_face.py`):
- Added missing `merge_threshold` and `attach_threshold` attributes to `__init__`

---

### 2026-02-18 00:00:00
**Files**:
- `CLAUDE.md` (modified)
- `docs/LEARNINGS.md` (created)
- `docs/architecture.md` (created)
- `docs/requirements.md` (created)

**Change**: Improved CLAUDE.md and created missing documentation files

**Reason**: User ran `/init` command to improve the Claude Code guidance file

**Details**:
- Fixed typos in CLAUDE.md: "agains" → "against", "architeture" → "architecture"
- Fixed path separator: `docs\LEARNINGS.md` → `docs/LEARNINGS.md`
- Added full paths to referenced docs: `architecture.md` → `docs/architecture.md`, `requirements.md` → `docs/requirements.md`
- Added "Windows Development Notes" section with path and testing guidance
- Added reference to README.md for detailed benchmarking information
- Created `docs/LEARNINGS.md` - template for bug learnings log
- Created `docs/architecture.md` - system architecture documentation
- Created `docs/requirements.md` - requirements tracking log

---

### 2026-02-11 00:00:00
**Files**: `CLAUDE.md`
**Change**: Improved CLAUDE.md documentation with fixes and enhancements
**Reason**: User ran `/init` command to review and improve the Claude Code guidance file

**Details**:
- Fixed typos: "implementign" → "implementing", "hot" → "how", "prepor" → "proper", "centralizedd" → "centralized", "unpreditable" → "unpredictable"
- Added Python version requirement (3.10+)
- Added services layer documentation to architecture section
- Added pipeline data flow explanation (API → Executor → Steps → Context)
- Added caching system documentation (UniversalCacheHandler, mtime tracking)
- Added face embedding factory to factory pattern section
- Improved code example for full imports

---

### 2026-02-19 00:00:00
**Files**: `sim_bench/clustering/distance_utils.py` (new), `sim_bench/clustering/hybrid_hdbscan_knn.py`, `sim_bench/clustering/hybrid_closest_face.py`, `sim_bench/clustering/hdbscan.py`, `sim_bench/clustering/hdbscan_pca.py`
**Change**: Switched all facial clustering algorithms from Euclidean distance to cosine distance
**Reason**: User requested consistent use of cosine distance (1 - cosine_similarity) instead of Euclidean distance on normalized vectors

**Details**:
- Created `distance_utils.py` with shared cosine distance functions:
  - `cosine_distance_matrix(X, Y)` - distance matrix between two sets
  - `cosine_distance_pairwise(X)` - condensed pairwise distances (like pdist)
  - `cosine_distance_to_set(x, Y)` - single vector to set distances
- Updated HDBSCAN calls to use `metric='precomputed'` with cosine distance matrix
- Recalibrated thresholds using formula: t_c = (t_e²) / 2
  - hybrid_hdbscan_knn: floor 0.50→0.125, ceiling 0.90→0.405
  - hybrid_closest_face: floor 0.30→0.045, ceiling 0.90→0.405
- All distance values clipped to [0, 2] for numeric safety

**Additional fix (same change set)**:
- Updated `cluster_selection_epsilon` from 0.3 to 0.045 in hybrid methods (same conversion formula)

---

### 2026-02-19 01:00:00
**Files**: `sim_bench/clustering/hybrid_hdbscan_knn_Tcore2all.py` (new), `sim_bench/clustering/hybrid_hdbscan_knn_merge_twotier.py` (new), `sim_bench/clustering/base.py`
**Change**: Added two new clustering algorithm variants
**Reason**: User requested variants to reduce pose-mode splits in face clustering

**Details**:
- `hybrid_hdbscan_knn_tcore2all`: Computes threshold T from exemplar→all-faces distances instead of exemplar↔exemplar pairwise. Uses 95th percentile. Captures pose spread better.
- `hybrid_hdbscan_knn_merge_twotier`: Adds secondary merge rule - if ≥5 pairs pass <= max(T_A, T_B), merge even when primary rule fails. Helps merge when one cluster has tighter T.
- Both registered in clustering factory

---

### 2026-02-19 01:30:00
**Files**: `sim_bench/clustering/hybrid_hdbscan_knn_attach_strong1.py` (new), `sim_bench/clustering/base.py`
**Change**: Added strong single-exemplar attachment variant
**Reason**: Reduce leftover noise fragments that would form tiny clusters

**Details**:
- `hybrid_hdbscan_knn_attach_strong1`: Adds secondary attach rule
  - Primary (unchanged): noise joins if ≥2 exemplars within T
  - Secondary (new): noise joins if 1 exemplar within 0.8×T (stricter threshold)
- New param: `attach_strong1_multiplier` (default: 0.8)
- Prioritizes primary matches over strong1 matches when choosing cluster

---

### 2026-02-19 02:00:00
**Files**: `configs/clustering_benchmark.yaml`
**Change**: Updated benchmark config with cosine distance thresholds and new variants
**Reason**: Align config with code changes and add new algorithm variants to benchmark

**Details**:
- Updated all `cluster_selection_epsilon` from 0.3/0.35 to 0.045 (cosine distance)
- Updated `threshold_floor`/`threshold_ceiling` to cosine distance values
- Added three new variants:
  - `hybrid_knn_tcore2all`: Threshold from exemplar→all-faces (95th percentile)
  - `hybrid_knn_merge_twotier`: Two-tier merge with secondary max(T) rule
  - `hybrid_knn_attach_strong1`: Strong single-exemplar attachment (0.8×T)

### 2026-02-19 10:10:13
**Files**: `sim_bench/clustering/hybrid_closest_face.py`, `app/face_clustering_debug/services/clustering_runner.py`, `app/face_clustering_debug/models/schemas.py`
**Change**: Fixed hybrid_closest debug output - added d3_cross values per face, separate fits_a/fits_b counts, merge_threshold_multiplier parameter
**Reason**: Debug app was showing misleading MinDist (exemplar distance) instead of the actual d3_cross values used for merge decisions

### 2026-02-19 10:20:04
**Files**: `app/face_clustering_debug/services/clustering_runner.py`, `app/face_clustering_debug/services/file_loader.py`, `app/face_clustering_debug/main.py`, `app/face_clustering_debug/components/decision_card.py`, `app/face_clustering_debug/components/algorithm_explanation.py`
**Change**: Added 3 new clustering variants to Parameter Tuning, added error handling to prevent blank pages, updated merge decision UI for hybrid_closest_face d3_cross values, added method comparison documentation
**Reason**: Complete debug app improvements - show actual merge criteria, document algorithm differences, improve error visibility

### 2026-02-27 10:00:00
**Files**: 
- `face_cluster/config.py`
- `face_cluster/merge.py`
- `face_cluster/analysis.py`
- `docs/FACE_CLUSTERING_COMPLETE_GUIDE.md`
- `notebooks/analyze_merge_decisions.ipynb` (NEW)

**Change**: Added `merge_global_percentile` parameter to control global threshold percentile

**Reason**: User requested ability to experiment with global threshold calculation (previously hardcoded to median)

**What changed**:

1. **New parameter** `merge_global_percentile` (default 50):
   - 25 = more conservative (use P25 of cluster thresholds)
   - 50 = median (default, previous behavior)
   - 75 = more permissive (use P75, allows more merging)
   - 90 = very permissive

2. **Updated `_compute_global_threshold()`**:
   - Changed from `np.median()` to `np.percentile(values, config.merge_global_percentile)`
   - Allows experimentation with different global threshold strategies

3. **Updated analysis.py**:
   - `get_close_clusters_df()` uses configurable percentile
   - `get_merge_decisions_df()` uses configurable percentile
   - `plot_decision_boundaries()` shows "P50" or "P75" labels instead of just "median"

4. **Created tutorial notebook** `notebooks/analyze_merge_decisions.ipynb`:
   - Shows how to see failed merge criteria in DataFrame
   - Shows how to create ClusterSnapshot after merging
   - Shows how to access distance matrix
   - Shows how to experiment with different global percentiles
   - Includes code examples for all common analysis tasks

5. **Updated documentation**:
   - Added `merge_global_percentile` to configuration table
   - Updated adaptive threshold formula to show percentile is configurable
   - Added section on experimenting with global threshold values

**Why this helps**:
- T_global now configurable: can make it more conservative (P25) or permissive (P75)
- Formula: T_merge = α × MAX(T_A, T_B) + (1-α) × percentile(all cluster thresholds)
- Higher percentiles → higher T_global → more lenient merging
- Lower percentiles → lower T_global → more conservative merging

**Example impact**:
If cluster thresholds are [0.20, 0.25, 0.30, 0.35, 0.40]:
- P25 = 0.25 (conservative)
- P50 = 0.30 (median, default)
- P75 = 0.35 (permissive)

For pair with T_local=0.25, alpha=0.7:
- With P25: T_merge = 0.7×0.25 + 0.3×0.25 = 0.250
- With P50: T_merge = 0.7×0.25 + 0.3×0.30 = 0.265
- With P75: T_merge = 0.7×0.25 + 0.3×0.35 = 0.280

Higher global percentile → more pairs pass exemplar distance check → more merging.


### 2026-02-27 10:30:00
**Files**: 
- `face_cluster/analysis.py`
- `docs/LEARNINGS.md`

**Change**: Added `get_cluster_distances()` method to ClusterSnapshot

**Reason**: User said merge_decisions_df was "useless" - they needed cluster-to-cluster distance matrix, not just failure criteria

**What the method does**:
Returns DataFrame with ALL cluster pairs showing:
- `Exemplar_Dist`: Min distance between exemplars (used for merge proposal threshold 0.45)
- `Min_Dist`: Min distance between any two faces
- `Mean_Dist`: Mean distance across all face pairs
- `Max_Dist`: Max distance between any two faces

**Why this is better**:
- Shows WHY pairs weren't even proposed (e.g., "Exemplar_Dist=0.52 > 0.45")
- merge_decisions_df only shows already-proposed candidates
- Missing info: pairs with exemplar_dist > 0.45 never appear in merge_decisions

**Usage**:
```python
df = snapshot.get_cluster_distances()
print(df.head(20))  # Sorted by Exemplar_Dist

# Why didn't (4, 23) merge?
row = df[(df['C1']==4) & (df['C2']==23)].iloc[0]
if row['Exemplar_Dist'] > 0.45:
    print("Not proposed - exemplar distance too large")
```

**Learning added**: For "why didn't X happen?" provide input data (distances) first, decision logic (criteria) second.

