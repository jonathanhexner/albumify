# ML-Based Cluster Merging Implementation Plan

**Date**: 2026-02-27
**Feature Request**: ML-Based Cluster Merging Pipeline
**Status**: Draft - Awaiting Approval

---

## 1. Objective

Replace the heuristic-based cluster merging in the face clustering pipeline with a **logistic regression model** trained on manually corrected clustering results. This will enable:
- Data-driven merge decisions instead of hand-tuned thresholds
- Iterative improvement through additional labeled data
- Interpretable feature importance for understanding merge criteria
- Baseline comparison between heuristic and ML approaches

---

## 2. Requirements

### Functional Requirements
- **FR-1**: Export clustering results to 3 CSV files (faces, clusters, candidate pairs) after initial mutual kNN clustering
- **FR-2**: Provide visual labeling interface for correcting cluster assignments
- **FR-3**: Generate binary training labels (merge/no-merge) from corrected cluster identities
- **FR-4**: Train logistic regression model with comprehensive evaluation metrics
- **FR-5**: Deploy trained model as alternative merge strategy alongside existing heuristic merger

### Non-Functional Requirements
- **NFR-1**: Support batch processing on multiple datasets
- **NFR-2**: Export format must be human-readable CSV for manual review/editing
- **NFR-3**: Model training should complete in < 5 minutes for typical datasets (< 1000 cluster pairs)
- **NFR-4**: Labeling interface should load and display clusters in < 3 seconds

---

## 3. Architecture Overview

### 3.1 Data Flow

```
┌─────────────────┐
│ Image Dataset   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│ Stage 1: Clustering Export                  │
│ (script or notebook)                        │
├─────────────────────────────────────────────┤
│ • Load embeddings OR detect faces           │
│ • Quality gating (pose + blur)              │
│ • Mutual kNN graph → connected components   │
│ • D10 exemplar selection                    │
│ • Export 3 CSVs (NO merging yet)            │
└────────┬────────────────────────────────────┘
         │
         ├──► faces.csv (one row per face)
         ├──► clusters.csv (one row per cluster)
         └──► candidate_pairs.csv (cluster pair features)
         │
         ▼
┌─────────────────────────────────────────────┐
│ Stage 2: Manual Labeling                   │
│ (Streamlit app)                             │
├─────────────────────────────────────────────┤
│ • Load CSVs + display cluster images        │
│ • User assigns corrected_identity per       │
│   cluster (e.g., "person_A", "person_B")    │
│ • Save corrected_labels.csv                 │
└────────┬────────────────────────────────────┘
         │
         └──► corrected_labels.csv (cluster_id → identity)
         │
         ▼
┌─────────────────────────────────────────────┐
│ Stage 3: Training Data Generation           │
│ (part of training script)                   │
├─────────────────────────────────────────────┤
│ • Join candidate_pairs.csv + corrected      │
│   labels                                    │
│ • For each pair: label=1 if same identity,  │
│   label=0 if different                      │
│ • Save merge_training_data.csv              │
└────────┬────────────────────────────────────┘
         │
         └──► merge_training_data.csv (features + binary label)
         │
         ▼
┌─────────────────────────────────────────────┐
│ Stage 4: Model Training & Evaluation        │
│ (training script)                           │
├─────────────────────────────────────────────┤
│ • Train logistic regression                 │
│ • Cross-validation                          │
│ • Feature importance analysis               │
│ • Confusion matrix, precision/recall        │
│ • Save model + metrics                      │
└────────┬────────────────────────────────────┘
         │
         ├──► merge_classifier.pkl (trained model)
         └──► training_report.json (metrics)
         │
         ▼
┌─────────────────────────────────────────────┐
│ Stage 5: Deployment                         │
│ (MLMerger class in face_cluster)            │
├─────────────────────────────────────────────┤
│ • Load trained model                        │
│ • Compute same features as training         │
│ • Predict merge probability                 │
│ • Apply threshold (e.g., p > 0.5)           │
│ • Iterative merging                         │
└─────────────────────────────────────────────┘
```

### 3.2 Integration Points

**Existing Components**:
- `face_cluster/` module - all clustering logic (knn_graph, clustering, exemplars, merge, analysis)
- `notebooks/debug_knn_graph_clustering.ipynb` - reference pipeline implementation
- `face_cluster/analysis.py` - `ClusterSnapshot` class for metrics

**New Components**:
1. **Export Script** (`scripts/export_clustering_data.py`)
   - CLI interface for batch processing
   - Uses `face_cluster` library
   - Outputs to `results/face_clustering_training/<dataset_name>/`

2. **Export Notebook** (`notebooks/export_clustering_data.ipynb`)
   - Interactive version for exploration
   - Same logic as script, with visualization

3. **Labeling App** (`app/face_clustering_labeling.py`)
   - Streamlit interface
   - Reads CSVs, displays cluster grids
   - Saves corrected_labels.csv

4. **Training Script** (`scripts/train_merge_classifier.py`)
   - Generates training data
   - Trains logistic regression
   - Evaluates and saves model

5. **ML Merger** (`face_cluster/ml_merge.py`)
   - `MLMerger` class (similar to `ConservativeMerger`)
   - Loads trained model
   - Implements merge logic using predictions

---

## 4. Data Schema

### 4.1 faces.csv

One row per face detected in the dataset.

| Column | Type | Description |
|--------|------|-------------|
| `face_id` | int | Unique face identifier (0-indexed) |
| `image_path` | str | Path to source image |
| `cluster_id` | int | Initial cluster assignment from Stage 1 (-1 for noise) |
| `bbox_x` | float | Bounding box top-left x (pixels) |
| `bbox_y` | float | Bounding box top-left y (pixels) |
| `bbox_w` | float | Bounding box width (pixels) |
| `bbox_h` | float | Bounding box height (pixels) |
| `blur_score` | float | Laplacian variance blur metric |
| `pose_yaw` | float | Yaw angle (degrees, optional) |
| `pose_pitch` | float | Pitch angle (degrees, optional) |
| `pose_roll` | float | Roll angle (degrees, optional) |
| `is_core` | bool | Whether face passed quality gating |
| `embedding_index` | int | Index into embeddings array (for linking) |

### 4.2 clusters.csv

One row per cluster from Stage 1 (before merging).

| Column | Type | Description |
|--------|------|-------------|
| `cluster_id` | int | Cluster identifier |
| `cluster_size` | int | Number of faces in cluster |
| `exemplar_ids` | str | Comma-separated face IDs of exemplars (e.g., "5,8,11") |
| `diameter` | float | Max pairwise distance within cluster |
| `T_A` | float | Per-cluster adaptive threshold (P90 of exemplar distances) |
| `mean_blur` | float | Mean blur score of faces in cluster |
| `face_ids` | str | Comma-separated list of all face IDs in cluster |

### 4.3 candidate_pairs.csv

One row per candidate cluster pair (filtered by reasonable distance threshold).

**Features** (all float):
| Column | Description |
|--------|-------------|
| `cluster_id_1` | First cluster ID |
| `cluster_id_2` | Second cluster ID |
| `min_exemplar_dist` | Min distance between any exemplar pair |
| `p10_cross_dist` | 10th percentile of all cross-cluster distances |
| `p50_cross_dist` | 50th percentile (median) of all cross-cluster distances |
| `support_fraction` | Fraction of cross-cluster pairs below threshold |
| `diameter_ratio` | max(diameter_1, diameter_2) / min(diameter_1, diameter_2) |
| `cluster_size_min` | min(size_1, size_2) |
| `cluster_size_ratio` | max(size_1, size_2) / min(size_1, size_2) |
| `T_A` | Threshold for cluster 1 |
| `T_B` | Threshold for cluster 2 |
| `T_local` | MAX(T_A, T_B) |
| `T_global` | Global median threshold across all clusters |

**Candidate Filtering**: Only export pairs where `min_exemplar_dist < CANDIDATE_THRESHOLD` (default 0.45) to reduce file size.

### 4.4 corrected_labels.csv

Manual corrections provided by user.

| Column | Type | Description |
|--------|------|-------------|
| `cluster_id` | int | Cluster ID from Stage 1 |
| `corrected_identity` | str | User-assigned identity (e.g., "person_A", "person_B", "noise") |

All faces in a cluster share the same corrected_identity.

### 4.5 merge_training_data.csv

Generated from `candidate_pairs.csv` + `corrected_labels.csv`.

**Columns**: All features from candidate_pairs.csv + `label` (int, 0 or 1)

**Label Logic**:
```python
if corrected_identity(cluster_1) == corrected_identity(cluster_2):
    label = 1  # Should merge
else:
    label = 0  # Should not merge
```

---

## 5. Component Details

### 5.1 Export Script (`scripts/export_clustering_data.py`)

**CLI Interface**:
```bash
python scripts/export_clustering_data.py \
    --embeddings results/face_clustering_benchmark/embeddings_*.npy \
    --output results/face_clustering_training/dataset1 \
    --k 5 \
    --distance-threshold 0.35 \
    --candidate-threshold 0.45
```

**Algorithm**:
1. Load embeddings + metadata (reuse logic from notebook)
2. Quality gating (pose + blur filtering)
3. Build distance matrix
4. Build mutual kNN graph (k neighbors, distance threshold)
5. Connected components clustering
6. D10 exemplar selection
7. **Compute cluster statistics** (diameter, T_A per cluster, T_global)
8. **Generate candidate pairs** (all pairs with min_exemplar_dist < candidate_threshold)
9. **Compute pair features** (p10/p50 cross distances, support fraction, etc.)
10. Export 3 CSVs

**Key Functions**:
- `load_embeddings_and_metadata(path)` - Load from .npy + .json
- `run_clustering_pipeline(embeddings, config)` - Run Stage 1 clustering
- `compute_cluster_stats(cluster_result, distance_matrix)` - Compute diameter, T_A
- `generate_candidate_pairs(cluster_result, distance_matrix, threshold)` - Filter pairs
- `compute_pair_features(cluster_a, cluster_b, distance_matrix)` - All 12 features
- `export_csvs(faces, clusters, pairs, output_dir)` - Write CSVs

### 5.2 Export Notebook (`notebooks/export_clustering_data.ipynb`)

**Structure** (similar to debug_knn_graph_clustering):
1. Config cell (editable parameters)
2. Load data
3. Quality gating (with visualizations)
4. Build graph (show kNN table, edge list)
5. Clustering (show cluster grid)
6. Exemplar selection (show d10 histogram)
7. **Cluster statistics** (show T_A distribution, diameter histogram)
8. **Candidate pair generation** (show distance matrix between clusters, feature distributions)
9. Export CSVs
10. Summary (counts, file paths)

**Visualization Enhancements**:
- Cluster-to-cluster distance heatmap
- Feature distribution plots (p10, p50, support_fraction)
- Candidate pair scatter plots (e.g., min_exemplar_dist vs support_fraction)

### 5.3 Labeling App (`app/face_clustering_labeling.py`)

**dolit UI**:

```
┌─────────────────────────────────────────────┐
│ Face Clustering Labeling Interface          │
├─────────────────────────────────────────────┤
│ [Load CSV Directory] [📁 Browse]            │
│ Dataset: results/face_clustering_training/  │
│         dataset1                            │
│                                             │
│ ┌─────────────────────────────────────────┐ │
│ │ Cluster 0 (Size: 39)                    │ │
│ │ Corrected Identity: [person_A ▼]        │ │
│ │ ┌───┬───┬───┬───┬───┐                  │ │
│ │ │img│img│img│img│img│  (face grid)     │ │
│ │ └───┴───┴───┴───┴───┘                  │ │
│ └─────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────┐ │
│ │ Cluster 1 (Size: 44)                    │ │
│ │ Corrected Identity: [person_B ▼]        │ │
│ │ ┌───┬───┬───┬───┬───┐                  │ │
│ │ │img│img│img│img│img│                  │ │
│ │ └───┴───┴───┴───┴───┘                  │ │
│ └─────────────────────────────────────────┘ │
│                                             │
│ [💾 Save Corrected Labels]                  │
│ [📊 Show Statistics] [🔄 Reset All]         │
└─────────────────────────────────────────────┘
```

**Features**:
- Load faces.csv, clusters.csv from directory
- Display face images using `image_path` + `bbox` (crop from original)
- Dropdown per cluster to assign identity (auto-suggests based on cluster ID)
- Save/load corrected_labels.csv
- Statistics: how many clusters per identity, label distribution
- Export summary: "10 clusters → 3 identities (person_A: 4 clusters, person_B: 3 clusters, noise: 3 clusters)"

**Implementation Notes**:
- Use `st.columns()` for face grid layout
- Cache loaded images with `@st.cache_data`
- Auto-save on change (write to temp file, confirm with button)

### 5.4 Training Script (`scripts/train_merge_classifier.py`)

**CLI Interface**:
```bash
python scripts/train_merge_classifier.py \
    --candidate-pairs results/face_clustering_training/dataset1/candidate_pairs.csv \
    --corrected-labels results/face_clustering_training/dataset1/corrected_labels.csv \
    --output results/face_clustering_training/dataset1/model \
    --test-split 0.2
```

**Algorithm**:
1. Load `candidate_pairs.csv`
2. Load `corrected_labels.csv` (cluster_id → identity mapping)
3. **Generate training labels**:
   - For each row in candidate_pairs:
     - Look up identity_1, identity_2 from corrected_labels
     - label = 1 if identity_1 == identity_2, else 0
4. Save `merge_training_data.csv` (features + label)
5. **Train-test split** (stratified by label to handle imbalance)
6. **Feature engineering** (optional):
   - Normalize features (StandardScaler)
   - Interaction terms (e.g., `min_exemplar_dist * support_fraction`)
7. **Train logistic regression**:
   - Class weights to handle imbalance (more "no merge" than "merge" typically)
   - L2 regularization (tune C parameter with cross-validation)
8. **Evaluation**:
   - Confusion matrix
   - Precision, recall, F1 for both classes
   - ROC curve, AUC
   - Feature importance (coefficients)
   - Classification report
9. **Save artifacts**:
   - `merge_classifier.pkl` (trained model + scaler)
   - `training_report.json` (all metrics)
   - `feature_importance.png` (bar chart)
   - `confusion_matrix.png`
   - `roc_curve.png`

**Key Functions**:
- `generate_training_labels(pairs_df, labels_df)` - Create merge_training_data.csv
- `prepare_features(df)` - Standardize, create interaction terms
- `train_model(X, y, test_split)` - Train with cross-validation
- `evaluate_model(model, X_test, y_test)` - Compute all metrics
- `plot_diagnostics(model, X_test, y_test, feature_names)` - Generate plots
- `save_artifacts(model, scaler, metrics, output_dir)` - Pickle + JSON

**Hyperparameter Tuning**:
- Use `GridSearchCV` or `RandomizedSearchCV` to tune:
  - `C` (regularization strength)
  - `class_weight` (auto or custom)
  - `penalty` (l2 or l1 with saga solver)

### 5.5 ML Merger (`face_cluster/ml_merge.py`)

**Class**: `MLMerger` (similar API to `ConservativeMerger`)

```python
class MLMerger:
    def __init__(self, config: PipelineConfig, model_path: str):
        """Load trained logistic regression model."""
        self.config = config
        self.model, self.scaler = self._load_model(model_path)
        self.merge_threshold = config.ml_merge_threshold  # Default 0.5

    def merge_clusters(
        self,
        cluster_result: ClusterResult,
        graph_result: GraphResult
    ) -> ClusterResult:
        """Iteratively merge clusters using ML predictions."""
        # Similar structure to ConservativeMerger.merge_clusters()
        # 1. Compute cluster stats (exemplars, diameter, T_A)
        # 2. Generate all candidate pairs (min_exemplar_dist < threshold)
        # 3. Compute features for each pair
        # 4. Predict merge probability with model
        # 5. Sort by probability (descending)
        # 6. Iteratively merge pairs with p > merge_threshold
        # 7. Re-compute stats after each merge
        # 8. Return updated ClusterResult
        pass

    def _compute_pair_features(
        self,
        cluster_a: list[int],
        cluster_b: list[int],
        distance_matrix: np.ndarray
    ) -> np.ndarray:
        """Compute same 12 features as training data."""
        # min_exemplar_dist, p10_cross_dist, p50_cross_dist, ...
        pass
```

**Integration**:
- Add `ml_merge_enabled` and `ml_merge_model_path` to `PipelineConfig`
- In notebook/script, replace `ConservativeMerger` with `MLMerger` when `ml_merge_enabled=True`
- Keep both classes for comparison

---

## 6. Edge Cases & Constraints

### 6.1 Edge Cases

1. **Empty clusters after quality gating**
   - Handle: Skip export, log warning

2. **No candidate pairs (all clusters too far)**
   - Handle: Export empty candidate_pairs.csv, labeling app shows "No pairs to evaluate"

3. **Single-face clusters**
   - Handle: Include in faces.csv and clusters.csv, but may have no exemplars (use single face as exemplar)

4. **Class imbalance in training data**
   - Handle: Use `class_weight='balanced'` in logistic regression, report metrics for both classes

5. **Missing corrected labels for some clusters**
   - Handle: Only generate training data for clusters with labels, log warning for unlabeled clusters

6. **Model predicts merge for already-merged clusters**
   - Handle: Track merged clusters, skip pairs where both IDs are in same merged cluster

### 6.2 Constraints

1. **Memory**: Distance matrix is O(n²) - limit to ~10,000 faces per dataset
2. **Candidate pairs**: Filter aggressively (threshold < 0.45) to avoid quadratic explosion
3. **Model deployment**: Keep model file size small (< 1 MB) for fast loading
4. **Feature computation**: Must match training exactly (same percentile calculation, same threshold definitions)

---

## 7. Risks & Trade-offs

### 7.1 Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Overfitting to single dataset** | Model doesn't generalize to new datasets | Train on multiple diverse datasets, use cross-validation |
| **Feature mismatch between training and inference** | Poor predictions | Unit tests to verify feature computation, assert same column names/order |
| **Imbalanced training data** | Model always predicts "no merge" | Use class weights, monitor precision/recall for both classes |
| **Manual labeling errors** | Model learns wrong patterns | Double-check labels, start with small high-confidence dataset |
| **Iterative merge order dependency** | Different results based on merge order | Sort by probability, merge highest confidence first |

### 7.2 Trade-offs

| Decision | Pros | Cons |
|----------|------|------|
| **Logistic regression vs tree-based models** | Interpretable coefficients, fast training | May not capture complex interactions (but simpler is better for start) |
| **Filter candidate pairs vs all pairs** | Faster, smaller files | May miss distant clusters that should merge (acceptable - very distant pairs unlikely to merge) |
| **Keep heuristic merger vs replace** | Easy comparison, fallback option | More code to maintain |
| **Standalone scripts vs integrated pipeline** | Easier to iterate, debug | Less integrated UX (acceptable for ML workflow) |

---

## 8. Testing Strategy

### 8.1 Unit Tests

| Test | File | Description |
|------|------|-------------|
| `test_compute_pair_features()` | `tests/face_cluster/test_ml_merge.py` | Verify 12 features computed correctly for sample cluster pair |
| `test_generate_training_labels()` | `tests/scripts/test_train_merge_classifier.py` | Check label=1 for same identity, label=0 for different |
| `test_ml_merger_inference()` | `tests/face_cluster/test_ml_merge.py` | Verify ML predictions match feature computation |
| `test_export_csvs()` | `tests/scripts/test_export_clustering_data.py` | Check CSV columns, data types, row counts |

### 8.2 Integration Tests

1. **End-to-end pipeline**:
   - Run export script on test dataset
   - Manually create corrected_labels.csv
   - Run training script
   - Load model and predict on same data
   - Verify merge decisions match expected

2. **Labeling app**:
   - Load test CSVs
   - Assign identities
   - Save and reload corrected_labels.csv
   - Verify persistence

### 8.3 Validation

1. **Baseline comparison**:
   - Run heuristic merger on dataset
   - Run ML merger on same dataset
   - Compare cluster counts, precision/recall (requires ground truth)

2. **Feature distribution check**:
   - Plot feature distributions from candidate_pairs.csv
   - Check for outliers, NaNs, unreasonable values

3. **Model sanity checks**:
   - Verify high probability for obviously-similar clusters (same exemplar faces)
   - Verify low probability for obviously-different clusters (large distance)

---

## 9. Implementation Plan

### Phase 1: Export Pipeline (2-3 days)

**Tasks**:
1. Create `scripts/export_clustering_data.py`:
   - CLI argument parsing (embeddings path, output dir, config params)
   - Reuse `face_cluster` library (knn_graph, clustering, exemplars)
   - Implement `compute_cluster_stats()` - diameter, T_A
   - Implement `generate_candidate_pairs()` - filter by threshold
   - Implement `compute_pair_features()` - all 12 features
   - Implement `export_csvs()` - write faces, clusters, pairs
   - Add logging and progress bars
2. Create `notebooks/export_clustering_data.ipynb`:
   - Config cell with editable params
   - Copy pipeline from debug_knn_graph_clustering
   - Add visualization for cluster stats
   - Add visualization for candidate pairs (heatmap, scatter plots)
   - Add export cell
3. Test on sample dataset:
   - Run on existing benchmark embeddings
   - Verify CSV output (column names, data types, row counts)
   - Inspect features for sanity (no NaNs, reasonable ranges)

**Deliverables**:
- ✅ `scripts/export_clustering_data.py`
- ✅ `notebooks/export_clustering_data.ipynb`
- ✅ Sample output in `results/face_clustering_training/test_dataset/`

### Phase 2: Labeling Interface (1-2 days)

**Tasks**:
1. Create `app/face_clustering_labeling.py`:
   - Streamlit page layout (sidebar for config, main area for clusters)
   - Load CSVs from directory
   - Display cluster grids (load images from image_path + bbox)
   - Dropdown per cluster for corrected_identity
   - Save/load corrected_labels.csv
   - Statistics display (clusters per identity, label distribution)
2. Test labeling workflow:
   - Load test dataset
   - Assign identities to 5-10 clusters
   - Save corrected_labels.csv
   - Reload and verify persistence
   - Check for UI bugs (slow loading, image errors)

**Deliverables**:
- ✅ `app/face_clustering_labeling.py`
- ✅ Sample `corrected_labels.csv` for test dataset

### Phase 3: Training Pipeline (2-3 days)

**Tasks**:
1. Create `scripts/train_merge_classifier.py`:
   - Implement `generate_training_labels()` - join pairs + labels → binary label
   - Implement `prepare_features()` - standardize, handle NaNs
   - Implement `train_model()` - logistic regression with cross-validation
   - Implement `evaluate_model()` - confusion matrix, precision/recall, ROC
   - Implement `plot_diagnostics()` - feature importance, confusion matrix, ROC curve
   - Implement `save_artifacts()` - pickle model, save metrics JSON
   - CLI argument parsing
2. Train on test dataset:
   - Use sample corrected_labels.csv
   - Run training script
   - Inspect metrics (AUC, precision/recall)
   - Inspect feature importance (which features matter most?)
   - Check for overfitting (train vs test metrics)
3. Hyperparameter tuning:
   - Try different C values (0.1, 1.0, 10.0)
   - Try different class weights
   - Select best model based on F1 score

**Deliverables**:
- ✅ `scripts/train_merge_classifier.py`
- ✅ `merge_training_data.csv` (features + labels)
- ✅ `merge_classifier.pkl` (trained model)
- ✅ `training_report.json` (metrics)
- ✅ Diagnostic plots (feature importance, confusion matrix, ROC)

### Phase 4: Deployment (2-3 days)

**Tasks**:
1. Create `face_cluster/ml_merge.py`:
   - Implement `MLMerger` class
   - Implement `_load_model()` - load pickle
   - Implement `_compute_pair_features()` - same as training
   - Implement `merge_clusters()` - iterative merge with ML predictions
   - Add unit tests for feature computation
2. Integrate into pipeline:
   - Add `ml_merge_enabled`, `ml_merge_model_path`, `ml_merge_threshold` to `PipelineConfig`
   - Update notebook to support ML merger
3. Comparison test:
   - Run heuristic merger on test dataset → count clusters
   - Run ML merger on test dataset → count clusters
   - Compare results (which pairs were merged differently?)
   - Manual inspection: did ML fix any obvious heuristic mistakes?

**Deliverables**:
- ✅ `face_cluster/ml_merge.py`
- ✅ Unit tests in `tests/face_cluster/test_ml_merge.py`
- ✅ Comparison report (heuristic vs ML metrics)

### Phase 5: Documentation & Polish (1 day)

**Tasks**:
1. Update `README.md`:
   - Add section on ML-based clustering
   - Link to this plan document
   - CLI usage examples
2. Create `docs/ML_CLUSTER_MERGING_GUIDE.md`:
   - Step-by-step workflow (export → label → train → deploy)
   - Best practices (how much data to label, how to check model quality)
   - Troubleshooting (class imbalance, feature errors, model not improving)
3. Add docstrings to all functions
4. Run full test suite
5. Mark feature request as Done in FEATURE_REQUESTS.md

**Deliverables**:
- ✅ Updated README.md
- ✅ `docs/ML_CLUSTER_MERGING_GUIDE.md`
- ✅ All tests passing

---

## 10. Success Criteria

1. ✅ Can run export script on multiple datasets and produce valid CSVs
2. ✅ Can load CSVs in labeling app, assign identities, and save corrected labels
3. ✅ Can generate training data from corrected labels with correct binary labels
4. ✅ Can train logistic regression model with AUC > 0.80 on test set
5. ✅ Feature importance shows interpretable patterns (e.g., min_exemplar_dist has high weight)
6. ✅ ML merger can run inference and merge clusters
7. ✅ ML merger produces different (hopefully better) results than heuristic merger
8. ✅ All unit tests pass
9. ✅ Documentation complete and clear

---

## 11. Open Questions

1. **Should we standardize features (StandardScaler) or use raw values?**
   - Recommendation: Standardize for logistic regression (helps with convergence and coefficient interpretation)

2. **Should we add interaction terms (e.g., `min_exemplar_dist * support_fraction`)?**
   - Recommendation: Start without, add if model underperforms (simpler is better)

3. **What if two clusters have corrected_identity = "noise" - should they merge?**
   - Recommendation: No, label=0 for noise-noise pairs (noise is "unknown identity", not same person)

4. **How to handle new clusters at inference time that weren't in training?**
   - Recommendation: Fine - features are cluster-level (size, diameter, distances), not cluster-specific. Model should generalize.

5. **Should we re-train model as we get more labeled data?**
   - Recommendation: Yes, incremental training workflow:
     - Export from multiple datasets
     - Label batches over time
     - Concatenate all merge_training_data.csv files
     - Re-train on combined data

6. **What's the minimum labeled data needed?**
   - Recommendation: Start with 1 dataset (~50-200 cluster pairs, ~10-50 unique identities). Iterate from there.

---

## 12. Future Enhancements

**After initial implementation**:
1. **Active learning**: Suggest most uncertain cluster pairs for manual review
2. **Tree-based models**: Try Random Forest, XGBoost (may capture non-linear patterns)
3. **Deep learning**: Train Siamese network on cluster exemplars (overkill for start)
4. **Online learning**: Update model as user corrects errors in production
5. **Multi-dataset training**: Train on diverse datasets, evaluate generalization
6. **Automated evaluation**: Track precision/recall against ground truth labels over time

---

## Approval Required

**Before proceeding, please confirm**:
1. ✅ Overall approach (export → label → train → deploy)
2. ✅ Data schema (CSV columns, feature list)
3. ✅ Component architecture (scripts, notebook, Streamlit app, MLMerger class)
4. ✅ Integration strategy (keep both heuristic and ML mergers)
5. ❓ Any missing requirements or edge cases?
6. ❓ Any concerns about feasibility, complexity, or timeline?

**Please provide feedback on**:
- Are there additional features you want in candidate_pairs.csv?
- Should we use a different model (e.g., Random Forest instead of logistic regression)?
- Any specific evaluation metrics you care about (e.g., false positive rate)?
- Any other datasets you want to test on initially?

---

**Next Steps After Approval**:
1. Create task list from Phase 1-5
2. Begin Phase 1 implementation (export pipeline)
3. Regular check-ins after each phase for feedback
