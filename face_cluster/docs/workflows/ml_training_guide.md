# ML-Based Cluster Merging - Getting Started Guide

**Complete workflow for training a logistic regression model to replace heuristic cluster merging.**

---

## Overview

Instead of hand-tuning merge thresholds, you can train a machine learning model to decide which clusters should merge based on labeled examples.

**Benefits**:
- Data-driven decisions (learns from your corrections)
- Iterative improvement (add more labeled data over time)
- Interpretable (see which features matter most)
- Extensible (easy to try Random Forest, XGBoost, etc.)

**Workflow**: Prepare Embeddings → Export Features → Label Clusters → Train Model

---

## Prerequisites

**Required**:
- Python 3.10+
- Installed dependencies: `pip install -r requirements.txt`
- Face embeddings (either pre-computed or images to process)

**Optional**:
- InsightFace for face detection: `pip install insightface onnxruntime`
- sixdrepnet for pose estimation: `pip install sixdrepnet`

---

## Quick Start (5 Steps)

### 1. Prepare Embeddings from Images

If you have a folder of images with faces:

```bash
python scripts/prepare_embeddings.py \
    --images D:/my_photos \
    --output results/my_dataset
```

**What this does**:
- Detects faces using InsightFace
- Extracts 512-dim embeddings
- Computes pose (yaw/pitch/roll) and blur scores
- Saves: `embeddings_<timestamp>.npy`, `metadata.json`, `face_crops/`

**Output**:
```
results/my_dataset/
├── embeddings_2026-02-27_14-30-00.npy
├── benchmark_2026-02-27_14-30-00.json  (metadata)
├── face_crops/
│   ├── face_0000_aligned.jpg
│   ├── face_0001_aligned.jpg
│   └── ...
└── logs/
```

**Time**: ~1-2 seconds per image (CPU), faster on GPU

---

### 2. Export Clustering Data

Run clustering and export features for ML training:

```bash
python scripts/export_clustering_data.py \
    --embeddings results/my_dataset/embeddings_*.npy \
    --k 5 \
    --distance-threshold 0.35 \
    --feature-version 2 \
    --output results/training/run1
```

**What this does**:
- Quality gating (filters blurry/non-frontal faces)
- Mutual kNN graph → connected components clustering
- D10 exemplar selection
- Computes 16 V2 features for each candidate cluster pair
- Exports 3 CSVs

**Output**:
```
results/training/run1/
├── faces.csv              (254 rows: metadata per face)
├── clusters.csv           (39 rows: stats per cluster)
├── candidate_pairs.csv    (39 rows x 16 cols: features for ML)
├── export_summary.json
└── logs/
```

**Key Parameters**:
- `--k`: Number of nearest neighbors (default: 5)
- `--distance-threshold`: Max distance for kNN edges (default: 0.35)
- `--feature-version`: 1 (11 features) or 2 (16 features with pose, default: 2)
- `--blur-min`: Min blur score (default: 50)

**Time**: ~1-5 seconds (fast - no face detection)

---

### 3. Label Clusters

Manually assign corrected identities to clusters:

```bash
streamlit run app/face_clustering_labeling.py
```

**In the web UI**:
1. Enter path: `results/training/run1`
2. Click "Load Data"
3. Review face images for each cluster
4. Assign identities:
   - Same person across clusters → same identity (e.g., "person_A")
   - Different people → different identities (e.g., "person_A", "person_B")
   - Bad clusters → "noise"
5. Click "💾 Save Labels"

**Example Labeling**:
```
Cluster 0 (39 faces) → person_A
Cluster 1 (44 faces) → person_B
Cluster 2 (3 faces)  → person_A  (same as cluster 0!)
Cluster 3 (6 faces)  → person_C
Cluster 4 (27 faces) → person_B  (same as cluster 1!)
...
```

**Output**:
```
results/training/run1/
├── corrected_labels.csv       (cluster_id → corrected_identity)
└── labeling_metadata.json
```

**Time**: ~5-10 minutes for 50 clusters (depends on how well you know your dataset)

---

### 4. Train Model

Train logistic regression on labeled data:

```bash
python scripts/train_merge_classifier.py \
    --candidate-pairs results/training/run1/candidate_pairs.csv \
    --corrected-labels results/training/run1/corrected_labels.csv \
    --output results/training/run1/model \
    --test-split 0.2
```

**What this does**:
- Generates binary labels: 1 if same identity (should merge), 0 if different
- Standardizes features (StandardScaler)
- Hyperparameter tuning with GridSearchCV (5-fold CV)
- Trains logistic regression
- Evaluates on test set
- Saves model + metrics + plots

**Output**:
```
results/training/run1/model/
├── merge_classifier.pkl         (model + scaler + metadata)
├── training_metrics.json        (precision/recall/F1/AUC)
├── merge_training_data.csv      (features + labels)
├── feature_names.txt
├── plots/
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   └── feature_importance.png
└── logs/
```

**Example Output**:
```
Test Performance:
  Accuracy: 0.923
  Precision (merge): 0.857
  Recall (merge): 0.750
  F1 (merge): 0.800
  ROC AUC: 0.942

Top 5 Most Important Features:
  1. min_exemplar_dist: -2.4567  (negative = lower dist → more likely to merge)
  2. p50_cross_dist: -1.8932
  3. support_fraction: 1.2345    (positive = more support → more likely to merge)
  4. pose_diff: -0.9876          (negative = similar pose → more likely to merge)
  5. T_local: 0.7654
```

**Time**: ~5-30 seconds (depends on dataset size)

---

### 5. Validate Export (Optional)

Check that CSVs are valid:

```bash
python scripts/validate_export.py results/training/run1
```

**What this checks**:
- CSV schema (column names, data types)
- No NaN values
- Feature ranges are valid
- Summary statistics

---

## Complete Example

Here's a full end-to-end example:

```bash
# Dataset: Family photos from Budapest trip
DATASET_NAME="budapest_2025"
IMAGES_DIR="D:/Budapest2025_Google"

# Step 1: Prepare embeddings (run once)
python scripts/prepare_embeddings.py \
    --images "$IMAGES_DIR" \
    --output "results/$DATASET_NAME" \
    --detector buffalo_l

# Step 2: Export clustering data
python scripts/export_clustering_data.py \
    --embeddings "results/$DATASET_NAME/embeddings_*.npy" \
    --k 5 \
    --distance-threshold 0.35 \
    --feature-version 2 \
    --output "results/training/${DATASET_NAME}_run1"

# Step 3: Label clusters (opens browser)
streamlit run app/face_clustering_labeling.py
# → Load: results/training/budapest_2025_run1
# → Assign identities, save

# Step 4: Train model
python scripts/train_merge_classifier.py \
    --candidate-pairs "results/training/${DATASET_NAME}_run1/candidate_pairs.csv" \
    --corrected-labels "results/training/${DATASET_NAME}_run1/corrected_labels.csv" \
    --output "results/training/${DATASET_NAME}_run1/model"

# Step 5: Review results
cat "results/training/${DATASET_NAME}_run1/model/training_metrics.json"
open "results/training/${DATASET_NAME}_run1/model/plots/feature_importance.png"
```

---

## Understanding the Features

**V2 Features (16 total)**:

### Distance Features (4)
- `min_exemplar_dist`: Min distance between cluster exemplars (lower → more likely to merge)
- `p10_cross_dist`: 10th percentile of cross-cluster distances
- `p50_cross_dist`: Median cross-cluster distance
- `support_fraction`: Fraction of pairs below threshold (higher → more likely to merge)

### Size Features (3)
- `diameter_ratio`: max(dia_A, dia_B) / min(dia_A, dia_B) (higher → less likely to merge)
- `cluster_size_min`: Smaller cluster size
- `cluster_size_ratio`: Size ratio between clusters

### Threshold Features (4)
- `T_A`, `T_B`: Per-cluster adaptive thresholds (P90 of exemplar distances)
- `T_local`: max(T_A, T_B)
- `T_global`: Median threshold across all clusters

### Pose Features (3) - New in V2
- `frontal_frac_A`, `frontal_frac_B`: Fraction of frontal faces per cluster
- `pose_diff`: Euclidean distance of mean (yaw, pitch) between clusters

### Interaction Features (2) - New in V2
- `min_exemplar_dist_x_pose`: Distance weighted by pose difference
- `p50_cross_dist_x_pose`: Median distance weighted by pose

**Why V2 Features Help**:
- Same person at different angles (e.g., profile vs frontal) should still merge
- Pose features tell the model: "clusters have different viewing angles, but low distance → likely same person"

---

## Tuning Clustering Parameters

You can re-run Step 2 with different parameters **without re-detecting faces**:

```bash
# Try tighter clustering (higher k, lower threshold)
python scripts/export_clustering_data.py \
    --embeddings results/my_dataset/embeddings_*.npy \
    --k 7 \
    --distance-threshold 0.30 \
    --output results/training/run2_tight

# Try looser clustering
python scripts/export_clustering_data.py \
    --embeddings results/my_dataset/embeddings_*.npy \
    --k 3 \
    --distance-threshold 0.40 \
    --output results/training/run3_loose
```

Then label and train on each variant to see which clustering works best for your dataset.

---

## Iterative Improvement

As you get more data:

1. **Export from new dataset**:
   ```bash
   python scripts/prepare_embeddings.py --images D:/new_photos --output results/dataset2
   python scripts/export_clustering_data.py --embeddings results/dataset2/embeddings_*.npy --output results/training/dataset2_run1
   ```

2. **Label new clusters** (in labeling app)

3. **Combine training data**:
   ```bash
   # Concatenate candidate pairs and labels from multiple datasets
   cat results/training/*/candidate_pairs.csv > combined_pairs.csv
   cat results/training/*/corrected_labels.csv > combined_labels.csv
   ```

4. **Re-train on combined data**:
   ```bash
   python scripts/train_merge_classifier.py \
       --candidate-pairs combined_pairs.csv \
       --corrected-labels combined_labels.csv \
       --output results/training/combined_model
   ```

Model gets better with more diverse labeled data!

---

## Troubleshooting

### "No faces detected"
- Check images are valid (JPG/PNG)
- Try different InsightFace model: `--detector buffalo_s` (smaller, faster)
- Lower detection confidence: `--min-confidence 0.3`

### "Embeddings file not found"
- Ensure you ran `prepare_embeddings.py` first
- Check the path matches the output from Step 1
- Use wildcard: `--embeddings results/my_dataset/embeddings_*.npy`

### "No candidate pairs (empty candidate_pairs.csv)"
- Clustering produced no pairs below threshold
- Try higher `--candidate-threshold` (default 0.45, try 0.55)
- Or adjust clustering params (lower `--distance-threshold`)

### "Only one class in training data"
- All cluster pairs have same label (all merge or all don't merge)
- Need both positive and negative examples
- Re-label: ensure some clusters share identity (positive) and some don't (negative)

### "Poor test performance (F1 < 0.5)"
- Not enough training data (need 20+ pairs minimum, 50+ ideal)
- Inconsistent labeling (same person labeled differently)
- Try V1 features if pose data is unreliable: `--feature-version 1`

### "Face crops not showing in labeling app"
- Check `face_crops/` directory exists in export directory
- Ensure `prepare_embeddings.py` was used (not just raw embeddings)
- Try loading from parent directory (app auto-searches)

---

## Next Steps

### Deploy Model (Phase 4 - Optional)

To use the trained model in production:

1. Create `face_cluster/ml_merge.py` with `MLMerger` class
2. Load model and predict merge decisions
3. Integrate into clustering pipeline

(This is optional - the training pipeline is the core value)

### Try Different Models

The feature engineering is model-agnostic. Easy to try:

**Random Forest**:
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, class_weight='balanced')
```

**XGBoost**:
```python
import xgboost as xgb
model = xgb.XGBClassifier(scale_pos_weight=negative/positive)
```

**Neural Network**:
```python
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000)
```

Just replace the model in `train_merge_classifier.py`!

---

## FAQ

**Q: How much data do I need?**
A: Minimum 20-30 cluster pairs (both merge and non-merge examples). Ideal: 100+ pairs from diverse datasets.

**Q: Can I use pre-computed embeddings from benchmark results?**
A: Yes! If you already have `embeddings_*.npy` files, skip Step 1 and start at Step 2.

**Q: Do I need pose data?**
A: No. Use `--feature-version 1` to get 11 features without pose. V2 (16 features) is better if pose is available.

**Q: What if I don't have InsightFace?**
A: You can use any embedding extractor. Just save as `embeddings.npy` (N x 512 array) + metadata JSON in the same format.

**Q: Can I add custom features?**
A: Yes! Edit `face_cluster/features.py` and add to `POSE_FEATURES` or create new feature group. Features are modular.

**Q: How do I know if the model is good?**
A: Check test metrics:
- **ROC AUC > 0.85**: Good
- **F1 score > 0.70**: Decent
- **Precision/Recall balanced**: Not biased to one class
- **Feature importance**: Makes sense (distance features should be important)

**Q: What if I want to use the model for new data?**
A: Load the model and predict:
```python
import pickle
with open('merge_classifier.pkl', 'rb') as f:
    data = pickle.load(f)
model = data['model']
scaler = data['scaler']
feature_names = data['feature_names']

# New cluster pair features
X_new = [[0.35, 0.40, 0.45, 0.1, 1.2, 4, 1.5, ...]]  # 16 features
X_scaled = scaler.transform(X_new)
prediction = model.predict(X_scaled)[0]  # 1 = merge, 0 = don't merge
probability = model.predict_proba(X_scaled)[0, 1]  # confidence
```

---

## Summary

**Your ML Workflow**:
1. `prepare_embeddings.py` → embeddings.npy
2. `export_clustering_data.py` → candidate_pairs.csv
3. `app/face_clustering_labeling.py` → corrected_labels.csv
4. `train_merge_classifier.py` → merge_classifier.pkl

**Benefits over heuristics**:
- Learns from your data (not hand-tuned thresholds)
- Improves with more labeled examples
- Interpretable (see feature importance)
- Extensible (easy to try different models)

**Time Investment**:
- First run: ~30 min (setup + labeling)
- Additional datasets: ~10 min (just labeling)
- Model training: ~30 seconds

Start with one small dataset (~50 clusters) to validate the workflow, then scale up!

---

**Questions?** Check `docs/LEARNINGS.md` for tips or open an issue.
