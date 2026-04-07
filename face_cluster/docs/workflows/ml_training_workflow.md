# ML-Based Cluster Merging: Complete Workflow Guide

**Date**: 2026-03-01
**Status**: Active
**Related**: `docs/PLAN_ML_CLUSTER_MERGING.md`

This guide provides step-by-step instructions for the ML-based cluster merging pipeline, including troubleshooting for common issues encountered during implementation.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Phase 1: Data Preparation](#phase-1-data-preparation)
4. [Phase 2: Manual Labeling](#phase-2-manual-labeling)
5. [Phase 3: Model Training](#phase-3-model-training-not-yet-implemented)
6. [Troubleshooting](#troubleshooting)
7. [Validation & Quality Checks](#validation--quality-checks)

---

## Overview

The ML cluster merging pipeline replaces heuristic-based cluster merging with a machine learning model trained on manually corrected clustering results.

**Workflow:**
```
Images → Embeddings → Clustering → Labeling → Training → ML Merger
```

---

## Prerequisites

### Required Software
- Python 3.10+
- InsightFace models installed (`buffalo_l` recommended)
- All dependencies from `requirements.txt`

### Recommended: Clear Old Data
```bash
# Remove old/corrupted embeddings to avoid confusion
rm results/my_dataset/embeddings_*.npy
rm results/my_dataset/benchmark_*.json
```

---

## Phase 1: Data Preparation

### Step 1.1: Generate Face Embeddings

**Purpose**: Detect faces, extract embeddings, compute quality scores, save aligned crops.

**Command:**
```bash
python scripts/prepare_embeddings.py \
  --images /path/to/your/photos \
  --output results/my_dataset \
  --detector buffalo_l \
  --device cpu
```

**What this creates:**
- `embeddings_<timestamp>.npy` - 512-dim face embeddings (numpy array)
- `benchmark_<timestamp>.json` - Metadata with:
  - `image_path` - Full path to source image
  - `face_index` - Index of face within image (0-based)
  - `bbox` - Bounding box coordinates
  - `pose` - Yaw/pitch/roll angles
  - `blur_score` - Laplacian variance
- `face_crops/` - Aligned 112x112 face images
  - `face_0000_aligned.jpg`, `face_0001_aligned.jpg`, ...

**Output example:**
```
Images processed: 471
Faces detected: 727
  - embeddings_2026-03-01_01-16-27.npy
  - benchmark_2026-03-01_01-16-27.json
  - face_crops/ (727 images)
```

**⚠️ Critical Bug Fix:**
If you get `TypeError: Path(None)` errors, the `InsightFaceEmbedder` had a bug where it didn't set `image_path` and `face_index`. This was fixed in `face_cluster/embedding.py` by adding:
```python
image_path=str(image_path),
face_index=face_idx
```
to the `FaceRecord` creation. Regenerate embeddings after this fix.

---

### Step 1.2: Run Clustering Export

**Purpose**: Perform initial clustering (mutual kNN + connected components + merge) and export cluster data for labeling.

**Command:**
```bash
python scripts/export_clustering_data.py \
  --embeddings results/my_dataset/embeddings_2026-03-01_01-16-27.npy \
  --output results/my_dataset/clustering_export
```

**Default parameters (can override with CLI args):**
- `--k 5` - Mutual kNN neighbors
- `--distance-threshold 0.35` - Max cosine distance for edges
- `--blur-min 50.0` - Min blur score (quality gating)
- `--yaw-max 30.0`, `--pitch-max 25.0`, `--roll-max 25.0` - Pose filtering
- `--merge-enabled` - Conservative merge enabled by default
- `--candidate-threshold 0.45` - Max distance for candidate pairs
- `--feature-version 2` - Use V2 features (17 features including pose)

**What this creates:**
- `faces.csv` - One row per face with metadata
- `clusters.csv` - One row per cluster with statistics
- `candidate_pairs.csv` - Cluster pair features for ML training
- `export_summary.json` - Metadata including:
  - `embeddings_dir` - Path to embeddings (for face_crops lookup)
  - `n_faces`, `n_core`, `n_holdout`
  - `n_clusters`, `n_noise`

**Output example:**
```
Quality gating: core=200, holdout=74
Clustering: 26 clusters, 66 noise
Merging: 26 → 11 clusters (merged 15)
Exported:
  - faces.csv (274 faces)
  - clusters.csv (11 clusters)
  - candidate_pairs.csv (1 pair)
```

---

### Step 1.3: Validate Clustering Results

**Purpose**: Compare notebook clustering vs script clustering to verify consistency.

#### Option A: Use Existing Notebook

Run `notebooks/debug_knn_graph_clustering.ipynb`:

1. **Update config cell:**
   ```python
   RESULTS_DIR = Path("../results/my_dataset")
   USE_EXISTING_EMBEDDINGS = True
   ```

2. **Run all cells** - The notebook will:
   - Load embeddings
   - Run quality gating
   - Build kNN graph
   - Cluster with connected components
   - Select exemplars
   - Apply conservative merge
   - Export results to `notebook_clustering_results.csv`

3. **Check visualization** - Review cluster face grids to verify quality

#### Option B: Use Extracted Script

```bash
python scripts/export_from_notebook_logic.py \
  --embeddings results/my_dataset/embeddings_*.npy \
  --output results/my_dataset
```

**Compare Results:**
```bash
python scripts/compare_notebook_vs_export.py \
  --export-dir results/my_dataset/clustering_export
```

**Expected output:**
```
Mismatched assignments: 0 / 274 faces (0.0%)
[OK] All cluster assignments match perfectly!
```

**⚠️ Common Issue:**
If comparison shows mismatches, ensure:
1. Both notebook and script use the **same embeddings file** (not different timestamps)
2. Both use the **same parameters** (K, distance_threshold, merge settings)
3. Check `export_summary.json` for which embeddings were used

---

## Phase 2: Manual Labeling

### Step 2.1: Launch Labeling App

**Purpose**: Manually assign corrected identities to clusters for ML training.

**Command:**
```bash
streamlit run app/face_clustering_labeling.py
```

**Browser opens at:** `http://localhost:8501`

---

### Step 2.2: Load Clustering Data

1. **Enter export directory:**
   ```
   results/my_dataset/clustering_export
   ```

2. **Click "Load Data"**

3. **Verify face crops are found:**
   - App reads `export_summary.json` to find `embeddings_dir`
   - Looks for `<embeddings_dir>/face_crops/`
   - If not found, shows warning with checked paths

**How face crops are located:**
1. Reads `export_summary.json` → `embeddings_dir` field
2. Looks for `<embeddings_dir>/face_crops/`
3. Fallback: checks export directory itself, then parent directory

**⚠️ If face crops not found:**
- Check that `export_summary.json` has `embeddings_dir` field
- Verify `face_crops/` exists at that location
- Re-run `prepare_embeddings.py` if crops are missing

---

### Step 2.3: Assign Identities

**For each cluster:**

1. **Review face grid** - Shows up to 20 faces per cluster
2. **Assign identity via dropdown:**
   - `person_A`, `person_B`, ... for distinct people
   - Assign **same identity** to clusters that should merge
   - Use `noise` for clusters to discard
3. **Merge strategy:**
   - Cluster 0 and Cluster 5 both show "John" → assign both to `person_A`
   - Cluster 2 shows mix of people → assign to `noise`

**Example labeling:**
```
Cluster 0 (39 faces) → person_A
Cluster 1 (44 faces) → person_B
Cluster 2 (3 faces)  → noise
Cluster 3 (6 faces)  → person_A  (same person as cluster 0)
Cluster 4 (27 faces) → person_C
```

---

### Step 2.4: Save Labels

1. **Click "Save Corrected Labels"**
2. **Verify files created:**
   - `corrected_labels.csv`
   - `labeling_metadata.json`

**corrected_labels.csv format:**
```csv
cluster_id,corrected_identity
0,person_A
1,person_B
2,noise
3,person_A
4,person_C
```

**labeling_metadata.json:**
```json
{
  "timestamp": "2026-03-01T02:30:00",
  "n_clusters": 5,
  "n_identities": 4,
  "identity_counts": {
    "person_A": 2,
    "person_B": 1,
    "person_C": 1,
    "noise": 1
  }
}
```

---

## Phase 3: Model Training (Not Yet Implemented)

**Planned command:**
```bash
python scripts/train_merge_classifier.py \
  --candidate-pairs results/my_dataset/clustering_export/candidate_pairs.csv \
  --corrected-labels results/my_dataset/clustering_export/corrected_labels.csv \
  --output results/my_dataset/model
```

**Will create:**
- `merge_training_data.csv` - Features + binary labels (merge/no-merge)
- `merge_classifier.pkl` - Trained logistic regression model
- `training_report.json` - Metrics (AUC, precision, recall, F1)
- Diagnostic plots (feature importance, confusion matrix, ROC curve)

**Status:** Training script not yet implemented (see `PLAN_ML_CLUSTER_MERGING.md` Phase 3).

---

## Troubleshooting

### Issue 1: `TypeError: Path(None)` in Notebook

**Symptoms:**
```python
image_id = Path(meta['image_path']).name
TypeError: cannot instantiate Path from None
```

**Root Cause:**
`InsightFaceEmbedder.detect_and_embed()` didn't set `image_path` and `face_index` fields in FaceRecord objects.

**Fix Applied:**
Updated `face_cluster/embedding.py`:
```python
# Before (line 70)
for face in faces:

# After
for face_idx, face in enumerate(faces):

# Before (line 94-105)
face_record = FaceRecord(
    face_id=face_id_counter,
    ...
    is_core=False
)

# After
face_record = FaceRecord(
    face_id=face_id_counter,
    ...
    is_core=False,
    image_path=str(image_path),
    face_index=face_idx
)
```

**Resolution:**
Regenerate embeddings with fixed script:
```bash
python scripts/prepare_embeddings.py --images /path/to/photos --output results/my_dataset_fixed
```

---

### Issue 2: Clustering Results Don't Match Between Notebook and Script

**Symptoms:**
```
Mismatched assignments: 115 / 254 faces (45.3%)
```

**Root Cause:**
Comparing outputs from **different embeddings files**.

**Check:**
```bash
# What embeddings did notebook use?
grep "EMBEDDINGS_FILE" notebooks/debug_knn_graph_clustering.ipynb

# What embeddings did export script use?
cat results/my_dataset/clustering_export/export_summary.json | grep embeddings_source
```

**Fix:**
Ensure both use the **exact same** embeddings file:
```python
# In notebook config cell
RESULTS_DIR = Path("../results/my_dataset")
```

```bash
# In export script
python scripts/export_clustering_data.py \
  --embeddings results/my_dataset/embeddings_2026-03-01_01-16-27.npy
```

---

### Issue 3: Face Crops Not Found in Labeling App

**Symptoms:**
```
⚠️ Face crops directory not found. Checked:
- results/my_dataset/clustering_export/embeddings_dir/face_crops (from export_summary.json)
- results/my_dataset/clustering_export/face_crops
- results/my_dataset/face_crops
```

**Root Cause:**
Export script ran on embeddings from different directory.

**Fix:**
Check `export_summary.json`:
```json
{
  "embeddings_dir": "results/my_dataset",  // Should point to where face_crops/ exists
  ...
}
```

If incorrect:
1. Re-run export script with correct embeddings path
2. Or manually copy `face_crops/` to expected location

---

### Issue 4: Metadata File Missing or Empty

**Symptoms:**
```
Warning: metadata file not found: benchmark_2026-03-01_01-16-27.json
```

**Root Cause:**
`prepare_embeddings.py` failed to save metadata, or wrong filename pattern.

**Check:**
```bash
ls results/my_dataset/benchmark_*.json
ls results/my_dataset/embeddings_*.npy
```

**Fix:**
Re-run `prepare_embeddings.py` and check logs for errors:
```bash
tail -f results/my_dataset/logs/prepare_embeddings_*.log
```

---

## Validation & Quality Checks

### Check 1: Verify Embeddings Metadata

```bash
# Check metadata structure
python -c "
import json
with open('results/my_dataset/benchmark_*.json') as f:
    data = json.load(f)
    print('Total faces:', len(data['face_metadata']))
    print('First face:', data['face_metadata'][0])
    print('Has image_path:', data['face_metadata'][0].get('image_path') is not None)
    print('Has face_index:', data['face_metadata'][0].get('face_index') is not None)
"
```

**Expected output:**
```
Total faces: 727
First face: {'face_index': 0, 'image_path': '20240816_080028.jpg', ...}
Has image_path: True
Has face_index: True
```

---

### Check 2: Verify Clustering Output

```bash
# Check exported CSVs
wc -l results/my_dataset/clustering_export/*.csv

# Expected:
# 275 faces.csv         (274 faces + 1 header)
# 12 clusters.csv       (11 clusters + 1 header)
# 2 candidate_pairs.csv (1 pair + 1 header)
```

---

### Check 3: Verify Face Crops Exist

```bash
ls results/my_dataset/face_crops/ | wc -l
# Should match number of faces in embeddings

# Check naming pattern
ls results/my_dataset/face_crops/face_0000_aligned.jpg
ls results/my_dataset/face_crops/face_0001_aligned.jpg
```

---

### Check 4: Compare Notebook vs Script (100% Match Expected)

```bash
python scripts/compare_notebook_vs_export.py --export-dir results/my_dataset/clustering_export
```

**Expected:**
```
============================================================
CLUSTERING RESULTS COMPARISON
============================================================

Notebook: 274 faces, 12 unique clusters, 140 noise
Export:   274 faces, 12 unique clusters, 140 noise

Mismatched assignments: 0 / 274 faces (0.0%)

[OK] All cluster assignments match perfectly!
```

---

## File Structure Reference

After completing Phase 1 & 2, your directory should look like:

```
results/my_dataset/
├── embeddings_2026-03-01_01-16-27.npy       # Face embeddings (numpy array)
├── benchmark_2026-03-01_01-16-27.json       # Metadata (image_path, bbox, pose, blur)
├── face_crops/                               # Aligned face images
│   ├── face_0000_aligned.jpg
│   ├── face_0001_aligned.jpg
│   └── ...
├── clustering_export/                        # Export script output
│   ├── faces.csv                             # All faces with cluster assignments
│   ├── clusters.csv                          # Cluster statistics
│   ├── candidate_pairs.csv                   # Cluster pair features
│   ├── export_summary.json                   # Export metadata
│   ├── corrected_labels.csv                  # Manual labels (Phase 2)
│   └── labeling_metadata.json                # Labeling stats (Phase 2)
├── logs/
│   └── prepare_embeddings_*.log
└── notebook_clustering_results.csv           # Notebook export (for validation)
```

---

## Quick Reference Commands

```bash
# 1. Generate embeddings
python scripts/prepare_embeddings.py \
  --images /path/to/photos \
  --output results/my_dataset

# 2. Export clustering data
python scripts/export_clustering_data.py \
  --embeddings results/my_dataset/embeddings_*.npy

# 3. Run notebook for validation
jupyter notebook notebooks/debug_knn_graph_clustering.ipynb

# 4. Compare results
python scripts/compare_notebook_vs_export.py \
  --export-dir results/my_dataset/clustering_export

# 5. Launch labeling app
streamlit run app/face_clustering_labeling.py
```

---

## Next Steps

1. ✅ **Phase 1 Complete**: Data preparation working
2. ✅ **Phase 2 Complete**: Labeling interface working
3. ⏳ **Phase 3 Pending**: Implement `scripts/train_merge_classifier.py`
4. ⏳ **Phase 4 Pending**: Implement `face_cluster/ml_merge.py` (MLMerger class)

See `docs/PLAN_ML_CLUSTER_MERGING.md` for detailed implementation plan for remaining phases.

---

## Related Documents

- `docs/PLAN_ML_CLUSTER_MERGING.md` - Full implementation plan
- `docs/SIGHTINGS.md` - Bug reports and resolutions
- `docs/LEARNINGS.md` - Lessons learned from debugging
- `CHANGES_LOG.md` - Change history

---

**Last Updated**: 2026-03-01
**Contributors**: Claude Code, Jonathan Hexner
