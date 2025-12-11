# PhotoTriage Data Management Guide

## Overview

This guide consolidates all PhotoTriage dataset operations and clarifies which scripts to use.

---

## Unified Data Interface (RECOMMENDED)

### Use: `sim_bench.datasets.phototriage_data.PhotoTriageData`

**This is the new unified interface for all PhotoTriage operations.**

```python
from sim_bench.datasets.phototriage_data import PhotoTriageData

# Initialize
data = PhotoTriageData(
    root_dir="D:/Similar Images/automatic_triage_photo_series",
    min_agreement=0.7,
    min_reviewers=2
)

# Get official train/val/test splits
train_df, val_df, test_df = data.get_official_splits(use_official_test=True)

# Verify images exist
data.verify_image_exists(train_df, split='train_val')
data.verify_image_exists(test_df, split='test')

# Get series statistics
series_info = data.get_series_info()
```

**Features**:
- ✓ Loads pairwise comparisons with filtering
- ✓ Handles official train_val / test split
- ✓ Normalizes filenames automatically
- ✓ Verifies image existence
- ✓ Provides series statistics

---

## Directory Structure

```
D:/Similar Images/automatic_triage_photo_series/
├── train_val/                          # Training + Validation set
│   ├── train_val_imgs/                 # Images (12,988 files)
│   ├── reviews_trainval/               # Raw review JSONs
│   ├── train_pairlist.txt
│   └── val_pairlist.txt
│
├── test/                               # Held-out test set
│   ├── test_imgs/                      # Test images
│   └── test_pairlist.txt
│
├── photo_triage_pairs_embedding_labels.csv  # Main CSV (RECOMMENDED)
├── photo_triage_pairs_keyword_labels.csv
├── photo_triage_pairs_with_labels.csv
└── reviews_df.csv                      # Aggregated reviews
```

**Key points**:
1. **Two separate image directories**: `train_val_imgs` and `test_imgs`
2. **Use `photo_triage_pairs_embedding_labels.csv`** - most complete version
3. **Official test set**: Images in `test/test_imgs` should NOT be used for training

---

## Existing Preprocessing Scripts

### 1. `create_labeled_pairs.py` ✓ KEEP

**Purpose**: Creates the main CSV from raw review JSONs

**When to use**: Only when regenerating CSV from scratch (rare)

**Input**:
- Raw review JSON files in `reviews_trainval/`

**Output**:
- `photo_triage_pairs_embedding_labels.csv`
- `photo_triage_pairs_keyword_labels.csv`

**Command**:
```bash
python create_labeled_pairs.py \
    --input-dir "D:/Similar Images/automatic_triage_photo_series/train_val/reviews_trainval" \
    --output-dir "D:/Similar Images/automatic_triage_photo_series"
```

**Status**: ✓ Keep - needed for regenerating dataset

---

### 2. `create_labeled_pairs_dual_method.py` ⚠️ REDUNDANT?

**Purpose**: Alternative labeling method (unclear difference from above)

**Status**: ⚠️ Review - may be redundant with `create_labeled_pairs.py`

**Recommendation**: Archive or delete if not actively used

---

### 3. `convert_pairs_csv_to_jsonl.py` ⚠️ REDUNDANT

**Purpose**: Converts CSV to JSONL format

**Status**: ⚠️ Probably not needed - CSV is fine for training

**Recommendation**: Archive or delete

---

### 4. `analyze_phototriage_results.py` ✓ KEEP

**Purpose**: Streamlit app for analyzing training results

**Status**: ✓ Essential for analysis

**Command**:
```bash
streamlit run analyze_phototriage_results.py
```

---

## Training Scripts

### 1. `train_multifeature_ranker.py` ✓ PRIMARY

**Purpose**: Train multi-feature pairwise ranker (CLIP + CNN + IQA)

**Status**: ✓ Main training script

**Usage**:
```bash
python train_multifeature_ranker.py --output_dir outputs/phototriage_multifeature
```

**Or use launch config**: "Train Multi-Feature Ranker - Full (Recommended)"

---

### 2. `sim_bench/quality_assessment/trained_models/train_binary_classifier.py`

**Purpose**: Train CLIP-only binary classifier

**Status**: ⚠️ Superseded by multi-feature ranker

**Recommendation**: Keep for comparison, but multi-feature is better

---

### 3. `sim_bench/quality_assessment/trained_models/train_series_classifier.py`

**Purpose**: Train series-softmax classifier

**Status**: ⚠️ Had labeling issues (see IMPLEMENTATION_SUMMARY.md)

**Recommendation**: Pairwise ranking (multi-feature) works better

---

## Recommended Workflow

### For Training

```python
# 1. Load data using unified interface
from sim_bench.datasets.phototriage_data import PhotoTriageData

data = PhotoTriageData(
    root_dir="D:/Similar Images/automatic_triage_photo_series",
    min_agreement=0.7,
    min_reviewers=2
)

# 2. Get official splits
train_df, val_df, test_df = data.get_official_splits(use_official_test=True)

# 3. Verify data
print(f"Train: {len(train_df)} pairs")
print(f"Val: {len(val_df)} pairs")
print(f"Test: {len(test_df)} pairs")

data.verify_image_exists(train_df, split='train_val')
data.verify_image_exists(test_df, split='test')

# 4. Use in training (example)
# Pass train_df, val_df, test_df to your training script
```

### For Reproducing Results

```bash
# 1. Train with multi-feature ranker (auto-generates timestamped directory)
python train_multifeature_ranker.py \
    --batch_size 64 \
    --max_epochs 30

# 2. Analyze results
streamlit run analyze_phototriage_results.py
```

**Expected outputs** (in timestamped directory like `outputs/phototriage_20251128_135055/`):
- `best_model.pt` - Best model checkpoint
- `detailed_predictions.csv` - Analysis database with per-pair predictions
- `training_curves.png` - Training/validation curves
- `test_results.json` - Summary metrics

**Note**: Each run creates a new timestamped directory automatically. To specify a custom directory, add `--output_dir outputs/my_experiment`.

---

## Data Statistics

From `photo_triage_pairs_embedding_labels.csv`:

| Metric | Value |
|--------|-------|
| Total pairs | 24,186 |
| Pairs (agreement ≥ 0.7, reviewers ≥ 2) | 12,073 |
| Unique images | ~5,000 |
| Series | ~2,000 |
| Avg images per series | 2.5-3 |

**Winner distribution** (after filtering):
- Should be ~50/50 split (balanced)
- If all winners = 0 or 1 → DATA BUG

---

## Common Issues & Solutions

### Issue: All winners are 0 or 1

**Problem**: Incorrect winner label computation

**Solution**: Use `PhotoTriageData` which handles this correctly:
```python
# Correct way:
df['winner'] = (df['MaxVote'] == df['compareID2']).astype(int)

# WRONG:
df['winner'] = (df['MaxVote'] == 'RIGHT').astype(int)  # MaxVote is numeric!
```

### Issue: Images not found

**Problem**: Filename normalization mismatch

**CSV format**: `"1-1.JPG"`, `"12-3.JPG"`
**Actual files**: `"000001-01.JPG"`, `"000012-03.JPG"`

**Solution**: Use `PhotoTriageData.normalize_filename()`:
```python
normalized = PhotoTriageData.normalize_filename("1-1.JPG")
# Returns: "000001-01.JPG"
```

### Issue: Train/test contamination

**Problem**: Test images leaked into training set

**Solution**: Use `get_official_splits(use_official_test=True)`:
```python
data = PhotoTriageData(...)
train_df, val_df, test_df = data.get_official_splits(use_official_test=True)

# This ensures:
# - train_df and val_df only use images from train_val_imgs/
# - test_df only uses images from test_imgs/
```

---

## Cleanup Recommendations

### Scripts to Archive

Move to `scripts/archive/`:
1. `create_labeled_pairs_dual_method.py` (if redundant)
2. `convert_pairs_csv_to_jsonl.py` (not needed)

### Scripts to Keep

Keep in root:
1. `create_labeled_pairs.py` - Essential for dataset regeneration
2. `analyze_phototriage_results.py` - Essential for analysis
3. `train_multifeature_ranker.py` - Primary training script

### Scripts in `sim_bench/quality_assessment/trained_models/`

Keep all (for comparison and reference):
- `train_binary_classifier.py` - CLIP-only baseline
- `train_series_classifier.py` - Series-softmax approach
- `phototriage_binary.py` - Binary classifier model
- `phototriage_series.py` - Series classifier model
- `phototriage_multifeature.py` - Multi-feature ranker (BEST)

---

## Migration Guide

### Old Code

```python
# OLD: Scattered preprocessing
from sim_bench.quality_assessment.trained_models.train_binary_classifier import load_and_filter_data

df = load_and_filter_data(csv_path, min_agreement, min_reviewers)
# Manual train/val/test split
# Manual filename normalization
# No official test set support
```

### New Code

```python
# NEW: Unified interface
from sim_bench.datasets.phototriage_data import PhotoTriageData

data = PhotoTriageData(root_dir, min_agreement=0.7, min_reviewers=2)
train_df, val_df, test_df = data.get_official_splits(use_official_test=True)

# Everything handled:
# ✓ Filename normalization
# ✓ Winner label computation
# ✓ Official train/val/test splits
# ✓ Image verification
```

---

## Summary

**For most users:**
1. Use `PhotoTriageData` for all data loading
2. Use `train_multifeature_ranker.py` for training
3. Use `analyze_phototriage_results.py` for analysis
4. Ignore other preprocessing scripts unless regenerating dataset

**CSV to use**: `photo_triage_pairs_embedding_labels.csv`

**Test set**: Always use `use_official_test=True` to avoid contamination

**Expected performance**: 60-70% pairwise accuracy (vs 56.4% sharpness baseline)
