# Reproducing PhotoTriage Multi-Feature Ranker Results

**Goal**: Reproduce 69.4% pairwise accuracy with detailed per-image results database

---

## Quick Start

### 1. Run Training

```bash
# Option A: Command line (auto-generates timestamped directory)
python train_multifeature_ranker.py \
    --batch_size 64 \
    --max_epochs 30

# Option B: Command line with custom output directory
python train_multifeature_ranker.py \
    --output_dir outputs/my_experiment \
    --batch_size 64 \
    --max_epochs 30

# Option C: VS Code (F5) - auto-generates timestamped directory
# Select: "Train Multi-Feature Ranker - Full (Recommended)"
```

**Time**: ~3-4 hours total
- Feature extraction: 30-60 min (one-time, cached)
- Training: 2-3 hours (30 epochs)

### 2. Check Results

After training completes, you'll have a timestamped directory (e.g., `outputs/phototriage_20251128_135055/`):

```
outputs/phototriage_YYYYMMDD_HHMMSS/
├── best_model.pt                    # Best model checkpoint
├── config.yaml                      # Configuration used
├── training.log                     # Full training log
├── training_curves.png              # Loss/accuracy plots
├── test_results.json                # Summary metrics
├── detailed_predictions.csv         # ← PER-IMAGE RESULTS DATABASE
└── features_cache.pkl               # Cached features (reusable)
```

**Note**: Each run automatically creates a new timestamped directory to avoid overwriting previous results. To use a specific directory, pass `--output_dir outputs/my_experiment`.

**Key file**: `detailed_predictions.csv` contains:
- Image pairs (image1, image2)
- Model scores (score1, score2, score_diff)
- Ground truth (true_winner)
- Prediction (predicted_winner)
- Correctness (correct: True/False)
- Quality attributes (label_sharpness, label_exposure, etc.)
- Human agreement metrics

### 3. Analyze Results

```bash
streamlit run analyze_phototriage_results.py
```

This opens an interactive dashboard with:
- Performance overview (accuracy, confusion matrix)
- Breakdown by quality attributes
- Error analysis
- Individual pair browser with images

---

## Expected Results

### Performance Targets

| Metric | Target (Series-Based) | Old (Random Split) | Baseline |
|--------|--------|----------|----------|
| **Test Accuracy** | **55-65%** | 69.4% | 50% (random) |
| vs Sharpness | TBD | +13% | 56.4% |
| Winner Distribution | 50/50 | 50/50 | Balanced |

**IMPORTANT**: Accuracy with series-based split will likely be **lower** than the previous 69.4% because:
1. **Previous 69.4% had data leakage** - images from same series appeared in both train and test
2. **Series-based split is harder** - model must generalize to completely new photo series
3. **More realistic** - reflects real-world performance on unseen series

Expected accuracy drop: 5-10% (from 69.4% to 59-64%)

### Training Curves

**Good signs**:
- Training accuracy climbs from ~50% to 60-65%
- Validation accuracy follows training (gap < 5%)
- Loss decreases steadily
- Winner distribution: {0: ~6000, 1: ~6000}

**Bad signs**:
- Accuracy stuck at 50% → Data issue
- Large train/val gap (>10%) → Overfitting
- All winners = 0 or 1 → Labeling bug

---

## Data Configuration

### Current Setup (Series-Based Split)

**IMPORTANT**: We split by `series_id` to prevent data leakage. This ensures that all pairs from a given photo series are in the same split (train/val/test), preventing the model from learning series-specific patterns.

```python
# From train_multifeature_ranker.py
train_ratio = 0.8  # ~9,760 pairs from ~3,466 series
val_ratio = 0.1    # ~1,114 pairs from ~433 series
test_ratio = 0.1   # ~1,199 pairs from ~434 series
```

**Why series-based split?**
- Prevents images from the same series appearing in both train and test
- Ensures the model learns generalizable quality assessment, not series-specific patterns
- More realistic evaluation: real-world usage is on new photo series

**Source**: `photo_triage_pairs_embedding_labels.csv` (filtered by agreement ≥ 0.7, reviewers ≥ 2)

**Total after filtering**: 12,073 pairs from 4,333 unique series

### Alternative Split Methods

**Option 1: Random Split (NOT RECOMMENDED - Data Leakage)**
```python
# WARNING: This may have images from same series in train and test!
train_df, val_df, test_df = data.get_random_splits(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
```

**Option 2: Official Test Set (Currently Not Usable)**
```python
# Official test set exists but doesn't have labels in CSV yet
train_df, val_df, test_df = data.get_official_splits(use_official_test=True)
```
Currently returns 0 test pairs because the CSV doesn't contain labeled pairs for the official test set.

---

## Detailed Predictions Database

### Schema

The `detailed_predictions.csv` file contains:

| Column | Type | Description |
|--------|------|-------------|
| image1 | str | First image filename (normalized) |
| image2 | str | Second image filename (normalized) |
| score1 | float | Model score for image1 (higher = better quality) |
| score2 | float | Model score for image2 |
| score_diff | float | score1 - score2 (positive = image1 predicted better) |
| true_winner | int | Ground truth winner (0=image1, 1=image2) |
| predicted_winner | int | Model prediction (0=image1, 1=image2) |
| correct | bool | Whether prediction matches ground truth |
| agreement | float | Human agreement on this pair (0-1) |
| num_reviewers | int | Number of reviewers for this pair |
| label_sharpness | int | Count of sharpness issues mentioned |
| label_exposure_quality | int | Count of exposure issues |
| label_lighting_quality | int | Count of lighting issues |
| label_motion_blur | int | Count of motion blur issues |
| label_... | int | Other quality attributes |

### Example Queries

**Find high-confidence errors**:
```python
import pandas as pd

df = pd.read_csv('outputs/phototriage_multifeature/detailed_predictions.csv')

# High confidence errors (large score difference, but wrong)
high_conf_errors = df[
    (~df['correct']) &
    (df['score_diff'].abs() > 0.5)
].sort_values('score_diff', ascending=False)

print(high_conf_errors[['image1', 'image2', 'score1', 'score2', 'true_winner']])
```

**Performance by attribute**:
```python
# Accuracy on pairs with sharpness issues
sharpness_pairs = df[df['label_sharpness'] > 0]
sharpness_accuracy = sharpness_pairs['correct'].mean()

print(f"Sharpness pairs: {len(sharpness_pairs)}")
print(f"Accuracy: {100*sharpness_accuracy:.1f}%")
```

**Agreement vs accuracy**:
```python
import matplotlib.pyplot as plt

# Bin by agreement level
agreement_bins = pd.cut(df['agreement'], bins=[0.7, 0.8, 0.9, 1.0])
accuracy_by_agreement = df.groupby(agreement_bins)['correct'].mean()

accuracy_by_agreement.plot(kind='bar')
plt.ylabel('Accuracy')
plt.xlabel('Human Agreement')
plt.title('Model Accuracy vs Human Agreement')
plt.show()
```

---

## Troubleshooting

### Issue: All winners are 0

**Problem**: Incorrect winner label computation

**Fix**: Already fixed in current code (line 82):
```python
df_filtered['winner'] = (df_filtered['MaxVote'] == df_filtered['compareID2']).astype(int)
```

**Verify**: Check log output for `Winner distribution: {0: ~6000, 1: ~6000}`

### Issue: Feature extraction fails

**Problem**: Images not found

**Solution**: Verify image directory:
```bash
ls "D:/Similar Images/automatic_triage_photo_series/train_val/train_val_imgs" | head
# Should show: 000001-01.JPG, 000001-02.JPG, etc.
```

### Issue: Cache is corrupted

**Solution**: Delete and regenerate:
```bash
rm outputs/phototriage_multifeature/features_cache.pkl
# Rerun training - will regenerate cache
```

### Issue: Out of memory

**Solution**: Reduce batch size:
```bash
python train_multifeature_ranker.py --batch_size 32
```

---

## Next Steps After Training

### 1. Verify Results Match Previous Run

```bash
# Check test accuracy (replace YYYYMMDD_HHMMSS with your run's timestamp)
cat outputs/phototriage_YYYYMMDD_HHMMSS/test_results.json

# Should show: "accuracy": 0.69 or similar
```

### 2. Analyze Failure Cases

```bash
streamlit run analyze_phototriage_results.py
```

Go to "Error Analysis" tab to see:
- Where model fails (low agreement pairs?)
- Which attributes are hardest (motion blur? exposure?)
- High-confidence errors (model very wrong)

### 3. Compare to Baselines

From benchmark results:
- Random: 50.0%
- Sharpness only: 56.4%
- **Multi-feature (yours)**: 69.4% ✓

**Improvement**: +13% over sharpness, +19.4% over random

### 4. Plan Next Improvements

See [NEXT_STEPS_PLAN.md](docs/quality_assessment/NEXT_STEPS_PLAN.md):
- Phase 1: Bradley-Terry loss (+1-3%)
- Phase 2: Facial features (+2-5%)
- Target: 70-78% accuracy

---

## File Organization

### Training Scripts (Keep)

- ✓ `train_multifeature_ranker.py` - Main training script
- ✓ `analyze_phototriage_results.py` - Streamlit analysis

### Data Management (Use)

- ✓ `sim_bench/datasets/phototriage_data.py` - Unified data interface
- ✓ `create_labeled_pairs.py` - Dataset creation (rarely needed)

### Documentation (Reference)

- ✓ `docs/quality_assessment/PHOTOTRIAGE_DATA_GUIDE.md` - Data guide
- ✓ `docs/quality_assessment/NEXT_STEPS_PLAN.md` - Future improvements
- ✓ `REPRODUCE_RESULTS.md` - This file

---

## Summary

**To reproduce results:**

1. Run: `python train_multifeature_ranker.py`
2. Wait: ~3-4 hours
3. Check: `outputs/phototriage_multifeature/test_results.json`
4. Analyze: `streamlit run analyze_phototriage_results.py`

**Expected output**: 60-70% accuracy with detailed per-pair predictions database

**Previous result**: 69.4% ✓ Excellent performance
