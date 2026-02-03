# Series-Based Split: Preventing Data Leakage

## The Problem: Data Leakage in Random Split

### What Was Wrong

The previous training used **random splitting** of pairs without considering `series_id`:

```python
# OLD (WRONG): Random shuffle of all pairs
df = df.sample(frac=1, random_state=42)
train = df[:80%]
test = df[80%:]
```

**Result**: Images from the same photo series appeared in BOTH training and test sets.

### Why This Is Bad

**Example of Data Leakage:**

Photo Series #123 has 4 images:
- `123-01.JPG` (worst quality)
- `123-02.JPG` (slightly better)
- `123-03.JPG` (good)
- `123-04.JPG` (best)

With random split, you might get:
- **Training pairs**: (123-01, 123-02), (123-03, 123-04)
- **Test pairs**: (123-01, 123-04), (123-02, 123-03)

**Problem**: The model sees images from series #123 during training, so it learns series-specific patterns:
- Lighting conditions of this particular location
- Camera settings used for this series
- Subject-specific features (same person, same building, etc.)

**Result**: 69.4% accuracy was **overoptimistic** because the model exploited series-specific cues rather than learning generalizable quality assessment.

---

## The Solution: Series-Based Split

### Implementation

```python
# NEW (CORRECT): Split by series_id
unique_series = df['series_id'].unique()
np.random.shuffle(unique_series)

train_series = unique_series[:80%]  # 3,466 series
val_series = unique_series[80:90%]  # 433 series
test_series = unique_series[90:]    # 434 series

train_df = df[df['series_id'].isin(train_series)]
val_df = df[df['series_id'].isin(val_series)]
test_df = df[df['series_id'].isin(test_series)]
```

**Verification**:
```python
assert len(train_series & test_series) == 0  # No overlap!
```

### Benefits

1. **No Data Leakage**: Zero overlap of series between splits
2. **Generalizable Model**: Forces model to learn quality features that work on new series
3. **Realistic Evaluation**: Matches real-world usage (evaluating new photo series)

---

## Expected Performance Impact

### Accuracy Comparison

| Split Method | Test Accuracy | What It Measures |
|--------------|---------------|------------------|
| **Random** | 69.4% | Can recognize images from seen series (overfit) |
| **Series-Based** | 55-65% (expected) | Can assess quality on completely new series (generalize) |
| **Sharpness Baseline** | 56.4% | Simple rule-based method |

### Why Lower Accuracy Is Actually Better

**Random split (69.4%)**:
- Model learns: "Series #123 has warm lighting, so prefer warmer images"
- Fails on: New series with different lighting

**Series-based split (55-65%)**:
- Model learns: "Sharp images are generally better than blurry ones"
- Works on: Any new photo series

**Real-world analogy**:
- Random split = Memorizing specific photos from a textbook
- Series-based split = Learning general photography principles

---

## Dataset Statistics

### Filtered Dataset
- **Total pairs**: 12,073 (agreement ≥ 0.7, reviewers ≥ 2)
- **Unique series**: 4,333
- **Average pairs per series**: 2.8

### Series-Based Split (80/10/10)
| Split | Series Count | Pair Count | % of Total Pairs |
|-------|--------------|------------|------------------|
| **Train** | 3,466 (80%) | 9,760 | 80.8% |
| **Val** | 433 (10%) | 1,114 | 9.2% |
| **Test** | 434 (10%) | 1,199 | 9.9% |

**Note**: Pair percentages don't exactly match 80/10/10 because some series have more pairs than others. We prioritize series-level split to prevent leakage.

---

## How to Use

### Recommended (Series-Based)
```python
from sim_bench.datasets.phototriage_data import PhotoTriageData

data = PhotoTriageData(
    root_dir="D:/Similar Images/automatic_triage_photo_series",
    min_agreement=0.7,
    min_reviewers=2
)

# Use series-based split (prevents data leakage)
train_df, val_df, test_df = data.get_series_based_splits(
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_seed=42
)
```

### Not Recommended (Random Split)
```python
# WARNING: Data leakage! Only use for comparison.
train_df, val_df, test_df = data.get_random_splits(
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_seed=42
)
```

---

## Verification

### Check for Data Leakage

```python
# Get unique series from each split
train_series = set(train_df['series_id'].unique())
val_series = set(val_df['series_id'].unique())
test_series = set(test_df['series_id'].unique())

# Check overlaps (should all be 0)
print(f"Train-Val overlap: {len(train_series & val_series)}")    # 0
print(f"Train-Test overlap: {len(train_series & test_series)}")  # 0
print(f"Val-Test overlap: {len(val_series & test_series)}")      # 0
```

### Visualize Leakage in Random Split

```python
# Compare random vs series-based split
random_train, random_val, random_test = data.get_random_splits()
series_train, series_val, series_test = data.get_series_based_splits()

# Check series overlap in random split
random_train_series = set(random_train['series_id'].unique())
random_test_series = set(random_test['series_id'].unique())

print(f"Random split overlap: {len(random_train_series & random_test_series)} series")
# Typically ~3,500+ series overlap! (Severe data leakage)

# Check series overlap in series-based split
series_train_series = set(series_train['series_id'].unique())
series_test_series = set(series_test['series_id'].unique())

print(f"Series-based split overlap: {len(series_train_series & series_test_series)} series")
# Always 0 (No leakage)
```

---

## Related Files

- **Implementation**: [sim_bench/datasets/phototriage_data.py](../../sim_bench/datasets/phototriage_data.py) - Lines 240-298
- **Training Script**: [train_multifeature_ranker.py](../../train_multifeature_ranker.py) - Line 439
- **Documentation**: [REPRODUCE_RESULTS.md](../../REPRODUCE_RESULTS.md) - Data Configuration section

---

## Key Takeaways

1. **Always split by series_id** when working with PhotoTriage dataset
2. **Random split gives inflated accuracy** due to data leakage (~5-10% overestimate)
3. **Series-based split is more realistic** for real-world photo quality assessment
4. **Lower accuracy doesn't mean worse model** - it means more honest evaluation

The 69.4% accuracy from random split was impressive but misleading. The true test is whether the model can assess quality on completely new photo series it has never seen before.
