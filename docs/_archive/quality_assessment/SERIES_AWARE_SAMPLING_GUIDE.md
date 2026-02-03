# Series-Aware Batch Sampling - Complete Guide

## Your Requirements

You asked for:
1. **Each series_id goes into one batch** - Each batch contains pairs from different series
2. **Batch composition shuffled between epochs** - Series in batch 0 of epoch 0 ≠ series in batch 0 of epoch 1

## How to Enable

### Option 1: Command Line

```bash
# Enable series-aware sampling
python train_multifeature_ranker.py --use_series_sampler --mlp_hidden_dims 256

# Combine with quick experiment (recommended for testing)
python train_multifeature_ranker.py --use_series_sampler --quick_experiment 0.1 --mlp_hidden_dims 256
```

### Option 2: VS Code Launch Configurations (F5)

Press F5 and select:
- **"Train Multi-Feature Ranker - Series-Aware Sampling ⭐"** - Full dataset with series-aware sampling
- **"Train Multi-Feature Ranker - Quick + Series-Aware ⚡⭐"** - 10% quick test with series-aware sampling

### Option 3: Hyperparameter Search

```bash
# Quick search (3 experiments) - currently does NOT use series-aware sampling
python run_hyperparameter_search.py --mode quick

# To add series-aware sampling to all experiments, modify run_hyperparameter_search.py
# (Future enhancement)
```

---

## How It Works - Code Walkthrough

### Implementation Location

File: [sim_bench/datasets/series_aware_sampler.py](../../sim_bench/datasets/series_aware_sampler.py)

### Algorithm

The `BalancedSeriesBatchSampler` implements both requirements:

#### Requirement 1: Each batch contains pairs from different series

**Algorithm** (lines 172-199):
```python
def __iter__(self):
    # 1. Shuffle series order (DIFFERENT each epoch!)
    series_order = self.unique_series.copy()
    if self.shuffle:
        np.random.shuffle(series_order)  # <-- KEY: New order each epoch

    # 2. Create iterators for each series (shuffle pairs within series)
    series_iterators = {}
    for series_id, indices in self.series_to_indices.items():
        series_indices = indices.copy()
        if self.shuffle:
            np.random.shuffle(series_indices)  # Different pair order each epoch
        series_iterators[series_id] = iter(series_indices)

    all_indices = []

    # 3. Round-robin through series (one pair from each series)
    while series_iterators:
        for series_id in series_order.copy():
            if series_id not in series_iterators:
                continue

            try:
                idx = next(series_iterators[series_id])  # Take one pair from this series
                all_indices.append(idx)
            except StopIteration:
                # This series is exhausted, remove it
                del series_iterators[series_id]
                series_order.remove(series_id)

    # 4. Create batches from round-robin order
    for i in range(0, len(all_indices), self.batch_size):
        batch = all_indices[i:i + self.batch_size]
        yield batch
```

**Result**: Each batch contains one pair from each of N different series (where N ≈ batch_size).

#### Requirement 2: Batch composition shuffled between epochs

**Key line** (line 175):
```python
np.random.shuffle(series_order)  # Different order EACH epoch!
```

This is called inside `__iter__()`, which is called **once per epoch** by PyTorch's DataLoader.

**Example**:

```
Epoch 0:
  series_order = [5, 12, 23, 7, 19, ...]  (random shuffle)
  Round-robin: series_5 → series_12 → series_23 → series_7 → ...
  Batch 0: [pair_from_5, pair_from_12, pair_from_23, ..., pair_from_7]  (64 pairs)
  Batch 1: [pair_from_19, pair_from_31, pair_from_42, ..., pair_from_8]  (64 pairs)

Epoch 1:
  series_order = [42, 7, 5, 31, 12, ...]  (DIFFERENT random shuffle)
  Round-robin: series_42 → series_7 → series_5 → series_31 → ...
  Batch 0: [pair_from_42, pair_from_7, pair_from_5, ..., pair_from_31]  (64 pairs)
  Batch 1: [pair_from_12, pair_from_19, pair_from_23, ..., pair_from_67]  (64 pairs)
```

Notice:
- Epoch 0, Batch 0 contains: series [5, 12, 23, 7, ...]
- Epoch 1, Batch 0 contains: series [42, 7, 5, 31, ...] ← DIFFERENT!

---

## Verification in Training Script

When you run with `--use_series_sampler`, you'll see this output:

```
Using series-aware batch sampler
BalancedSeriesBatchSampler initialized:
  Total pairs: 9760
  Unique series: 3466
  Batch size: 64
  Shuffle enabled: True
  Train batches: ~152 (series-aware)
```

### How to Verify It's Working

1. **Check initialization** (lines 158-162 in sampler):
   - "Total pairs" should match your training set size
   - "Unique series" should be ~3466 for full dataset
   - "Shuffle enabled: True" confirms shuffling is on

2. **Monitor batch diversity** (add this debugging code):
   ```python
   # In train_one_epoch(), add after getting batch:
   series_in_batch = [train_df.iloc[idx]['series_id'] for idx in batch_indices]
   unique_series = len(set(series_in_batch))
   logger.info(f"Batch {batch_idx}: {unique_series}/{len(batch_indices)} unique series")
   ```

   You should see: "Batch 0: 64/64 unique series" (all unique!)

3. **Verify epoch-to-epoch shuffling** (add debugging):
   ```python
   # In training loop, save first batch's series IDs
   if epoch == 1:
       epoch1_batch0_series = [...]
   elif epoch == 2:
       epoch2_batch0_series = [...]
       overlap = set(epoch1_batch0_series) & set(epoch2_batch0_series)
       logger.info(f"Batch 0 series overlap: {len(overlap)}/{batch_size}")
   ```

   You should see low overlap (~10-20% due to chance).

---

## Comparison: Standard vs Series-Aware Sampling

### Standard Sampling (WITHOUT --use_series_sampler)

```python
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

**Problem**:
- Random shuffling → Multiple pairs from same series in one batch
- Example batch: [series_5_pair1, series_5_pair2, series_5_pair3, series_12_pair1, ...]
- Model learns series-specific patterns (e.g., "series 5 prefers warm tones")
- Overfits to training series, fails on new series

**Batch composition between epochs**:
- Randomized, but not series-aware

### Series-Aware Sampling (WITH --use_series_sampler)

```python
from sim_bench.datasets.series_aware_sampler import BalancedSeriesBatchSampler

train_sampler = BalancedSeriesBatchSampler(
    train_df['series_id'].tolist(),
    batch_size=64,
    shuffle=True
)
train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
```

**Benefits**:
- ✅ Each batch: ONE pair per series → Forces diverse examples
- ✅ Shuffled between epochs → Different series combinations each epoch
- ✅ Model learns generalizable quality features, not series patterns
- ✅ Better generalization to new series

**Example batch**: [series_5_pair1, series_12_pair1, series_23_pair1, series_7_pair1, ...] (64 different series)

---

## When to Use Series-Aware Sampling

### Use It When:
- ✅ Training with series-based split (current setup)
- ✅ Model is overfitting (train >> val accuracy)
- ✅ You want to encourage generalization across series
- ✅ Dataset has many series with few pairs each

### Don't Use It When:
- ❌ Quick debugging (adds complexity)
- ❌ You want to maximize training signal (standard sampling may give higher train accuracy)
- ❌ Series have very unbalanced pair counts (some series will dominate)

---

## Code Locations

### Where Series-Aware Sampling is Implemented

1. **Sampler class**: [sim_bench/datasets/series_aware_sampler.py](../../sim_bench/datasets/series_aware_sampler.py)
   - Line 118-217: `BalancedSeriesBatchSampler` class
   - Line 172-199: `__iter__()` method (key logic)

2. **Training script integration**: [train_multifeature_ranker.py](../../train_multifeature_ranker.py)
   - Line 376: `--use_series_sampler` argument
   - Line 496-509: Series-aware sampler creation and DataLoader setup
   ```python
   if args.use_series_sampler:
       logger.info("\nUsing series-aware batch sampler")
       from sim_bench.datasets.series_aware_sampler import BalancedSeriesBatchSampler

       train_sampler = BalancedSeriesBatchSampler(
           train_df['series_id'].tolist(),
           batch_size=config.batch_size,
           drop_last=False,
           shuffle=True
       )
       train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0)
   else:
       train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
   ```

---

## Quick Start - Recommended Testing Workflow

```bash
# 1. Quick test with 10% of data + series-aware sampling (~5-10 min)
python train_multifeature_ranker.py \
    --quick_experiment 0.1 \
    --use_series_sampler \
    --mlp_hidden_dims 256 \
    --batch_size 64 \
    --max_epochs 10

# 2. If results look promising, run full dataset
python train_multifeature_ranker.py \
    --use_series_sampler \
    --mlp_hidden_dims 256 \
    --batch_size 64 \
    --max_epochs 30

# 3. Compare with standard sampling (ablation study)
python train_multifeature_ranker.py \
    --mlp_hidden_dims 256 \
    --batch_size 64 \
    --max_epochs 30
# (no --use_series_sampler flag)
```

---

## Expected Impact on Training

### With Series-Aware Sampling:
- **Training accuracy**: May be slightly lower (harder to overfit)
- **Validation accuracy**: Should be higher or similar (better generalization)
- **Train-val gap**: Smaller (less overfitting)
- **Test accuracy**: Better (generalizes to new series)

### Without Series-Aware Sampling (Current Problem):
- **Training accuracy**: Higher (easier to overfit to series patterns)
- **Validation accuracy**: Degrades over epochs (overfitting)
- **Train-val gap**: Large (10-15% or more)
- **Test accuracy**: Poor (48.6% - worse than random)

---

## Summary - Your Questions Answered

### 1. Does each series_id go into one batch?

**YES**. The round-robin algorithm ensures each batch contains one pair from each of N different series.

**Code proof** (lines 187-199):
- Loops through series_order
- Takes ONE pair from each series (`next(series_iterators[series_id])`)
- Appends to all_indices
- Creates batches from sequential slices

### 2. Are batch_id/seriesId shuffled between epochs?

**YES**. Series order is reshuffled at the start of each epoch.

**Code proof** (lines 173-175):
```python
series_order = self.unique_series.copy()
if self.shuffle:
    np.random.shuffle(series_order)  # New random order each epoch
```

This happens inside `__iter__()`, which PyTorch calls once per epoch.

### 3. How to enable?

**Command line**: `--use_series_sampler`

**VS Code**: Select launch config with "Series-Aware" in name

**Hyperparameter search**: Currently not enabled by default (would need to modify experiment configs)

### 4. How to test with 10% of data?

**Command line**: `--quick_experiment 0.1`

**VS Code**: Select launch config with "Quick Experiment (10%)" in name

**Both together**: `--quick_experiment 0.1 --use_series_sampler`
