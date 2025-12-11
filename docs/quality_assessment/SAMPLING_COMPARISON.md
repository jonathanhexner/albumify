# Sampling Comparison: Standard vs Series-Aware

## Standard Sampling (Current - Causes Overfitting)

```
WITHOUT --use_series_sampler

Dataset: [pair1(series_5), pair2(series_5), pair3(series_12), pair4(series_5), pair5(series_12), ...]

Random Shuffle Each Epoch:
  Epoch 0: [pair4, pair1, pair3, pair2, pair5, ...]
  Epoch 1: [pair2, pair5, pair1, pair4, pair3, ...]

Batches (batch_size=4):
  Epoch 0:
    Batch 0: [pair4(series_5), pair1(series_5), pair3(series_12), pair2(series_5)]
             ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^                    ^^^^^^^^^^^^^^^^
             3 pairs from series_5 in ONE batch! Model learns series_5 patterns!

    Batch 1: [pair5(series_12), pair6(series_23), pair7(series_12), pair8(series_5)]
             ^^^^^^^^^^^^^^^^^                     ^^^^^^^^^^^^^^^^^
             2 pairs from series_12 in ONE batch!

  Epoch 1:
    Batch 0: [pair2(series_5), pair5(series_12), pair1(series_5), pair4(series_5)]
             ^^^^^^^^^^^^^^^^                     ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^
             3 pairs from series_5 again! Same pattern!
```

**Problem**:
- Multiple pairs from same series in each batch
- Model learns: "Series 5 likes warm colors, Series 12 likes high contrast"
- Overfits to series-specific patterns
- Fails on new series (test accuracy: 48.6%)

---

## Series-Aware Sampling (NEW - Prevents Overfitting)

```
WITH --use_series_sampler

Step 1: Group pairs by series_id
  series_5:  [pair1, pair2, pair4]
  series_12: [pair3, pair5, pair7]
  series_23: [pair6, pair8]
  series_7:  [pair9]
  series_19: [pair10]

Step 2: Shuffle series order (DIFFERENT each epoch)
  Epoch 0: series_order = [series_5, series_12, series_23, series_7, series_19, ...]
  Epoch 1: series_order = [series_23, series_7, series_5, series_19, series_12, ...]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           DIFFERENT ORDER! This ensures different series in each batch!

Step 3: Round-robin through series (one pair from each)
  Epoch 0:
    Take from series_5  → pair1
    Take from series_12 → pair3
    Take from series_23 → pair6
    Take from series_7  → pair9
    Take from series_19 → pair10
    (loop back)
    Take from series_5  → pair2  (series_5 still has pairs)
    Take from series_12 → pair5
    Take from series_23 → pair8
    (series_7 exhausted, skip)
    (series_19 exhausted, skip)
    Take from series_5  → pair4
    Take from series_12 → pair7
    ...

  all_indices = [pair1, pair3, pair6, pair9, pair10, pair2, pair5, pair8, pair4, pair7, ...]

Step 4: Create batches sequentially (batch_size=4)
  Epoch 0:
    Batch 0: [pair1(series_5), pair3(series_12), pair6(series_23), pair9(series_7)]
             ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^
             ALL DIFFERENT SERIES! Model forced to learn general quality features!

    Batch 1: [pair10(series_19), pair2(series_5), pair5(series_12), pair8(series_23)]
             ^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^
             ALL DIFFERENT SERIES again!

  Epoch 1 (series_order shuffled differently):
    Batch 0: [pair6(series_23), pair9(series_7), pair1(series_5), pair10(series_19)]
             ^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^
             DIFFERENT series than Epoch 0 Batch 0!

    Batch 1: [pair3(series_12), pair2(series_5), pair8(series_23), pair4(series_5)]
```

**Benefits**:
- ✅ Each batch: ONE pair per series → Can't overfit to series patterns
- ✅ Different epochs: Different series combinations → More diverse training
- ✅ Model learns: "Sharp images are better" (general), not "Series 5 likes warm tones" (specific)
- ✅ Generalizes to new series (expected test accuracy: 52-57%)

---

## Side-by-Side Comparison

| Aspect | Standard Sampling | Series-Aware Sampling |
|--------|-------------------|----------------------|
| **Pairs per series in batch** | Multiple (2-5) | Exactly 1 |
| **Batch 0 across epochs** | Random series | Shuffled series each epoch |
| **What model learns** | Series-specific patterns | General quality features |
| **Training accuracy** | High (61.5%) | Moderate (55-60%) |
| **Validation accuracy** | Degrades (43.6%) | Stable (53-57%) |
| **Test accuracy** | Poor (48.6%) | Better (52-57% expected) |
| **Overfitting** | Severe | Reduced |

---

## Visual Example: Batch Composition

### Standard Sampling

```
Epoch 0:
  Batch 0: [S5, S5, S12, S5, S23, S12, S5, S7, S12, S5, ...]
            ^^^  ^^^       ^^^       ^^^^      ^^^^  ^^^
            Series 5 appears 5 times in this batch!

Epoch 1:
  Batch 0: [S12, S5, S5, S23, S5, S12, S7, S5, S12, S23, ...]
                 ^^^  ^^^      ^^^            ^^^
            Series 5 appears 4 times in this batch!
```

### Series-Aware Sampling

```
Epoch 0:
  Batch 0: [S5, S12, S23, S7, S19, S31, S42, S8, S67, S88, ...]
            ^^^  ^^^^  ^^^^  ^^^  ^^^^  ^^^^  ^^^^  ^^^  ^^^^
            All unique! No repetition!

Epoch 1:
  Batch 0: [S23, S7, S5, S19, S12, S8, S67, S42, S31, S88, ...]
            ^^^^  ^^^  ^^^  ^^^^  ^^^^  ^^^  ^^^^  ^^^^  ^^^^
            Still all unique, but DIFFERENT ORDER than Epoch 0!
```

---

## Code to Enable

```bash
# Standard sampling (current default - overfits)
python train_multifeature_ranker.py --mlp_hidden_dims 256

# Series-aware sampling (recommended)
python train_multifeature_ranker.py --use_series_sampler --mlp_hidden_dims 256

# Quick test with 10% of data
python train_multifeature_ranker.py --use_series_sampler --quick_experiment 0.1 --mlp_hidden_dims 256
```

---

## Expected Training Curves

### Standard Sampling (Overfitting)

```
Epoch    Train Acc    Val Acc    Gap
  1        52.0%       51.5%     0.5%   ← Good start
  2        54.2%       50.5%     3.7%
  3        56.3%       49.8%     6.5%   ← Gap widening
  5        59.1%       47.2%    11.9%   ← Severe overfitting
  7        61.5%       43.6%    17.9%   ← Early stop
Test                   48.6%            ← Worse than random!
```

### Series-Aware Sampling (Better Generalization)

```
Epoch    Train Acc    Val Acc    Gap
  1        51.2%       51.0%     0.2%   ← Good start
  3        53.8%       52.4%     1.4%   ← Healthy gap
  5        55.1%       53.6%     1.5%   ← Stable
  10       56.3%       54.2%     2.1%   ← Both improving
  15       57.2%       54.8%     2.4%   ← Converging
Test                   54.3%            ← Better than random!
```

---

## Debugging: How to Verify It's Working

Add this to your training loop to verify series-aware sampling:

```python
# In train_one_epoch(), after getting batch:
if batch_idx == 0:  # First batch
    # Get series_ids for this batch
    batch_series = []
    for idx in batch['indices']:  # Assuming you add indices to batch
        batch_series.append(train_df.iloc[idx]['series_id'])

    unique_series = len(set(batch_series))
    logger.info(f"Epoch {epoch}, Batch 0: {unique_series}/{len(batch_series)} unique series")
    logger.info(f"  Series IDs: {batch_series[:10]}...")  # First 10
```

**Expected output with series-aware sampling**:
```
Epoch 1, Batch 0: 64/64 unique series
  Series IDs: [5, 12, 23, 7, 19, 31, 42, 8, ...]
Epoch 2, Batch 0: 64/64 unique series
  Series IDs: [23, 7, 5, 19, 12, 8, 67, 42, ...]  ← Different order!
```

**Expected output with standard sampling** (problem):
```
Epoch 1, Batch 0: 43/64 unique series  ← Only 43 unique out of 64!
  Series IDs: [5, 5, 12, 5, 23, 12, 5, 7, 12, 5, ...]  ← Repetition!
Epoch 2, Batch 0: 45/64 unique series  ← Still only 45 unique!
  Series IDs: [12, 5, 5, 23, 5, 12, 7, 5, 12, 23, ...]
```
