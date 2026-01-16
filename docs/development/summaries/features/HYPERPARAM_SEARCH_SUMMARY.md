# Hyperparameter Search - Quick Reference

## What I Just Created

### 1. **Series-Aware Batch Sampler** ([series_aware_sampler.py](sim_bench/datasets/series_aware_sampler.py))

**Problem**: With random batch sampling, the model sees multiple pairs from the same series in each batch, making it easy to overfit to series-specific patterns.

**Solution**: Two new samplers ensure each batch contains pairs from different series:

```python
from sim_bench.datasets.series_aware_sampler import BalancedSeriesBatchSampler

# Ensures each batch has pairs from different series
sampler = BalancedSeriesBatchSampler(
    series_ids=train_df['series_id'].tolist(),
    batch_size=64,
    shuffle=True
)

train_loader = DataLoader(train_dataset, batch_sampler=sampler)
```

**How it works**:
- Round-robin through series, taking one pair from each
- Shuffles series order between epochs
- When a series is exhausted, moves to next series
- Result: Model learns from diverse examples in each batch

### 2. **Hyperparameter Search Script** ([run_hyperparameter_search.py](run_hyperparameter_search.py))

**Problem**: Need to systematically try different configurations to find what works with series-based split.

**Solution**: Automated search script with pre-configured experiments:

```bash
# Quick search (3 experiments, ~2-3 hours)
python run_hyperparameter_search.py --mode quick

# Full search (10 experiments, ~6-8 hours)
python run_hyperparameter_search.py --mode full

# Specific experiments
python run_hyperparameter_search.py --experiments simple_mlp clip_only
```

**Features**:
- 10 pre-configured experiments
- Automatic result tracking and summarization
- Resume capability for interrupted searches
- Generates comparison table automatically

### 3. **Extended Training Script** ([train_multifeature_ranker.py](train_multifeature_ranker.py))

**Added command-line arguments**:
```bash
--dropout 0.5                        # Regularization
--use_clip true/false                # Feature ablation
--use_cnn_features true/false        # Feature ablation
--use_iqa_features true/false        # Feature ablation
--use_series_sampler                 # Use series-aware sampling
```

## Quick Start

### Recommended: Run Quick Search

```bash
python run_hyperparameter_search.py --mode quick
```

This will test:
1. **simple_mlp**: Simpler model (1-layer MLP)
2. **tiny_mlp**: Very small model with large batches
3. **clip_only**: CLIP features only

**Expected time**: 2-3 hours (30-45 min per experiment)

**Success looks like**: Test accuracy > 50%, ideally 52-57%

### Alternative: Manual Single Experiment

```bash
# Try simple MLP
python train_multifeature_ranker.py \
    --mlp_hidden_dims 256 \
    --batch_size 128 \
    --learning_rate 0.0005

# Try CLIP-only
python train_multifeature_ranker.py \
    --mlp_hidden_dims 256 \
    --use_cnn_features false \
    --use_iqa_features false \
    --batch_size 128

# Try with series-aware sampling
python train_multifeature_ranker.py \
    --mlp_hidden_dims 128 \
    --batch_size 128 \
    --use_series_sampler
```

## Why We Need This

### The Problem

**Severe overfitting** with original configuration:
```
Epoch 1: Train 52.0%, Val 51.5% ✓ Good start
Epoch 2: Train 54.2%, Val 50.5%
Epoch 3: Train 53.6%, Val 49.8%
Epoch 7: Train 61.5%, Val 43.6% ❌ Massive overfitting
Test:    48.6% (worse than random!)
```

**Root causes**:
1. **Model too complex**: 1540 → [512, 256] → 1 for limited training data
2. **Series-based split is harder**: Can't exploit series-specific patterns
3. **Limited effective data**: Only 3,466 unique series for training

### The Solutions

**Option 1: Simpler Model** (Recommended)
```python
# Original: 1540 → 512 → 256 → 1 (too deep)
mlp_hidden_dims = [512, 256]

# New: 1540 → 256 → 1 (simpler)
mlp_hidden_dims = [256]

# Or: 1540 → 128 → 1 (very simple)
mlp_hidden_dims = [128]
```

**Option 2: Feature Ablation**
```python
# Use CLIP only (512-dim instead of 1540-dim)
use_cnn_features = False
use_iqa_features = False
```

**Option 3: Stronger Regularization**
```python
dropout = 0.5           # Up from 0.3
batch_size = 256        # Larger batches
```

**Option 4: Series-Aware Sampling**
```python
--use_series_sampler    # Ensures diverse examples per batch
```

## Experiment Catalog

| Experiment | MLP Dims | Batch | LR | Features | Notes |
|------------|----------|-------|-----|----------|-------|
| `baseline` | [512, 256] | 64 | 0.0001 | All | Original (overfits) |
| `simple_mlp` | [256] | 64 | 0.0001 | All | **Recommended first try** |
| `tiny_mlp` | [128] | 128 | 0.0005 | All | Very simple + large batches |
| `clip_only` | [256] | 128 | 0.0005 | CLIP | **Good alternative** |
| `iqa_only` | [32] | 256 | 0.001 | IQA | Simplest baseline |
| `high_dropout` | [512, 256] | 64 | 0.0001 | All | Dropout 0.5 |
| `larger_batches` | [256] | 256 | 0.001 | All | Implicit regularization |
| `high_lr` | [256] | 128 | 0.001 | All | Faster learning |
| `low_lr` | [256] | 128 | 0.00005 | All | Careful learning |

## Interpreting Results

### After Search Completes

Check `summary.csv`:
```bash
cat outputs/hyperparameter_search/search_*/summary.csv
```

**Look for**:
- `test_accuracy` > 50% (better than random)
- Small train-val gap (< 5%)
- Stable validation accuracy (not degrading)

### Example Good Result

```
Experiment: simple_mlp
Test Accuracy: 0.5423 (54.23%)
Test Loss: 0.9234
Duration: 32.4 min

Training progression:
Epoch 1: Train 51.2%, Val 51.0%
Epoch 5: Train 54.8%, Val 53.1%
Epoch 10: Train 56.3%, Val 54.2% ✓ Both improving
```

### Example Bad Result (Still Overfitting)

```
Experiment: baseline
Test Accuracy: 0.4856 (48.56%)
Test Loss: 0.9970

Training progression:
Epoch 1: Train 52.0%, Val 51.5%
Epoch 7: Train 61.5%, Val 43.6% ❌ Diverging
```

## Series-Aware Sampling Explained

### Standard Sampling (Current - Bad)

```
Batch 1: [series_5_pair_1, series_5_pair_2, series_12_pair_1, ...]
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Same series = easy to overfit
```

Model learns: "Series 5 prefers warm tones, series 12 prefers high contrast"

### Series-Aware Sampling (New - Better)

```
Batch 1: [series_5_pair_1, series_12_pair_1, series_23_pair_1, ...]
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ All different series
```

Model learns: "Good photos are sharp, well-exposed, well-composed" (generalizes!)

### How to Use

```bash
# Add flag to any experiment
python train_multifeature_ranker.py --use_series_sampler --mlp_hidden_dims 256

# Or in search script (need to modify EXPERIMENTS dict)
```

## Expected Performance Targets

| Split Type | Expected Accuracy | Why |
|------------|-------------------|-----|
| **Random split** | 65-70% | Can exploit series patterns (data leakage) |
| **Series-based (overfit)** | 48-50% | Memorizes training, fails test |
| **Series-based (good)** | 52-57% | Generalizes to new series ✓ |
| **Sharpness baseline** | 56.4% | Simple rule-based method |

**Target**: Beat random (50%) and approach sharpness baseline (56.4%)

## Next Steps

1. **Run quick search**:
   ```bash
   python run_hyperparameter_search.py --mode quick
   ```

2. **Check results** after 2-3 hours:
   ```bash
   cat outputs/hyperparameter_search/search_*/summary.csv
   ```

3. **If best accuracy > 52%**: You found a good config!
   - Run 2-3 more times with different seeds to verify
   - Update default config
   - Move to next improvements (data augmentation, Bradley-Terry loss)

4. **If best accuracy < 52%**: Need more radical changes
   - Try data augmentation
   - Consider pre-training on random split then fine-tuning
   - Look at error analysis to understand what's failing

## Files Created

1. [sim_bench/datasets/series_aware_sampler.py](sim_bench/datasets/series_aware_sampler.py) - Series-aware batch samplers
2. [run_hyperparameter_search.py](run_hyperparameter_search.py) - Automated search script
3. [docs/quality_assessment/HYPERPARAMETER_SEARCH_GUIDE.md](docs/quality_assessment/HYPERPARAMETER_SEARCH_GUIDE.md) - Detailed guide
4. [train_multifeature_ranker.py](train_multifeature_ranker.py) - Extended with new arguments

## Key Insights

1. **Series-based split is much harder** than random split (no data leakage)
2. **Original model is too complex** for the amount of generalization needed
3. **Simple models + strong regularization** likely to work better
4. **Series-aware sampling** may help by forcing diverse examples per batch
5. **Target is 52-57%** - beating random and approaching sharpness baseline

Good luck with the search! 🚀
