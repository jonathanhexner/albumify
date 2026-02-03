# Hyperparameter Search Guide

## Overview

After discovering severe overfitting with series-based split (48.6% test accuracy - worse than random!), we need to find better hyperparameters.

**Problem**: Original config (MLP [512, 256], batch=64, LR=0.0001) overfits:
- Epoch 1: Train 52.0%, Val 51.5% ✓
- Epoch 7: Train 61.5%, Val 43.6% ❌ (early stop)
- Test: 48.6% (worse than random 50%)

## Quick Start

### Run Quick Search (Recommended First)

```bash
python run_hyperparameter_search.py --mode quick
```

This runs 3 promising configurations:
1. **simple_mlp**: Simpler 1-layer MLP (256)
2. **tiny_mlp**: Very small MLP (128) with larger batches
3. **clip_only**: CLIP features only (no CNN/IQA)

**Time**: ~2-3 hours total (30 min per experiment)

### Run Full Search

```bash
python run_hyperparameter_search.py --mode full
```

Runs all 10 experiments. **Time**: ~6-8 hours

### Run Specific Experiments

```bash
python run_hyperparameter_search.py --experiments simple_mlp clip_only high_dropout
```

### Resume Interrupted Search

```bash
python run_hyperparameter_search.py --mode full --resume
```

## Experiment Configurations

### Recommended Experiments

#### 1. **simple_mlp** (Best First Try)
```python
{
    "mlp_hidden_dims": [256],        # Simpler: 1540 → 256 → 1
    "batch_size": 64,
    "learning_rate": 0.0001,
    "max_epochs": 30,
}
```
**Why**: Reduces model capacity to prevent overfitting

#### 2. **tiny_mlp** (Aggressive Simplification)
```python
{
    "mlp_hidden_dims": [128],        # Very simple: 1540 → 128 → 1
    "batch_size": 128,               # Larger batches for regularization
    "learning_rate": 0.0005,         # Higher LR for smaller model
    "max_epochs": 30,
}
```
**Why**: Even simpler model + larger batches = strong regularization

#### 3. **clip_only** (Feature Ablation)
```python
{
    "mlp_hidden_dims": [256],
    "batch_size": 128,
    "learning_rate": 0.0005,
    "use_clip": true,
    "use_cnn_features": false,       # Disable CNN features
    "use_iqa_features": false,       # Disable IQA features
}
```
**Why**: CLIP alone (512-dim) may be sufficient and less prone to overfitting

### Other Experiments

#### **high_dropout** (Regularization)
```python
{
    "mlp_hidden_dims": [512, 256],
    "dropout": 0.5,                  # Up from 0.3
}
```

#### **larger_batches** (Implicit Regularization)
```python
{
    "mlp_hidden_dims": [256],
    "batch_size": 256,               # Much larger
    "learning_rate": 0.001,
}
```

#### **iqa_only** (Simplest Baseline)
```python
{
    "mlp_hidden_dims": [32],         # Tiny for 4-dim input
    "use_clip": false,
    "use_cnn_features": false,
    # Only IQA features (sharpness, exposure, contrast, colorfulness)
}
```

## Search Modes

| Mode | Experiments | Focus | Time |
|------|-------------|-------|------|
| `quick` | 3 | Capacity reduction + feature ablation | ~2-3 hours |
| `regularization` | 3 | Different regularization strategies | ~2-3 hours |
| `features` | 3 | Feature combinations | ~2-3 hours |
| `learning_rate` | 3 | LR variations | ~2-3 hours |
| `full` | 10 | All experiments | ~6-8 hours |

## Output Structure

```
outputs/hyperparameter_search/
└── search_YYYYMMDD_HHMMSS/
    ├── results.json              # Full results with all metrics
    ├── summary.csv               # Quick summary table
    ├── simple_mlp_YYYYMMDD_HHMMSS/
    │   ├── best_model.pt
    │   ├── training.log
    │   ├── training_curves.png
    │   ├── test_results.json
    │   └── detailed_predictions.csv
    ├── tiny_mlp_YYYYMMDD_HHMMSS/
    │   └── ...
    └── clip_only_YYYYMMDD_HHMMSS/
        └── ...
```

## Analyzing Results

After search completes, check `summary.csv`:

```bash
# View summary
cat outputs/hyperparameter_search/search_*/summary.csv

# Or use pandas
python -c "import pandas as pd; df = pd.read_csv('outputs/hyperparameter_search/search_*/summary.csv'); print(df.sort_values('test_accuracy', ascending=False))"
```

**Key columns**:
- `test_accuracy`: Test set accuracy (target: >50%, ideally 55-60%)
- `test_loss`: Test loss
- `duration_min`: Training time in minutes

## Expected Results

Based on the overfitting pattern, we expect:

| Configuration | Expected Accuracy | Rationale |
|---------------|-------------------|-----------|
| **simple_mlp** | 52-56% | Simpler model = less overfitting |
| **tiny_mlp** | 50-54% | May be too simple, but won't overfit |
| **clip_only** | 53-57% | CLIP features are powerful, less prone to overfitting |
| **iqa_only** | 52-55% | Very simple baseline (like sharpness-only: 56.4%) |
| **baseline** | 48-50% | Known to overfit badly |

**Success criteria**:
- ✓ Test accuracy > 50% (better than random)
- ✓ Val accuracy doesn't degrade during training
- ✓ Train-Val gap < 5%

## Series-Aware Batch Sampling (Experimental)

Add `--use_series_sampler` to any experiment:

```bash
python run_hyperparameter_search.py --experiments simple_mlp --use_series_sampler
```

**What it does**: Ensures each batch contains pairs from different series, preventing the model from seeing multiple pairs from the same series in one batch.

**Why it might help**: Prevents overfitting to series-specific patterns by forcing the model to learn from diverse examples in each batch.

## Troubleshooting

### "Experiment FAILED"
Check the individual experiment's `training.log`:
```bash
cat outputs/hyperparameter_search/search_*/EXPERIMENT_NAME_*/training.log
```

### Out of Memory
Reduce batch size or use simpler model:
```bash
python train_multifeature_ranker.py --mlp_hidden_dims 128 --batch_size 32
```

### Training Too Slow (CPU)
Use quicker experiments or reduce max_epochs:
```python
EXPERIMENTS["quick_test"] = {
    "mlp_hidden_dims": [128],
    "batch_size": 128,
    "max_epochs": 5,  # Very fast
}
```

## Next Steps After Finding Best Config

1. **Verify best config** by running 3 times with different seeds
2. **Update default config** in `phototriage_multifeature.py`
3. **Run analysis** with Streamlit app:
   ```bash
   streamlit run analyze_phototriage_results.py
   ```
4. **Consider next improvements**:
   - Data augmentation (synthetic degradations)
   - Bradley-Terry loss
   - Facial features
   - Pre-training on random split

## Key Insights

**Why series-based split is harder**:
- Random split: Model can exploit series-specific patterns (e.g., "this photographer prefers warm tones")
- Series-based split: Model must learn generalizable quality features

**Why we're overfitting**:
- Limited effective training data (3,466 series, but each with 2-3 pairs on average)
- High model capacity (1540 input features → [512, 256])
- Series-specific patterns are easy to memorize but don't generalize

**Solution**: Simpler models + stronger regularization + better sampling
