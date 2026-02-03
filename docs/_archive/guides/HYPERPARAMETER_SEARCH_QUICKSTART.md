# Hyperparameter Search - Quick Start Guide

## ✅ Changes Made

**Series-aware sampling is now the DEFAULT behavior** for `train_multifeature_ranker.py`!

All training runs automatically use series-aware batch sampling unless you explicitly disable it with `--no_series_sampler`.

---

## How to Run Hyperparameter Search

### Option 1: Command Line (Recommended)

```bash
# Quick search (3 experiments, ~2-3 hours total)
python run_hyperparameter_search.py --mode quick

# Full search (10 experiments, ~6-8 hours total)
python run_hyperparameter_search.py --mode full

# Specific experiments only
python run_hyperparameter_search.py --experiments simple_mlp clip_only tiny_mlp
```

### Option 2: VS Code (F5)

Press **F5** and select:
- **"Hyperparameter Search - Quick (3 experiments)"** - Quick search
- **"Hyperparameter Search - Full (10 experiments)"** - Full search

### Option 3: Background Mode

```bash
# Run in background and save output to log
nohup python run_hyperparameter_search.py --mode full > hyperparam_search.log 2>&1 &

# Monitor progress
tail -f hyperparam_search.log
```

---

## What Gets Tested

### Quick Mode (3 experiments, ~2-3 hours)

1. **simple_mlp**: Simpler 1-layer MLP (256)
2. **tiny_mlp**: Very small MLP (128) with larger batches
3. **clip_only**: CLIP features only (no CNN/IQA)

### Full Mode (10 experiments, ~6-8 hours)

All of the above PLUS:
4. **baseline**: Original config (known to overfit - for comparison)
5. **high_dropout**: Higher dropout (0.5)
6. **larger_batches**: Batch size 256
7. **iqa_only**: IQA features only (simplest baseline)
8. **high_lr**: Learning rate 0.001
9. **low_lr**: Learning rate 0.00005
10. **quick_test**: 5 epochs only (fast sanity check)

---

## Output Structure

```
outputs/phototriage_multifeature/hyperparameter_search/
└── search_YYYYMMDD_HHMMSS/
    ├── results.json              # Full results with all metrics
    ├── summary.csv               # Quick summary table ← CHECK THIS FIRST
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

---

## Checking Results

### Quick Check (After Search Completes)

```bash
# View summary table
cat outputs/phototriage_multifeature/hyperparameter_search/search_*/summary.csv

# Or use Python to sort by accuracy
python -c "
import pandas as pd
df = pd.read_csv('outputs/phototriage_multifeature/hyperparameter_search/search_*/summary.csv')
print(df.sort_values('test_accuracy', ascending=False).to_string())
"
```

### What to Look For

**Success criteria**:
- ✅ `test_accuracy` > 0.50 (better than random)
- ✅ `test_accuracy` > 0.52 (good generalization)
- ✅ `test_accuracy` > 0.56 (approaching sharpness baseline)

**Example good result**:
```
experiment    name              test_accuracy  test_loss  duration_min
clip_only     CLIP Only         0.5423        0.9234     32.4
simple_mlp    Simple MLP (256)  0.5381        0.9456     35.1
tiny_mlp      Tiny MLP (128)    0.5212        0.9678     28.7
```

**Example bad result** (still overfitting):
```
experiment    name              test_accuracy  test_loss  duration_min
baseline      Baseline          0.4856        0.9970     42.3  ← Worse than random!
```

---

## Monitoring Progress (While Running)

### Check Which Experiment Is Running

```bash
# View live output
tail -f outputs/phototriage_multifeature/hyperparameter_search/search_*/simple_mlp_*/training.log
```

### Check Overall Progress

The script prints progress like:
```
[1/3] Running experiment: simple_mlp
Experiment: Simple MLP (256)
...
Experiment COMPLETED: test_accuracy=0.5423

[2/3] Running experiment: tiny_mlp
Experiment: Tiny MLP (128)
...
```

---

## Series-Aware Sampling in Hyperparameter Search

**IMPORTANT**: The hyperparameter search script does NOT automatically add `--use_series_sampler` to experiments!

However, since we just made it the **default behavior**, all experiments will automatically use series-aware sampling.

### To Verify

Each experiment should log:
```
Using series-aware batch sampler
BalancedSeriesBatchSampler initialized:
  Total pairs: 9760
  Unique series: 3466
  Batch size: 64
  Shuffle enabled: True
```

---

## Advanced: Customizing Experiments

Edit [run_hyperparameter_search.py](run_hyperparameter_search.py) to add new experiments:

```python
EXPERIMENTS = {
    # Your custom experiment
    "custom_config": {
        "name": "Custom Configuration",
        "args": {
            "--mlp_hidden_dims": ["128"],
            "--batch_size": "64",
            "--learning_rate": "0.0003",
            "--max_epochs": "20",
            "--dropout": "0.4",
        },
        "notes": "Custom config for testing",
    },
}
```

Then run:
```bash
python run_hyperparameter_search.py --experiments custom_config
```

---

## Recommended Workflow

### Step 1: Quick Search (2-3 hours)

```bash
python run_hyperparameter_search.py --mode quick
```

**While it runs**: Go do something else. It will automatically:
- Run 3 experiments
- Save results after each experiment
- Generate summary.csv at the end

### Step 2: Check Results (5 minutes)

```bash
cat outputs/phototriage_multifeature/hyperparameter_search/search_*/summary.csv
```

**Look for**:
- Best test_accuracy
- Which config performed best
- Whether any config exceeded 52% (better than random baseline)

### Step 3: Decide Next Steps

**If best accuracy > 52%**:
- ✅ Found a working config!
- Run it 2-3 more times with different seeds to verify
- Update default config
- Move to improvements (data augmentation, Bradley-Terry loss)

**If best accuracy < 52%**:
- Run full search (`--mode full`)
- Consider more radical changes:
  - Pre-training on random split then fine-tuning
  - Different loss function (Bradley-Terry)
  - Data augmentation

### Step 4: Run Full Search (If Needed)

```bash
python run_hyperparameter_search.py --mode full
```

This tests 10 configurations to find the best one.

---

## Training With Series-Aware Sampling (Default)

Now that series-aware sampling is the default:

```bash
# This automatically uses series-aware sampling
python train_multifeature_ranker.py --mlp_hidden_dims 256

# Quick test (10% of data)
python train_multifeature_ranker.py --quick_experiment 0.1 --mlp_hidden_dims 256

# Disable series-aware sampling (if you want old behavior)
python train_multifeature_ranker.py --no_series_sampler --mlp_hidden_dims 256
```

---

## VS Code Launch Configs

Updated configurations:

| Config Name | Series-Aware? | Quick? | Use For |
|------------|---------------|--------|---------|
| **"Train Multi-Feature Ranker - Quick Experiment (10%) ⚡"** | ✅ YES (default) | ✅ 10% | Quick testing (5-10 min) |
| **"Train Multi-Feature Ranker - Full Dataset ⭐"** | ✅ YES (default) | ❌ Full | Production training (2-3 hrs) |
| **"Train Multi-Feature Ranker - No Series Sampling (OLD)"** | ❌ NO | ❌ Full | Comparison/ablation |
| **"Hyperparameter Search - Quick (3 experiments)"** | ✅ YES (default) | ❌ Full | Find best config (2-3 hrs) |
| **"Hyperparameter Search - Full (10 experiments)"** | ✅ YES (default) | ❌ Full | Comprehensive search (6-8 hrs) |

---

## Summary

### What Changed
- ✅ Series-aware sampling is now **DEFAULT** for all training
- ✅ Use `--no_series_sampler` to disable (not recommended)
- ✅ All hyperparameter search experiments automatically use series-aware sampling

### How to Run Hyperparameter Search
```bash
# Quick (3 experiments, 2-3 hours)
python run_hyperparameter_search.py --mode quick

# Full (10 experiments, 6-8 hours)
python run_hyperparameter_search.py --mode full
```

### Expected Results
- Target: test_accuracy > 52% (better than random 50%)
- Good: test_accuracy > 56% (approaching sharpness baseline)
- Best configs will be saved in summary.csv

### Next Steps
1. Run quick search
2. Check summary.csv for best config
3. If accuracy > 52%, verify with multiple runs
4. If accuracy < 52%, run full search
