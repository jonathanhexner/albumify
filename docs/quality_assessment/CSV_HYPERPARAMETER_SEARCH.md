# CSV-Based Hyperparameter Search Guide

## Overview

The hyperparameter search now uses **CSV input files** for experiment configuration and produces **detailed CSV output files** tracking all metrics.

## Quick Start

```bash
# Run all experiments defined in CSV
python run_hyperparameter_search.py

# Run specific experiments
python run_hyperparameter_search.py --experiments simple_mlp clip_only tiny_mlp

# Use custom CSV file
python run_hyperparameter_search.py --config configs/my_experiments.csv
```

---

## Input: Experiment Configuration CSV

### Location
[configs/hyperparameter_experiments.csv](../../configs/hyperparameter_experiments.csv)

### Format

```csv
experiment_name,mlp_hidden_dims,batch_size,learning_rate,max_epochs,dropout,use_clip,use_cnn_features,use_iqa_features,notes
simple_mlp,256,64,0.0001,30,0.3,true,true,true,Simpler 1-layer MLP to reduce overfitting
clip_only,256,128,0.0005,30,0.3,true,false,false,CLIP features only (512-dim)
```

### Required Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `experiment_name` | string | Unique identifier | `simple_mlp` |
| `mlp_hidden_dims` | string | Comma-separated dimensions | `256` or `512,256` |
| `batch_size` | int | Batch size | `64` |
| `learning_rate` | float | Learning rate | `0.0001` |
| `max_epochs` | int | Maximum epochs | `30` |

### Optional Columns

| Column | Type | Description | Default |
|--------|------|-------------|---------|
| `dropout` | float | Dropout rate | `0.3` |
| `use_clip` | bool | Use CLIP features | `true` |
| `use_cnn_features` | bool | Use CNN features | `true` |
| `use_iqa_features` | bool | Use IQA features | `true` |
| `notes` | string | Description | (empty) |

---

## Output: Results CSV

### Location
`outputs/phototriage_multifeature/hyperparameter_search/search_YYYYMMDD_HHMMSS/results.csv`

### Columns

| Column | Description |
|--------|-------------|
| `experiment_name` | Name from input CSV |
| `timestamp` | When experiment started |
| `success` | True/False |
| `error` | Error message if failed |
| `duration_seconds` | Total training time (seconds) |
| `duration_minutes` | Total training time (minutes) |
| `output_dir` | Path to experiment output |
| **Configuration** | |
| `mlp_hidden_dims` | MLP architecture |
| `batch_size` | Batch size used |
| `learning_rate` | Learning rate used |
| `max_epochs` | Max epochs configured |
| `dropout` | Dropout rate |
| **Training Metrics** | |
| `final_epoch` | Epoch where training stopped |
| `best_val_acc` | Best validation accuracy |
| `best_val_loss` | Best validation loss |
| **Test Metrics** | |
| `test_acc` | Test accuracy (PRIMARY METRIC) |
| `test_loss` | Test loss |
| `notes` | Notes from input CSV |

### Example Output

```csv
experiment_name,timestamp,success,test_acc,best_val_acc,final_epoch,duration_minutes,mlp_hidden_dims,batch_size,learning_rate
clip_only,2025-11-30 14:23:10,True,0.5423,0.5381,12,32.4,256,128,0.0005
simple_mlp,2025-11-30 15:01:45,True,0.5381,0.5312,15,35.1,256,64,0.0001
tiny_mlp,2025-11-30 15:40:22,True,0.5212,0.5156,10,28.7,128,128,0.0005
baseline,2025-11-30 16:15:03,True,0.4856,0.4987,7,42.3,"512,256",64,0.0001
```

---

## Usage Examples

### 1. Run All Experiments

```bash
python run_hyperparameter_search.py
```

**What it does**:
- Loads all experiments from `configs/hyperparameter_experiments.csv`
- Runs them sequentially
- Saves results after each experiment to `results.csv`

### 2. Run Specific Experiments

```bash
python run_hyperparameter_search.py --experiments simple_mlp clip_only
```

**What it does**:
- Only runs experiments named `simple_mlp` and `clip_only`
- Faster for testing specific configurations

### 3. Use Custom CSV File

```bash
python run_hyperparameter_search.py --config configs/my_custom_experiments.csv
```

**What it does**:
- Loads experiments from your custom CSV file
- Must follow the same format

### 4. Resume Interrupted Search

```bash
python run_hyperparameter_search.py --resume
```

**What it does**:
- Loads existing `results.csv`
- Skips already completed experiments
- Continues from where it left off

---

## VS Code Integration

### Option 1: Run All Experiments
Press **F5** → Select **"Hyperparameter Search - All Experiments (CSV)"**

### Option 2: Run Selected Experiments
Press **F5** → Select **"Hyperparameter Search - Selected Experiments"**

Edit [.vscode/launch.json](../../.vscode/launch.json) to change which experiments:
```json
{
    "name": "Hyperparameter Search - Selected Experiments",
    "args": [
        "--experiments", "simple_mlp", "clip_only", "tiny_mlp"
    ]
}
```

---

## Adding New Experiments

### Method 1: Edit CSV File

Open [configs/hyperparameter_experiments.csv](../../configs/hyperparameter_experiments.csv) and add a row:

```csv
my_experiment,128,64,0.0003,20,0.4,true,true,true,My custom configuration
```

### Method 2: Create New CSV File

```csv
experiment_name,mlp_hidden_dims,batch_size,learning_rate,max_epochs,notes
exp1,256,64,0.0001,30,First test
exp2,128,128,0.0005,20,Second test
```

Save as `configs/my_experiments.csv`, then:

```bash
python run_hyperparameter_search.py --config configs/my_experiments.csv
```

---

## Checking Results

### During Execution

Results are saved **after each experiment** to:
```
outputs/phototriage_multifeature/hyperparameter_search/search_YYYYMMDD_HHMMSS/results.csv
```

You can open this file while the search is running to see completed experiments.

### After Completion

```bash
# View results
cat outputs/phototriage_multifeature/hyperparameter_search/search_*/results.csv

# Sort by test accuracy (best first)
python -c "
import pandas as pd
df = pd.read_csv('outputs/phototriage_multifeature/hyperparameter_search/search_*/results.csv')
print(df[['experiment_name', 'test_acc', 'best_val_acc', 'duration_minutes']].sort_values('test_acc', ascending=False))
"
```

### In Excel/Google Sheets

Simply open `results.csv` in your favorite spreadsheet application.

---

## Advanced Features

### Incremental Results

Results are saved after **each experiment**, not at the end. This means:
- ✅ You can monitor progress in real-time
- ✅ If search crashes, you don't lose results
- ✅ You can resume with `--resume`

### Resume Capability

```bash
# First run (interrupted after 3 experiments)
python run_hyperparameter_search.py
# Ctrl+C after experiment 3

# Resume (skips first 3, continues from 4)
python run_hyperparameter_search.py --resume
```

### Parallel Feature Caching

Because experiments share the cache directory:
- First experiment extracts all features (~30-60 min)
- Subsequent experiments reuse cached features (instant)
- Saves hours of computation time!

---

## Experiment Design Tips

### Start Small

```csv
experiment_name,mlp_hidden_dims,batch_size,learning_rate,max_epochs,notes
quick_test,128,128,0.0005,5,Fast sanity check (5-10 min)
```

Run this first to verify everything works.

### Grid Search Example

```csv
experiment_name,mlp_hidden_dims,batch_size,learning_rate,max_epochs
mlp256_bs64_lr1e4,256,64,0.0001,30
mlp256_bs64_lr5e4,256,64,0.0005,30
mlp256_bs128_lr1e4,256,128,0.0001,30
mlp256_bs128_lr5e4,256,128,0.0005,30
mlp128_bs64_lr1e4,128,64,0.0001,30
mlp128_bs64_lr5e4,128,64,0.0005,30
```

Systematic exploration of hyperparameter space.

### Feature Ablation Study

```csv
experiment_name,mlp_hidden_dims,use_clip,use_cnn_features,use_iqa_features,notes
all_features,256,true,true,true,All features (1540-dim)
clip_only,256,true,false,false,CLIP only (512-dim)
cnn_only,256,false,true,false,CNN only (1024-dim)
iqa_only,32,false,false,true,IQA only (4-dim)
clip_cnn,256,true,true,false,CLIP + CNN (1536-dim)
clip_iqa,256,true,false,true,CLIP + IQA (516-dim)
```

Understand which features are most important.

---

## Troubleshooting

### Error: "Missing required columns in CSV"

Make sure your CSV has these columns:
- `experiment_name`
- `mlp_hidden_dims`
- `batch_size`
- `learning_rate`
- `max_epochs`

### Error: "No experiments found matching"

Check that `experiment_name` in your `--experiments` argument matches the CSV exactly.

### Experiment Shows "success: False"

Check the `error` column in results.csv for details, or look at the experiment's `training.log`.

---

## Complete Workflow Example

```bash
# 1. Edit experiment CSV
nano configs/hyperparameter_experiments.csv

# 2. Run quick test first (single experiment, 5 epochs)
python run_hyperparameter_search.py --experiments quick_test

# 3. Check if it worked
cat outputs/phototriage_multifeature/hyperparameter_search/search_*/results.csv

# 4. Run full search (all experiments)
python run_hyperparameter_search.py

# 5. Monitor progress (in another terminal)
tail -f outputs/phototriage_multifeature/hyperparameter_search/search_*/results.csv

# 6. After completion, find best config
python -c "
import pandas as pd
df = pd.read_csv('outputs/phototriage_multifeature/hyperparameter_search/search_*/results.csv')
best = df.loc[df['test_acc'].idxmax()]
print(f'Best: {best[\"experiment_name\"]} with test_acc={best[\"test_acc\"]:.4f}')
print(f'Config: MLP={best[\"mlp_hidden_dims\"]}, BS={best[\"batch_size\"]}, LR={best[\"learning_rate\"]}')
"
```

---

## Summary

### Input: CSV File
- **Location**: `configs/hyperparameter_experiments.csv`
- **Format**: One row per experiment with configuration
- **Easy to edit**: Use any text editor or Excel

### Output: CSV File
- **Location**: `outputs/.../search_YYYYMMDD_HHMMSS/results.csv`
- **Format**: One row per experiment with all metrics
- **Updated live**: After each experiment completes
- **Columns**: experiment_name, test_acc, best_val_acc, duration_minutes, configuration, notes

### Key Features
- ✅ **Incremental saves**: Results saved after each experiment
- ✅ **Resume capability**: Continue interrupted searches
- ✅ **Flexible filtering**: Run all or selected experiments
- ✅ **Custom configs**: Use any CSV file
- ✅ **Rich metrics**: Train/val/test scores, timing, configuration
