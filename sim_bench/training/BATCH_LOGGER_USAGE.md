# Batch Prediction Logger - Usage Guide

## Overview

The Batch Prediction Logger is a lightweight CSV logger for debugging training differences between two models/runs. It logs per-batch predictions (filenames, winners, logits) to help identify where training diverges.

**Location**: `sim_bench/utils/batch_logger.py` - Independent module that can be imported from any framework without circular dependencies.

## Features

- **Simple CSV format**: Easy to compare in Excel, Beyond Compare, or diff tools
- **Per-sample logging**: One row per sample with all details
- **Configurable frequency**: Log every N batches to control file size
- **Independent**: Separate from telemetry system for simplicity

## Usage

### 1. Enable in Config

Add to your YAML config file:

```yaml
# Batch prediction logging (for debugging)
log_batch_predictions: true  # Enable batch prediction logging
log_predictions_every_n: 1   # Log every N batches (1 = log all, 10 = every 10th batch)
```

### 2. Run Training

```bash
python -m sim_bench.training.train_siamese_e2e --config configs/siamese_e2e/resnet50.yaml
```

### 3. Find Output

Predictions are saved to: `outputs/siamese_e2e/<timestamp>/telemetry/batch_predictions.csv`

The file is written **after each batch** (or every N batches if configured), so you can monitor it during training.

## CSV Format

| Column | Description |
|--------|-------------|
| `batch_idx` | Batch index (1-indexed) |
| `epoch` | Current epoch number |
| `sample_idx` | Sample index within batch (0-indexed) |
| `image1` | Filename of first image |
| `image2` | Filename of second image |
| `winner` | Ground truth label (0 or 1) |
| `logit0` | Model logit for class 0 |
| `logit1` | Model logit for class 1 |

## Example Output

```csv
batch_idx,epoch,sample_idx,image1,image2,winner,logit0,logit1
1,0,0,000554-05.jpg,000554-06.jpg,1,-0.1234567890,0.9876543210
1,0,1,002516-01.jpg,002516-06.jpg,0,1.2345678901,-0.8765432109
2,0,0,000357-01.jpg,000357-02.jpg,1,-0.5432109876,0.6543210987
```

## Debugging Training Differences

### Compare Two Runs

1. Run model A with `log_batch_predictions: true`
2. Run model B with same config and seed
3. Compare the CSV files:
   - Use Beyond Compare, Excel, or command-line diff
   - Look for first batch where logits diverge
   - Check if filenames match (batch order)

### Common Issues to Check

- **Different batch order**: Filenames don't match → check seeds/shuffling
- **Different initial logits**: Models initialized differently
- **Logits diverge at batch N**: Training diverged after batch N-1
- **Same filenames, different logits**: Different model weights or forward pass

## Performance Tips

- Set `log_predictions_every_n: 10` or higher for large datasets
- Only enable during debugging (adds I/O overhead)
- CSV files can be large for full training runs (~1MB per 1000 samples)

## Using from Other Frameworks

Simple module-level API (no need to pass logger around):

```python
from sim_bench.utils import batch_logger

# Initialize once at the start
batch_logger.init('predictions.csv', log_every_n=10)

# In your training loop - just call the function
batch_logger.log_batch(batch_idx, epoch, image1_list, image2_list, winners_tensor, logits_tensor)
```

The logger is a module-level singleton, so you don't need to pass it through function parameters.

## Example: Finding Divergence Point

```python
import pandas as pd

# Load predictions from two runs
df_a = pd.read_csv('run_a/batch_predictions.csv')
df_b = pd.read_csv('run_b/batch_predictions.csv')

# Merge on batch_idx, epoch, sample_idx
merged = df_a.merge(df_b, on=['batch_idx', 'epoch', 'sample_idx'], suffixes=('_a', '_b'))

# Compute logit differences
merged['logit0_diff'] = abs(merged['logit0_a'] - merged['logit0_b'])
merged['logit1_diff'] = abs(merged['logit1_a'] - merged['logit1_b'])

# Find first significant divergence (threshold: 0.001)
diverged = merged[(merged['logit0_diff'] > 0.001) | (merged['logit1_diff'] > 0.001)]
print(f"First divergence at batch {diverged.iloc[0]['batch_idx']}")
```

