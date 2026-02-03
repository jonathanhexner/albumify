# Per-Batch Model Comparison Guide

This guide explains how to compare two models trained in parallel by dumping and analyzing their states.

## Overview

Instead of comparing against a frozen reference model (which tells us nothing), we:
1. Train both models independently on the same data
2. Dump model states after each batch
3. Compare the dumps offline to see where they diverge

This keeps both codebases clean and separate while allowing detailed comparison.

## Step 1: Enable Batch Dumping in Our Code

Edit `configs/siamese_e2e/resnet50.yaml`:

```yaml
batch_comparison_interval: 10  # Dump every 10 batches
```

Then run training:

```bash
python -m sim_bench.training.train_siamese_e2e --config configs/siamese_e2e/resnet50.yaml
```

This creates dumps in: `outputs/siamese_e2e/TIMESTAMP/batch_dumps/epoch_XXX/batch_XXXX.pt`

## Step 2: Train Reference Model with Dumps

The reference model uses different code. We wrap it minimally to add dumping:

```bash
python train_reference_with_dumps.py \
    --output_dir outputs/reference_model \
    --batch_dump_interval 10
```

**Note**: You need to implement `make_loader()` in `train_reference_with_dumps.py` to match their data loading.

This creates dumps in: `outputs/reference_model/batch_dumps/epoch_XXX/batch_XXXX.pt`

## Step 3: Compare the Models

Once both models are trained (or after a few epochs), compare them:

```bash
python -m sim_bench.training.analyze_batch_comparisons \
    outputs/siamese_e2e/20231215_120000 \
    outputs/reference_model \
    --filter all
```

Options for `--filter`:
- `all`: Compare all parameters (backbone + head)
- `mlp`: Compare only MLP head parameters
- `backbone`: Compare only backbone parameters

## What You Get

The comparison produces:

### 1. Visual Plots (`model_comparison.png`)
6 subplots showing:
- Weight divergence (L2 distance over time)
- Weight similarity (cosine similarity over time)
- Loss comparison (both models on same data)
- Loss difference (absolute difference)
- Accuracy comparison
- Accuracy difference

### 2. CSV Data (`model_comparison.csv`)
Full time series of all metrics for further analysis

### 3. JSON Report (`detailed_comparison.json`)
Per-checkpoint, per-parameter detailed comparison

### 4. Summary Statistics
Printed to console:
- Average/max L2 distance
- Average cosine similarity
- Loss/accuracy divergence statistics

## What This Tells You

- **Weight divergence**: If L2 distance grows quickly, models are learning different features
- **Loss/accuracy divergence**: If losses differ significantly, one model is optimizing better
- **Sudden jumps**: Indicate gradient issues or learning rate problems
- **Consistent patterns**: If one model consistently has lower loss, investigate why

## Example Workflow

```bash
# 1. Train our model with dumps
python -m sim_bench.training.train_siamese_e2e \
    --config configs/siamese_e2e/resnet50.yaml

# 2. Train reference model with dumps (in parallel or sequentially)
python train_reference_with_dumps.py \
    --output_dir outputs/reference \
    --batch_dump_interval 10

# 3. Compare after training (or mid-training)
python -m sim_bench.training.analyze_batch_comparisons \
    outputs/siamese_e2e/20231215_120000 \
    outputs/reference \
    --filter mlp

# 4. Analyze the plots and identify divergence points
```

## Storage Considerations

Dumping every batch creates many files:
- Each dump: ~100MB (depends on model size)
- 10 batches/epoch × 100 epochs = 1000 dumps
- Total: ~100GB

**Recommendations:**
- Start with `batch_comparison_interval: 10` (every 10 batches)
- Only dump for first few epochs to identify early divergence
- Delete dumps after analysis

## Implementation Details

### Our Model Dumps
- Location: `train_siamese_e2e.py:146-158`
- Format: PyTorch checkpoint with `model_state_dict`, `optimizer_state_dict`, `loss`, `acc`

### Reference Model Wrapper
- Location: `train_reference_with_dumps.py`
- Wraps their training loop with minimal changes (just adds torch.save)

### Comparison Logic
- Location: `analyze_batch_comparisons.py`
- Matches dumps by (epoch, batch) key
- Computes L2, cosine similarity, max absolute difference
- Filters parameters (all/mlp/backbone)
