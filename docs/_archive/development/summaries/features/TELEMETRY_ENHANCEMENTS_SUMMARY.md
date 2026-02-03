# Telemetry Enhancements - Implementation Summary

**Date**: 2026-01-13  
**Status**: ✅ COMPLETE

---

## Overview

Enhanced the telemetry system with comprehensive weight tracking and analysis tools to enable deep debugging of model training issues.

---

## ✅ Completed Features

### 1. Weight Snapshots

**Added**: Full model checkpoint saving at configurable intervals

**Files Modified**:
- `sim_bench/telemetry/config.py`: Added `track_weight_snapshots` and `weight_snapshot_every_n` options
- `sim_bench/telemetry/training_telemetry.py`: Added `_save_weight_snapshot()` method

**Configuration**:
```yaml
telemetry:
  enabled: true
  track_weight_snapshots: true
  weight_snapshot_every_n: 500  # Save every 500 batches
```

**Output**: 
- Snapshots saved to: `outputs/<exp>/telemetry/weight_snapshots/weights_epoch{N}_batch{M}.pt`
- Each snapshot contains:
  - `model_state_dict`: Full model weights
  - `batch_idx`: Training step
  - `epoch`: Current epoch

**Storage**: ~100MB per snapshot (depends on model size)

---

### 2. Per-Layer Weight Statistics

**Added**: Detailed statistics for each model layer

**Files Modified**:
- `sim_bench/telemetry/config.py`: Added `track_per_layer_stats` option
- `sim_bench/telemetry/training_telemetry.py`: Added `_compute_per_layer_stats()` method

**Configuration**:
```yaml
telemetry:
  enabled: true
  track_per_layer_stats: true
```

**Output**:
- CSV file: `outputs/<exp>/telemetry/per_layer_stats.csv`
- Columns:
  - `batch_idx`, `epoch`: Training progress
  - `layer_name`: Parameter name
  - `mean`, `std`, `min`, `max`: Distribution statistics
  - `l2_norm`: Overall magnitude
  - `num_params`: Layer size

**Use Cases**:
- Identify layers with abnormal weight distributions
- Track layer-specific learning dynamics
- Detect gradient vanishing/exploding in specific layers

---

### 3. Enhanced Batch Statistics

**Status**: ✅ Already implemented (confirmed)

**Existing Features**:
- `batch_stats.csv` already tracks `winner_0_pct` and `winner_1_pct`
- Shows label distribution per batch
- Helps detect data bias

**No changes needed** - this feature was already complete!

---

### 4. Weight Comparison Tool

**Created**: `scripts/compare_weights.py`

**Purpose**: Compare weight evolution between two experiments

**Features**:
- Per-layer L2 distance (how much weights differ)
- Per-layer cosine similarity (same direction?)
- Weight distribution comparisons
- Divergence timeline
- Automated visualization generation

**Usage**:
```bash
python scripts/compare_weights.py \
    --exp1 outputs/siamese_e2e/20260113_073023 \
    --exp2 outputs/siamese_e2e/20260111_005327 \
    --output outputs/weight_comparison
```

**Outputs**:
- `layer_divergence_detailed.csv`: Full per-layer comparison
- `summary_stats.csv`: Aggregated statistics
- `divergence_heatmap.png`: Layer × checkpoint heatmap
- `divergence_timeline.png`: Average divergence over time
- `layer_ranking.png`: Top 20 most divergent layers
- `SUMMARY.txt`: Human-readable summary

**Requirements**:
- Both experiments must have weight snapshots enabled
- Snapshots must be aligned (same checkpoint indices)

---

### 5. Debug Notebook

**Created**: `notebooks/debug_dataloader_issue.ipynb`

**Purpose**: Interactive analysis of training issues

**Sections**:
1. **Training Curves**: Side-by-side loss/accuracy comparison
2. **Gradient Norms**: Optimization dynamics analysis
3. **Batch Statistics**: Label distribution tracking
4. **Holdout Predictions**: Model confidence and accuracy evolution
5. **Summary & Conclusions**: Automated diagnosis

**Features**:
- Loads telemetry data from both experiments
- Generates comprehensive visualizations
- Automatically detects label bias
- Provides actionable conclusions

**Usage**:
```bash
jupyter notebook notebooks/debug_dataloader_issue.ipynb
```

---

## 📊 How to Use

### Enable Enhanced Telemetry

Add to your config YAML:

```yaml
telemetry:
  enabled: true
  collect_every_n: 10
  
  # Basic metrics (already available)
  track_gradients: true
  track_weight_delta: true
  track_learning_rates: true
  track_holdout_logits: true
  track_batch_stats: true
  
  # NEW: Weight snapshots
  track_weight_snapshots: true
  weight_snapshot_every_n: 500
  
  # NEW: Per-layer stats
  track_per_layer_stats: true
  
  # Holdout settings
  holdout_size: 50
```

### Run Training with Telemetry

```bash
python sim_bench/training/train_siamese_e2e.py \
    --config configs/siamese_e2e/resnet50.yaml
```

### Analyze Results

**Option 1: Compare weights between experiments**
```bash
python scripts/compare_weights.py \
    --exp1 outputs/siamese_e2e/exp1 \
    --exp2 outputs/siamese_e2e/exp2
```

**Option 2: Interactive notebook**
```bash
jupyter notebook notebooks/debug_dataloader_issue.ipynb
```

---

## 🎯 Key Benefits

### 1. Identify Divergence Points
- Weight snapshots show **exactly when** models start to differ
- Pin-point critical training steps

### 2. Per-Layer Insights
- See which layers learn differently
- Identify problematic layers (vanishing gradients, dead neurons)

### 3. Label Bias Detection
- Batch stats reveal data distribution issues
- Notebook automatically flags bias problems

### 4. Comprehensive Comparison
- Side-by-side analysis of experiments
- Visualizations make patterns obvious

---

## 🔧 Storage Considerations

### Weight Snapshots

**Default**: Save every 500 batches

**Example calculation**:
- Model size: ~100MB
- Batches per epoch: 300
- Snapshots per epoch: 300 / 500 ≈ 1 (rounded up to nearest)
- 3 epochs × 1 snapshot = 3 snapshots
- **Total**: ~300MB

**Adjust frequency** based on storage:
```yaml
weight_snapshot_every_n: 1000  # Less frequent → less storage
```

### Per-Layer Stats

**Size**: Minimal (~1KB per collection step)

**Example**:
- 100 layers × 50 collections = 5,000 rows
- CSV size: ~500KB

**No storage concerns** - very lightweight

---

## 📁 Output Structure

```
outputs/siamese_e2e/<experiment>/
├── telemetry/
│   ├── gradient_norms.csv
│   ├── weight_deltas.csv
│   ├── learning_rates.csv
│   ├── batch_stats.csv
│   ├── holdout_predictions.csv
│   ├── per_layer_stats.csv          ← NEW
│   └── weight_snapshots/             ← NEW
│       ├── weights_epoch0_batch500.pt
│       ├── weights_epoch0_batch1000.pt
│       └── ...
├── training_history.json
├── training_curves.png
└── config.yaml
```

---

## 🚀 Next Steps

### Recommended Workflow

1. **Enable telemetry** in both experiment configs
2. **Run training** with weight snapshots enabled
3. **Compare experiments** using `compare_weights.py`
4. **Analyze interactively** using the debug notebook
5. **Identify root cause** from visualizations and statistics

### Example Investigation

**Problem**: Internal (51.8%) underperforms External (69.6%)

**Steps**:
```bash
# 1. Run weight comparison
python scripts/compare_weights.py \
    --exp1 outputs/siamese_e2e/20260113_073023 \
    --exp2 outputs/siamese_e2e/20260111_005327

# 2. Open debug notebook
jupyter notebook notebooks/debug_dataloader_issue.ipynb

# 3. Check batch_stats.csv for label bias
# 4. Review weight divergence heatmap
# 5. Identify which layers diverge most
```

---

## ✅ Testing

All features tested and working:

- ✅ Weight snapshots saved correctly
- ✅ Per-layer stats captured
- ✅ Comparison script generates all plots
- ✅ Notebook runs without errors
- ✅ Storage overhead is acceptable

---

## 📚 References

**Modified Files**:
- `sim_bench/telemetry/config.py`
- `sim_bench/telemetry/training_telemetry.py`

**Created Files**:
- `scripts/compare_weights.py`
- `notebooks/debug_dataloader_issue.ipynb`

**Documentation**:
- `sim_bench/telemetry/ARCHITECTURE.md` (existing)
- This summary document

---

## 🎉 Summary

All telemetry enhancements are **complete and ready to use**!

The system now provides:
- Deep insight into weight evolution
- Automated comparison tools
- Interactive analysis notebooks
- Comprehensive debugging capabilities

These tools will help diagnose training issues, compare experiments, and understand model behavior in detail.
