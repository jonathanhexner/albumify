# Command-Line Overrides Solution - The Clean Way

## Problem Solved!

Instead of copying/modifying YAML configs in the Kaggle notebook, we now support **command-line overrides** so you can use the YAML files **directly from the repo**.

---

## Changes Made

### 1. Added Command-Line Overrides to Training Scripts

**Both `train_siamese_e2e.py` and `train_frozen.py` now support:**

```bash
--config      <path>    # Path to YAML config (required)
--output-dir  <path>    # Override output directory
--data-dir    <path>    # Override data root directory (NEW!)
--device      <device>  # Override device (cpu/cuda) (NEW!)
--quick-experiment <fraction>  # Use fraction of data
```

### 2. Updated Kaggle Notebook

**Cell 8 - Simplified (No more config recreation!):**

```python
import yaml
from pathlib import Path

# Check available configs
configs_dir = Path('/kaggle/working/sim-bench/configs/siamese_e2e')

print("Available configs:")
for config_file in configs_dir.glob('*.yaml'):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\n✓ {config_file.name}")
    print(f"  Model: {config['model']['cnn_backbone']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Differential LR: {config['training'].get('differential_lr', False)}")

print("\nWe'll use these configs directly with command-line overrides!")
```

**Cell 10 - Quick Test:**

```bash
!python -m sim_bench.training.train_siamese_e2e \
    --config configs/siamese_e2e/vgg16.yaml \
    --data-dir /kaggle/input/triage \
    --device cuda \
    --output-dir /kaggle/working/outputs/quick_test \
    --quick-experiment 0.1
```

**Cell 12 - Full Training:**

```bash
# Train VGG16 (paper replication)
!python -m sim_bench.training.train_siamese_e2e \
    --config configs/siamese_e2e/vgg16.yaml \
    --data-dir /kaggle/input/triage \
    --device cuda \
    --output-dir /kaggle/working/outputs/siamese_e2e_vgg16

# Or train ResNet50 (uncomment to use)
# !python -m sim_bench.training.train_siamese_e2e \
#     --config configs/siamese_e2e/resnet50.yaml \
#     --data-dir /kaggle/input/triage \
#     --device cuda \
#     --output-dir /kaggle/working/outputs/siamese_e2e_resnet50
```

---

## Benefits

### ✅ Single Source of Truth
- YAML configs live in the repo
- No duplication or manual copying
- Update once, works everywhere

### ✅ Consistency
- Local training: Uses YAMLs directly
- Kaggle training: Uses same YAMLs with overrides
- Same differential LR settings everywhere

### ✅ Flexibility
- Override any parameter from command line
- No need to edit configs for platform-specific settings
- Easy to experiment with different settings

### ✅ Maintainability
- Less code in the notebook
- Clearer what's being overridden
- Easier to debug

---

## Usage Examples

### Local Training (Original)

```bash
# Use config as-is
python -m sim_bench.training.train_siamese_e2e \
    --config configs/siamese_e2e/resnet50.yaml
```

### Kaggle Training (Override paths)

```bash
# Override data directory and device
python -m sim_bench.training.train_siamese_e2e \
    --config configs/siamese_e2e/resnet50.yaml \
    --data-dir /kaggle/input/triage \
    --device cuda \
    --output-dir /kaggle/working/outputs/resnet50
```

### Quick Experiment (Override data fraction)

```bash
# Use 10% of data for quick testing
python -m sim_bench.training.train_siamese_e2e \
    --config configs/siamese_e2e/resnet50.yaml \
    --quick-experiment 0.1
```

### Custom Output Directory

```bash
# Save results to custom location
python -m sim_bench.training.train_siamese_e2e \
    --config configs/siamese_e2e/resnet50.yaml \
    --output-dir experiments/resnet50_run_1
```

---

## What Gets Overridden?

| Config Value | Original (YAML) | Override Flag | Kaggle Value |
|--------------|----------------|---------------|--------------|
| `data.root_dir` | `D:\Similar Images\...` | `--data-dir` | `/kaggle/input/triage` |
| `device` | `cpu` | `--device` | `cuda` |
| `output_dir` | `outputs/siamese_e2e_resnet50` | `--output-dir` | `/kaggle/working/outputs/...` |
| `data.quick_experiment` | `null` | `--quick-experiment` | `0.1` (for testing) |

**Everything else** (batch size, learning rate, differential_lr, etc.) comes from the YAML file unchanged!

---

## Comparison: Before vs After

### Before (Duplicated Config in Notebook) ❌

```python
# Cell 8: 50+ lines recreating the config in Python
vgg16_config = {
    'name': 'siamese_e2e_vgg16_kaggle',
    'data': {...},
    'model': {...},
    'training': {
        'batch_size': 32,          # ❌ Wrong!
        'learning_rate': 0.001,    # ❌ Wrong!
        # ... manual copying/pasting from YAML
    },
    # ... more duplication
}
yaml.dump(vgg16_config, ...)
```

**Problems:**
- Config values out of sync with repo YAMLs
- Manual copying = errors (wrong batch size, missing differential_lr)
- Hard to maintain

### After (Command-Line Overrides) ✅

```python
# Cell 8: Simple verification (5 lines)
print("Using configs from: configs/siamese_e2e/")
for config in ['vgg16.yaml', 'resnet50.yaml']:
    print(f"  ✓ {config} - ready to use!")
```

```bash
# Training: Override only what's needed
!python -m sim_bench.training.train_siamese_e2e \
    --config configs/siamese_e2e/vgg16.yaml \
    --data-dir /kaggle/input/triage \
    --device cuda
```

**Benefits:**
- Config values stay in sync automatically
- No manual copying
- Clear what's being overridden
- Easy to maintain

---

## Summary

**Your suggestion was spot-on!** We now:

1. ✅ Use the same YAML files everywhere (no copying)
2. ✅ Support command-line overrides for platform-specific settings
3. ✅ Maintain a single source of truth for configs
4. ✅ Ensure consistency between local and Kaggle training

**The Kaggle notebook is now much simpler** - just verify configs exist and run training with appropriate overrides. No more config duplication!

---

## Files Modified

- ✅ `sim_bench/training/train_siamese_e2e.py` - Added `--data-dir` and `--device` flags
- ✅ `sim_bench/training/train_frozen.py` - Added `--data-dir` and `--device` flags
- ✅ `kaggle_siamese_training.ipynb` - Simplified to use overrides instead of config duplication

All training scripts now support the same override flags for maximum flexibility!


















