# Training Scripts

Clean, simple training scripts for pairwise image quality ranking using Siamese networks.

## Overview

All training uses **Siamese architecture** (shared weights) with an MLP comparison head. The key difference is whether the feature extractor (CNN/CLIP) is frozen or trainable.

### Architecture

```
Image1 → Feature Extractor (shared) → embedding1 ─┐
                                                   ├→ MLP → P(img1 > img2)
Image2 → Feature Extractor (shared) → embedding2 ─┘
```

## Scripts

### 1. `train_frozen.py` - Frozen Features (Fast)

Pre-extracts features once, caches them, then trains only the MLP head.

**Use when:**
- You want fast experimentation
- Testing different MLP architectures
- Combining multiple feature types (CLIP + CNN + IQA)

**Features:**
- CLIP embeddings (512-dim)
- ResNet50 layer4 (2048-dim) or layer3 (1024-dim)
- VGG16 features (4096-dim)
- IQA metrics (4-dim: sharpness, brightness, contrast, color variance)

**Usage:**
```bash
# Multi-feature fusion
python -m sim_bench.training.train_frozen --config configs/frozen/multifeature.yaml

# Single feature type
python -m sim_bench.training.train_frozen --config configs/frozen/resnet50.yaml

# Quick test (10% of data)
python -m sim_bench.training.train_frozen --config configs/frozen/clip_aesthetic.yaml --quick-experiment 0.1
```

**Configs:**
- `configs/frozen/multifeature.yaml` - CLIP + ResNet50 + IQA
- `configs/frozen/resnet50.yaml` - ResNet50 only
- `configs/frozen/vgg16.yaml` - VGG16 only
- `configs/frozen/clip_aesthetic.yaml` - CLIP only

### 2. `train_siamese_e2e.py` - End-to-End (Slow)

Trains both the CNN and MLP. Images go through the CNN each batch (no cache).

**Use when:**
- You want to fine-tune the CNN on your specific data
- Maximum performance is needed
- You have GPU resources and time

**Supported CNNs:**
- ResNet50
- VGG16

**Usage:**
```bash
# Train ResNet50 end-to-end
python -m sim_bench.training.train_siamese_e2e --config configs/siamese_e2e/resnet50.yaml

# Train VGG16
python -m sim_bench.training.train_siamese_e2e --config configs/siamese_e2e/vgg16.yaml

# Quick test
python -m sim_bench.training.train_siamese_e2e --config configs/siamese_e2e/resnet50.yaml --quick-experiment 0.1
```

**Configs:**
- `configs/siamese_e2e/resnet50.yaml` - Fine-tune ResNet50
- `configs/siamese_e2e/vgg16.yaml` - Fine-tune VGG16

## YAML Configuration

All scripts use YAML configs. Key sections:

```yaml
data:
  root_dir: "data/phototriage"
  min_agreement: 0.7         # Agreement threshold for labels
  min_reviewers: 2           # Minimum reviewers per pair
  quick_experiment: null     # Set to 0.1 for 10% of data

model:
  # Feature types (frozen mode only)
  use_clip: true
  use_cnn: true
  use_iqa: false

  # CNN settings
  cnn_backbone: "resnet50"   # or "vgg16"
  cnn_layer: "layer4"        # ResNet: "layer3" or "layer4"
  cnn_freeze_mode: "none"    # E2E only: "none", "partial", "full"

  # MLP architecture (always trainable)
  mlp_hidden_dims: [512, 256]
  dropout: 0.5

training:
  batch_size: 32
  learning_rate: 0.001
  optimizer: "adamw"         # or "sgd"
  weight_decay: 0.01
  max_epochs: 50
  early_stop_patience: 10
```

## Code Structure

Both scripts follow the same clean pattern:

1. **Helper Functions**: Small, focused functions
   - `load_config()` - Parse YAML
   - `create_optimizer()` - Factory for optimizers
   - `create_model()` - Instantiate model
   - `load_data()` - Load and split data
   - `create_dataloaders()` - Create PyTorch loaders
   - `train_epoch()` - Single training epoch
   - `evaluate()` - Validation/test evaluation
   - `train_model()` - Full training loop with early stopping

2. **Main Function**: Orchestrates the training pipeline
   - Parse arguments
   - Load config
   - Setup output directory
   - Call helper functions in order
   - Save results

3. **Logging**: All output via `logger.info()` (not print)

## Data Handling

### Series-Based Splitting

PhotoTriage data is split by `series_id` to prevent data leakage (images from same series never appear in different splits).

### Quick Experiment Mode

Use `--quick-experiment 0.1` to subsample to 10% of series. Useful for:
- Testing config changes
- Debugging
- Fast iteration

### Feature Caching (Frozen Mode Only)

Features are cached at `cache/phototriage_features/`:
- `clip_cache.pkl` - CLIP embeddings
- `cnn_cache.pkl` - ResNet/VGG features
- `iqa_cache.pkl` - IQA metrics

Cache is reused across runs. Delete to force re-extraction.

## Performance

| Mode | Feature Extraction | Training Time | Typical Accuracy |
|------|-------------------|---------------|------------------|
| Frozen (multi-feature) | 5-10 min | 2-5 min | 68-72% |
| Frozen (single) | 2-5 min | 1-3 min | 64-68% |
| End-to-End | N/A | 30-60 min | 70-75% |

Times are for full dataset on GPU. Quick experiment mode is 10x faster.

## Output

Each training run creates:
```
outputs/
└── [mode]/
    └── [timestamp]/
        ├── config.yaml        # Config used
        ├── best_model.pt      # Best checkpoint
        └── results.json       # Final test metrics
```

Checkpoint contains:
```python
{
    'epoch': 15,
    'model_state_dict': ...,
    'val_acc': 0.694,
    'config': {...}
}
```

## Migration from Old Scripts

The old `train_multifeature_ranker.py` (1094 lines, 35+ CLI args) has been replaced with these clean scripts.

**Old way:**
```bash
python train_multifeature_ranker.py --use_clip true --use_cnn_features true --cnn_backbone resnet50 --mlp_hidden_dims 256 128 --dropout 0.5 --batch_size 32 --learning_rate 0.001 ...
```

**New way:**
```bash
python -m sim_bench.training.train_frozen_improved --config configs/frozen/multifeature.yaml
```

See `MIGRATION_GUIDE.md` in project root for full migration details.

## Design Principles

These scripts follow clean code principles:

1. **Simple**: ~200 lines each, easy to understand
2. **YAML-first**: Configuration separate from code
3. **Reusable**: Complex logic in `PhotoTriageData` class
4. **Maintainable**: Helper functions with single responsibilities
5. **No over-abstraction**: Inline training loop, no excessive try/except
6. **Logging**: All output via logger for production use

## Examples

### Quick test of CLIP-only model
```bash
python -m sim_bench.training.train_frozen \
    --config configs/frozen/clip_aesthetic.yaml \
    --quick-experiment 0.1
```

### Train multi-feature fusion
```bash
python -m sim_bench.training.train_frozen \
    --config configs/frozen/multifeature.yaml
```

### Fine-tune ResNet50 end-to-end
```bash
python -m sim_bench.training.train_siamese_e2e \
    --config configs/siamese_e2e/resnet50.yaml
```

### Custom output directory
```bash
python -m sim_bench.training.train_frozen \
    --config configs/frozen/resnet50.yaml \
    --output-dir outputs/my_experiment
```
