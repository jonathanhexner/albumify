# Migration Guide: Old to New Training Scripts

## Overview

The old `train_multifeature_ranker.py` (1094 lines) has been replaced with clean, simple alternatives:
- **`train_frozen.py`** (378 lines) - For frozen features training
- **`train_siamese_e2e.py`** (coming soon) - For end-to-end CNN fine-tuning

## Quick Migration

### Old Way ❌
```bash
python train_multifeature_ranker.py \
    --output_dir outputs/my_experiment \
    --batch_size 32 \
    --learning_rate 0.001 \
    --max_epochs 50 \
    --mlp_hidden_dims 256 128 \
    --dropout 0.5 \
    --use_clip true \
    --use_cnn_features true \
    --use_iqa_features true \
    --cnn_backbone resnet50 \
    --cnn_layer layer4 \
    --optimizer adamw \
    --weight_decay 0.01 \
    --quick_experiment 0.1
    # ... and 20+ more arguments!
```

### New Way ✅
```bash
# Create a YAML config once
python train_frozen.py --config configs/frozen/multifeature.yaml --quick-experiment 0.1
```

## Detailed Migration

### 1. Command-Line Arguments → YAML Config

**Old**: 35+ command-line arguments
**New**: YAML configuration file

#### Example Conversion

**Old command:**
```bash
python train_multifeature_ranker.py \
    --output_dir outputs/test \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --use_clip true \
    --use_cnn_features true \
    --cnn_backbone resnet50 \
    --mlp_hidden_dims 512 256 128
```

**New YAML config** (`configs/my_experiment.yaml`):
```yaml
name: "my_experiment"

data:
  root_dir: "data/phototriage"
  min_agreement: 0.7
  min_reviewers: 2
  split_ratios: [0.8, 0.1, 0.1]
  quick_experiment: null

model:
  use_clip: true
  use_cnn: true
  use_iqa: false
  cnn_backbone: "resnet50"
  cnn_layer: "layer4"
  mlp_hidden_dims: [512, 256, 128]  # Your custom MLP layers
  dropout: 0.5

training:
  batch_size: 64  # Your custom batch size
  learning_rate: 0.0005  # Your custom learning rate
  optimizer: "adamw"
  weight_decay: 0.01
  max_epochs: 50
  early_stop_patience: 10

cache_dir: "cache/phototriage_features"
output:
  output_dir: "outputs/test"  # Your custom output dir
device: "cuda"
seed: 42
```

**New command:**
```bash
python train_frozen.py --config configs/my_experiment.yaml
```

### 2. Common Use Cases

#### Use Case 1: ResNet50 Baseline

**Old:**
```bash
python train_multifeature_ranker.py \
    --use_clip false \
    --use_cnn_features true \
    --use_iqa_features false \
    --cnn_backbone resnet50 \
    --cnn_layer layer4
```

**New:**
```bash
python train_frozen.py --config configs/frozen/resnet50.yaml
```

#### Use Case 2: Multi-Feature Fusion (CLIP + CNN + IQA)

**Old:**
```bash
python train_multifeature_ranker.py \
    --use_clip true \
    --use_cnn_features true \
    --use_iqa_features true \
    --cnn_backbone resnet50
```

**New:**
```bash
python train_frozen.py --config configs/frozen/multifeature.yaml
```

#### Use Case 3: Quick Experiment (10% of data)

**Old:**
```bash
python train_multifeature_ranker.py \
    --quick_experiment 0.1 \
    --use_clip true \
    --use_cnn_features true
```

**New:**
```bash
python train_frozen.py --config configs/frozen/multifeature.yaml --quick-experiment 0.1
```

#### Use Case 4: VGG16 Backbone

**Old:**
```bash
python train_multifeature_ranker.py \
    --use_clip false \
    --use_cnn_features true \
    --cnn_backbone vgg16
```

**New:**
```bash
python train_frozen.py --config configs/frozen/vgg16.yaml
```

#### Use Case 5: CLIP Aesthetic Only

**Old:**
```bash
python train_multifeature_ranker.py \
    --use_clip true \
    --use_cnn_features false \
    --use_iqa_features false
```

**New:**
```bash
python train_frozen.py --config configs/frozen/clip_aesthetic.yaml
```

### 3. End-to-End Training (CNN Fine-Tuning)

**Old:**
```bash
python train_multifeature_ranker.py \
    --cnn_freeze_mode none \  # Train the CNN
    --use_cnn_features true \
    --cnn_backbone resnet50
```

**New:**
```bash
# Coming soon:
python train_siamese_e2e.py --config configs/siamese_e2e/resnet50.yaml
```

### 4. Custom MLP Architecture

**Old:**
```bash
python train_multifeature_ranker.py \
    --mlp_hidden_dims 512 256 128 64 \
    --dropout 0.3
```

**New YAML:**
```yaml
model:
  mlp_hidden_dims: [512, 256, 128, 64]
  dropout: 0.3
```

### 5. Different Optimizers

#### SGD with Momentum

**Old:**
```bash
python train_multifeature_ranker.py \
    --optimizer sgd \
    --learning_rate 0.01 \
    --momentum 0.9 \
    --weight_decay 0.0001
```

**New YAML:**
```yaml
training:
  optimizer: "sgd"
  learning_rate: 0.01
  momentum: 0.9
  weight_decay: 0.0001
```

#### AdamW (default)

**Old:**
```bash
python train_multifeature_ranker.py \
    --optimizer adamw \
    --learning_rate 0.001 \
    --weight_decay 0.01
```

**New YAML:**
```yaml
training:
  optimizer: "adamw"
  learning_rate: 0.001
  weight_decay: 0.01
```

## Argument Mapping Table

| Old Argument | New YAML Location | Example |
|-------------|-------------------|---------|
| `--output_dir` | `output.output_dir` | `"outputs/test"` |
| `--batch_size` | `training.batch_size` | `32` |
| `--learning_rate` | `training.learning_rate` | `0.001` |
| `--max_epochs` | `training.max_epochs` | `50` |
| `--mlp_hidden_dims` | `model.mlp_hidden_dims` | `[256, 128]` |
| `--dropout` | `model.dropout` | `0.5` |
| `--use_clip` | `model.use_clip` | `true` |
| `--use_cnn_features` | `model.use_cnn` | `true` |
| `--use_iqa_features` | `model.use_iqa` | `true` |
| `--cnn_backbone` | `model.cnn_backbone` | `"resnet50"` |
| `--cnn_layer` | `model.cnn_layer` | `"layer4"` |
| `--optimizer` | `training.optimizer` | `"adamw"` |
| `--momentum` | `training.momentum` | `0.9` |
| `--weight_decay` | `training.weight_decay` | `0.01` |
| `--quick_experiment` | CLI or `data.quick_experiment` | `0.1` |
| `--seed` | `seed` | `42` |
| `--device` | `device` | `"cuda"` |

## What Changed Under the Hood

### 1. Data Loading Simplified

**Old (in training script):**
```python
# 30 lines of series subsampling
train_series = train_df['series_id'].unique()
n_train = max(1, int(len(train_series) * args.quick_experiment))
selected_train_series = np.random.choice(train_series, n_train, replace=False)
train_df = train_df[train_df['series_id'].isin(selected_train_series)].reset_index(drop=True)
# ... repeat for val and test
```

**New (one parameter):**
```python
train_df, val_df, test_df = data_loader.get_series_based_splits(
    quick_experiment=0.1  # That's it!
)
```

### 2. Feature Caching Simplified

**Old (in training script):**
```python
# 184 lines of feature caching logic
clip_cache_path = cache_dir / 'clip_cache.pkl'
if clip_cache_path.exists():
    with open(clip_cache_path, 'rb') as f:
        clip_cache = pickle.load(f)
# ... 180 more lines
```

**New (one line):**
```python
feature_cache = data_loader.precompute_features(
    all_df, model.feature_extractor, config['cache_dir']
)
```

### 3. No More Config Override Boilerplate

**Old (50+ lines):**
```python
if args.output_dir:
    config.output_dir = args.output_dir
if args.batch_size:
    config.batch_size = args.batch_size
if args.learning_rate:
    config.learning_rate = args.learning_rate
# ... 40+ more
```

**New:**
```python
config = load_config(args.config)  # Done!
```

## Benefits of Migration

1. **Simplicity**: 378 lines vs 1094 lines (65% reduction)
2. **Readability**: YAML config is self-documenting
3. **Maintainability**: Easy to modify and extend
4. **Reusability**: Data loader methods used by all scripts
5. **Version Control**: Config files in git, easy to track experiments
6. **No Duplication**: Logic moved to PhotoTriageData class

## Troubleshooting

### "ModuleNotFoundError: No module named 'tensorboardX'"
```bash
pip install tensorboardX
```

### "Can't find config file"
Make sure you're in the project root directory:
```bash
cd /path/to/sim-bench
python train_frozen.py --config configs/frozen/multifeature.yaml
```

### "Feature extractor doesn't have 'config' attribute"
This is expected if using old MultiFeaturePairwiseRanker. The precompute_features method will use defaults. No action needed.

## Getting Help

- See [TRAINING_REFACTORING_COMPLETE.md](TRAINING_REFACTORING_COMPLETE.md) for complete documentation
- See [REFACTORING_PROGRESS.md](REFACTORING_PROGRESS.md) for technical details
- Check example configs in `configs/frozen/`

## Next Steps

1. Try a quick experiment:
   ```bash
   python train_frozen.py --config configs/frozen/resnet50.yaml --quick-experiment 0.1
   ```

2. Create your own config based on the examples in `configs/frozen/`

3. Run full training when ready:
   ```bash
   python train_frozen.py --config configs/frozen/multifeature.yaml
   ```

---

**Note**: The old `train_multifeature_ranker.py` is now `train_multifeature_ranker_OLD.py` for reference only. Please use the new scripts going forward!
