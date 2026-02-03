# PhotoTriage Paper Exact Replication Guide

This guide explains how to replicate the exact training method from the PhotoTriage paper using the newly implemented features.

## Two Critical Implementation Details

We identified and implemented **two subtle but critical** details from the PhotoTriage paper that were initially missing:

### Issue #1: End-to-End CNN Training

**Paper Quote:**
> "We concatenate these two stages of network and train them together."

**What This Means:**
- The paper trains BOTH the VGG16 CNN and the MLP end-to-end
- Our initial implementation only trained the MLP (VGG16 frozen)
- **Solution**: Added `--cnn_freeze_mode` parameter with three options:
  - `all`: Freeze VGG16 completely (original behavior)
  - `fc_layers`: Freeze conv layers, train FC layers only
  - `none`: Train everything end-to-end (paper method)

### Issue #2: Aspect-Ratio Preserving Preprocessing

**Paper Quote:**
> "we resize each image so that its larger dimension is the required size, while maintaining the original aspect ratio and padding with the mean pixel color in the training set."

**What This Means:**
- Standard preprocessing: Resize → Center Crop (CROPS the image)
- Paper preprocessing: Resize larger dim → Pad to square (NO CROPPING)
- Padding uses training set mean color, not ImageNet mean
- **Solution**: Added `--use_paper_preprocessing` and `--padding_mean_color` parameters

## Training Set Mean Color

The paper uses the **training set mean color** for padding, not ImageNet mean.

**Computed Values:**
- **Mean RGB [0,1]**: `0.460 0.450 0.430`
- **Mean RGB [0,255]**: `117.3 114.7 109.6`
- **ImageNet mean [0,1]**: `0.485 0.456 0.406`

The PhotoTriage dataset is slightly darker than ImageNet.

**How to Regenerate:**
```bash
python scripts/phototriage/compute_mean_color.py
```

This creates `data/phototriage/training_mean_color.json`.

## Launch Configurations

We've added three VS Code launch configurations for paper replication:

### 1. Quick Test (5 epochs, 5% data) ⚡
**Name**: `Paper Exact - Quick Test (5 epochs) ⚡`

**Purpose**: Fast validation that implementation works correctly

**Features**:
- End-to-end CNN training (`cnn_freeze_mode=none`)
- Paper preprocessing (`use_paper_preprocessing=true`)
- Training set mean color for padding
- All other paper settings (SGD, lr=0.001, etc.)
- Only 5% of data, 5 epochs (~10-15 minutes)

**Use Case**: Quick sanity check before running full experiment

### 2. Full Paper Replication (30 epochs) 🎯
**Name**: `Paper Exact - Full 30 Epochs (End-to-End CNN + Paper Preprocessing) 🎯`

**Purpose**: Exact paper replication for comparison with 73% accuracy

**Features**:
- End-to-end CNN training (`cnn_freeze_mode=none`)
- Paper preprocessing (`use_paper_preprocessing=true`)
- Training set mean color: `0.460 0.450 0.430`
- SGD optimizer with momentum=0.9, weight_decay=0.0005
- Learning rate: 0.001
- MLP: [128, 128] with tanh activation
- Batch size: 64
- 30 epochs (full dataset)

**Expected Runtime**: 4-6 hours on GPU

**Expected Accuracy**: ~73% (paper's reported accuracy)

### 3. Frozen CNN Baseline (30 epochs) 📊
**Name**: `Paper Exact - Frozen CNN Baseline (30 epochs) 📊`

**Purpose**: Compare frozen CNN vs end-to-end training

**Features**:
- Frozen VGG16 (`cnn_freeze_mode=all`)
- Paper preprocessing (same as above)
- All other settings identical to full replication

**Use Case**: Understand the impact of CNN fine-tuning on accuracy

## Command-Line Usage

If you prefer running from command line instead of VS Code:

### Quick Test
```bash
python train_multifeature_ranker.py \
    --use_clip false \
    --use_cnn_features true \
    --use_iqa_features false \
    --cnn_backbone vgg16 \
    --cnn_freeze_mode none \
    --use_paper_preprocessing true \
    --padding_mean_color 0.460 0.450 0.430 \
    --activation tanh \
    --optimizer sgd \
    --momentum 0.9 \
    --weight_decay 0.0005 \
    --learning_rate 0.001 \
    --mlp_hidden_dims 128 128 \
    --dropout 0.0 \
    --use_visual_tower false \
    --max_epochs 5 \
    --batch_size 64 \
    --quick_experiment 0.05 \
    --output_dir outputs/phototriage_multifeature/paper_exact_quick_test
```

### Full Paper Replication
```bash
python train_multifeature_ranker.py \
    --use_clip false \
    --use_cnn_features true \
    --use_iqa_features false \
    --cnn_backbone vgg16 \
    --cnn_freeze_mode none \
    --use_paper_preprocessing true \
    --padding_mean_color 0.460 0.450 0.430 \
    --activation tanh \
    --optimizer sgd \
    --momentum 0.9 \
    --weight_decay 0.0005 \
    --learning_rate 0.001 \
    --mlp_hidden_dims 128 128 \
    --dropout 0.0 \
    --use_visual_tower false \
    --max_epochs 30 \
    --batch_size 64 \
    --output_dir outputs/phototriage_multifeature/paper_exact_full
```

## Parameter Explanation

### New Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `--cnn_freeze_mode` | `all`, `fc_layers`, `none` | CNN training mode. `none` = end-to-end (paper) |
| `--use_paper_preprocessing` | `true`, `false` | Use aspect-ratio preserving resize + padding |
| `--padding_mean_color` | 3 floats in [0,1] | RGB mean for padding (e.g., `0.460 0.450 0.430`) |

### Paper Settings (must match exactly)

| Parameter | Value | Source |
|-----------|-------|--------|
| `--cnn_backbone` | `vgg16` | Paper: "VGG16 architecture" |
| `--activation` | `tanh` | Paper: "tanh activation" |
| `--optimizer` | `sgd` | Paper: "SGD optimizer" |
| `--momentum` | `0.9` | Paper: "momentum 0.9" |
| `--weight_decay` | `0.0005` | Paper: "weight decay 0.0005" |
| `--learning_rate` | `0.001` | Paper: "learning rate 0.001" |
| `--mlp_hidden_dims` | `128 128` | Paper: "two hidden layers, 128 units each" |
| `--dropout` | `0.0` | Paper: "no dropout" |
| `--batch_size` | `64` | Paper: "batch size 64" |
| `--max_epochs` | `30` | Paper: "30 epochs" |

## Verification Checklist

Before running the full experiment, verify:

- [ ] `training_mean_color.json` exists in `data/phototriage/`
- [ ] Mean color values are `[0.460, 0.450, 0.430]`
- [ ] Training pairs exist: `data/phototriage/pairs_train.jsonl`
- [ ] Validation pairs exist: `data/phototriage/pairs_val.jsonl`
- [ ] GPU available (check with `nvidia-smi` or similar)
- [ ] Quick test runs successfully (5 epochs, 5% data)

## Expected Output

### Training Logs
```
Model parameters:
  Total: 138,378,242
  Trainable: 138,378,242 (100.0%)  # All VGG16 + MLP parameters trainable
  Frozen: 0 (0.0%)

Loading VGG16 (freeze_mode=none, paper_preproc=True)
Using paper preprocessing (aspect-ratio + padding with mean=[0.460, 0.450, 0.430])
VGG16 fully trainable (end-to-end fine-tuning)

Epoch 1/30
  Train - Loss: 0.6234, Accuracy: 65.3%
  Val   - Loss: 0.5891, Accuracy: 68.7%
  ...
```

### Expected Accuracy Trajectory

Based on the paper's 73% validation accuracy:

| Epoch | Expected Val Accuracy |
|-------|----------------------|
| 5 | ~65-68% |
| 10 | ~68-70% |
| 15 | ~70-72% |
| 20 | ~71-73% |
| 25 | ~72-73% |
| 30 | ~73% (paper) |

## Comparison with Baseline

| Configuration | CNN Training | Preprocessing | Expected Acc |
|---------------|--------------|---------------|--------------|
| **Paper Exact** | End-to-end | Aspect-ratio + padding | ~73% |
| **Frozen Baseline** | Frozen | Aspect-ratio + padding | ~65-68% |
| **Original (before fixes)** | Frozen | Resize + crop | ~60-65% |

The combination of both fixes should yield **~8-13% improvement** over the original implementation.

## Troubleshooting

### Issue: "KeyError: padding_mean_color"
**Solution**: The mean color file doesn't exist. Run:
```bash
python scripts/phototriage/compute_mean_color.py
```

### Issue: OOM (Out of Memory)
**Solution**: Reduce batch size to 32 or 16:
```bash
--batch_size 32
```

### Issue: Training very slow
**Cause**: End-to-end CNN training computes gradients for all VGG16 parameters.

**Expected**: ~2-3x slower than frozen CNN training

**Solution**: This is normal. Ensure GPU is being used.

### Issue: Accuracy not reaching 73%
**Possible causes**:
1. **Different data split**: Paper used specific train/val split
2. **Learning rate**: Try lr=0.0005 or lr=0.002
3. **Batch size**: Paper used 64, try 32 or 128
4. **Seed**: Results may vary by random seed

## Next Steps After Replication

Once you've replicated the paper's 73% accuracy:

1. **Ablation studies**: Test impact of each component
   - End-to-end training vs frozen CNN
   - Paper preprocessing vs standard preprocessing
   - Training set mean vs ImageNet mean

2. **Hyperparameter tuning**:
   - Learning rate grid: [0.0005, 0.001, 0.002]
   - Batch size grid: [32, 64, 128]
   - Dropout: [0.0, 0.1, 0.3]

3. **Architecture experiments**:
   - Freeze only conv layers (`cnn_freeze_mode=fc_layers`)
   - Different CNN backbones (ResNet50, EfficientNet)
   - MLP variations: [256, 256], [512, 256, 128]

## Reference

Chang, H., Yu, F., Wang, J., Ashley, D., & Finkelstein, A. (2016).
Automatic Triage for a Photo Series. ACM Transactions on Graphics, 35(6).
https://dl.acm.org/doi/10.1145/2980179.2982433

## Files Modified

Implementation spans these files:

1. `sim_bench/feature_extraction/vgg_features.py` - VGG16 with paper preprocessing
2. `sim_bench/quality_assessment/trained_models/phototriage_multifeature.py` - Config + model
3. `train_multifeature_ranker.py` - Training script with new args
4. `scripts/phototriage/compute_mean_color.py` - Mean color computation
5. `data/phototriage/training_mean_color.json` - Computed mean color
6. `.vscode/launch.json` - VS Code launch configurations

See [README_MEAN_COLOR.md](../../data/phototriage/README_MEAN_COLOR.md) for more details on the mean color computation.
