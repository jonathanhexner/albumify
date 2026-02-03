# Kaggle Notebook Updates - Differential Learning Rates

## Changes Made

The Kaggle notebook has been updated to use the **fixed differential learning rates** that match the reference implementation.

### Before (Broken ❌)

```python
'training': {
    'batch_size': 32,              # Too large
    'learning_rate': 0.001,        # 100x too high for VGG16!
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'max_epochs': 30,
    'early_stop_patience': 5
}
```

**Result:** ~52% accuracy (random guessing)

### After (Fixed ✅)

```python
'training': {
    'batch_size': 8,               # Match reference
    'learning_rate': 0.00001,      # Base LR for backbone (1e-5)
    'differential_lr': True,       # Head gets 10x LR (1e-4)
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'max_epochs': 30,
    'early_stop_patience': 10
}
```

**Expected Result:** 70%+ accuracy from epoch 1!

## Key Updates

### 1. VGG16 Config (Updated)
- ✅ `learning_rate: 0.00001` (was 0.001, 100x too high!)
- ✅ `differential_lr: true` (new flag)
- ✅ `batch_size: 8` (was 32)
- ✅ `early_stop_patience: 10` (was 5)

### 2. ResNet50 Config (New!)
- ✅ Added ResNet50 configuration as an alternative
- ✅ Same differential LR strategy
- ✅ Standard ResNet preprocessing (not paper-style)
- ✅ Different MLP architecture: [256, 128] with dropout=0.3

### 3. Training Options
The notebook now supports training either network:

```bash
# Train VGG16 (paper replication)
!python -m sim_bench.training.train_siamese_e2e \
    --config /kaggle/working/configs/vgg16_kaggle.yaml

# Train ResNet50 (alternative)
!python -m sim_bench.training.train_siamese_e2e \
    --config /kaggle/working/configs/resnet50_kaggle.yaml
```

## Expected Performance

### Training Progress (Fixed)

```
Epoch 1/30:
  Train: loss=0.662, acc=0.622
  Val: loss=0.639, acc=0.705     ✅ Learning immediately!

Epoch 5/30:
  Train: loss=0.597, acc=0.690
  Val: loss=0.606, acc=0.707     ✅ Continuing to improve

Epoch 10/30:
  Train: loss=0.568, acc=0.711
  Val: loss=0.613, acc=0.695     ✅ Stable high performance

Final Test: 70%+ accuracy
```

### What Changed?

**Differential Learning Rates:**
- **Backbone (CNN layers)**: `1e-5` (10x slower, preserves pretrained features)
- **Head (MLP layers)**: `1e-4` (10x faster, learns task-specific features)

This prevents **catastrophic forgetting** where the pretrained CNN forgets ImageNet features.

## Running on Kaggle

### Prerequisites (Same as before)
1. ✅ Add dataset: `ericwolter/triage`
2. ✅ Enable GPU: Settings → Accelerator → GPU T4 x2
3. ✅ Enable Internet: Settings → Internet → On

### Training Time (GPU)
- Quick test (10% data): ~5 min
- Full VGG16 (30 epochs): ~2-3 hours
- Full ResNet50 (30 epochs): ~3-4 hours

### Output Files
```
/kaggle/working/outputs/
├── siamese_e2e_vgg16/
│   ├── training.log           ← Complete logs with batch-level progress
│   ├── training_curves.png
│   ├── best_model.pt
│   ├── config.yaml
│   ├── results.json
│   └── training_history.json
└── siamese_e2e_resnet50/
    └── (same structure)
```

## Verification

The notebook will print the differential LR configuration:

```
✓ VGG16 config saved: /kaggle/working/configs/vgg16_kaggle.yaml
✓ ResNet50 config saved: /kaggle/working/configs/resnet50_kaggle.yaml

Device: cuda
Differential LR: backbone=1e-05, head=0.0001
```

During training, you'll see:

```
Using differential LR: backbone=1e-05, head=0.0001
Training end-to-end vgg16 | Output: /kaggle/working/outputs/siamese_e2e_vgg16
```

## Summary

✅ **Critical Fix Applied**: Differential learning rates now match the reference implementation  
✅ **Both Networks Supported**: VGG16 and ResNet50 configs available  
✅ **Performance Expected**: 70%+ accuracy (vs 52% before)  
✅ **Complete Logging**: Training logs saved to file for analysis  

The notebook is now ready for high-quality training results! 🚀


















