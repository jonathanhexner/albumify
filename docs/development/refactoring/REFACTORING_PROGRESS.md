# Training Scripts Refactoring Progress

## Goal
Reduce complexity from 1094-line monolithic script to clean, simple ~200-300 line scripts following the `train_resnet.py` pattern.

## Completed ✅

### 1. Data Loader Enhancements (sim_bench/datasets/phototriage_data.py)

**Added `quick_experiment` parameter to `get_series_based_splits()`**
- Lines 246: New optional parameter
- Lines 302-306: Automatic series subsampling when enabled
- Usage: `get_series_based_splits(quick_experiment=0.1)` for 10% of data

**Added `_subsample_series()` method**
- Lines 310-342: Clean subsampling implementation
- Removes 30+ lines from training scripts
- Reusable across all training scripts

**Added `precompute_features()` method**
- Lines 477-643: Complete feature caching logic (~167 lines)
- Handles CLIP, CNN, and IQA feature caches separately
- Removes ~180 lines from training scripts
- Single source of truth for feature extraction

**Total removed from training scripts: ~220 lines**

### 2. Clean Training Script (train_frozen.py)

**Created new frozen features training script**
- **378 lines** (down from 1094 - 65% reduction!)
- YAML-based configuration (no 35+ argparse arguments)
- Inline training loop (like `train_resnet.py`)
- No over-abstraction
- Clear separation of concerns

**Key features:**
- Data loading: ONE LINE
  ```python
  train_df, val_df, test_df = data_loader.get_series_based_splits(
      quick_experiment=config['data'].get('quick_experiment')
  )
  ```

- Feature caching: ONE LINE
  ```python
  feature_cache = data_loader.precompute_features(
      all_df, model.feature_extractor, config['cache_dir']
  )
  ```

- Optimizer: Simple if/else (no factory)
  ```python
  if config['training']['optimizer'].lower() == 'sgd':
      optimizer = torch.optim.SGD(...)
  else:
      optimizer = torch.optim.AdamW(...)
  ```

- Training loop: Inline, easy to read

### 3. YAML Configuration Files

Created 4 config files in `configs/frozen/`:

1. **multifeature.yaml** - CLIP + ResNet50 + IQA → MLP
2. **resnet50.yaml** - ResNet50 only → MLP
3. **vgg16.yaml** - VGG16 only → MLP
4. **clip_aesthetic.yaml** - CLIP only → MLP

**Benefits:**
- No more 35+ command-line arguments
- Easy to create new experiments
- Version-controlled configurations
- Clear and readable

**Usage:**
```bash
# Fast training with frozen features
python train_frozen.py --config configs/frozen/multifeature.yaml

# Quick experiment (10% of data)
python train_frozen.py --config configs/frozen/resnet50.yaml --quick-experiment 0.1

# Override output directory
python train_frozen.py --config configs/frozen/vgg16.yaml --output-dir outputs/test
```

## Results Summary

### Code Reduction

**Before:**
- `train_multifeature_ranker.py`: 1094 lines
- 35+ argparse arguments
- 184 lines of feature caching logic
- 30+ lines of series subsampling logic
- Duplicate code with `phototriage_data.py`
- Complex config override logic (50+ lines)

**After:**
- `train_frozen.py`: 378 lines (65% reduction!)
- 3 simple CLI arguments (config, output-dir, quick-experiment)
- Feature caching delegated to data loader
- Series subsampling delegated to data loader
- No duplicate code
- YAML-based configuration

### Clarity Improvements

**Before:**
- Mixed frozen/end-to-end modes in one script
- Unclear where normalization happens
- Hard to understand control flow
- Difficult to modify

**After:**
- Single clear purpose: frozen features mode
- Clean data pipeline
- Easy to read top-to-bottom
- Easy to modify

### Architecture

**Always Siamese:**
```
Image1 → Feature Extractor (frozen) → feat1 ─┐
                                               ├→ [feat1; feat2; diff] → MLP → P(img1>img2)
Image2 → Feature Extractor (frozen) → feat2 ─┘
```

**Frozen mode:**
- Feature Extractor = Pre-trained CNN or CLIP (frozen)
- Uses cached features
- Only trains MLP head
- Fast (~minutes)

## Remaining Work ⏳

### 1. Create `train_siamese_e2e.py`
- End-to-end CNN fine-tuning
- Similar structure to `train_frozen.py`
- Uses raw images (no cache)
- Trains both CNN + MLP
- ~300-400 lines

### 2. Create end-to-end configs
- `configs/siamese_e2e/resnet50.yaml`
- `configs/siamese_e2e/vgg16.yaml`

### 3. Test both scripts
- Run `train_frozen.py`
- Compare with old `train_multifeature_ranker.py`
- Verify same accuracy
- Check normalization behavior

### 4. Deprecate old script
- Rename `train_multifeature_ranker.py` → `train_multifeature_ranker_OLD.py`
- Add deprecation notice
- Update documentation

## Next Steps

1. Create `train_siamese_e2e.py` for end-to-end training
2. Test the new scripts
3. Compare performance with old implementation
4. Document normalization strategy
5. Deprecate old script

## Files Modified

### Edited:
- `sim_bench/datasets/phototriage_data.py` (added 220 lines of reusable logic)

### Created:
- `train_frozen.py` (378 lines - clean and simple!)
- `configs/frozen/multifeature.yaml`
- `configs/frozen/resnet50.yaml`
- `configs/frozen/vgg16.yaml`
- `configs/frozen/clip_aesthetic.yaml`

### To Deprecate:
- `train_multifeature_ranker.py` (1094 lines - complex and hard to maintain)

## Success Metrics

- ✅ Training script < 400 lines (target was 300)
- ✅ YAML-based configuration
- ✅ No code duplication
- ✅ Clear separation of concerns
- ✅ Reusable data loader methods
- ⏳ Same or better performance (to be tested)
- ⏳ Clear normalization strategy (to be documented)

---

**Status**: Frozen features mode complete! Ready to create end-to-end training script.
