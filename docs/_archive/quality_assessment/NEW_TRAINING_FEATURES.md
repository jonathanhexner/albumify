# New Training Features

Three new features have been added to the training script to improve experimentation workflow.

## 1. Quick Experiment Mode

**Flag**: `--quick_experiment <fraction>`

Run fast experiments on a subset of photo series (e.g., 10% for testing hyperparameters).

**Example**:
```bash
# Test with 10% of series (~5-10 minutes instead of 30-60 minutes)
python train_multifeature_ranker.py --quick_experiment 0.1 --mlp_hidden_dims 256

# Test with 20% of series
python train_multifeature_ranker.py --quick_experiment 0.2 --mlp_hidden_dims 128 --batch_size 64
```

**How it works**:
- After series-based split, subsamples each split (train/val/test) to the specified fraction
- Maintains series-based split integrity (no data leakage)
- Example: 0.1 = 347 train series + 43 val series + 43 test series (instead of 3466/433/434)

**Use cases**:
- Quick sanity checks (does training run without errors?)
- Hyperparameter testing (is this configuration promising?)
- Development (testing code changes)

**Important**: Results from quick experiments may not generalize to full dataset!

---

## 2. Smart Feature Caching

**What changed**: Features are now cached separately by type instead of all together.

**Old behavior** (single monolithic cache):
```
outputs/experiment_dir/
└── features_cache.pkl  # All features (CLIP+CNN+IQA) for all images
```

**New behavior** (separate caches per feature type):
```
outputs/phototriage_multifeature/experiment_dir/
├── clip_cache.pkl  # CLIP features only
├── cnn_cache.pkl   # CNN features only
└── iqa_cache.pkl   # IQA features only
```

**Benefits**:

1. **Extendable**: Add new feature types without recalculating existing ones
   ```bash
   # First run: Extract CLIP + CNN (no IQA yet)
   python train_multifeature_ranker.py --use_iqa_features false

   # Later: Add IQA features (reuses cached CLIP + CNN)
   python train_multifeature_ranker.py --use_iqa_features true
   # Only extracts IQA, loads CLIP+CNN from cache
   ```

2. **Feature ablation experiments**: Skip unneeded feature extraction
   ```bash
   # CLIP-only experiment (skips CNN/IQA extraction entirely)
   python train_multifeature_ranker.py \
       --use_cnn_features false \
       --use_iqa_features false \
       --mlp_hidden_dims 256

   # Uses cached CLIP features, doesn't extract CNN/IQA
   ```

3. **Reusable across experiments**: All experiments in same output directory share caches
   ```bash
   # First experiment extracts all features
   python train_multifeature_ranker.py --output_dir outputs/phototriage_multifeature/exp1

   # Second experiment reuses cached features from exp1
   python train_multifeature_ranker.py --output_dir outputs/phototriage_multifeature/exp2
   # Loads: outputs/phototriage_multifeature/exp1/clip_cache.pkl, etc.
   ```

**Technical details**:
- Each cache file: `{image_name: feature_tensor}` dictionary
- CLIP: 512-dim tensor per image
- CNN (ResNet layer3): 1024-dim tensor per image
- IQA: 4-dim tensor per image (sharpness, exposure, colorfulness, contrast)
- Final features: Concatenation of enabled feature types

---

## 3. Output Directory Reorganization

**What changed**: All training outputs now go to `outputs/phototriage_multifeature/` by default.

**Old behavior**:
```
outputs/
├── phototriage_20251128_233309/
├── phototriage_20251129_101520/
└── phototriage_20251129_143722/
```

**New behavior**:
```
outputs/
└── phototriage_multifeature/
    ├── 20251130_101520/
    ├── 20251130_143722/
    ├── 20251130_152033/
    ├── clip_cache.pkl     # Shared cache (optional)
    ├── cnn_cache.pkl      # Shared cache (optional)
    └── iqa_cache.pkl      # Shared cache (optional)
```

**Benefits**:
- Cleaner organization (all multifeature experiments in one place)
- Easy to compare experiments side-by-side
- Shared feature caches across experiments (saves time!)

**Custom output directory** (still supported):
```bash
python train_multifeature_ranker.py --output_dir outputs/my_custom_experiment
```

---

## Combining All Three Features

**Example workflow**: Quick hyperparameter search

```bash
# 1. Quick sanity check with 10% of data
python train_multifeature_ranker.py \
    --quick_experiment 0.1 \
    --mlp_hidden_dims 256 \
    --batch_size 64
# Takes ~5-10 min, extracts features for ~10% of images

# 2. If results look promising, run on full dataset
python train_multifeature_ranker.py \
    --mlp_hidden_dims 256 \
    --batch_size 64
# Reuses cached features from step 1 where they exist
# Only extracts features for remaining 90% of images

# 3. Try different architecture (features already cached)
python train_multifeature_ranker.py \
    --mlp_hidden_dims 128 \
    --batch_size 128
# Uses cached features, only trains MLP (very fast)

# 4. Try CLIP-only experiment
python train_multifeature_ranker.py \
    --use_cnn_features false \
    --use_iqa_features false \
    --mlp_hidden_dims 256
# Loads CLIP cache, skips CNN/IQA
```

---

## Integration with Hyperparameter Search

The hyperparameter search script can benefit from these features:

```bash
# Quick search on 10% of data (very fast prototyping)
python run_hyperparameter_search.py --mode quick

# Modify run_hyperparameter_search.py to add --quick_experiment 0.1 to all experiments
# (Future enhancement)
```

---

## Cache Management

**Location of caches**:
- By default: In each experiment's output directory
- Example: `outputs/phototriage_multifeature/20251130_101520/clip_cache.pkl`

**Sharing caches across experiments**:
- Place caches in parent directory: `outputs/phototriage_multifeature/`
- All experiments will check parent directory for existing caches

**Cache size**:
- CLIP cache: ~50 MB (10,000 images × 512 dims × 4 bytes)
- CNN cache: ~100 MB (10,000 images × 1024 dims × 4 bytes)
- IQA cache: ~160 KB (10,000 images × 4 dims × 4 bytes)
- Total: ~150 MB for full dataset

**Invalidating caches**:
- Delete specific cache file to force re-extraction
- Example: `rm outputs/phototriage_multifeature/*/iqa_cache.pkl`

---

## Summary

| Feature | Flag | Benefit |
|---------|------|---------|
| **Quick experiment** | `--quick_experiment 0.1` | 10x faster experimentation |
| **Smart caching** | Automatic | Reusable features, extensible |
| **Organized output** | `outputs/phototriage_multifeature/` | Clean organization |

**Recommended workflow**:
1. Quick experiment (10%) to test configuration
2. Full run if promising
3. Reuse cached features for architectural variants
4. Share caches across experiments
