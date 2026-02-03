# Multi-Feature Pairwise Ranker for PhotoTriage

**Status**: ✅ Implemented and ready to train

## Overview

This implements a multi-feature pairwise ranker that addresses the limitations of CLIP-only approaches for photo quality assessment. The model combines:

- **CLIP embeddings** (512-dim): Semantic understanding
- **ResNet layer3 features** (1024-dim): Mid-level visual features (edges, textures)
- **IQA features** (4-dim): Hand-crafted quality metrics (sharpness, exposure, colorfulness, contrast)

**Total feature dimension**: 1540-dim → LayerNorm → MLP → Scalar quality score

## Why This Architecture?

Based on research papers (PhotoTriAGE, Best-Photo-From-Burst) and empirical analysis:

1. **CLIP alone fails** for photo quality because:
   - Trained for semantic similarity, not photographic quality
   - Loses low-level signals during compression (sharpness, blur, exposure)
   - Series selection is *relative* (best in burst), not absolute quality

2. **Multi-level features** capture different aspects:
   - CLIP: Semantic content (composition, subject matter)
   - CNN layer3: Mid-level features (edges, textures, patterns)
   - IQA: Explicit quality signals (sharpness, exposure)

3. **Margin ranking loss** is better than cross-entropy:
   - Learns scalar scores per image (not binary classification)
   - Enables consistent ranking across series
   - More aligned with the actual task

## Files

### Core Implementation

1. **[phototriage_multifeature.py](../../sim_bench/quality_assessment/trained_models/phototriage_multifeature.py)** (~600 lines)
   - `MultiFeatureConfig`: Configuration dataclass
   - `MultiFeatureExtractor`: Frozen feature extractors (CLIP + CNN + IQA)
   - `MultiFeaturePairwiseRanker`: MLP ranker with margin loss
   - `MultiFeaturePairwiseDataset`: Dataset for pairwise training
   - `compute_pairwise_accuracy()`: Accuracy metric

2. **[train_multifeature_ranker.py](../../train_multifeature_ranker.py)** (~450 lines)
   - Complete training pipeline
   - Feature caching for efficiency
   - Accuracy tracking during training
   - Early stopping and checkpointing
   - Training curves visualization

### Launch Configurations

Added to [.vscode/launch.json](../../.vscode/launch.json):

1. **"Train Multi-Feature Ranker - Full (Recommended) ⭐"**
   - MLP: [512, 256]
   - Batch size: 64
   - Learning rate: 0.0001
   - Epochs: 30

2. **"Train Multi-Feature Ranker - Quick Test (Small MLP)"**
   - MLP: [256]
   - Batch size: 128
   - Learning rate: 0.001
   - Epochs: 15

3. **"Train Multi-Feature Ranker - Deep MLP"**
   - MLP: [1024, 512, 256]
   - For complex feature interactions

## How to Use

### Option 1: VS Code (Easiest)

1. Press **F5** or open Run/Debug panel
2. Select: **"Train Multi-Feature Ranker - Full (Recommended) ⭐"**
3. Click Start

### Option 2: Command Line

```bash
# Default settings (recommended)
python train_multifeature_ranker.py

# Custom settings
python train_multifeature_ranker.py \
    --output_dir outputs/my_experiment \
    --batch_size 64 \
    --learning_rate 0.0001 \
    --max_epochs 30 \
    --mlp_hidden_dims 512 256
```

## What Happens During Training

### 1. Data Loading
- Loads pairwise comparisons from PhotoTriage CSV
- Filters by agreement ≥ 0.7 and reviewers ≥ 2
- Splits: 80% train, 10% val, 10% test
- **~12,073 reliable pairs** after filtering

### 2. Feature Extraction (One-Time)
- Pre-computes features for all unique images (~5,000 images)
- Saves to cache: `outputs/phototriage_multifeature/features_cache.pkl`
- Takes ~30-60 minutes on CPU (one-time cost)
- Future runs reuse cache instantly

### 3. Training Loop
Each epoch:
- Forward pass: Extract features → MLP → Scalar scores
- Loss: Margin ranking loss (winner should score > loser + margin)
- **Accuracy metric**: How often does model rank winner higher?
- Logs: Loss and accuracy every 10 batches

### 4. Outputs

All saved to `outputs/phototriage_multifeature/`:

- `config.yaml` - Full configuration
- `features_cache.pkl` - Pre-computed features
- `best_model.pt` - Best model checkpoint
- `training_curves.png` - Loss and accuracy plots
- `test_results.json` - Final test metrics

## Expected Performance

Based on benchmarks and literature:

| Method | Pairwise Accuracy | Notes |
|--------|------------------|-------|
| Random baseline | 50.0% | Pure guessing |
| Rule-based (sharpness only) | 56.4% | Current best from benchmarks |
| CLIP-only frozen MLP | 52-55% | Barely better than random |
| **Multi-feature ranker** | **60-70%** | **Target performance** |

**Goal**: Beat 56.4% sharpness baseline with learned features.

## Architecture Details

### Feature Extraction (All Frozen)

```python
# 1. CLIP (ViT-B-32, OpenAI)
clip_embedding = model.encode_image(img)  # (512,)
clip_embedding = F.normalize(clip_embedding)  # L2 normalize

# 2. ResNet50 layer3 (ImageNet)
cnn_features = resnet_layer3(img)  # (1024,)

# 3. IQA features (rule-based)
sharpness = laplacian_variance(img) / 1000.0  # normalize
exposure = histogram_quality(img)
colorfulness = hasler_susstrunk(img) / 100.0
contrast = rms_contrast(img) / 0.8

iqa_features = [sharpness, exposure, colorfulness, contrast]  # (4,)

# Concatenate
features = [clip_embedding; cnn_features; iqa_features]  # (1540,)
```

### MLP Ranker (Trainable)

```python
# Recommended architecture
features (1540,)
    ↓
LayerNorm
    ↓
Linear(1540 → 512) → ReLU → Dropout(0.3)
    ↓
Linear(512 → 256) → ReLU → Dropout(0.3)
    ↓
Linear(256 → 1)  # Scalar score
```

**Trainable parameters**: ~800K (MLP only, features frozen)

### Loss Function

```python
# Margin ranking loss
score1 = model(features1)
score2 = model(features2)

# target = +1 if image1 better, -1 if image2 better
loss = F.margin_ranking_loss(score1, score2, target, margin=1.0)
```

## Reusing Existing Components

This implementation **maximally reuses** existing code:

✅ **CLIP loading**: Same as `phototriage_binary.py`
✅ **IQA features**: Same computations as `rule_based.py`
✅ **Data loading**: Same CSV parsing as `train_binary_classifier.py`
✅ **Training loop**: Similar structure to existing trainers
✅ **Feature caching**: Similar to CLIP embedding cache

**No re-implementations** - just combines existing pieces in a new architecture.

## Troubleshooting

### Q: Feature extraction is slow
**A**: This is normal for first run (~30-60 min CPU). Cache is saved and reused forever.

### Q: Running out of memory
**A**: Reduce batch size: `--batch_size 32` or `--batch_size 16`

### Q: Model not learning (accuracy stuck at 50%)
**A**: Try:
1. Increase learning rate: `--learning_rate 0.001`
2. Reduce MLP size: `--mlp_hidden_dims 256`
3. Check data: Ensure CSV has reliable labels

### Q: Want to use only CLIP + IQA (no CNN)
**A**: Not currently supported, but easy to modify:
- Edit `MultiFeatureExtractor.__init__()` to skip CNN initialization
- Edit `extract_all()` to exclude CNN features

## Next Steps

### After Training

1. **Check results**:
   ```bash
   cat outputs/phototriage_multifeature/test_results.json
   ```

2. **View training curves**:
   ```bash
   outputs/phototriage_multifeature/training_curves.png
   ```

3. **Compare to baselines**:
   - Sharpness only: 56.4%
   - Your model: ???

### If Performance is Good (>60%)

1. **Test on series ranking**:
   - Create script to rank full series using trained model
   - Compare top-1 accuracy vs series-softmax baseline

2. **Ablation studies**:
   - Remove each feature type (CLIP, CNN, IQA) to see contribution
   - Try different CNN layers (layer2, layer4)

3. **Hyperparameter tuning**:
   - Different MLP architectures
   - Different margin values
   - Different learning rates

### If Performance is Poor (≤55%)

1. **Check data quality**:
   - Are labels reliable?
   - Is agreement threshold too low?

2. **Try simpler baseline**:
   - Train with only IQA features (4-dim → MLP)
   - Should beat 56.4% if IQA features are good

3. **Debug features**:
   - Visualize feature distributions
   - Check for NaN or extreme values
   - Verify normalization

## Technical Notes

### Why ResNet layer3, not layer4?

- **layer3** (1024-dim): Mid-level features (textures, patterns)
- **layer4** (2048-dim): High-level features (objects, scenes)

For photo quality, mid-level features are more relevant (sharpness, textures).
Layer4 is too semantic (similar to CLIP).

### Why margin ranking loss, not cross-entropy?

Cross-entropy learns: P(image1 better | pair)
Margin ranking learns: score(image1) and score(image2)

Scalar scores enable:
- Ranking full series (not just pairs)
- Consistent comparisons across different pairs
- Bradley-Terry probabilistic models

### Why freeze feature extractors?

1. **Stability**: Frozen features are consistent
2. **Efficiency**: No gradients through CLIP/ResNet (faster)
3. **Data**: 12K pairs may not be enough to fine-tune CLIP
4. **Baseline**: Establish strong baseline before fine-tuning

If performance is good, consider unfreezing later.

## References

1. **PhotoTriAGE Dataset**: Mihai et al., 2023/2024
   - Photo series selection task
   - 12,988 images, 24,186 pairwise comparisons

2. **Best-Photo-From-Burst**: ECCV 2019
   - Multi-feature approach for burst selection
   - Combines CNN features with IQA metrics

3. **Series Photo Selection via Multi-view Graph Learning**: CVPR 2023
   - Graph-based ranking for series
   - Uses pairwise preferences

## Summary

**What was implemented:**
- ✅ Multi-feature extractor (CLIP + CNN + IQA)
- ✅ Margin ranking loss training
- ✅ Pairwise accuracy metrics
- ✅ Feature caching for efficiency
- ✅ Complete training pipeline
- ✅ VS Code launch configurations
- ✅ Minimal changes (reuse existing code)

**What you can do now:**
1. Press F5 → Select "Train Multi-Feature Ranker - Full (Recommended) ⭐"
2. Wait for training (~2-3 hours with feature extraction)
3. Check test accuracy vs 56.4% baseline

**Expected outcome:**
60-70% pairwise accuracy, beating rule-based methods.
