# Architecture Split: Frozen vs End-to-End Training

## Overview

We now have **two clean, focused training pipelines** for different use cases:

### 1. Frozen Features Training (`train_frozen.py`)
**Use Case**: Multi-feature fusion with frozen backbones
- **Model**: `MultiFeaturePairwiseRanker`
- **Features**: CLIP + CNN + IQA (any combination)
- **Architecture**: Multi-feature fusion with late fusion
- **Training**: Only MLP is trainable, features are cached
- **Speed**: ⚡ Fast (features pre-extracted)

```
CLIP (frozen) ───┐
CNN (frozen)  ───┼──→ Fusion MLP (trainable) ──→ Output
IQA (rule-based) ┘
```

### 2. End-to-End Training (`train_siamese_e2e.py`)
**Use Case**: Paper replication - Siamese CNN + MLP end-to-end
- **Model**: `SiameseCNNRanker`
- **Backbones**: VGG16 or ResNet50
- **Architecture**: Simple Siamese CNN + MLP
- **Training**: Entire network is trainable (CNN + MLP)
- **Speed**: 🐌 Slower (images processed each batch)

```
Image1 ──→ CNN (trainable) ──→ feat1 ─┐
                                       ├──→ diff ──→ MLP (trainable) ──→ Output
Image2 ──→ CNN (shared weights) ──→ feat2 ─┘
```

## Model Comparison

| Feature | MultiFeaturePairwiseRanker | SiameseCNNRanker |
|---------|---------------------------|------------------|
| **Purpose** | Multi-feature fusion | Paper replication |
| **Complexity** | High (790 lines) | Low (~220 lines) |
| **Features** | CLIP + CNN + IQA | CNN only |
| **Training** | Frozen features + MLP | End-to-end CNN + MLP |
| **Config** | Complex dict with many options | Simple 6 parameters |
| **Use in** | `train_frozen.py` | `train_siamese_e2e.py` |

## Usage Examples

### Frozen Features Training
```bash
# Multi-feature (CLIP + ResNet50 + IQA)
python -m sim_bench.training.train_frozen --config configs/frozen/multifeature.yaml

# ResNet50 only (frozen)
python -m sim_bench.training.train_frozen --config configs/frozen/resnet50.yaml

# VGG16 only (frozen)
python -m sim_bench.training.train_frozen --config configs/frozen/vgg16.yaml
```

### End-to-End Training
```bash
# VGG16 Siamese (trainable)
python -m sim_bench.training.train_siamese_e2e --config configs/siamese_e2e/vgg16.yaml

# ResNet50 Siamese (trainable)
python -m sim_bench.training.train_siamese_e2e --config configs/siamese_e2e/resnet50.yaml
```

## Configuration Files

### For `train_frozen.py`
```yaml
# configs/frozen/multifeature.yaml
model:
  use_clip: true
  use_cnn: true
  use_iqa: true
  cnn_backbone: resnet50
  mlp_hidden_dims: [256, 128]
  dropout: 0.3
```

### For `train_siamese_e2e.py`
```yaml
# configs/siamese_e2e/vgg16.yaml
model:
  cnn_backbone: vgg16
  mlp_hidden_dims: [128, 128]
  dropout: 0.0
  activation: tanh
  use_paper_preprocessing: true
  padding_mean_color: [0.460, 0.450, 0.430]
```

## Key Design Decisions

### Why Two Separate Models?

1. **Different Use Cases**: 
   - Frozen features → exploratory, fast experimentation
   - End-to-end → paper replication, max performance

2. **Different Architectures**:
   - MultiFeaturePairwiseRanker → complex multi-feature fusion
   - SiameseCNNRanker → simple Siamese CNN + MLP

3. **Code Clarity**:
   - Each model is focused on its specific use case
   - No if/else logic for different modes
   - Easier to maintain and debug

### What About the SiameseResNet50 We Created Earlier?

That's a **standalone reference implementation** (`sim_bench/models/siamese_resnet.py`) with:
- Custom Bottleneck architecture
- Built-in difference layer before FC
- Differential learning rate helpers

It's kept as a **reference** but `SiameseCNNRanker` is more flexible (works with both VGG16 and ResNet50).

## Migration Notes

### Old Code (DELETED)
- ❌ `train_multifeature_ranker.py` - tried to do everything, 800+ lines
- ❌ Complex argparse with 30+ arguments
- ❌ Manual feature caching logic in training script

### New Code (CURRENT)
- ✅ `train_frozen.py` - focused on frozen features, 262 lines
- ✅ `train_siamese_e2e.py` - focused on e2e training, 276 lines
- ✅ YAML configuration files
- ✅ Feature caching in `PhotoTriageData`

## Benefits of New Architecture

1. **Simplicity**: Each script does one thing well
2. **Clarity**: No mode switching or complex if/else
3. **Maintainability**: Focused, small modules
4. **Correctness**: No config mismatches
5. **Performance**: Each optimized for its use case

## Common Pitfalls (NOW FIXED)

❌ **OLD MISTAKE**: Using `MultiFeaturePairwiseRanker` for e2e training
- Wrong: Heavy multi-feature fusion when you just need CNN
- Config mismatch: Paper preprocessing params don't match model expectations

✅ **NEW APPROACH**: Use `SiameseCNNRanker` for e2e training
- Simple, clean Siamese CNN + MLP
- Direct config mapping from YAML
- Paper preprocessing built-in

