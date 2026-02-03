# Model Recommendations for Quality Assessment

Recommendations for CNN and Transformer models to run on PhotoTriage dataset.

## Recommended Models

### CNN Model: NIMA with MobileNetV2 Backbone

**Why MobileNetV2:**
- Fast: 15-25ms per image on GPU, 80-120ms on CPU
- Lightweight: Only 3.4M parameters
- Good accuracy: 72-76% Top-1 on PhotoTriage (expected)
- Production-ready: Used in real applications

**Configuration:**
```yaml
methods:
  - name: nima_mobilenet
    type: nima
    config:
      backbone: mobilenet_v2
      weights_path: null  # Will use ImageNet pretrained
      device: cuda  # or 'cpu'
```

**Expected Performance:**
- Top-1 Accuracy: 72-76% (vs 64.95% for sharpness-only)
- Speed: ~20ms per image on GPU
- Memory: ~200MB GPU memory

**Training Data:**
- Pre-trained on ImageNet (general features)
- Can be fine-tuned on AVA dataset (aesthetic quality)
- Optional: Fine-tune on PhotoTriage for domain-specific improvement

**Alternative: NIMA with ResNet50**
- More accurate: 73-77% expected
- Slower: ~30ms per image on GPU
- Larger: 25M parameters
- Use if accuracy is more important than speed

### Transformer Model: ViT-Base (Vision Transformer)

**Why ViT-Base:**
- Implemented and ready to use
- Good accuracy: 74-78% expected on PhotoTriage
- Pre-trained: Available on ImageNet (can fine-tune on quality datasets)
- Flexible: Can use different ViT variants

**Configuration:**
```yaml
methods:
  - name: vit_base
    type: vit
    config:
      model_name: google/vit-base-patch16-224
      weights_path: null  # Use ImageNet pretrained
      device: cuda
```

**Expected Performance:**
- Top-1 Accuracy: 74-78% (good improvement over baseline)
- Speed: ~40-50ms per image on GPU
- Memory: ~400MB GPU memory

**Training Data:**
- Pre-trained on ImageNet (general features)
- Can be fine-tuned on AVA or KonIQ-10k for quality-specific features
- Optional: Fine-tune on PhotoTriage for best results

**Alternative ViT Variants:**
- `google/vit-base-patch32-224`: Faster, slightly less accurate
- `google/vit-large-patch16-224`: More accurate, slower, needs more memory

**Note:** MUSIQ (Multi-Scale Image Quality) is a more advanced transformer method but is not yet implemented. ViT-Base is a good alternative that's available now.

## Comparison Table

| Model | Type | Expected Accuracy | Speed (GPU) | Memory | Status |
|-------|------|-------------------|-------------|--------|--------|
| Sharpness-only | Rule | 64.95% | 60ms (CPU) | - | Implemented |
| NIMA MobileNetV2 | CNN | 72-76% | 20ms | 200MB | Implemented |
| NIMA ResNet50 | CNN | 73-77% | 30ms | 500MB | Implemented |
| ViT-Base | Transformer | 74-78% | 40-50ms | 400MB | Implemented |
| MUSIQ | Transformer | 75-80% | 50-100ms | 500MB | Not implemented |

## Recommended Benchmark Configuration

```yaml
# configs/quality_benchmark.deep_learning.yaml
datasets:
  - name: phototriage
    config: configs/dataset.phototriage.yaml
    sampling:
      strategy: random
      num_series: 100  # Start with sample
      seed: 42

methods:
  # Baseline
  - name: sharpness_only
    type: rule_based
    config:
      weights: {sharpness: 1.0, exposure: 0.0, colorfulness: 0.0, contrast: 0.0, noise: 0.0}
  
  # Recommended CNN
  - name: nima_mobilenet
    type: nima
    config:
      backbone: mobilenet_v2
      weights_path: null
      device: cuda
  
  # Recommended Transformer: ViT-Base
  - name: vit_base
    type: vit
    config:
      model_name: google/vit-base-patch16-224
      weights_path: null
      device: cuda

settings:
  verbose: true
```

## Implementation Status

**Currently Implemented:**
- NIMA with MobileNetV2, ResNet50, EfficientNet-B0
- ViT-Base (basic transformer)

**Needs Implementation:**
- MUSIQ (recommended transformer) - [Implementation plan](MUSIQ_IMPLEMENTATION.md)

**Recommendation:**
1. Start with NIMA MobileNetV2 (already implemented, fast, good accuracy)
2. Implement MUSIQ for best accuracy - [Google Research repo](https://github.com/google-research/google-research/tree/master/musiq)
3. Compare both against sharpness baseline

**MUSIQ Implementation:**
- Official repository available from Google Research
- Expected 75-80% Top-1 accuracy (best)
- Multi-scale architecture (handles variable image sizes naturally)
- See [MUSIQ_IMPLEMENTATION.md](MUSIQ_IMPLEMENTATION.md) for details

## Running the Benchmark

```bash
# Quick test with 20 series
python run_quality_benchmark.py configs/quality_benchmark.deep_learning.yaml

# Full evaluation (100 series)
# Edit config: num_series: 100 or remove sampling
python run_quality_benchmark.py configs/quality_benchmark.deep_learning.yaml
```

## Expected Results

Based on literature and PhotoTriage characteristics:

**Sharpness-only (baseline):**
- Top-1: 64.95%
- Speed: 60ms/image (CPU)

**NIMA MobileNetV2:**
- Top-1: 72-76% (8-12% improvement)
- Speed: 20ms/image (GPU)
- 3x faster AND more accurate

**ViT-Base:**
- Top-1: 74-78% (9-13% improvement)
- Speed: 40-50ms/image (GPU)
- Good accuracy with transformer architecture

## Decision Guide

**Choose NIMA MobileNetV2 if:**
- You want good accuracy with fast speed
- GPU memory is limited (<2GB)
- You need real-time processing
- You want something that works out-of-the-box

**Choose ViT-Base if:**
- You want transformer architecture
- You need good accuracy (74-78%)
- You have GPU memory available (>2GB)
- You want something that works out-of-the-box
- MUSIQ is not available (ViT is the implemented alternative)

## Next Steps

1. Run NIMA MobileNetV2 benchmark (already implemented)
2. Run ViT-Base benchmark (already implemented)
3. Compare results against sharpness baseline
4. Fine-tune best model on PhotoTriage if needed (optional)

## Related Documentation

- [Output Format](output_format.md) - Understanding results
- [Benchmark Guide](benchmark.md) - How to run benchmarks
- [Literature Survey](literature_survey.md) - Detailed method descriptions
