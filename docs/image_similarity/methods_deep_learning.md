# DINOv2 and OpenCLIP Methods

## Overview

This document describes the DINOv2 and OpenCLIP feature extraction methods added to sim-bench.

---

## DINOv2 (Meta)

**DINOv2** is Meta's self-supervised vision transformer that produces high-quality image embeddings without requiring labeled data.

### Features

- **Self-supervised learning**: Trained on large-scale unlabeled images
- **Multiple model sizes**: small (384-dim), base (768-dim), large (1024-dim), giant (1536-dim)
- **Strong performance**: Excellent for similarity tasks and clustering
- **No domain-specific fine-tuning needed**

### Configuration

```yaml
# configs/methods/dinov2.yaml
method: dinov2
variant: "base"           # Options: "small", "base", "large", "giant"
batch_size: 16
device: "cpu"             # Options: "cpu", "cuda"
normalize: true           # L2 normalization (recommended)
distance: "cosine"        # Distance measure
cache_dir: "artifacts/dinov2"
```

### Model Variants

| Variant | Embedding Dim | Parameters | Speed | Accuracy |
|---------|--------------|------------|-------|----------|
| **small** | 384 | 22M | Fastest | Good |
| **base** | 768 | 86M | Fast | Better |
| **large** | 1024 | 307M | Moderate | Excellent |
| **giant** | 1536 | 1.1B | Slow | Best |

### Usage

```bash
# Basic usage
python -m sim_bench.cli --methods dinov2 --datasets ukbench

# With GPU acceleration
# Edit configs/methods/dinov2.yaml: device: "cuda"
python -m sim_bench.cli --methods dinov2 --datasets holidays

# Different model size
# Edit configs/methods/dinov2.yaml: variant: "large"
python -m sim_bench.cli --methods dinov2 --datasets ukbench
```

### Installation

```bash
pip install torch torchvision
```

### References

- **Paper**: [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
- **Code**: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)

---

## OpenCLIP (LAION)

**OpenCLIP** is an open-source implementation of CLIP (Contrastive Language-Image Pre-training) with various model architectures and pre-trained weights.

### Features

- **Vision-language model**: Learns joint image-text embeddings
- **Multiple architectures**: ViT-B-32, ViT-L-14, ViT-H-14, and more
- **Various pre-training datasets**: LAION-2B, LAION-5B, OpenAI weights
- **Strong zero-shot capabilities**: Works well without fine-tuning

### Configuration

```yaml
# configs/methods/openclip.yaml
method: openclip
model: "ViT-B-32"                      # Model architecture
pretrained: "laion2b_s34b_b79k"        # Pretrained checkpoint
batch_size: 16
device: "cpu"                          # Options: "cpu", "cuda"
normalize: true                        # L2 normalization (recommended)
distance: "cosine"                     # Distance measure
cache_dir: "artifacts/openclip"
```

### Popular Model Configurations

**Fast Models** (512-dim embeddings):
```yaml
model: "ViT-B-32"
pretrained: "laion2b_s34b_b79k"  # Fast, good balance
```

```yaml
model: "ViT-B-16"
pretrained: "laion2b_s34b_b88k"  # Better quality
```

**High-Quality Models**:
```yaml
model: "ViT-L-14"
pretrained: "laion2b_s32b_b82k"  # 768-dim, very strong
```

```yaml
model: "ViT-H-14"
pretrained: "laion2b_s32b_b79k"  # 1024-dim, best quality
```

**OpenAI Original**:
```yaml
model: "ViT-B-32"
pretrained: "openai"              # Original CLIP weights
```

### Usage

```bash
# Basic usage with default config
python -m sim_bench.cli --methods openclip --datasets ukbench

# Quick test on samples
python -m sim_bench.cli --quick --methods openclip --datasets holidays

# Compare with other methods
python -m sim_bench.cli --methods deep,dinov2,openclip --datasets ukbench
```

### Installation

```bash
pip install torch torchvision open-clip-torch
```

### References

- **Paper**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- **Code**: [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)
- **Weights**: [LAION pretrained models](https://github.com/mlfoundations/open_clip#pretrained-model-interface)

---

## Performance Comparison

### Expected Performance (approximate)

**UKBench Dataset (N-S Score out of 4.0):**

| Method | Score | Speed | Memory |
|--------|-------|-------|--------|
| Chi-Square | 2.5-2.8 | Very Fast | Low |
| EMD | 2.7-2.9 | Slow | Low |
| ResNet50 | 2.9-3.2 | Fast | Medium |
| **DINOv2-base** | **3.1-3.4** | **Fast** | **Medium** |
| **DINOv2-large** | **3.3-3.5** | **Moderate** | **High** |
| **OpenCLIP (ViT-B)** | **3.0-3.3** | **Fast** | **Medium** |
| **OpenCLIP (ViT-L)** | **3.2-3.4** | **Moderate** | **High** |
| SIFT BoVW | 2.2-2.7 | Very Slow | Medium |

**INRIA Holidays Dataset (mAP@10):**

| Method | mAP@10 | Speed | Memory |
|--------|--------|-------|--------|
| Chi-Square | 0.35-0.45 | Very Fast | Low |
| EMD | 0.40-0.50 | Slow | Low |
| ResNet50 | 0.65-0.75 | Fast | Medium |
| **DINOv2-base** | **0.75-0.85** | **Fast** | **Medium** |
| **DINOv2-large** | **0.80-0.90** | **Moderate** | **High** |
| **OpenCLIP (ViT-B)** | **0.70-0.80** | **Fast** | **Medium** |
| **OpenCLIP (ViT-L)** | **0.75-0.85** | **Moderate** | **High** |
| SIFT BoVW | 0.45-0.60 | Very Slow | Medium |

---

## Best Practices

### Model Selection

**For speed**:
- DINOv2-small or OpenCLIP ViT-B-32
- Use `batch_size: 32` with GPU

**For accuracy**:
- DINOv2-large or OpenCLIP ViT-L-14
- Enable GPU acceleration

**For balance**:
- DINOv2-base or OpenCLIP ViT-B-16

### GPU Usage

Both methods benefit significantly from GPU acceleration:

```yaml
# In config file
device: "cuda"
batch_size: 32  # Increase for GPU
```

```bash
# Or check available devices
python -c "import torch; print(torch.cuda.is_available())"
```

### Caching

Both methods support feature caching:

```bash
# First run extracts and caches features
python -m sim_bench.cli --methods dinov2 --datasets ukbench

# Subsequent runs with same config load from cache (instant)
python -m sim_bench.cli --methods dinov2 --datasets ukbench
```

Cache files stored in:
- `artifacts/dinov2/*.pkl`
- `artifacts/openclip/*.pkl`

---

## Comparison with Existing Methods

| Aspect | ResNet50 | DINOv2 | OpenCLIP |
|--------|----------|---------|----------|
| **Training** | Supervised (ImageNet) | Self-supervised (large-scale) | Contrastive (image-text) |
| **Features** | Task-specific | General-purpose | Semantic + visual |
| **Zero-shot** | Limited | Excellent | Excellent |
| **Embedding Dim** | 2048 | 384-1536 | 512-1024 |
| **Pre-training Data** | 1.2M images | 142M images | 2B+ image-text pairs |

**When to use each:**
- **ResNet50**: Baseline, well-understood, good for supervised tasks
- **DINOv2**: Best general-purpose visual similarity, no text needed
- **OpenCLIP**: When semantic understanding matters, can leverage text

---

## Troubleshooting

### Import Errors

```
ImportError: PyTorch is required for DINOv2/OpenCLIP
```

**Solution**: Install PyTorch
```bash
pip install torch torchvision
```

### OpenCLIP-specific

```
ImportError: open_clip is required for OpenCLIP method
```

**Solution**: Install open-clip-torch
```bash
pip install open-clip-torch
```

### Model Download Issues

Models are downloaded automatically on first use via torch.hub (DINOv2) or from HuggingFace (OpenCLIP).

**If download fails:**
1. Check internet connection
2. Verify torch.hub cache: `~/.cache/torch/hub/`
3. Verify HuggingFace cache: `~/.cache/huggingface/`

### Memory Issues

```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size: `batch_size: 8`
2. Use smaller model: `variant: "small"` (DINOv2) or `model: "ViT-B-32"` (OpenCLIP)
3. Use CPU: `device: "cpu"`

---

## Future Enhancements

Potential additions:
- [ ] Support for DINOv2 with register tokens
- [ ] OpenCLIP text-based filtering
- [ ] Multi-scale feature extraction
- [ ] Fine-tuning on custom datasets
- [ ] Ensemble methods combining multiple models




