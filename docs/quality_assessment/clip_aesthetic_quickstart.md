# CLIP Aesthetic Assessment - Quick Start

## Overview

CLIP Aesthetic is a zero-shot quality assessment method that uses OpenCLIP to compare images against text prompts describing aesthetic qualities.

**Prompts used**:
- Composition: "well-composed" vs "poorly-composed"
- Subject placement: "subject well placed in frame" vs "not well placed"
- Cropping: "well cropped" vs "poorly cropped"
- Overall quality: "Good Quality photo" vs "Poor Quality photo"

## Running the Benchmark

### Add to Deep Learning Benchmark

CLIP Aesthetic is already configured in `configs/quality_benchmark.deep_learning.yaml`:

```bash
python run_quality_benchmark.py configs/quality_benchmark.deep_learning.yaml
```

This will run:
- NIMA (MobileNetV2 and ResNet50)
- ViT (Vision Transformer)
- **CLIP Aesthetic** (NEW)

### Standalone Test

Quick test on sample images:

```bash
python test_clip_aesthetic.py
```

### Use in Python

```python
from sim_bench.quality_assessment import CLIPAestheticAssessor

# Initialize
assessor = CLIPAestheticAssessor(
    model_name="ViT-B-32",
    pretrained="laion2b_s34b_b79k",
    device="cuda",  # or "cpu"
    aggregation_method="weighted"
)

# Assess image
score = assessor.assess_image("path/to/image.jpg")
print(f"Aesthetic score: {score:.4f}")

# Get detailed breakdown
detailed = assessor.get_detailed_scores("path/to/image.jpg")
for prompt, score in detailed.items():
    print(f"  {prompt}: {score:.4f}")
```

## Configuration Options

### Model Variants

```yaml
# Faster, smaller model (default)
model_name: ViT-B-32
pretrained: laion2b_s34b_b79k

# Larger, more accurate model
model_name: ViT-L-14
pretrained: laion2b_s34b_b79k
```

### Aggregation Methods

```yaml
# Weighted average (recommended, default)
aggregation_method: weighted
# - 50% weight to contrastive pairs
# - 30% weight to positive attributes
# - 20% weight to negative attributes

# Only contrastive pairs (simpler)
aggregation_method: contrastive_only

# Simple mean of all prompts
aggregation_method: mean
```

## Expected Performance

Based on research literature (CLIP-IQA, LIQE):

- **Zero-shot CLIP**: ~0.3-0.5 correlation with human judgments
- **NIMA/MUSIQ**: ~0.7-0.8 correlation (trained models)

**Best use cases**:
- Ensemble with other methods
- Interpretable analysis (see which prompts activate)
- Specific aesthetic dimensions (composition, framing)

## Interpreting Results

### Score Range

- **Typical range**: -0.2 to +0.3
- **Higher is better**: More similarity to positive prompts
- **Relative ranking** matters more than absolute values

### Example Output

```
Image: IMG_1234.jpg
  Overall Score: 0.1567

Contrastive scores:
  contrast_a well-composed photograph: 0.1823
  contrast_a photo with the subject well placed: 0.1456
  contrast_a photo that is well cropped: 0.1389
  contrast_Good Quality photo: 0.1601
```

## Advantages

✅ **Zero-shot**: No training data needed
✅ **Interpretable**: Text prompts explain scores
✅ **Fast**: Single forward pass
✅ **Flexible**: Easy to add new prompts

## Limitations

⚠️ **Not trained for aesthetics**: Uses general vision-language model
⚠️ **Prompt-dependent**: Results vary with phrasing
⚠️ **Semantic confusion**: May conflate content with quality

## Troubleshooting

### "open_clip module not found"

```bash
pip install open-clip-torch
```

### Low correlation with other methods

Try:
1. Different aggregation method (`contrastive_only`)
2. Larger CLIP model (`ViT-L-14`)
3. More focused prompts (edit `CONTRASTIVE_PAIRS` in `clip_aesthetic.py`)

### Scores all similar

- CLIP may not be discriminative for your dataset
- Try more diverse test images
- Check detailed scores to see which prompts activate

## Further Reading

- **Implementation**: `sim_bench/quality_assessment/clip_aesthetic.py`
- **Full analysis**: `docs/quality_assessment/clip_aesthetic_analysis.md`
- **Test script**: `test_clip_aesthetic.py`
- **Research**: CLIP-IQA (2022), LIQE (2023)
