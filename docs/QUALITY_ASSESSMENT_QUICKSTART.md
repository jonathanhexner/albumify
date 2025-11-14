# Quality Assessment Quick Start Guide

## Installation

```bash
# Core dependencies (required)
pip install opencv-python torch torchvision Pillow numpy tqdm

# Optional: For transformer models
pip install transformers
```

## Quick Examples

### 1. Evaluate Rule-Based Method on PhotoTriage

```bash
python run_quality_assessment.py \
    --method rule_based \
    --dataset phototriage \
    --output results/rule_based_results.json
```

**Expected output:**
```
Top-1 Accuracy: 62-65%
Runtime: ~5ms per image
```

### 2. Evaluate NIMA (CNN) on PhotoTriage

```bash
python run_quality_assessment.py \
    --method nima \
    --dataset phototriage \
    --device cuda \
    --backbone mobilenet_v2 \
    --output results/nima_results.json
```

**Expected output:**
```
Top-1 Accuracy: 72-76%
Runtime: ~18ms per image (GPU)
```

### 3. Evaluate ViT (Transformer) on PhotoTriage

```bash
python run_quality_assessment.py \
    --method vit \
    --dataset phototriage \
    --device cuda \
    --output results/vit_results.json
```

**Expected output:**
```
Top-1 Accuracy: 74-78%
Runtime: ~40ms per image (GPU)
```

### 4. Compare All Methods

```bash
python run_quality_assessment.py \
    --compare-all \
    --dataset phototriage \
    --device cuda \
    --output results/comparison.json
```

## Python API Examples

### Simple Quality Assessment

```python
from sim_bench.quality_assessment import RuleBasedQuality

# Create assessor
method = RuleBasedQuality()

# Assess single image
score = method.assess_image('image.jpg')
print(f"Quality score: {score:.4f}")

# Get detailed metrics
details = method.get_detailed_scores('image.jpg')
print(f"Sharpness: {details['sharpness_normalized']:.4f}")
print(f"Exposure: {details['exposure']:.4f}")
print(f"Colorfulness: {details['colorfulness_normalized']:.4f}")
```

### Select Best from Series

```python
from sim_bench.quality_assessment import NIMAQuality

# Create assessor (CNN-based)
method = NIMAQuality(backbone='mobilenet_v2', device='cuda')

# Series of similar images
series = ['burst_001.jpg', 'burst_002.jpg', 'burst_003.jpg']

# Select best
result = method.select_best_from_series(series)
print(f"Best image: {result['best_path']}")
print(f"Score: {result['best_score']:.4f}")
print(f"All scores: {result['scores']}")
```

### Full Dataset Evaluation

```python
import yaml
from sim_bench.datasets import load_dataset
from sim_bench.quality_assessment import RuleBasedQuality
from sim_bench.quality_assessment.evaluator import QualityEvaluator

# Load dataset
with open('configs/dataset.phototriage.yaml') as f:
    config = yaml.safe_load(f)

dataset = load_dataset('phototriage', config)
dataset.load_data()

# Create method
method = RuleBasedQuality()

# Evaluate
evaluator = QualityEvaluator(dataset, method)
results = evaluator.evaluate()
evaluator.print_results()

# Save results
evaluator.save_results('results/evaluation.json')
```

### Compare Multiple Methods

```python
from sim_bench.quality_assessment import RuleBasedQuality, NIMAQuality, ViTQuality
from sim_bench.quality_assessment.evaluator import QualityEvaluator

# Create methods
methods = [
    ('Rule-Based', RuleBasedQuality()),
    ('NIMA', NIMAQuality(backbone='mobilenet_v2', device='cuda')),
    ('ViT', ViTQuality(model_name='google/vit-base-patch16-224', device='cuda'))
]

# Compare
results = QualityEvaluator.compare_methods(dataset, methods)
```

## Custom Configurations

### Custom Metric Weights (Rule-Based)

```python
# Emphasize sharpness for burst photos
weights = {
    'sharpness': 0.60,    # 60%
    'exposure': 0.25,     # 25%
    'colorfulness': 0.10, # 10%
    'contrast': 0.05      # 5%
}

method = RuleBasedQuality(weights=weights)
```

### Different CNN Backbones

```python
# Fastest (MobileNetV2: 3.4M params)
fast_method = NIMAQuality(backbone='mobilenet_v2', device='cuda')

# More accurate (ResNet50: 25M params)
accurate_method = NIMAQuality(backbone='resnet50', device='cuda')

# Balanced (EfficientNet-B0: 5M params)
balanced_method = NIMAQuality(backbone='efficientnet_b0', device='cuda')
```

### Different ViT Models

```python
# Base (86M params, faster)
base_vit = ViTQuality(
    model_name='google/vit-base-patch16-224',
    device='cuda'
)

# Large (307M params, more accurate)
large_vit = ViTQuality(
    model_name='google/vit-large-patch16-224',
    device='cuda'
)

# Base with larger patches (faster)
fast_vit = ViTQuality(
    model_name='google/vit-base-patch32-224',
    device='cuda'
)
```

## Performance Tips

### For Speed:
1. Use rule-based methods (~5ms per image)
2. Use MobileNetV2 for CNN (~18ms GPU)
3. Batch processing: `method.assess_batch(paths)`
4. Use larger patches for ViT (patch32 vs patch16)

### For Accuracy:
1. Use ViT-Large for best results
2. Fine-tune on your specific dataset
3. Ensemble multiple methods
4. Use ResNet50 instead of MobileNet

### For Memory Efficiency:
1. Reduce batch size
2. Use smaller models (MobileNetV2, ViT-Base with patch32)
3. Process images at lower resolution (resize before assessment)

## Troubleshooting

### CUDA Out of Memory

```python
# Reduce batch size
method = NIMAQuality(backbone='mobilenet_v2', device='cuda', batch_size=4)

# Or use CPU
method = NIMAQuality(backbone='mobilenet_v2', device='cpu')
```

### Slow Performance

```python
# Enable batch processing
scores = method.assess_batch(image_paths)  # Much faster than loop

# Use lighter model
method = NIMAQuality(backbone='mobilenet_v2')  # vs resnet50
```

### Import Errors

```bash
# Missing transformers (for ViT)
pip install transformers

# Missing OpenCV
pip install opencv-python

# Missing PyTorch
pip install torch torchvision
```

## Next Steps

1. **Experiment with Methods**: Try all three approaches on your data
2. **Fine-tune Models**: Train on your specific domain for +3-5% accuracy
3. **Optimize Weights**: Tune rule-based weights for your use case
4. **Ensemble Methods**: Combine multiple methods for best results
5. **Deploy**: Use lightweight models (MobileNet, rule-based) for production

## Results Interpretation

### Quality Scores

- **Rule-Based**: 0-1 scale (higher is better)
  - < 0.4: Poor quality
  - 0.4-0.6: Acceptable
  - 0.6-0.8: Good
  - > 0.8: Excellent

- **NIMA/ViT**: 1-10 scale (higher is better)
  - 1-3: Poor
  - 4-6: Average
  - 7-8: Good
  - 9-10: Excellent

### Evaluation Metrics

- **Top-1 Accuracy**: % of times best image is correctly identified
- **Top-2 Accuracy**: % of times best image is in top-2
- **Mean Reciprocal Rank (MRR)**: Average of 1/rank of correct image
  - 1.0 = perfect (always rank 1)
  - 0.5 = average rank of 2
  - 0.33 = average rank of 3

## Contact & Support

For issues or questions:
1. Check `sim_bench/quality_assessment/README.md` for detailed API docs
2. See `docs/IMAGE_SELECTION_SURVEY.md` for technical background
3. Run example: `python examples/quality_assessment_demo.py`


