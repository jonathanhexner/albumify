# Image Quality Assessment

Automatically select the best photo from a series of similar images.

## Quick Links

- [Understanding Evaluation](evaluation_explained.html) - **START HERE** - How it all works (HTML)
- [Output Format](output_format.md) - Understanding CSV output files
- [Model Recommendations](model_recommendations.md) - CNN and Transformer recommendations
- [MUSIQ Implementation](MUSIQ_IMPLEMENTATION.md) - Plan for adding MUSIQ
- [Quick Start](quickstart.md) - Get started in 5 minutes
- [Benchmark Guide](benchmark.md) - Run performance tests
- [Literature Survey](literature_survey.md) - Academic methods review
- [PhotoTriage Results](../image_similarity/datasets_phototriage.md) - Our findings

## What is Quality Assessment?

When you have multiple similar photos (burst/series), automatically pick the best one:
- Sharpest photo
- Best exposure
- Most aesthetically pleasing
- Best composition

## Available Methods

### Rule-Based (Fast, No GPU)
- Sharpness, exposure, colorfulness, contrast, noise
- **Speed**: 40-60ms/image (CPU)
- **Accuracy**: 64-65% Top-1

### CNN-Based (Good Balance)
- NIMA with MobileNetV2 or ResNet50
- **Speed**: 20-30ms/image (GPU)
- **Accuracy**: 72-77% Top-1

### Transformer-Based
- ViT-Base (implemented)
  - **Speed**: 40-50ms/image (GPU)
  - **Accuracy**: 74-78% Top-1
- MUSIQ (to be implemented)
  - **Speed**: 50-100ms/image (GPU)
  - **Accuracy**: 75-80% Top-1 (expected)

## Quick Start

```bash
# Run benchmark
python run_quality_benchmark.py configs/quality_benchmark.deep_learning.yaml

# Results in: outputs/quality_benchmarks/benchmark_YYYY-MM-DD_HH-MM-SS/
```

## Documentation Notes

**Markdown Files:** If you have trouble viewing `.md` files in Cursor:
- Use the HTML version: `evaluation_explained.html`
- Open in EmEditor or another markdown viewer
- All content is also available in the HTML version

## Related Documentation

- [Understanding Evaluation](evaluation_explained.html) - How metrics are calculated
- [Output Format](output_format.md) - Understanding CSV output files
- [Benchmark Guide](benchmark.md) - How to run benchmarks
- [Quick Start](quickstart.md) - Getting started
