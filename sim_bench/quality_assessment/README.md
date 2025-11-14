# Quality Assessment Module

This module provides image quality assessment methods to select the best images from a series of similar images, plus a comprehensive benchmarking framework.

## Methods Implemented

### 1. Rule-Based Quality Assessment
Fast, interpretable methods based on traditional image quality metrics:
- Sharpness (Laplacian variance)
- Exposure (brightness distribution)
- Colorfulness
- Contrast
- Noise level

### 2. CNN-Based Methods (NIMA)
Neural Image Assessment using MobileNet or Inception backbones.
Trained on aesthetic and technical quality datasets.

### 3. Transformer-Based Methods (MUSIQ)
Multi-Scale Image Quality Transformer for state-of-the-art performance.

## Quick Start

### Single Method Evaluation

```python
from sim_bench.quality_assessment import load_quality_method
from sim_bench.quality_assessment.evaluator import QualityEvaluator
from sim_bench.datasets import load_dataset

# Load method
method = load_quality_method('rule_based', config={
    'weights': {
        'sharpness': 0.3,
        'exposure': 0.2,
        'colorfulness': 0.2,
        'contrast': 0.15,
        'noise': 0.15
    }
})

# Load dataset
dataset = load_dataset('phototriage', config)
dataset.load_data()

# Evaluate
evaluator = QualityEvaluator(dataset, method)
results = evaluator.evaluate()

print(f"Top-1 Accuracy: {results['metrics']['top1_accuracy']:.4f}")
```

### Benchmark Multiple Methods

```bash
# Quick test
python run_quality_benchmark.py configs/quality_benchmark.quick.yaml

# Full PhotoTriage benchmark
python run_quality_benchmark.py configs/quality_benchmark.phototriage.yaml

# Generate visualizations
python visualize_quality_benchmark.py outputs/quality_benchmarks/benchmark_*
```

## Benchmark Framework

The benchmark framework provides:
- **Flexible YAML configuration** for datasets and methods
- **Automatic evaluation** with comprehensive metrics
- **Performance comparison** across methods and datasets
- **Visualization generation** (charts and reports)
- **CSV/JSON export** for further analysis

See `docs/QUALITY_BENCHMARK_GUIDE.md` for complete documentation.

## Method Comparison

| Method | Speed | Accuracy | Hardware |
|--------|-------|----------|----------|
| Rule-Based | Very Fast (~2-5ms) | Good | CPU |
| CNN (NIMA) | Fast (~20-50ms) | Better | GPU recommended |
| Transformer (MUSIQ) | Moderate (~50-200ms) | Best | GPU required |

## Evaluation Metrics

- **Top-1 Accuracy**: % where best image is ranked #1
- **Top-2 Accuracy**: % where best image is in top 2
- **Top-3 Accuracy**: % where best image is in top 3
- **Mean Reciprocal Rank (MRR)**: Average of 1/rank for best images
- **Mean Rank**: Average position of best image
- **Processing Time**: Per-image and per-series timing
- **Throughput**: Images processed per second
- **Efficiency**: Accuracy/Time ratio

## Module Structure

```
sim_bench/quality_assessment/
├── __init__.py              # Module interface
├── base.py                  # Base class
├── rule_based.py           # Rule-based methods
├── cnn_methods.py          # CNN methods (NIMA)
├── transformer_methods.py  # Transformer methods (MUSIQ)
├── evaluator.py            # Evaluation framework
├── benchmark.py            # Benchmark runner
├── visualization.py        # Visualization tools
└── README.md              # This file

configs/
├── quality_benchmark.quick.yaml        # Quick test
├── quality_benchmark.phototriage.yaml  # Full benchmark
└── quality_benchmark.multi_dataset.yaml # Multi-dataset

scripts/
├── run_quality_benchmark.py      # Run benchmarks
└── visualize_quality_benchmark.py # Generate visualizations

docs/
├── QUALITY_BENCHMARK_GUIDE.md        # Complete guide
├── QUALITY_ASSESSMENT_QUICKSTART.md  # Quick start
└── IMAGE_SELECTION_SURVEY.md         # Literature review
```

## Example Workflows

### Development Testing
```bash
# Test with small sample
python run_quality_benchmark.py configs/quality_benchmark.quick.yaml
```

### Method Comparison
```bash
# Compare all methods on PhotoTriage
python run_quality_benchmark.py configs/quality_benchmark.phototriage.yaml
python visualize_quality_benchmark.py outputs/quality_benchmarks/benchmark_*
```

### Custom Configuration
```yaml
# my_benchmark.yaml
datasets:
  - name: phototriage
    config: configs/dataset.phototriage.yaml
    sampling: {num_series: 100, seed: 42}

methods:
  - name: my_method
    type: rule_based
    config: {...}
```

```bash
python run_quality_benchmark.py my_benchmark.yaml
```

## Extending the Framework

### Add New Method

1. Implement `QualityAssessor` interface:
```python
# sim_bench/quality_assessment/my_method.py
from sim_bench.quality_assessment.base import QualityAssessor

class MyMethod(QualityAssessor):
    def assess_image(self, image_path):
        # Implementation
        return quality_score
```

2. Register in `load_quality_method()`:
```python
def load_quality_method(method_name, config):
    if method_name == 'my_method':
        return MyMethod(**config)
```

3. Use in config:
```yaml
methods:
  - name: my_method
    type: my_method
    config: {...}
```

### Add New Dataset

Implement `BaseDataset` interface with:
- `get_images()` - List of image paths
- `get_evaluation_data()` - Dict with 'series' containing evaluation data

## Documentation

- **[Benchmark Guide](../../docs/QUALITY_BENCHMARK_GUIDE.md)** - Complete benchmark documentation
- **[Quick Start](../../docs/QUALITY_ASSESSMENT_QUICKSTART.md)** - Getting started guide
- **[Literature Survey](../../docs/IMAGE_SELECTION_SURVEY.md)** - Methods and references
- **[Main README](../../README_QUALITY_BENCHMARK.md)** - Overview and examples

## References

For detailed academic references and method descriptions, see `docs/IMAGE_SELECTION_SURVEY.md`.
