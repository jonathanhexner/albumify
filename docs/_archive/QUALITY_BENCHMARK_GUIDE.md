# Quality Assessment Benchmark Guide

Complete guide to benchmarking image quality assessment methods.

## Overview

The quality assessment benchmark framework provides:
- **Flexible configuration** - Easy YAML-based setup
- **Multiple methods** - Compare rule-based, CNN, and transformer approaches
- **Multiple datasets** - Test on PhotoTriage and other datasets
- **Comprehensive metrics** - Accuracy, speed, efficiency
- **Visualizations** - Automatic charts and reports

## Quick Start

### 1. Run a Quick Test

```bash
# Fast test with sampled data (20 series)
python run_quality_benchmark.py configs/quality_benchmark.quick.yaml
```

### 2. Full PhotoTriage Benchmark

```bash
# Complete evaluation on PhotoTriage dataset
python run_quality_benchmark.py configs/quality_benchmark.phototriage.yaml
```

### 3. Multi-Dataset Comparison

```bash
# Compare across multiple datasets
python run_quality_benchmark.py configs/quality_benchmark.multi_dataset.yaml
```

### 4. Generate Visualizations

```bash
# After benchmark completes, generate charts and report
python visualize_quality_benchmark.py outputs/quality_benchmarks/benchmark_2025-11-12_12-34-56
```

## Configuration

### Basic Structure

```yaml
datasets:
  - name: phototriage
    config: configs/dataset.phototriage.yaml
    sampling:  # Optional
      strategy: random
      num_series: 100
      seed: 42

methods:
  - name: composite_rule_based
    type: rule_based
    config:
      weights:
        sharpness: 0.3
        exposure: 0.2
        colorfulness: 0.2
        contrast: 0.15
        noise: 0.15

settings:
  verbose: true
  save_visualizations: false
```

### Datasets Configuration

```yaml
datasets:
  # Full dataset
  - name: phototriage
    config: configs/dataset.phototriage.yaml
  
  # Sampled dataset
  - name: phototriage
    config: configs/dataset.phototriage.yaml
    sampling:
      strategy: random
      num_series: 50  # Use only 50 series
      seed: 42        # Reproducible sampling
  
  # Multiple datasets
  - name: phototriage
    config: configs/dataset.phototriage.yaml
  - name: budapest
    config: configs/dataset.budapest.yaml
```

### Methods Configuration

#### Rule-Based Methods

```yaml
methods:
  # Composite scoring
  - name: composite_rule_based
    type: rule_based
    config:
      weights:
        sharpness: 0.3
        exposure: 0.2
        colorfulness: 0.2
        contrast: 0.15
        noise: 0.15
  
  # Single metric
  - name: sharpness_only
    type: rule_based
    config:
      weights:
        sharpness: 1.0
        exposure: 0.0
        colorfulness: 0.0
        contrast: 0.0
        noise: 0.0
```

#### CNN Methods (NIMA)

```yaml
methods:
  - name: nima_mobilenet
    type: nima
    config:
      backbone: mobilenet  # or 'inception'
      weights_path: null   # Auto-download pretrained
      device: cuda         # or 'cpu'
```

#### Transformer Methods (MUSIQ)

```yaml
methods:
  - name: musiq_aesthetic
    type: musiq
    config:
      variant: ava      # or 'koniq', 'spaq'
      weights_path: null
      device: cuda
```

## Output Structure

After running a benchmark, you'll get:

```
outputs/quality_benchmarks/benchmark_2025-11-12_12-34-56/
├── config.yaml                    # Configuration used
├── summary.json                   # Complete results summary
├── methods_summary.csv            # Method performance table
├── detailed_results.csv           # Per-dataset, per-method results
├── phototriage_results.json       # Full results for PhotoTriage
├── phototriage_composite_series.json  # Per-series details
└── ... (per-dataset, per-method files)
```

### Key Files

**summary.json**: Complete results with rankings
```json
{
  "benchmark_info": {...},
  "datasets": {...},
  "methods": {...},
  "comparison": {
    "accuracy_ranking": [...],
    "speed_ranking": [...],
    "efficiency_ranking": [...]
  }
}
```

**methods_summary.csv**: Quick comparison table
```csv
method,avg_top1_accuracy,avg_top2_accuracy,avg_mrr,avg_time_ms,datasets_tested
composite_rule_based,0.7234,0.8912,0.7823,2.34,1
nima_mobilenet,0.7891,0.9234,0.8345,45.67,1
```

**detailed_results.csv**: Per-dataset results
```csv
dataset,method,top1_accuracy,top2_accuracy,mrr,avg_time_ms,throughput
phototriage,composite_rule_based,0.7234,0.8912,0.7823,2.34,427.35
phototriage,nima_mobilenet,0.7891,0.9234,0.8345,45.67,21.90
```

## Metrics Explained

### Accuracy Metrics

- **Top-1 Accuracy**: % of series where best image is ranked #1
- **Top-2 Accuracy**: % of series where best image is in top 2
- **Top-3 Accuracy**: % of series where best image is in top 3
- **Mean Reciprocal Rank (MRR)**: Average of 1/rank for best images
- **Mean Rank**: Average position of best image in rankings

### Speed Metrics

- **Total Time (s)**: Total evaluation time
- **Per-Series Time (ms)**: Average time per image series
- **Per-Image Time (ms)**: Average time per image
- **Throughput (img/s)**: Images processed per second

### Efficiency Score

- **Efficiency**: `Accuracy / Time` - balances quality and speed
- Higher is better - rewards high accuracy with low time

## Visualizations

After running `visualize_quality_benchmark.py`, you'll get:

### 1. Accuracy Comparison
Bar charts comparing Top-1 and Top-2 accuracy across methods.

### 2. Speed Comparison
Bar chart of processing time per image (log scale).
- Green: < 10ms (fast)
- Orange: 10-50ms (moderate)
- Red: > 50ms (slow)

### 3. Accuracy vs Speed
Scatter plot showing the tradeoff between accuracy and speed.

### 4. Per-Dataset Comparison
Separate charts for each dataset (if multiple).

### 5. Markdown Report
Complete report with tables, rankings, and embedded visualizations.

## Advanced Usage

### Custom Output Directory

```bash
python run_quality_benchmark.py config.yaml --output my_results/
```

### Quiet Mode

```bash
python run_quality_benchmark.py config.yaml --quiet
```

### GPU vs CPU Comparison

```yaml
methods:
  - name: nima_cpu
    type: nima
    config:
      device: cpu
  
  - name: nima_gpu
    type: nima
    config:
      device: cuda
```

### Sampling Strategies

```yaml
datasets:
  - name: phototriage
    config: configs/dataset.phototriage.yaml
    sampling:
      strategy: random     # Random sampling
      num_series: 100
      seed: 42
  
  - name: phototriage
    config: configs/dataset.phototriage.yaml
    sampling:
      strategy: balanced   # Balanced across categories
      num_series: 100
      seed: 42
```

## Best Practices

### 1. Start Small
Begin with `quality_benchmark.quick.yaml` to verify setup:
```bash
python run_quality_benchmark.py configs/quality_benchmark.quick.yaml
```

### 2. Use Appropriate Sampling
For development/testing:
- Use `num_series: 20-50` for quick iteration
- Use full dataset for final evaluation

### 3. Monitor GPU Memory
If running out of GPU memory:
```yaml
config:
  device: cpu  # Switch to CPU
  # or reduce batch size in method implementation
```

### 4. Compare Similar Methods
Group similar methods together:
```yaml
methods:
  # Rule-based variants
  - name: sharpness_only
    type: rule_based
    config: {...}
  - name: composite
    type: rule_based
    config: {...}
  
  # CNN variants
  - name: nima_mobilenet
    type: nima
    config: {backbone: mobilenet}
  - name: nima_inception
    type: nima
    config: {backbone: inception}
```

### 5. Document Your Runs
Add comments to configs:
```yaml
# Experiment: Testing impact of sharpness weight
# Date: 2025-11-12
# Hypothesis: Higher sharpness weight improves accuracy

methods:
  - name: high_sharpness
    type: rule_based
    config:
      weights: {sharpness: 0.5, exposure: 0.2, ...}
```

## Troubleshooting

### "Dataset not found"
Check that dataset config files exist:
```bash
ls configs/dataset.*.yaml
```

### "CUDA out of memory"
Switch to CPU or use smaller model:
```yaml
config:
  device: cpu
```

### "Method takes too long"
Use sampling to test on subset:
```yaml
sampling:
  num_series: 20  # Small sample
```

### "Empty results"
Check that dataset has evaluation data:
```python
from sim_bench.datasets import load_dataset
ds = load_dataset('phototriage', config)
ds.load_data()
print(ds.get_evaluation_data())
```

## Examples

### Example 1: Quick Development Test

```bash
# Create a quick test config
cat > configs/quality_benchmark.dev.yaml << EOF
datasets:
  - name: phototriage
    config: configs/dataset.phototriage.yaml
    sampling: {strategy: random, num_series: 10, seed: 42}

methods:
  - name: test_method
    type: rule_based
    config: {weights: {sharpness: 1.0}}
EOF

# Run it
python run_quality_benchmark.py configs/quality_benchmark.dev.yaml
```

### Example 2: Method Comparison Study

```bash
# Compare all rule-based weight configurations
python run_quality_benchmark.py configs/quality_benchmark.phototriage.yaml

# Generate report
python visualize_quality_benchmark.py outputs/quality_benchmarks/benchmark_*
```

### Example 3: Multi-Dataset Validation

```bash
# Test method generalization across datasets
python run_quality_benchmark.py configs/quality_benchmark.multi_dataset.yaml
```

## Integration with Existing Workflows

### Using with Existing Datasets

```yaml
datasets:
  - name: my_custom_dataset
    config: configs/dataset.my_custom_dataset.yaml
```

Your dataset must implement:
- `get_images()` - Return list of image paths
- `get_evaluation_data()` - Return dict with 'series' structure
- See `sim_bench/datasets/base.py` for interface

### Adding Custom Methods

1. Implement method in `sim_bench/quality_assessment/`
2. Add to `load_quality_method()` in `__init__.py`
3. Use in config:
```yaml
methods:
  - name: my_method
    type: my_custom_method
    config: {...}
```

## Performance Expectations

### PhotoTriage Dataset (~2000 series)

**Rule-Based Methods:**
- Time: 2-5ms per image
- Full dataset: ~5-10 minutes on CPU

**CNN Methods (NIMA):**
- Time: 20-50ms per image (GPU)
- Full dataset: ~40-100 minutes on GPU

**Transformer Methods (MUSIQ):**
- Time: 50-200ms per image (GPU)
- Full dataset: ~100-400 minutes on GPU

**Recommendations:**
- Development: Use sampled data (50-100 series)
- Final evaluation: Use full dataset
- GPU strongly recommended for deep learning methods

## Related Documentation

- [Quality Assessment Quickstart](QUALITY_ASSESSMENT_QUICKSTART.md)
- [Image Selection Survey](IMAGE_SELECTION_SURVEY.md)
- [Multi-Experiment Analysis](MULTI_EXPERIMENT_ANALYSIS.md)







