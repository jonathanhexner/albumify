# Running Quality Assessment Benchmarks

Quick guide to running comprehensive benchmarks on PhotoTriage.

## Quick Start

### Option 1: Quick Test (Recommended First)

Test all methods on 100 series (~5-10 minutes):

```bash
python run_comprehensive_benchmark.py --quick
```

Or directly:
```bash
python run_quality_benchmark.py configs/quality_benchmark.quick_test.yaml
```

### Option 2: Full Benchmark

Run all methods on full PhotoTriage dataset (~30-60 minutes):

```bash
python run_comprehensive_benchmark.py
```

Or directly:
```bash
python run_quality_benchmark.py configs/quality_benchmark.comprehensive.yaml
```

## What Gets Tested

### Rule-Based Methods (7 methods)
- `sharpness_only` - Sharpness metric only
- `exposure_only` - Exposure quality only
- `colorfulness_only` - Colorfulness metric only
- `contrast_only` - Contrast metric only
- `composite_balanced` - Balanced combination (40% sharpness, 30% exposure, 20% color, 10% contrast)
- `composite_sharpness_focused` - Sharpness-focused (60% sharpness)
- `composite_exposure_focused` - Exposure-focused (50% exposure)

### CNN Methods (2 methods)
- `nima_mobilenet` - NIMA with MobileNetV2 backbone (fast)
- `nima_resnet50` - NIMA with ResNet50 backbone (more accurate)

### Transformer Methods (1 method)
- `vit_base` - Vision Transformer Base

**Total: 10 methods**

## Configuration Files

### `configs/quality_benchmark.comprehensive.yaml`
- Full PhotoTriage dataset (all ~4,986 series)
- All 10 methods
- Use for final evaluation

### `configs/quality_benchmark.quick_test.yaml`
- 100 random series (sampled)
- Core methods (7 total)
- Use for quick testing/validation

### `configs/quality_benchmark.deep_learning.yaml`
- Deep learning methods only (NIMA + ViT)
- Baseline (sharpness_only) for comparison

## GPU vs CPU

**Default:** Uses GPU (`device: cuda`)

**To use CPU instead:**
1. Edit the config file
2. Change `device: cuda` to `device: cpu` for deep learning methods
3. Note: Rule-based methods always use CPU

**Check GPU availability:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Output Files

Results are saved to:
```
outputs/quality_benchmarks/benchmark_YYYY-MM-DD_HH-MM-SS/
```

**Key files:**
- `methods_summary.csv` - Overall comparison
- `detailed_results.csv` - Per-dataset, per-method
- `phototriage_[method]_series.csv` - Per-series results (one per method)
- `benchmark.log` - Full execution log
- `config.yaml` - Copy of configuration used

## Expected Runtime

**Quick test (100 series):**
- Rule-based: ~2-3 minutes
- NIMA MobileNetV2: ~3-5 minutes
- ViT-Base: ~5-8 minutes
- **Total: ~10-15 minutes**

**Full benchmark (~4,986 series):**
- Rule-based: ~10-15 minutes
- NIMA MobileNetV2: ~20-30 minutes
- NIMA ResNet50: ~30-45 minutes
- ViT-Base: ~45-60 minutes
- **Total: ~1.5-2 hours**

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in method config
- Use CPU instead: `device: cpu`
- Use smaller model (MobileNetV2 instead of ResNet50)

### Import Errors
```bash
# Install PyTorch (if missing)
pip install torch torchvision

# Install Transformers (for ViT)
pip install transformers
```

### Dataset Not Found
- Check `configs/dataset.phototriage.yaml`
- Verify path: `D:/Similar Images/automatic_triage_photo_series`

## Example Output

After running, you'll see:
```
================================================================================
BENCHMARK SUMMARY
================================================================================

Method Rankings by Accuracy:
  1. nima_mobilenet           0.7450 (74.50%)
  2. vit_base                 0.7320 (73.20%)
  3. sharpness_only           0.6495 (64.95%)
  ...

Results saved to: outputs/quality_benchmarks/benchmark_2025-01-15_10-30-00
```

## Next Steps

1. **Review results** in CSV files
2. **Check log file** for any errors: `benchmark.log`
3. **Analyze per-series results** for failure cases
4. **Compare methods** using `methods_summary.csv`

See [Output Format](output_format.md) for detailed explanation of all output files.




