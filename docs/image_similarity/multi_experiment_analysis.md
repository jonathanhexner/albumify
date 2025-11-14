# Multi-Experiment Analysis Guide

## Overview

The updated `methods_comparison.ipynb` notebook now supports analyzing results across multiple experiments, methods, and datasets simultaneously.

## Key Features

### 1. Automatic Experiment Discovery
Automatically scan a directory tree to find all experiment results:
```python
AUTO_SCAN = True
EXPERIMENT_BASE_DIR = PROJECT_ROOT / "outputs" / "baseline_runs" / "comprehensive_baseline"
```

### 2. Manual Experiment Selection
Specify exact experiment directories if you need precise control:
```python
AUTO_SCAN = False
EXPERIMENT_DIRS = [
    PROJECT_ROOT / "outputs" / "run1",
    PROJECT_ROOT / "outputs" / "run2"
]
```

### 3. Duplicate Detection
If the same method-dataset combination is found in multiple runs, the system:
- **Takes the first occurrence**
- **Shows a prominent warning** that cannot be missed:
```
================================================================================
⚠️  WARNING: DUPLICATE DETECTED!
================================================================================
Method: deep
Dataset: holidays
Already found in previous run, skipping: ...
================================================================================
```

## What Gets Analyzed

### Per-Dataset Analysis
- **Method comparison within each dataset**: Which method performs best on holidays? On ukbench?
- **Performance tables**: Methods × Metrics for each dataset
- **Rankings**: Methods ranked by average performance per dataset

### Cross-Dataset Analysis
- **Pivot tables**: Methods × Datasets comparison
- **Dataset difficulty**: Which datasets are harder (lower average scores)?
- **Method consistency**: Does a method perform well across all datasets?

### Correlation Analysis
- **Per-dataset**: Do methods agree on which queries are difficult within each dataset?
- **Heatmaps**: Visual correlation matrices for each dataset separately

## Visualizations

### Bar Charts
- Per-dataset method comparison
- Multiple metrics side-by-side
- Easy to see best performer per dataset

### Box Plots
Two implementations available:

**fivecentplots** (recommended, install with `pip install fivecentplots`):
- Advanced grouped box plots
- Methods grouped by dataset
- Better visual clarity for complex comparisons

**matplotlib/seaborn** (fallback):
- Standard box plots
- Automatic fallback if fivecentplots not installed

### Correlation Heatmaps
- One heatmap per dataset
- Shows method agreement on query difficulty
- Helps understand if methods fail on similar queries

## Example Directory Structure

```
outputs/
└── baseline_runs/
    └── comprehensive_baseline/
        ├── run_2025-10-24/
        │   ├── holidays_results/
        │   │   ├── deep/
        │   │   │   ├── metrics.csv
        │   │   │   ├── per_query.csv
        │   │   │   └── manifest.json  ← Contains dataset name
        │   │   ├── emd/
        │   │   └── sift_bovw/
        │   └── ukbench_results/
        │       ├── deep/
        │       ├── emd/
        │       └── sift_bovw/
        └── run_2025-10-27/
            └── holidays_results/
                └── dinov2/
```

## Dataset Detection

The system detects dataset names from `manifest.json` files in method directories:
```json
{
  "method": "deep",
  "dataset": {
    "name": "holidays"
  }
}
```

This ensures accurate dataset identification even with complex directory structures.

## Key Metrics

The notebook focuses on these metrics for comparison:
- `recall@1` - Accuracy (top-1 retrieval)
- `recall@10` - Recall in top-10 results
- `map@10` - Mean Average Precision at 10
- `map` - Full Mean Average Precision

Additional metrics from your experiments are also loaded and can be selected for comparison.

## Usage Example

### Scenario: Compare DINOv2, ResNet50, and SIFT on Holidays and UKBench

```python
# In notebook configuration cell:
AUTO_SCAN = True
EXPERIMENT_BASE_DIR = PROJECT_ROOT / "outputs" / "baseline_runs" / "comprehensive_baseline"
```

**Expected Output:**
```
Scanning: outputs/baseline_runs/comprehensive_baseline
  Found: deep on holidays in run_2025-10-24
  Found: deep on ukbench in run_2025-10-24
  Found: sift_bovw on holidays in run_2025-10-24
  Found: sift_bovw on ukbench in run_2025-10-24
  Found: dinov2 on holidays in run_2025-10-27
  Found: dinov2 on ukbench in run_2025-10-27

✓ Found 6 method-dataset combinations
  holidays: 3 methods (deep, dinov2, sift_bovw)
  ukbench: 3 methods (deep, dinov2, sift_bovw)
```

**Analysis Provided:**
1. Per-dataset tables showing which method is best
2. Cross-dataset pivot table (methods × datasets)
3. Dataset difficulty ranking
4. Box plots showing per-query distributions
5. Correlation heatmaps per dataset

## Backward Compatibility

The new notebook handles:
- **Old metric names**: Automatically converts `map_full` → `map`, `prec@10` → `precision@10`
- **Single experiment**: Works fine with just one experiment directory
- **Single method**: Skips correlation analysis if only one method per dataset

## Tips

### Best Practices
1. **Use AUTO_SCAN** for comprehensive analysis
2. **Check for warnings** about duplicates
3. **Install fivecentplots** for better visualizations: `pip install fivecentplots`
4. **Use key metrics** that work across all datasets

### Common Issues

**Issue**: "No valid experiments found"
**Solution**: Check that directories contain method subdirectories with `metrics.csv` files

**Issue**: "Dataset name not found"
**Solution**: Ensure `manifest.json` files exist in method directories with dataset information

**Issue**: Duplicate warnings appearing
**Solution**: This is expected if same method-dataset combination exists in multiple runs. First one is used, others skipped.

## Related Documentation

- [ANALYSIS_GUIDE.md](ANALYSIS_GUIDE.md) - Overview of all analysis notebooks
- [DATASETS.md](DATASETS.md) - Dataset details
- [METRICS_BUG_FIX.md](METRICS_BUG_FIX.md) - Metric naming issues (if applicable)
