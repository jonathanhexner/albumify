# Quick Start: Method Comparison Analysis

## In Your Notebook (5 Lines of Code)

```python
from sim_bench.analysis.multi_experiment import merge_per_query_metrics, find_all_method_wins
from sim_bench.analysis.comparison_viz import visualize_all_method_wins

# 1. Merge metrics (replaces your 14-line manual loop)
df_merged = merge_per_query_metrics(per_query_dict, ['deep', 'dinov2', 'openclip'], 'holidays', ['ap@10'])

# 2. Find winning queries
all_wins = find_all_method_wins(df_merged, ['deep', 'dinov2', 'openclip'], 'ap@10', threshold_high=0.9, threshold_low=0.3, top_n=3)

# 3. Visualize (generates all comparison figures automatically)
saved_figs = visualize_all_method_wins(all_wins, ['deep', 'dinov2', 'openclip'], 'ap@10', 'holidays', DATASET_CONFIG, EXPERIMENT_DIR, OUTPUT_DIR, top_n=5, show_plots=True)
```

## What You Get

**Output Files:**
```
method_comparison_wins/
├── comparison_deep_wins_q123.png       # Deep excels, others fail
├── comparison_deep_wins_q456.png
├── comparison_deep_wins_q789.png
├── comparison_dinov2_wins_q234.png     # Dinov2 excels, others fail
├── comparison_dinov2_wins_q567.png
├── comparison_dinov2_wins_q890.png
├── comparison_openclip_wins_q345.png   # OpenCLIP excels, others fail
├── comparison_openclip_wins_q678.png
└── comparison_openclip_wins_q901.png
```

**Each figure shows:**
- Query image at top
- 3 columns (one per method) with top-5 results
- Color-coded scores: 🟢 green (good) / 🔴 red (bad)
- Match indicators: ✓ (correct) / ✗ (incorrect)

## Configuration

```python
DATASET_CONFIG = {
    'name': 'holidays',
    'root': 'D:/Similar Images/DataSets/InriaHolidaysFull',
    'pattern': '*.jpg'
}
```

## Adjust Parameters

**More lenient (find more wins):**
```python
all_wins = find_all_method_wins(..., threshold_high=0.8, threshold_low=0.4, top_n=5)
```

**Compare different methods:**
```python
merge_per_query_metrics(per_query_dict, ['deep', 'emd', 'sift_bovw'], 'holidays', ['ap@10'])
```

**Show more results:**
```python
visualize_all_method_wins(..., top_n=10, show_plots=False)
```

## Run the Updated Notebook

1. Open: `sim_bench/analysis/methods_comparison_updated.ipynb`
2. Run cells up to Section 5 (loads experiment data)
3. Run new Section 6 (method-specific wins analysis)
4. Check output: `outputs/.../analysis_reports/method_comparison_wins/`

## Documentation

- **Complete guide**: `COMPARISON_ANALYSIS_GUIDE.md`
- **Notebook reference**: `NOTEBOOK_USAGE_SUMMARY.md`
- **Update details**: `NOTEBOOK_UPDATE_SUMMARY.md`
- **Standalone example**: `examples/method_comparison_example.py`

## Functions Reference

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `merge_per_query_metrics()` | Combine metrics from multiple methods | per_query_dict, methods, dataset, metrics | Wide DataFrame |
| `find_all_method_wins()` | Find queries where each method excels | merged DataFrame, methods, thresholds | Dict[method → wins] |
| `visualize_all_method_wins()` | Generate comparison figures | wins, methods, config | Dict[method → paths] |

That's it! 🎉
