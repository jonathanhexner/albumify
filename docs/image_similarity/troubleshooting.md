# Troubleshooting Guide

## ImportError: cannot import name 'merge_per_query_metrics'

**Problem:**
```python
ImportError: cannot import name 'merge_per_query_metrics' from 'sim_bench.analysis.multi_experiment'
```

**Cause:** Jupyter has cached the old version of the module before the new functions were added.

**Solution:** Restart the Jupyter kernel
1. Menu: **Kernel → Restart Kernel**
2. Re-run cells from the beginning
3. The new functions will now be available

**Alternative:** The notebook now includes a cell that clears the module cache automatically. Just run that cell before importing.

---

## No winning queries found

**Problem:**
```
WARNING: No queries found where deep wins. Try lowering threshold_high or raising threshold_low.
```

**Cause:** The thresholds are too strict - no queries meet the criteria.

**Solution:** Adjust thresholds to be more lenient:
```python
all_wins = find_all_method_wins(
    df_merged=df_merged_results,
    methods=['deep', 'dinov2', 'openclip'],
    metric='ap@10',
    threshold_high=0.8,   # Lower from 0.9
    threshold_low=0.4,    # Higher from 0.3
    top_n=5               # Get more results
)
```

**Explanation:**
- `threshold_high=0.9` means winner must score ≥ 0.9 (very strict)
- `threshold_low=0.3` means losers must score ≤ 0.3 (very strict)
- Lowering/raising these makes it easier to find "wins"

---

## Missing columns in df_merged

**Problem:**
```
ValueError: Missing columns in df_merged: ['ap@10_deep']
```

**Cause:** The metric doesn't exist in the per-query data, or method name is wrong.

**Solution 1:** Check available metrics
```python
# See what metrics are available
print(per_query_dict[('deep', 'holidays')].columns)
```

**Solution 2:** Use a metric that exists
```python
# Common metrics: 'ap@10', 'ap_full', 'recall@10'
df_merged = merge_per_query_metrics(
    per_query_dict,
    methods=['deep', 'dinov2', 'openclip'],
    dataset='holidays',
    metrics=['ap_full']  # Try different metric
)
```

**Solution 3:** Check method names
```python
# See available methods
print(list(set(method for method, dataset in per_query_dict.keys())))
```

---

## Method-dataset combination not found

**Problem:**
```
WARNING: Method-dataset combination not found: ('deep', 'ukbench')
```

**Cause:** The specified method-dataset combination doesn't exist in your loaded data.

**Solution:** Check what's available
```python
# See all available combinations
print("Available combinations:")
for (method, dataset) in per_query_dict.keys():
    print(f"  {method} on {dataset}")
```

Then use only combinations that exist:
```python
df_merged = merge_per_query_metrics(
    per_query_dict,
    methods=['deep', 'dinov2'],  # Only use methods that exist
    dataset='holidays',           # Only use datasets that exist
    metrics=['ap@10']
)
```

---

## Cannot find experiment directory

**Problem:**
```
⚠️  Could not find experiment directory for holidays
```

**Cause:** The visualization code can't find where the method subdirectories are.

**Solution:** Specify the directory explicitly
```python
# Find the correct experiment directory
holidays_experiment_dir = Path('outputs/baseline_runs/comprehensive_baseline/2025-11-02_00-12-26')

# Use it in visualization
saved_figures = visualize_all_method_wins(
    all_wins=all_wins,
    methods=['deep', 'dinov2', 'openclip'],
    metric_name='ap@10',
    dataset_name='holidays',
    dataset_config=DATASET_CONFIG,
    experiment_dir=holidays_experiment_dir,  # Explicit path
    output_dir=holidays_experiment_dir / "analysis_reports" / "method_wins",
    top_n=5
)
```

---

## Images not loading in visualization

**Problem:** Generated PNGs are empty or images don't show.

**Cause:** Image paths in DATASET_CONFIG don't match actual file locations.

**Solution:** Verify dataset configuration
```python
# Check if paths are correct
from pathlib import Path

dataset_root = Path('D:/Similar Images/DataSets/InriaHolidaysFull')
print(f"Dataset root exists: {dataset_root.exists()}")

# List some images
images = list(dataset_root.glob('*.jpg'))
print(f"Found {len(images)} images")
print(f"Sample: {images[0] if images else 'None'}")
```

Update DATASET_CONFIG with correct paths:
```python
DATASET_CONFIG = {
    'name': 'holidays',
    'root': 'D:/Similar Images/DataSets/InriaHolidaysFull',  # Must exist!
    'pattern': '*.jpg'
}
```

---

## Need Help?

Check these resources:
- **Quick start**: `QUICK_START.md`
- **Notebook guide**: `NOTEBOOK_USAGE_SUMMARY.md`
- **Complete guide**: `COMPARISON_ANALYSIS_GUIDE.md`
- **Function reference**: See docstrings in the code

Or create an issue with your error message and I'll help!
