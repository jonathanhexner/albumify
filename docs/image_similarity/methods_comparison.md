# Method Comparison Analysis Guide

This guide shows how to compare multiple methods and visualize cases where each method excels.

## Quick Start

```python
from pathlib import Path
from sim_bench.analysis.multi_experiment import (
    load_experiments,
    merge_per_query_metrics,
    find_all_method_wins
)
from sim_bench.analysis.comparison_viz import visualize_all_method_wins

# Load experiment data
base_dir = Path('outputs/baseline_runs/comprehensive_baseline')
metrics_df, per_query_dict, experiment_infos = load_experiments(
    base_dir=base_dir,
    auto_scan=True
)

# Merge per-query metrics for multiple methods
df_merged = merge_per_query_metrics(
    per_query_dict=per_query_dict,
    methods=['deep', 'dinov2', 'openclip'],
    dataset='holidays',
    metrics=['ap@10']
)

# Find queries where each method excels but others fail
all_wins = find_all_method_wins(
    df_merged=df_merged,
    methods=['deep', 'dinov2', 'openclip'],
    metric='ap@10',
    threshold_high=0.9,  # Winner must score >= 0.9
    threshold_low=0.3,   # Losers must score <= 0.3
    top_n=3              # Find top 3 per method
)

# Visualize comparisons
dataset_config = {
    'name': 'holidays',
    'root': 'D:/Similar Images/DataSets/InriaHolidaysFull',
    'pattern': '*.jpg'
}

saved_figures = visualize_all_method_wins(
    all_wins=all_wins,
    methods=['deep', 'dinov2', 'openclip'],
    metric_name='ap@10',
    dataset_name='holidays',
    dataset_config=dataset_config,
    experiment_dir=Path('outputs/baseline_runs/comprehensive_baseline/2025-11-02_00-12-26'),
    output_dir=Path('sim_bench/analysis/reports/method_wins'),
    top_n=5
)
```

## Functions

### `merge_per_query_metrics()`

Merges per-query metrics from multiple methods into a wide-format DataFrame.

**Before:**
```python
# Manual merging code (OLD)
metrics = ['ap@10']
dataset = 'holidays'
df_merged = pd.DataFrame()
methods = ['deep', 'dinov2', 'openclip']
for n_method, method in enumerate(methods):
    metrics_method = list(map(lambda x: x+ '_' + method, metrics))
    if n_method == 0:
        df_merged = per_query_dict[(method, dataset)][['query_idx', 'query_path', 'group_id', 'num_relevant']+metrics]
        df_merged.rename(columns=dict(zip(metrics, metrics_method)), inplace=True)
    else:
        df_query = per_query_dict[(method, dataset)][['query_idx']+metrics]
        df_query.rename(columns=dict(zip(metrics, metrics_method)), inplace=True)
        df_merged = df_merged.merge(df_query, on='query_idx', how='left')
```

**After:**
```python
# Clean function call (NEW)
df_merged = merge_per_query_metrics(
    per_query_dict=per_query_dict,
    methods=['deep', 'dinov2', 'openclip'],
    dataset='holidays',
    metrics=['ap@10']
)
```

**Output:**
| query_idx | query_path | group_id | num_relevant | ap@10_deep | ap@10_dinov2 | ap@10_openclip |
|-----------|------------|----------|--------------|------------|--------------|----------------|
| 0         | .../img.jpg| 1000     | 2            | 0.95       | 0.32         | 0.28           |
| 1         | .../img.jpg| 1001     | 3            | 0.88       | 0.92         | 0.85           |

### `find_method_wins()`

Finds queries where one method excels but competitors fail.

```python
# Find queries where 'deep' wins
deep_wins = find_method_wins(
    df_merged=df_merged,
    winning_method='deep',
    competing_methods=['dinov2', 'openclip'],
    metric='ap@10',
    threshold_high=0.9,  # deep >= 0.9
    threshold_low=0.3,   # others <= 0.3
    top_n=3
)
```

### `find_all_method_wins()`

Finds winning queries for all methods at once.

```python
# Find winners for each method
all_wins = find_all_method_wins(
    df_merged=df_merged,
    methods=['deep', 'dinov2', 'openclip'],
    metric='ap@10',
    threshold_high=0.9,
    threshold_low=0.3,
    top_n=3
)

# Returns: {'deep': DataFrame, 'dinov2': DataFrame, 'openclip': DataFrame}
```

### `plot_query_comparison_grid()`

Creates a side-by-side comparison visualization for one query.

```python
from sim_bench.analysis.comparison_viz import plot_query_comparison_grid

fig = plot_query_comparison_grid(
    methods=['deep', 'dinov2', 'openclip'],
    query_idx=123,
    query_path='/path/to/query.jpg',
    metric_scores={'deep': 0.95, 'dinov2': 0.32, 'openclip': 0.28},
    metric_name='ap@10',
    dataset_name='holidays',
    dataset_config=dataset_config,
    experiment_dir=Path('outputs/.../2025-11-02_00-12-26'),
    top_n=5
)
```

**Output:**
```
┌────────────────────────────────────────┐
│      QUERY: image.jpg (idx=123)        │
└────────────────────────────────────────┘
┌──────────┬──────────┬──────────┐
│  DEEP    │ DINOV2   │ OPENCLIP │
│ ap@10=   │ ap@10=   │ ap@10=   │
│  0.950   │  0.320   │  0.280   │
├──────────┼──────────┼──────────┤
│ #1 ✓     │ #1 ✗     │ #1 ✗     │
│ img1.jpg │ img9.jpg │ img7.jpg │
├──────────┼──────────┼──────────┤
│ #2 ✓     │ #2 ✗     │ #2 ✗     │
│ img2.jpg │ img8.jpg │ img6.jpg │
└──────────┴──────────┴──────────┘
```

### `visualize_all_method_wins()`

Generates comparison visualizations for all winning queries.

```python
saved_figures = visualize_all_method_wins(
    all_wins=all_wins,
    methods=['deep', 'dinov2', 'openclip'],
    metric_name='ap@10',
    dataset_name='holidays',
    dataset_config=dataset_config,
    experiment_dir=Path('outputs/.../2025-11-02_00-12-26'),
    output_dir=Path('reports/method_wins'),
    top_n=5,
    show_plots=True  # Display interactively
)

# Returns: {'deep': [path1, path2, path3], 'dinov2': [...], ...}
```

## Example Output Structure

```
sim_bench/analysis/reports/method_comparison_wins/
├── comparison_deep_wins_q123.png
├── comparison_deep_wins_q456.png
├── comparison_deep_wins_q789.png
├── comparison_dinov2_wins_q234.png
├── comparison_dinov2_wins_q567.png
├── comparison_dinov2_wins_q890.png
├── comparison_openclip_wins_q345.png
├── comparison_openclip_wins_q678.png
└── comparison_openclip_wins_q901.png
```

## Complete Example

See `examples/method_comparison_example.py` for a complete runnable example.

## Adjusting Thresholds

If you're not finding enough winning queries, try:

```python
# More lenient thresholds
all_wins = find_all_method_wins(
    df_merged=df_merged,
    methods=['deep', 'dinov2', 'openclip'],
    metric='ap@10',
    threshold_high=0.8,  # Lower winning threshold
    threshold_low=0.4,   # Higher losing threshold
    top_n=5              # Get more queries
)
```

## Multiple Metrics

You can merge and analyze multiple metrics at once:

```python
df_merged = merge_per_query_metrics(
    per_query_dict=per_query_dict,
    methods=['deep', 'dinov2', 'openclip'],
    dataset='holidays',
    metrics=['ap@10', 'recall@10']
)

# Now you have columns: ap@10_deep, ap@10_dinov2, ap@10_openclip,
#                        recall@10_deep, recall@10_dinov2, recall@10_openclip
```
