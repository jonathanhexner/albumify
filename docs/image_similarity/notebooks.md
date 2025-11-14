# Notebook Usage Summary

## Simplified Code for Your Notebook

Replace your manual merging code with these clean functions:

### Old Code (Manual Merging)
```python
from re import L

metrics = ['ap@10']
dataset = 'holidays'
df_merged_results = pd.DataFrame()
methods_merge = ['deep', 'dinov2', 'openclip']
for n_method, method in enumerate(methods_merge):
    metrics_method = list(map(lambda x: x+ '_' + method , metrics))
    if n_method == 0:
        df_merged_results = per_query_dict[(method, 'holidays')][['query_idx', 'query_path', 'group_id', 'num_relevant']+metrics]
        df_merged_results.rename(columns=dict(zip(metrics, metrics_method)), inplace=True)
    else:
        df_query = per_query_dict[(method, 'holidays')][['query_idx']+metrics]
        df_query.rename(columns=dict(zip(metrics, metrics_method)), inplace=True)
        df_merged_results = df_merged_results.merge(df_query, on='query_idx', how='left')

df_merged_results.head()
```

### New Code (Clean Functions)
```python
from sim_bench.analysis.multi_experiment import merge_per_query_metrics

# Merge metrics from multiple methods
df_merged_results = merge_per_query_metrics(
    per_query_dict=per_query_dict,
    methods=['deep', 'dinov2', 'openclip'],
    dataset='holidays',
    metrics=['ap@10']
)

df_merged_results.head()
```

## Finding Method Wins

```python
from sim_bench.analysis.multi_experiment import find_all_method_wins

# Find queries where each method excels but others fail
all_wins = find_all_method_wins(
    df_merged=df_merged_results,
    methods=['deep', 'dinov2', 'openclip'],
    metric='ap@10',
    threshold_high=0.9,  # Winner must score >= 0.9
    threshold_low=0.3,   # Losers must score <= 0.3
    top_n=3,             # Find top 3 per method
    verbose=True         # Print summary
)

# Access results for specific method
deep_wins = all_wins['deep']
dinov2_wins = all_wins['dinov2']
openclip_wins = all_wins['openclip']
```

## Visualizing Comparisons

```python
from pathlib import Path
from sim_bench.analysis.comparison_viz import visualize_all_method_wins

# Configuration
DATASET_CONFIG = {
    'name': 'holidays',
    'root': 'D:/Similar Images/DataSets/InriaHolidaysFull',
    'pattern': '*.jpg'
}
EXPERIMENT_DIR = Path('outputs/baseline_runs/comprehensive_baseline/2025-11-02_00-12-26')
OUTPUT_DIR = Path('sim_bench/analysis/reports/method_comparison_wins')

# Generate all visualizations
saved_figures = visualize_all_method_wins(
    all_wins=all_wins,
    methods=['deep', 'dinov2', 'openclip'],
    metric_name='ap@10',
    dataset_name='holidays',
    dataset_config=DATASET_CONFIG,
    experiment_dir=EXPERIMENT_DIR,
    output_dir=OUTPUT_DIR,
    top_n=5,
    show_plots=True  # Display in notebook
)

# saved_figures = {'deep': [path1, path2, path3], 'dinov2': [...], ...}
```

## Single Query Visualization

If you want to visualize just one specific query:

```python
from sim_bench.analysis.comparison_viz import plot_query_comparison_grid

# Pick a specific query from wins
query_idx = int(deep_wins.iloc[0]['query_idx'])
query_path = deep_wins.iloc[0]['query_path']

# Get scores for all methods
metric_scores = {
    'deep': deep_wins.iloc[0]['ap@10_deep'],
    'dinov2': deep_wins.iloc[0]['ap@10_dinov2'],
    'openclip': deep_wins.iloc[0]['ap@10_openclip']
}

# Create visualization
fig = plot_query_comparison_grid(
    methods=['deep', 'dinov2', 'openclip'],
    query_idx=query_idx,
    query_path=query_path,
    metric_scores=metric_scores,
    metric_name='ap@10',
    dataset_name='holidays',
    dataset_config=DATASET_CONFIG,
    experiment_dir=EXPERIMENT_DIR,
    top_n=5,
    save_path=OUTPUT_DIR / f'comparison_q{query_idx}.png'
)
plt.show()
```

## Complete Notebook Cell

Here's everything in one cell for your notebook:

```python
from pathlib import Path
from sim_bench.analysis.multi_experiment import merge_per_query_metrics, find_all_method_wins
from sim_bench.analysis.comparison_viz import visualize_all_method_wins

# 1. Merge per-query metrics
df_merged_results = merge_per_query_metrics(
    per_query_dict=per_query_dict,
    methods=['deep', 'dinov2', 'openclip'],
    dataset='holidays',
    metrics=['ap@10']
)

print("Merged DataFrame:")
print(df_merged_results.head())

# 2. Find winning queries
all_wins = find_all_method_wins(
    df_merged=df_merged_results,
    methods=['deep', 'dinov2', 'openclip'],
    metric='ap@10',
    threshold_high=0.9,
    threshold_low=0.3,
    top_n=3,
    verbose=True
)

# 3. Visualize comparisons
DATASET_CONFIG = {
    'name': 'holidays',
    'root': 'D:/Similar Images/DataSets/InriaHolidaysFull',
    'pattern': '*.jpg'
}

saved_figures = visualize_all_method_wins(
    all_wins=all_wins,
    methods=['deep', 'dinov2', 'openclip'],
    metric_name='ap@10',
    dataset_name='holidays',
    dataset_config=DATASET_CONFIG,
    experiment_dir=Path('outputs/baseline_runs/comprehensive_baseline/2025-11-02_00-12-26'),
    output_dir=Path('sim_bench/analysis/reports/method_wins'),
    top_n=5,
    show_plots=True
)

print(f"\n{len(sum(saved_figures.values(), []))} figures saved!")
```

## What Gets Generated

For each method that has winning queries (e.g., 3 queries where deep wins):
- Creates a side-by-side comparison grid showing all 3 methods
- Query image at top
- Top-5 results per method in columns
- Match indicators (✓/✗) showing correct/incorrect matches
- Method name and score highlighted in green (good) or red (bad)
- Saves as PNG: `comparison_{method}_wins_q{query_idx}.png`

Example output:
```
method_comparison_wins/
├── comparison_deep_wins_q123.png       # 3 deep wins
├── comparison_deep_wins_q456.png
├── comparison_deep_wins_q789.png
├── comparison_dinov2_wins_q234.png     # 3 dinov2 wins
├── comparison_dinov2_wins_q567.png
├── comparison_dinov2_wins_q890.png
├── comparison_openclip_wins_q345.png   # 3 openclip wins
├── comparison_openclip_wins_q678.png
└── comparison_openclip_wins_q901.png
```

## Adjusting Parameters

### Not finding enough wins?
```python
all_wins = find_all_method_wins(
    df_merged=df_merged_results,
    methods=['deep', 'dinov2', 'openclip'],
    metric='ap@10',
    threshold_high=0.8,  # Lower (more lenient)
    threshold_low=0.4,   # Higher (more lenient)
    top_n=5              # More queries
)
```

### Change visualization layout?
```python
saved_figures = visualize_all_method_wins(
    # ... other params ...
    top_n=10,  # Show top-10 results instead of top-5
    show_plots=False  # Don't display, just save
)
```

## See Also

- Full documentation: `COMPARISON_ANALYSIS_GUIDE.md`
- Complete example script: `examples/method_comparison_example.py`
