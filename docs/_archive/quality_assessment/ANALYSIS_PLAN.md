# Quality Assessment Analysis Notebook - Revised Plan

## Overview
Create an analysis notebook for image quality assessment results, following the structure of `methods_comparison_updated.ipynb` but adapted for quality assessment metrics and data format.

## Data Structure

### Input Files (from quality benchmark output):
1. `methods_summary.csv` - Overall method comparison
   - Columns: method, avg_top1_accuracy, avg_top2_accuracy, avg_mrr, avg_time_ms, datasets_tested
   
2. `detailed_results.csv` - Per-dataset, per-method results
   - Columns: dataset, method, top1_accuracy, top2_accuracy, mrr, avg_time_ms, throughput

3. `[dataset]_[method]_series.csv` - Per-series detailed results
   - Columns: group_id, num_images, predicted_idx, ground_truth_idx, correct, scores, ranking, image_ranks, image_paths

## Configuration Options

### Folder Selection (Flexible)
```python
# Option 1: Single benchmark folder
BENCHMARK_DIR = "outputs/quality_benchmarks/benchmark_2025-11-14_00-06-59"

# Option 2: Auto-scan from base directory (find latest or all)
AUTO_SCAN = True
BENCHMARK_BASE_DIR = "outputs/quality_benchmarks"
# Will scan for all benchmark_* folders and use latest (or allow selection)
```

## Analysis Sections

### 1. Load and Prepare Data
- **Flexible loading**: Support single folder or auto-scan multiple folders
- Load methods_summary.csv and detailed_results.csv
- Load all per-series CSV files for correlation analysis
- Parse image_paths and scores from per-series data
- Create merged dataframe with per-series Top-1 accuracy for each method

### 2. Performance Metrics Comparison
- **Bar charts**: Top-1 Accuracy, Top-2 Accuracy, MRR per method
- **Runtime comparison**: Bar chart showing avg_time_ms per method (from CSV, no recalculation)
- **Efficiency plot**: Scatter plot (Accuracy vs Time) or efficiency metric
- **Summary table**: All metrics side-by-side with highlighting

### 3. Correlation Matrix
- Use per-series data to compute Top-1 Accuracy correlation between methods
- Show which methods agree on which series are difficult
- Heatmap visualization (similar to methods_comparison notebook)
- **Focus metric**: Top-1 Accuracy (as requested)

### 4. Method Wins Visualization
- Find series where one method succeeds (`correct=True`) but others fail (`correct=False`)
- **Display format**:
  1. **Ground truth best image** (shown first)
  2. For each method (restricted to top N methods, default N=4):
     - Show **top 3 scoring images** with:
       - Image thumbnail
       - Quality score
       - Image filename
       - Rank indicator (1st, 2nd, 3rd)
- **Method selection**: Use top N methods by Top-1 Accuracy (default: 4)
- Show 3 example series total

### 5. Additional Analysis
- **Per-series accuracy distribution**: Box plots showing per-series accuracy distribution
- **Series size analysis**: Does performance vary by series size? (group by num_images)
- **Failure case analysis**: 
  - Analyze series where all methods fail (all `correct=False`)
  - Analyze series where only one method succeeds
  - Look for patterns: series size, score distributions, etc.
  - Show examples of common failure patterns

## HTML Export

### Simple Approach (Using Existing Function)
- Use `sim_bench.analysis.export.export_notebook_to_pdf()` function
- This function already supports HTML export (falls back to HTML if PDF fails)
- **Usage**: Just call at end of notebook:
  ```python
  from sim_bench.analysis.export import export_notebook_to_pdf
  export_notebook_to_pdf("quality_assessment_analysis.ipynb", output_dir="outputs/...")
  ```
- **No complicated code needed** - just one function call
- Exports notebook with all outputs (plots, tables) embedded in HTML

## Implementation Plan

### Phase 1: Core Analysis Functions
1. `load_quality_results(benchmark_dir, auto_scan=False, base_dir=None)` - Flexible loading
2. `merge_per_series_metrics(series_data_dict)` - Merge per-series data across methods
3. `compute_correlation_matrix(merged_df, metric='top1_accuracy')` - Correlation analysis
4. `find_method_wins(merged_df, methods, top_n_series=3)` - Find winning cases
5. `analyze_failures(merged_df, methods)` - Failure case analysis

### Phase 2: Visualization Functions
1. `plot_performance_comparison(methods_summary_df)` - Bar charts
2. `plot_runtime_comparison(detailed_results_df)` - Runtime bars
3. `plot_correlation_heatmap(corr_matrix, methods)` - Correlation heatmap
4. `visualize_method_wins(wins_data, dataset_config, top_n_methods=4, top_n_images=3)` - Image displays
5. `plot_failure_analysis(failure_stats)` - Failure patterns

### Phase 3: Notebook
1. Create Jupyter notebook with all sections
2. Add HTML export cell at the end (simple function call)

## File Structure

```
sim_bench/quality_assessment/analysis/
├── __init__.py
├── load_results.py          # Flexible data loading
├── correlation.py            # Correlation analysis
├── visualization.py          # Plotting functions
├── method_wins.py           # Find and visualize wins
└── failure_analysis.py      # Failure case analysis

notebooks/
└── quality_assessment_analysis.ipynb  # Main analysis notebook
```

## Key Differences from Similarity Benchmark Analysis

1. **Metrics**: Top-1/2 Accuracy, MRR (not mAP, Recall@k)
2. **Granularity**: Per-series (not per-query)
3. **Success criteria**: Binary (correct/incorrect) for Top-1
4. **Runtime**: Already in CSV (no need to recalculate)
5. **Visualization**: Show actual images from series with top 3 per method
6. **Method selection**: Focus on top N methods (default 4)

## Failure Analysis Approach

1. **All methods fail**: Find series where all methods have `correct=False`
   - Analyze patterns: series size, score variance, etc.
   - Show examples

2. **Single method succeeds**: Find series where only one method has `correct=True`
   - Identify which method "saves" these cases
   - Show examples

3. **Score analysis**: Compare score distributions for failed vs successful cases
   - Are failures due to low scores or ambiguous cases?

4. **Series size impact**: Does failure rate vary by series size?
   - Group by num_images and compute failure rates

## Questions to Answer

1. Which method has highest Top-1 Accuracy?
2. Which method is fastest?
3. Which methods agree on series difficulty (correlation on Top-1 Accuracy)?
4. Are there series where one method excels but others fail?
5. What's the accuracy vs speed tradeoff?
6. What patterns exist in failure cases?

