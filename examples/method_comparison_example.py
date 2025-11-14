"""
Example: Compare methods to find and visualize where each method excels.

This script demonstrates how to:
1. Merge per-query metrics from multiple methods
2. Find queries where each method performs well but others fail
3. Visualize side-by-side comparisons
"""

from pathlib import Path
from sim_bench.analysis.multi_experiment import (
    load_experiments,
    merge_per_query_metrics,
    find_all_method_wins
)
from sim_bench.analysis.comparison_viz import visualize_all_method_wins


# ============================================================================
# Configuration
# ============================================================================

# Experiment directory
EXPERIMENT_DIR = Path('outputs/baseline_runs/comprehensive_baseline/2025-11-02_00-12-26')

# Dataset configuration
DATASET_NAME = 'holidays'
DATASET_CONFIG = {
    'name': 'holidays',
    'root': 'D:/Similar Images/DataSets/InriaHolidaysFull',
    'pattern': '*.jpg'
}

# Methods to compare
METHODS = ['deep', 'dinov2', 'openclip']

# Metric to analyze
METRIC = 'ap@10'

# Output directory
OUTPUT_DIR = Path('sim_bench/analysis/reports/method_comparison_wins')


# ============================================================================
# Step 1: Load Experiment Data
# ============================================================================

print("Loading experiment data...")
metrics_df, per_query_dict, experiment_infos = load_experiments(
    base_dir=EXPERIMENT_DIR.parent,
    auto_scan=True,
    verbose=True
)


# ============================================================================
# Step 2: Merge Per-Query Metrics
# ============================================================================

print(f"\nMerging per-query metrics for {DATASET_NAME}...")
df_merged = merge_per_query_metrics(
    per_query_dict=per_query_dict,
    methods=METHODS,
    dataset=DATASET_NAME,
    metrics=[METRIC]
)

print(f"Merged DataFrame shape: {df_merged.shape}")
print(f"Columns: {list(df_merged.columns)}")
print("\nSample rows:")
print(df_merged.head())


# ============================================================================
# Step 3: Find Winning Queries for Each Method
# ============================================================================

print(f"\n{'='*80}")
print("FINDING WINNING QUERIES")
print(f"{'='*80}")

all_wins = find_all_method_wins(
    df_merged=df_merged,
    methods=METHODS,
    metric=METRIC,
    threshold_high=0.9,  # Winning method must have score >= 0.9
    threshold_low=0.3,   # Competing methods must have score <= 0.3
    top_n=3,             # Find top 3 queries per method
    verbose=True
)


# ============================================================================
# Step 4: Visualize Comparisons
# ============================================================================

print(f"\n{'='*80}")
print("GENERATING VISUALIZATIONS")
print(f"{'='*80}")

saved_figures = visualize_all_method_wins(
    all_wins=all_wins,
    methods=METHODS,
    metric_name=METRIC,
    dataset_name=DATASET_NAME,
    dataset_config=DATASET_CONFIG,
    experiment_dir=EXPERIMENT_DIR,
    output_dir=OUTPUT_DIR,
    top_n=5,
    show_plots=False  # Set to True to display plots interactively
)


# ============================================================================
# Summary
# ============================================================================

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

for method, figures in saved_figures.items():
    print(f"{method}: {len(figures)} visualizations")
    for fig_path in figures:
        print(f"  - {fig_path}")

print(f"\nAll figures saved to: {OUTPUT_DIR}")
