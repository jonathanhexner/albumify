"""
Test script for multi-experiment analysis on actual data.
"""

from pathlib import Path
from sim_bench.analysis.utils import get_project_root
from sim_bench.analysis.multi_experiment import load_experiments

PROJECT_ROOT = get_project_root()
BASE_DIR = PROJECT_ROOT / "outputs" / "baseline_runs" / "comprehensive_baseline"

print("="*80)
print("TESTING MULTI-EXPERIMENT LOADER")
print("="*80)
print(f"\nBase directory: {BASE_DIR}")
print(f"Exists: {BASE_DIR.exists()}\n")

# Load experiments
try:
    metrics_df, per_query_dict, experiment_infos = load_experiments(
        base_dir=BASE_DIR,
        auto_scan=True,
        verbose=True
    )

    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    print(f"\nTotal method-dataset combinations: {len(metrics_df)}")
    print(f"Unique methods: {sorted(metrics_df['method'].unique())}")
    print(f"Unique datasets: {sorted(metrics_df['dataset'].unique())}")

    print("\n" + "="*80)
    print("METRICS AVAILABILITY")
    print("="*80)

    # Check which metrics are available for which dataset
    for dataset in sorted(metrics_df['dataset'].unique()):
        df_dataset = metrics_df[metrics_df['dataset'] == dataset]
        print(f"\n{dataset.upper()}:")

        # Show available metrics (non-NaN)
        metric_cols = [c for c in df_dataset.columns
                      if c not in ['method', 'dataset', 'run_name', 'created_at', 'num_queries', 'num_gallery']]

        for col in sorted(metric_cols):
            non_null_count = df_dataset[col].notna().sum()
            if non_null_count > 0:
                print(f"  {col:<20}: {non_null_count}/{len(df_dataset)} methods have this metric")

    print("\n" + "="*80)
    print("DETAILED METRICS TABLE")
    print("="*80)
    print()

    # Display key metrics
    display_cols = ['method', 'dataset', 'ns', 'recall@1', 'recall@4', 'recall@10', 'map', 'map@10']
    available_display_cols = [c for c in display_cols if c in metrics_df.columns]

    print(metrics_df[available_display_cols].sort_values(['dataset', 'method']).to_string(index=False))

    print("\n" + "="*80)
    print("SUCCESS!")
    print("="*80)

except Exception as e:
    print(f"\n[ERROR]: {e}")
    import traceback
    traceback.print_exc()
