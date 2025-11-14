"""Quick test of the new comparison functions."""

from pathlib import Path
from sim_bench.analysis.multi_experiment import (
    load_experiments,
    merge_per_query_metrics,
    find_all_method_wins
)

print("="*80)
print("TESTING COMPARISON FUNCTIONS")
print("="*80)

# Load data
base_dir = Path('outputs/baseline_runs/comprehensive_baseline')
print(f"\n1. Loading experiments from: {base_dir}")

metrics_df, per_query_dict, experiment_infos = load_experiments(
    base_dir=base_dir,
    auto_scan=True,
    verbose=False
)
print(f"   Loaded {len(experiment_infos)} method-dataset combinations")

# Test merge_per_query_metrics
print("\n2. Testing merge_per_query_metrics()...")
df_merged = merge_per_query_metrics(
    per_query_dict=per_query_dict,
    methods=['deep', 'dinov2', 'openclip'],
    dataset='holidays',
    metrics=['ap@10']
)
print(f"   Merged DataFrame shape: {df_merged.shape}")
print(f"   Columns: {list(df_merged.columns)}")
print("\n   Sample rows:")
print(df_merged[['query_idx', 'ap@10_deep', 'ap@10_dinov2', 'ap@10_openclip']].head())

# Test find_all_method_wins
print("\n3. Testing find_all_method_wins()...")
all_wins = find_all_method_wins(
    df_merged=df_merged,
    methods=['deep', 'dinov2', 'openclip'],
    metric='ap@10',
    threshold_high=0.9,
    threshold_low=0.3,
    top_n=3,
    verbose=False
)

print("\n   Results:")
for method, df_wins in all_wins.items():
    print(f"   {method:10}: {len(df_wins)} winning queries")

print("\n" + "="*80)
print("ALL TESTS PASSED!")
print("="*80)
