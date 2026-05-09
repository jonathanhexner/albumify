"""
Investigate why merge_decisions.csv has duplicate merges.

Expected: 48 merges (75 → 27 clusters)
Actual: 1,454 "merged" entries

Usage:
    python scripts/debug_merge_duplicates.py --export-dir results/Google_Germany/clustering_export
"""

import argparse
import pandas as pd
from pathlib import Path
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description='Debug duplicate merge entries')
    parser.add_argument('--export-dir', type=Path, required=True, help='Export directory')
    args = parser.parse_args()

    export_dir = args.export_dir
    merge_path = export_dir / 'merge_decisions.csv'

    if not merge_path.exists():
        print(f"❌ ERROR: {merge_path} not found")
        return

    print("="*60)
    print("MERGE DUPLICATES INVESTIGATION")
    print("="*60)
    print()

    # Load merge decisions
    merge_df = pd.read_csv(merge_path)

    total_decisions = len(merge_df)
    merged = merge_df[merge_df['action'] == 'merged']
    rejected = merge_df[merge_df['action'] == 'rejected']

    print(f"OVERALL STATS:")
    print(f"  Total decisions: {total_decisions}")
    print(f"  Merged: {len(merged)}")
    print(f"  Rejected: {len(rejected)}")
    print()

    # Check iterations
    max_iteration = merge_df['iteration'].max()
    print(f"  Max iteration: {max_iteration}")
    print()

    # Question 1: How many UNIQUE cluster pairs were merged?
    print("="*60)
    print("QUESTION 1: How many UNIQUE cluster pairs merged?")
    print("="*60)

    merged_pairs = [(row['cluster_a'], row['cluster_b'])
                    for _, row in merged.iterrows()]

    unique_pairs = set(merged_pairs)
    print(f"  Total 'merged' entries: {len(merged)}")
    print(f"  Unique (cluster_a, cluster_b) pairs: {len(unique_pairs)}")
    print()

    # Question 2: Which pairs appear multiple times?
    pair_counts = Counter(merged_pairs)
    duplicates = {pair: count for pair, count in pair_counts.items() if count > 1}

    if duplicates:
        print(f"❌ FOUND DUPLICATES: {len(duplicates)} pairs merged multiple times!")
        print()
        print(f"  Top 10 duplicates:")
        for pair, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"    Cluster {pair[0]} + {pair[1]}: merged {count} times")
        print()

        # Show one example in detail
        example_pair = sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[0][0]
        print(f"  DETAILED EXAMPLE: Cluster {example_pair[0]} + {example_pair[1]}")
        print(f"  Merged {duplicates[example_pair]} times at iterations:")

        example_merges = merged[
            (merged['cluster_a'] == example_pair[0]) &
            (merged['cluster_b'] == example_pair[1])
        ]

        for _, row in example_merges.iterrows():
            print(f"    Iteration {row['iteration']}: "
                  f"dist={row['exemplar_dist']:.3f}, "
                  f"actually_merged={row.get('actually_merged', 'N/A')}")
        print()
    else:
        print(f"✅ No duplicates found - all {len(unique_pairs)} pairs merged exactly once")
        print()

    # Question 3: Check 'actually_merged' flag
    print("="*60)
    print("QUESTION 3: Check 'actually_merged' flag")
    print("="*60)

    if 'actually_merged' in merged.columns:
        actually_merged_count = len(merged[merged['actually_merged'] == True])
        print(f"  Entries with actually_merged=True: {actually_merged_count}")
        print(f"  Entries with actually_merged=False: {len(merged) - actually_merged_count}")
        print()

        if actually_merged_count != len(unique_pairs):
            print(f"  ⚠️  WARNING: actually_merged=True count ({actually_merged_count}) "
                  f"!= unique pairs ({len(unique_pairs)})")
        else:
            print(f"  ✅ actually_merged=True count matches unique pairs")
        print()
    else:
        print(f"  ❌ Column 'actually_merged' not found in CSV")
        print()

    # Question 4: Decisions per iteration
    print("="*60)
    print("QUESTION 4: How many decisions per iteration?")
    print("="*60)

    decisions_per_iter = merge_df.groupby('iteration').size()
    merged_per_iter = merged.groupby('iteration').size()

    print(f"  Average decisions per iteration: {decisions_per_iter.mean():.1f}")
    print(f"  Average merged per iteration: {merged_per_iter.mean():.1f}")
    print()

    # Show first 5 iterations
    print(f"  First 5 iterations:")
    for i in range(1, min(6, max_iteration + 1)):
        iter_decisions = merge_df[merge_df['iteration'] == i]
        iter_merged = len(iter_decisions[iter_decisions['action'] == 'merged'])
        iter_total = len(iter_decisions)
        print(f"    Iteration {i}: {iter_total} decisions, {iter_merged} merged")
    print()

    # Question 5: Check actually_merged flag per iteration
    print("="*60)
    print("QUESTION 5: How many actually_merged=True per iteration?")
    print("="*60)

    if 'actually_merged' in merged.columns:
        actually_merged_per_iter = merged[merged['actually_merged'] == True].groupby('iteration').size()
        print(f"  First 10 iterations with actually_merged=True:")
        for i in range(1, min(11, max_iteration + 1)):
            count = actually_merged_per_iter.get(i, 0)
            print(f"    Iteration {i}: {count}")

        if (actually_merged_per_iter > 1).any():
            print(f"  ⚠️  WARNING: Some iterations have multiple actually_merged=True!")
            bad_iters = actually_merged_per_iter[actually_merged_per_iter > 1]
            for iter_num, count in bad_iters.items():
                print(f"    Iteration {iter_num}: {count} actually_merged=True")
        else:
            print(f"  ✅ Each iteration has at most 1 actually_merged=True")
    print()

    # Summary and recommendations
    print("="*60)
    print("DIAGNOSIS")
    print("="*60)

    if len(unique_pairs) == actually_merged_count:
        print("✅ HYPOTHESIS: Merge logic is CORRECT")
        print("   - Each pair merged exactly once")
        print("   - actually_merged flag correctly identifies actual merges")
        print("   - The 1,454 'merged' entries include ALL CANDIDATES evaluated")
        print("   - Only entries with actually_merged=True are real merges")
        print()
        print("🔧 FIX NEEDED: UI should filter by actually_merged=True")
        print("   Current: shows all action='merged' (includes non-executed candidates)")
        print("   Should: show only actually_merged=True (actual merges)")
    else:
        print("❌ HYPOTHESIS: Merge logic has BUG")
        print("   - Cluster pairs merging multiple times")
        print("   - Clusters not being removed properly after merge")
        print("   - Need to debug ConservativeMerger._merge_two_clusters()")


if __name__ == '__main__':
    main()
