"""Compare notebook results vs export script results (both on same embeddings).

Usage:
    python scripts/compare_notebook_vs_export.py
    python scripts/compare_notebook_vs_export.py --export-dir results/path/to/export
"""

import pandas as pd
from pathlib import Path
import argparse
import sys

parser = argparse.ArgumentParser(description='Compare notebook vs export script clustering')
parser.add_argument('--notebook-csv', type=Path,
                    default=Path("results/face_clustering_benchmark/notebook_clustering_results.csv"),
                    help='Path to notebook CSV output')
parser.add_argument('--export-dir', type=Path,
                    default=Path("results/face_clustering_benchmark/clustering_export_test"),
                    help='Path to export script output directory')
args = parser.parse_args()

notebook_csv = args.notebook_csv
export_csv = args.export_dir / 'faces.csv'

# Check files exist
if not notebook_csv.exists():
    print(f"ERROR: Notebook CSV not found: {notebook_csv}")
    print("\nMake sure you ran the notebook and it exported notebook_clustering_results.csv")
    sys.exit(1)

if not export_csv.exists():
    print(f"ERROR: Export CSV not found: {export_csv}")
    print(f"\nMake sure you ran: python scripts/export_clustering_data.py --output {args.export_dir}")
    sys.exit(1)

notebook_df = pd.read_csv(notebook_csv)
export_df = pd.read_csv(export_csv)

print("="*60)
print("CLUSTERING RESULTS COMPARISON")
print("="*60)

print(f"\nNotebook: {len(notebook_df)} faces, {notebook_df['cluster_id'].nunique()} unique clusters, {(notebook_df['cluster_id'] == -1).sum()} noise")
print(f"Export:   {len(export_df)} faces, {export_df['cluster_id'].nunique()} unique clusters, {(export_df['cluster_id'] == -1).sum()} noise")

# Merge on face_id
merged = notebook_df.merge(export_df[['face_id', 'cluster_id']], on='face_id', suffixes=('_notebook', '_export'))

# Find mismatches
mismatches = merged[merged['cluster_id_notebook'] != merged['cluster_id_export']]

print(f"\nMismatched assignments: {len(mismatches)} / {len(merged)} faces ({len(mismatches)/len(merged)*100:.1f}%)")

if len(mismatches) > 0:
    print("\nFirst 20 mismatches:")
    print(mismatches[['face_id', 'cluster_id_notebook', 'cluster_id_export']].head(20))

    # Show some specific examples
    print("\n" + "="*60)
    print("DETAILED MISMATCH ANALYSIS")
    print("="*60)
    for i in range(min(5, len(mismatches))):
        row = mismatches.iloc[i]
        print(f"\nFace {row['face_id']}:")
        print(f"  Notebook cluster: {row['cluster_id_notebook']}")
        print(f"  Export cluster:   {row['cluster_id_export']}")
else:
    print("\n[OK] All cluster assignments match perfectly!")
    print(f"\nCluster distribution:")
    print(notebook_df['cluster_id'].value_counts().sort_index())
