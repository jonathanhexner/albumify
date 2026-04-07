"""
Compare clustering results from notebook vs export script.

Usage:
    python scripts/compare_clustering_results.py
"""

import pandas as pd
from pathlib import Path

# Load both results
notebook_csv = Path("results/face_clustering_benchmark/notebook_clustering_results.csv")
export_csv = Path("results/face_clustering_benchmark/clustering_export/faces.csv")

if not notebook_csv.exists():
    print(f"ERROR: Notebook results not found at: {notebook_csv}")
    print("Please run the notebook first to generate notebook_clustering_results.csv")
    exit(1)

if not export_csv.exists():
    print(f"ERROR: Export results not found at: {export_csv}")
    print("Please run the export script first")
    exit(1)

notebook_df = pd.read_csv(notebook_csv)
export_df = pd.read_csv(export_csv)

print("="*60)
print("CLUSTERING RESULTS COMPARISON")
print("="*60)

print(f"\nNotebook: {len(notebook_df)} faces, {notebook_df['cluster_id'].max() + 1} clusters, {(notebook_df['cluster_id'] == -1).sum()} noise")
print(f"Export:   {len(export_df)} faces, {export_df['cluster_id'].max() + 1} clusters, {(export_df['cluster_id'] == -1).sum()} noise")

# Merge on face_id
merged = notebook_df.merge(export_df, on='face_id', suffixes=('_notebook', '_export'))

# Find mismatches
mismatches = merged[merged['cluster_id_notebook'] != merged['cluster_id_export']]

print(f"\nMismatched assignments: {len(mismatches)} / {len(merged)} faces ({len(mismatches)/len(merged)*100:.1f}%)")

if len(mismatches) > 0:
    print("\nFirst 20 mismatches:")
    print(mismatches[['face_id', 'cluster_id_notebook', 'cluster_id_export']].head(20))

    # Analyze cluster 3 specifically (user's example)
    print("\n" + "="*60)
    print("CLUSTER 3 ANALYSIS")
    print("="*60)

    notebook_cluster3 = notebook_df[notebook_df['cluster_id'] == 3]
    export_cluster3 = export_df[export_df['cluster_id'] == 3]

    print(f"\nNotebook cluster 3: {len(notebook_cluster3)} faces")
    print(f"Face IDs: {sorted(notebook_cluster3['face_id'].tolist())}")

    print(f"\nExport cluster 3: {len(export_cluster3)} faces")
    print(f"Face IDs: {sorted(export_cluster3['face_id'].tolist())}")

    # Find faces that are in different clusters
    in_notebook_only = set(notebook_cluster3['face_id']) - set(export_cluster3['face_id'])
    in_export_only = set(export_cluster3['face_id']) - set(notebook_cluster3['face_id'])

    if in_notebook_only:
        print(f"\nFaces in notebook cluster 3 but NOT in export cluster 3:")
        for face_id in sorted(in_notebook_only):
            export_match = export_df[export_df['face_id'] == face_id]
            if len(export_match) > 0:
                export_cluster = export_match['cluster_id'].iloc[0]
                print(f"  Face {face_id}: export cluster = {export_cluster}")
            else:
                print(f"  Face {face_id}: NOT FOUND in export (missing from export entirely)")

    if in_export_only:
        print(f"\nFaces in export cluster 3 but NOT in notebook cluster 3:")
        for face_id in sorted(in_export_only):
            notebook_match = notebook_df[notebook_df['face_id'] == face_id]
            if len(notebook_match) > 0:
                notebook_cluster = notebook_match['cluster_id'].iloc[0]
                print(f"  Face {face_id}: notebook cluster = {notebook_cluster}")
            else:
                print(f"  Face {face_id}: NOT FOUND in notebook (missing from notebook entirely)")

else:
    print("\n✅ All cluster assignments match!")
