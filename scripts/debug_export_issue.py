"""
Debug script to investigate why pre_merge_cluster_id is not in faces.csv

Run this to gather diagnostic information:
    python scripts/debug_export_issue.py --export-dir results/face_clustering_training/diagnostic_test
"""

import argparse
import pandas as pd
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Debug export issue')
    parser.add_argument('--export-dir', type=Path, required=True, help='Export directory to check')
    args = parser.parse_args()

    export_dir = args.export_dir

    print("="*60)
    print("DIAGNOSTIC INFORMATION")
    print("="*60)
    print(f"Export directory: {export_dir}")
    print()

    # Check if directory exists
    if not export_dir.exists():
        print(f"❌ ERROR: Directory does not exist!")
        return

    # Check files
    print("FILE CHECK:")
    files_to_check = [
        'faces.csv',
        'clusters.csv',
        'candidate_pairs.csv',
        'pre_merge_clusters.csv',
        'merge_decisions.csv',
        'cluster_lineage.json',
        'export_summary.json'
    ]

    for filename in files_to_check:
        file_path = export_dir / filename
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"  ✅ {filename} ({size:,} bytes)")
        else:
            print(f"  ❌ {filename} MISSING")

    print()

    # Check faces.csv columns
    faces_path = export_dir / 'faces.csv'
    if faces_path.exists():
        print("FACES.CSV ANALYSIS:")
        faces_df = pd.read_csv(faces_path)
        print(f"  Rows: {len(faces_df)}")
        print(f"  Columns: {list(faces_df.columns)}")
        print()

        if 'pre_merge_cluster_id' in faces_df.columns:
            print(f"  ✅ pre_merge_cluster_id column EXISTS")
            print(f"  Sample values: {faces_df['pre_merge_cluster_id'].head(10).tolist()}")
            print(f"  Unique values: {faces_df['pre_merge_cluster_id'].nunique()}")
        else:
            print(f"  ❌ pre_merge_cluster_id column MISSING")
        print()

    # Check export_summary.json
    summary_path = export_dir / 'export_summary.json'
    if summary_path.exists():
        print("EXPORT SUMMARY:")
        with open(summary_path) as f:
            summary = json.load(f)

        print(f"  Timestamp: {summary.get('timestamp')}")
        print(f"  Embeddings source: {summary.get('embeddings_source')}")
        print(f"  N faces: {summary.get('n_faces')}")
        print(f"  N clusters: {summary.get('n_clusters')}")
        print(f"  N noise: {summary.get('n_noise')}")
        print(f"  Files: {list(summary.get('files', {}).keys())}")
        print()

    # Check merge_decisions.csv
    merge_path = export_dir / 'merge_decisions.csv'
    if merge_path.exists():
        print("MERGE DECISIONS:")
        merge_df = pd.read_csv(merge_path)
        print(f"  Total decisions: {len(merge_df)}")
        if len(merge_df) > 0:
            print(f"  Columns: {list(merge_df.columns)}")
            merged_count = len(merge_df[merge_df['action'] == 'merged'])
            rejected_count = len(merge_df[merge_df['action'] == 'rejected'])
            print(f"  Merged: {merged_count}")
            print(f"  Rejected: {rejected_count}")
        print()

    # Check pre_merge_clusters.csv
    pre_merge_path = export_dir / 'pre_merge_clusters.csv'
    if pre_merge_path.exists():
        print("PRE-MERGE CLUSTERS:")
        pre_merge_df = pd.read_csv(pre_merge_path)
        print(f"  N clusters: {len(pre_merge_df)}")
        print(f"  Columns: {list(pre_merge_df.columns)}")
        print()

    # Check logs
    log_dir = export_dir / 'logs'
    if log_dir.exists():
        print("LOG FILES:")
        log_files = sorted(log_dir.glob('*.log'))
        if log_files:
            latest_log = log_files[-1]
            print(f"  Latest log: {latest_log.name}")
            print(f"  Reading last 50 lines...")
            print()
            with open(latest_log, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[-50:]:
                    if 'merge' in line.lower() or 'pre_merge' in line.lower() or 'ERROR' in line or 'WARNING' in line:
                        print(f"    {line.rstrip()}")
        else:
            print(f"  No log files found")
    else:
        print("LOG DIRECTORY: Not found")

    print()
    print("="*60)
    print("RECOMMENDATIONS:")
    print("="*60)

    if not faces_path.exists():
        print("❌ faces.csv missing - export failed completely")
    elif 'pre_merge_cluster_id' not in faces_df.columns:
        print("❌ Column missing - likely pre_merge_result is None")
        if not merge_path.exists():
            print("   → No merge_decisions.csv - merge was disabled or failed")
            print("   → Check logs for 'Skipping merge (disabled in config)'")
        elif len(pd.read_csv(merge_path)) == 0:
            print("   → Empty merge_decisions.csv - no merge candidates found")
            print("   → This is OK, but pre_merge_cluster_id should still be added")
        else:
            print("   → Merge decisions exist, but column still missing")
            print("   → Check export_csvs() logic in export script")
    else:
        print("✅ All checks passed!")


if __name__ == '__main__':
    main()
