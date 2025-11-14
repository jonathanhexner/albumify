"""
Calculate missing recall@10 and map metrics for UKBench from rankings.csv.

This adds the missing metrics to existing metrics.csv files without re-running experiments.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import csv
from datetime import datetime


def calculate_recall_and_map_from_rankings(rankings_file: Path, groups: list, num_images: int):
    """
    Calculate recall@10 and mAP from rankings.csv.

    Args:
        rankings_file: Path to rankings.csv
        groups: List of group IDs for each image
        num_images: Total number of images

    Returns:
        dict with recall@10 and map values
    """

    # Read rankings (long format: query_filename, rank, result_filename, distance)
    df = pd.read_csv(rankings_file)

    # Get unique queries
    queries = df['query_filename'].unique()
    num_queries = len(queries)

    # Create filename to index mapping
    all_files = sorted(set(df['query_filename'].tolist() + df['result_filename'].tolist()))
    file_to_idx = {f: i for i, f in enumerate(all_files)}

    # Build rankings matrix
    recall_10_scores = []
    ap_scores = []

    for query_file in queries:
        query_idx = file_to_idx[query_file]
        query_group = groups[query_idx]

        # Get ranking for this query
        query_df = df[df['query_filename'] == query_file].sort_values('rank')
        result_files = query_df['result_filename'].tolist()
        result_indices = [file_to_idx[f] for f in result_files]

        # Calculate recall@10 (excluding self at rank 0)
        top_10_indices = result_indices[1:11]  # Skip rank 0 (self), take next 10
        relevant_in_top_10 = sum(1 for idx in top_10_indices if groups[idx] == query_group)
        recall_10_scores.append(1.0 if relevant_in_top_10 > 0 else 0.0)

        # Calculate AP (full ranking, excluding self)
        relevant_count = 0
        precision_sum = 0.0
        total_relevant = sum(1 for g in groups if g == query_group) - 1  # Exclude self

        for rank, result_idx in enumerate(result_indices[1:], start=1):  # Skip self at rank 0
            if groups[result_idx] == query_group:
                relevant_count += 1
                precision_at_k = relevant_count / rank
                precision_sum += precision_at_k

        if total_relevant > 0:
            ap = precision_sum / total_relevant
        else:
            ap = 0.0

        ap_scores.append(ap)

    return {
        'recall@10': np.mean(recall_10_scores),
        'map': np.mean(ap_scores)
    }


def add_missing_metrics(method_dir: Path):
    """Add missing recall@10 and map to a UKBench metrics.csv."""

    rankings_file = method_dir / "rankings.csv"
    metrics_file = method_dir / "metrics.csv"
    manifest_file = method_dir / "manifest.json"

    if not rankings_file.exists():
        return False, "No rankings.csv"

    if not metrics_file.exists():
        return False, "No metrics.csv"

    if not manifest_file.exists():
        return False, "No manifest.json"

    # Check if this is UKBench
    with open(manifest_file, 'r') as f:
        manifest = json.load(f)

    # Detect dataset
    if 'dataset' in manifest and 'name' in manifest['dataset']:
        dataset_name = manifest['dataset']['name']
    else:
        parent_name = method_dir.parent.name
        dataset_name = 'ukbench' if 'ukbench' in parent_name.lower() else 'holidays'

    if dataset_name != 'ukbench':
        return False, "Not UKBench"

    # Read current metrics
    current_metrics = pd.read_csv(metrics_file)

    # Check if already has these metrics
    if 'map' in current_metrics.columns and 'recall@10' in current_metrics.columns:
        return False, "Already has metrics"

    method_name = manifest['method']

    # UKBench: 10,200 images, groups of 4 (groups 0-2549, each with 4 images)
    num_images = 10200
    groups = [i // 4 for i in range(num_images)]  # 0,0,0,0, 1,1,1,1, 2,2,2,2, ...

    # Calculate missing metrics
    print(f"  Calculating from {len(pd.read_csv(rankings_file))} ranking entries...")
    new_metrics = calculate_recall_and_map_from_rankings(rankings_file, groups, num_images)

    # Merge with existing metrics
    method_row = current_metrics[current_metrics['method'] == method_name].iloc[0].to_dict()
    method_row['map'] = new_metrics['map']
    method_row['recall@10'] = new_metrics['recall@10']

    # Rebuild CSV with all metrics (sorted)
    metadata_keys = {'num_queries', 'num_gallery', 'created_at', 'method'}
    metric_names = [key for key in method_row.keys() if key not in metadata_keys]

    header = ["method"] + sorted(metric_names) + ["num_queries", "num_gallery", "created_at"]
    row = [method_name]
    for metric_name in sorted(metric_names):
        row.append(f"{method_row[metric_name]:.6f}")
    row.extend([
        int(method_row['num_queries']),
        int(method_row['num_gallery']),
        datetime.now().isoformat()
    ])

    with open(metrics_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerow(row)

    return True, new_metrics


def main():
    """Add missing metrics to all UKBench experiments."""

    base_dir = Path("outputs/baseline_runs/comprehensive_baseline")

    print("="*80)
    print("ADDING MISSING UKBENCH METRICS (recall@10, map)")
    print("="*80)
    print(f"\nBase directory: {base_dir}\n")

    # Find all UKBench method directories
    processed = 0
    skipped = 0

    for rankings_file in base_dir.rglob("*/rankings.csv"):
        method_dir = rankings_file.parent

        # Get method name
        manifest_file = method_dir / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
            method_name = manifest['method']
        else:
            method_name = method_dir.name

        # Detect if UKBench from manifest
        if manifest_file.exists():
            with open(manifest_file, 'r') as f:
                check_manifest = json.load(f)
            if 'dataset' in check_manifest and 'name' in check_manifest['dataset']:
                if check_manifest['dataset']['name'] != 'ukbench':
                    continue
            elif 'ukbench' not in method_dir.parent.name.lower():
                continue
        elif 'ukbench' not in method_dir.parent.name.lower():
            continue

        print(f"Processing: {method_name:12} on ukbench ... ", end='', flush=True)

        success, result = add_missing_metrics(method_dir)

        if success:
            print(f"[OK]")
            print(f"  recall@10 = {result['recall@10']:.6f}")
            print(f"  map       = {result['map']:.6f}")
            processed += 1
        else:
            print(f"[SKIP] {result}")
            skipped += 1

    print("\n" + "="*80)
    print(f"DONE! Processed: {processed}, Skipped: {skipped}")
    print("="*80)


if __name__ == "__main__":
    main()
