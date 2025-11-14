"""
Recalculate metrics.csv from existing rankings.csv files.

This script reads the rankings.csv and reconstructs all metrics
without re-running the experiments.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import csv
from datetime import datetime
from typing import Dict, Any

from sim_bench.datasets import load_dataset
from sim_bench.metrics import MetricFactory


def recalculate_metrics_for_method(
    method_dir: Path,
    dataset_name: str,
    dataset_root: str,
    metrics_list: list
) -> Dict[str, float]:
    """
    Recalculate metrics from rankings.csv.

    Args:
        method_dir: Path to method directory containing rankings.csv
        dataset_name: Name of dataset
        dataset_root: Path to dataset root
        metrics_list: List of metrics to compute

    Returns:
        Dictionary of computed metrics
    """

    # Load rankings
    rankings_file = method_dir / "rankings.csv"
    if not rankings_file.exists():
        raise FileNotFoundError(f"No rankings.csv in {method_dir}")

    rankings_df = pd.read_csv(rankings_file)

    # Load dataset to get groups
    dataset_config = {
        'name': dataset_name,
        'root': dataset_root
    }

    if dataset_name == 'ukbench':
        dataset_config['assume_groups_of_four'] = True
        dataset_config['pattern'] = 'ukbench*.jpg'
        dataset_config['subdirs'] = {'images': 'full'}
    elif dataset_name == 'holidays':
        dataset_config['pattern'] = '*.jpg'

    dataset = load_dataset(dataset_name, dataset_config)

    # Convert rankings to indices array
    num_queries = len(rankings_df)
    num_gallery = len(dataset.get_images())

    # Parse ranking strings to arrays
    ranking_indices = []
    for _, row in rankings_df.iterrows():
        # Rankings are stored as space-separated indices
        ranking_str = row['ranking']
        indices = [int(x) for x in ranking_str.split()]
        ranking_indices.append(indices)

    ranking_indices = np.array(ranking_indices)

    # Prepare evaluation data
    evaluation_data = {
        'groups': dataset.groups,
        'total_images': num_gallery
    }

    # Compute metrics
    config = {'metrics': metrics_list, 'k': 4}
    computed_metrics = MetricFactory.compute_all_metrics(
        ranking_indices,
        evaluation_data,
        config
    )

    return computed_metrics


def save_metrics_csv(method_dir: Path, method_name: str, computed_metrics: Dict[str, float], dataset):
    """Save metrics to CSV file."""

    metrics_file = method_dir / "metrics.csv"

    # Extract all metric columns (excluding metadata)
    metadata_keys = {'num_queries', 'num_images'}
    metric_names = [key for key in computed_metrics.keys() if key not in metadata_keys]

    # Build header: method, all metrics (sorted for consistency), metadata
    header = ["method"] + sorted(metric_names) + ["num_queries", "num_gallery", "created_at"]

    # Build row with corresponding values
    row = [method_name]
    for metric_name in sorted(metric_names):
        value = computed_metrics.get(metric_name, 0)
        row.append(f"{value:.6f}")

    # Add metadata
    row.extend([
        len(dataset.get_queries()),
        len(dataset.get_images()),
        datetime.now().isoformat()
    ])

    with open(metrics_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerow(row)

    print(f"  [OK] Saved: {metrics_file}")


def main():
    """Recalculate all metrics from existing experiment data."""

    # Comprehensive metrics list
    metrics_list = [
        'accuracy',
        'recall@1',
        'recall@4',
        'recall@10',
        'precision@10',
        'map',
        'map@10',
        'map@50',
        'ns_score'
    ]

    # Dataset configurations
    datasets_config = {
        'ukbench': 'D:/Similar Images/DataSets/ukbench',
        'holidays': 'D:/Similar Images/DataSets/InriaHolidaysFull'
    }

    # Find all experiment directories
    base_dir = Path("outputs/baseline_runs/comprehensive_baseline")

    print("="*80)
    print("RECALCULATING METRICS FROM EXISTING RANKINGS")
    print("="*80)
    print(f"\nBase directory: {base_dir}")
    print(f"Metrics to compute: {', '.join(metrics_list)}\n")

    # Scan for all method directories with rankings.csv
    method_dirs = []
    for method_dir in base_dir.rglob("*/rankings.csv"):
        method_dirs.append(method_dir.parent)

    print(f"Found {len(method_dirs)} method directories with rankings.csv\n")

    for method_dir in method_dirs:
        # Read manifest to get method name and dataset
        manifest_file = method_dir / "manifest.json"
        if not manifest_file.exists():
            print(f"[WARN] Skipping {method_dir} (no manifest.json)")
            continue

        with open(manifest_file, 'r') as f:
            manifest = json.load(f)

        method_name = manifest['method']

        # Detect dataset from manifest or directory name
        if 'dataset' in manifest and 'name' in manifest['dataset']:
            dataset_name = manifest['dataset']['name']
        else:
            # Infer from parent directory name
            parent_name = method_dir.parent.name
            if 'ukbench' in parent_name:
                dataset_name = 'ukbench'
            elif 'holidays' in parent_name:
                dataset_name = 'holidays'
            else:
                print(f"[WARN] Skipping {method_dir} (cannot detect dataset)")
                continue

        if dataset_name not in datasets_config:
            print(f"[WARN] Skipping {method_dir} (unknown dataset: {dataset_name})")
            continue

        dataset_root = datasets_config[dataset_name]

        print(f"Processing: {method_name} on {dataset_name}")
        print(f"  Directory: {method_dir}")

        try:
            # Recalculate metrics
            computed_metrics = recalculate_metrics_for_method(
                method_dir,
                dataset_name,
                dataset_root,
                metrics_list
            )

            # Save to CSV
            # Need to reload dataset for save function
            dataset_config = {
                'name': dataset_name,
                'root': dataset_root
            }
            if dataset_name == 'ukbench':
                dataset_config['assume_groups_of_four'] = True
                dataset_config['pattern'] = 'ukbench*.jpg'
                dataset_config['subdirs'] = {'images': 'full'}
            elif dataset_name == 'holidays':
                dataset_config['pattern'] = '*.jpg'

            dataset = load_dataset(dataset_name, dataset_config)

            save_metrics_csv(method_dir, method_name, computed_metrics, dataset)

            # Print summary
            print(f"  Metrics computed:")
            for key in sorted(computed_metrics.keys()):
                if key not in ['num_queries', 'num_images']:
                    print(f"    {key:15} = {computed_metrics[key]:.6f}")
            print()

        except Exception as e:
            print(f"  [ERROR]: {e}")
            import traceback
            traceback.print_exc()
            print()
            continue

    print("="*80)
    print("DONE!")
    print("="*80)
    print("\nAll metrics.csv files have been regenerated.")
    print("You can now run the multi-experiment analysis notebook.")


if __name__ == "__main__":
    main()
