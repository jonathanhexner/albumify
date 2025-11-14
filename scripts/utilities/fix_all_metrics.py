"""
Regenerate metrics.csv from per_query.csv files.
"""

import pandas as pd
from pathlib import Path
import json
import csv
from datetime import datetime


def aggregate_per_query_metrics(per_query_file):
    df = pd.read_csv(per_query_file)
    metrics = {}
    if 'ap_full' in df.columns:
        metrics['map'] = df['ap_full'].mean()
    if 'ap@10' in df.columns:
        metrics['map@10'] = df['ap@10'].mean()
    if 'ap@50' in df.columns:
        metrics['map@50'] = df['ap@50'].mean()
    if 'recall@10' in df.columns:
        metrics['recall@10'] = (df['recall@10'] > 0).mean()
    if 'ns_hitcount@4' in df.columns:
        metrics['ns_score'] = df['ns_hitcount@4'].mean()
    return metrics


def fix_metrics_csv(method_dir):
    per_query_file = method_dir / "per_query.csv"
    metrics_file = method_dir / "metrics.csv"
    manifest_file = method_dir / "manifest.json"
    
    if not per_query_file.exists() or not manifest_file.exists():
        return False, "Missing files"
    
    with open(manifest_file, 'r') as f:
        manifest = json.load(f)
    
    method_name = manifest['method']
    
    if 'dataset' in manifest and 'name' in manifest['dataset']:
        num_queries = manifest['dataset'].get('total_queries', 0)
        num_images = manifest['dataset'].get('total_images', 0)
    else:
        parent_name = method_dir.parent.name
        if 'ukbench' in parent_name:
            num_queries, num_images = 10200, 10200
        elif 'holidays' in parent_name:
            num_queries, num_images = 1491, 1491
        else:
            return False, "Cannot detect dataset"
    
    computed_metrics = aggregate_per_query_metrics(per_query_file)
    if not computed_metrics:
        return False, "No metrics"
    
    computed_metrics['num_queries'] = num_queries
    computed_metrics['num_images'] = num_images
    
    metadata_keys = {'num_queries', 'num_images'}
    metric_names = [key for key in computed_metrics.keys() if key not in metadata_keys]
    
    header = ["method"] + sorted(metric_names) + ["num_queries", "num_gallery", "created_at"]
    row = [method_name]
    for metric_name in sorted(metric_names):
        row.append(f"{computed_metrics[metric_name]:.6f}")
    row.extend([num_queries, num_images, datetime.now().isoformat()])
    
    with open(metrics_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerow(row)
    
    return True, computed_metrics


base_dir = Path("outputs/baseline_runs/comprehensive_baseline")
print("="*80)
print("FIXING METRICS.CSV")
print("="*80)

method_dirs = list(base_dir.rglob("*/per_query.csv"))
print(f"\nFound {len(method_dirs)} directories\n")

for per_query_file in method_dirs:
    method_dir = per_query_file.parent
    manifest_file = method_dir / "manifest.json"
    if manifest_file.exists():
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
        method_name = manifest['method']
        if 'dataset' in manifest and 'name' in manifest['dataset']:
            dataset_name = manifest['dataset']['name']
        else:
            dataset_name = 'ukbench' if 'ukbench' in method_dir.parent.name else 'holidays'
    else:
        method_name, dataset_name = method_dir.name, '?'
    
    print(f"{method_name:12} on {dataset_name:10} ... ", end='')
    success, result = fix_metrics_csv(method_dir)
    if success:
        print("[OK]")
    else:
        print(f"[SKIP] {result}")

print("\n" + "="*80)
print("DONE!")
