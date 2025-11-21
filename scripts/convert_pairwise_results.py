#!/usr/bin/env python
"""
One-time conversion script for existing pairwise benchmark results.

Converts existing pairwise benchmark output to unified format compatible
with quality benchmark analysis tools.

Usage:
    python scripts/convert_pairwise_results.py D:\\sim-bench\\outputs\\pairwise_benchmark_3hour\\pairwise_20251120_100520
"""

import json
import sys
from pathlib import Path
import pandas as pd


def extract_dataset_name(pairs_file_path: str) -> str:
    """Extract dataset name from pairs file path."""
    path = Path(pairs_file_path)
    parent_name = path.parent.name
    
    if 'phototriage' in parent_name.lower():
        return "phototriage"
    
    stem = path.stem.replace('pairs_', '').replace('_train', '').replace('_test', '')
    if not stem or stem == 'train' or stem == 'test':
        return "pairwise"
    return stem


def convert_pairwise_results(benchmark_dir: Path):
    """
    Convert existing pairwise benchmark results to unified format.
    
    Args:
        benchmark_dir: Path to pairwise benchmark output directory
    """
    benchmark_dir = Path(benchmark_dir)
    
    if not benchmark_dir.exists():
        raise ValueError(f"Benchmark directory not found: {benchmark_dir}")
    
    summary_json_path = benchmark_dir / "summary.json"
    if not summary_json_path.exists():
        raise ValueError(f"summary.json not found in {benchmark_dir}")
    
    overall_comparison_path = benchmark_dir / "overall_comparison.csv"
    if not overall_comparison_path.exists():
        raise ValueError(f"overall_comparison.csv not found in {benchmark_dir}")
    
    print(f"Converting pairwise results from: {benchmark_dir}")
    
    with open(summary_json_path, 'r') as f:
        summary = json.load(f)
    
    pairs_file = summary['benchmark_info']['pairs_file']
    dataset_name = extract_dataset_name(pairs_file)
    
    print(f"Dataset name: {dataset_name}")
    print(f"Number of methods: {len(summary['methods'])}")
    
    methods_summary_rows = []
    detailed_results_rows = []
    
    for method_name, method_data in summary['methods'].items():
        global_accuracy = method_data['global_accuracy']
        runtime_seconds = method_data.get('runtime_seconds', 0)
        num_pairs = method_data.get('num_pairs', 0)
        
        methods_summary_rows.append({
            'method': method_name,
            'avg_top1_accuracy': global_accuracy,
            'avg_top2_accuracy': 0.0,
            'avg_mrr': 0.0,
            'avg_time_ms': runtime_seconds * 1000,
            'datasets_tested': 1
        })
        
        throughput = num_pairs / runtime_seconds if runtime_seconds > 0 else 0
        
        detailed_results_rows.append({
            'dataset': dataset_name,
            'method': method_name,
            'top1_accuracy': global_accuracy,
            'top2_accuracy': 0.0,
            'mrr': 0.0,
            'avg_time_ms': runtime_seconds * 1000,
            'throughput': throughput
        })
    
    methods_summary_df = pd.DataFrame(methods_summary_rows)
    methods_summary_df = methods_summary_df.sort_values('avg_top1_accuracy', ascending=False)
    methods_summary_path = benchmark_dir / 'methods_summary.csv'
    methods_summary_df.to_csv(methods_summary_path, index=False)
    print(f"[OK] Created: {methods_summary_path}")
    
    detailed_results_df = pd.DataFrame(detailed_results_rows)
    detailed_results_path = benchmark_dir / 'detailed_results.csv'
    detailed_results_df.to_csv(detailed_results_path, index=False)
    print(f"[OK] Created: {detailed_results_path}")
    
    print("\nConversion complete!")
    print(f"Methods summary: {len(methods_summary_df)} methods")
    print(f"Detailed results: {len(detailed_results_df)} rows")
    
    # Print prompt information for CLIP methods
    print("\n" + "="*80)
    print("CLIP Method Prompts")
    print("="*80)
    for method_name, method_data in summary['methods'].items():
        if 'CLIP' in method_name.upper():
            config = method_data.get('config', {})
            prompts = config.get('prompt_texts', {})
            
            if prompts:
                print(f"\n{method_name}:")
                print(f"  Model: {config.get('model_name', 'N/A')}")
                print(f"  Pretrained: {config.get('pretrained', 'N/A')}")
                print(f"  Aggregation: {config.get('aggregation_method', 'N/A')}")
                
                if 'contrastive_pairs' in prompts:
                    print(f"  Contrastive Pairs ({len(prompts['contrastive_pairs'])}):")
                    for i, (pos, neg) in enumerate(prompts['contrastive_pairs'][:3], 1):
                        print(f"    {i}. +: {pos}")
                        print(f"       -: {neg}")
                    if len(prompts['contrastive_pairs']) > 3:
                        print(f"    ... and {len(prompts['contrastive_pairs']) - 3} more")
                
                if 'positive_attributes' in prompts and prompts['positive_attributes']:
                    print(f"  Positive Attributes: {', '.join(prompts['positive_attributes'])}")
                
                if 'negative_attributes' in prompts and prompts['negative_attributes']:
                    print(f"  Negative Attributes: {', '.join(prompts['negative_attributes'])}")
            else:
                print(f"\n{method_name}: No prompt information available")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/convert_pairwise_results.py <benchmark_dir>")
        print("\nExample:")
        print("  python scripts/convert_pairwise_results.py D:\\sim-bench\\outputs\\pairwise_benchmark_3hour\\pairwise_20251120_100520")
        sys.exit(1)
    
    benchmark_dir = Path(sys.argv[1])
    
    try:
        convert_pairwise_results(benchmark_dir)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

