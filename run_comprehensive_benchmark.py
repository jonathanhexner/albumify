#!/usr/bin/env python3
"""
Run comprehensive quality assessment benchmark on PhotoTriage.

Tests all rule-based methods, NIMA (CNN), and ViT (Transformer).
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sim_bench.quality_assessment.benchmark import run_benchmark_from_config


def main():
    """Run comprehensive benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run comprehensive quality assessment benchmark"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/quality_benchmark.comprehensive.yaml',
        help='Path to benchmark configuration file'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test (100 series) instead of full dataset'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/quality_benchmarks',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Use quick test config if requested
    if args.quick:
        config_path = 'configs/quality_benchmark.quick_test.yaml'
        print("Running QUICK TEST (100 series)...")
    else:
        config_path = args.config
        print("Running FULL BENCHMARK on PhotoTriage dataset...")
    
    print(f"Configuration: {config_path}")
    print(f"Output directory: {args.output_dir}")
    print("-" * 80)
    
    # Run benchmark
    results = run_benchmark_from_config(config_path, output_dir=args.output_dir)
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {results['run_dir']}")
    print("\nOutput files:")
    print("  - methods_summary.csv: Overall method comparison")
    print("  - detailed_results.csv: Per-dataset, per-method results")
    print("  - [dataset]_[method]_series.csv: Per-series detailed results")
    print("  - benchmark.log: Full execution log")
    print("=" * 80)


if __name__ == '__main__':
    main()






