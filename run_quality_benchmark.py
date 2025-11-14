#!/usr/bin/env python
"""
Command-line runner for quality assessment benchmarks.

Usage:
    # Run benchmark with config file
    python run_quality_benchmark.py configs/quality_benchmark.phototriage.yaml
    
    # Run quick test
    python run_quality_benchmark.py configs/quality_benchmark.quick.yaml
    
    # Run multi-dataset benchmark
    python run_quality_benchmark.py configs/quality_benchmark.multi_dataset.yaml
    
    # Specify custom output directory
    python run_quality_benchmark.py configs/quality_benchmark.phototriage.yaml --output outputs/my_benchmarks
"""

import argparse
import sys
from pathlib import Path

from sim_bench.quality_assessment import run_benchmark_from_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run quality assessment benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with sampled data
  python run_quality_benchmark.py configs/quality_benchmark.quick.yaml
  
  # Full PhotoTriage benchmark
  python run_quality_benchmark.py configs/quality_benchmark.phototriage.yaml
  
  # Multi-dataset comparison
  python run_quality_benchmark.py configs/quality_benchmark.multi_dataset.yaml
  
  # Custom output location
  python run_quality_benchmark.py configs/quality_benchmark.phototriage.yaml --output results/
        """
    )
    
    parser.add_argument(
        'config',
        type=str,
        help='Path to benchmark configuration YAML file'
    )
    
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default='outputs/quality_benchmarks',
        help='Output directory for results (default: outputs/quality_benchmarks)'
    )
    
    parser.add_argument(
        '--quiet',
        '-q',
        action='store_true',
        help='Suppress progress output'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Check config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        print(f"\nAvailable example configs:")
        print(f"  - configs/quality_benchmark.quick.yaml (fast test)")
        print(f"  - configs/quality_benchmark.phototriage.yaml (full PhotoTriage)")
        print(f"  - configs/quality_benchmark.multi_dataset.yaml (multiple datasets)")
        sys.exit(1)
    
    # Run benchmark
    try:
        print(f"Running benchmark from: {config_path}")
        print(f"Output directory: {args.output}")
        print()
        
        results = run_benchmark_from_config(str(config_path), args.output)
        
        print("\n" + "="*80)
        print("BENCHMARK COMPLETE")
        print("="*80)
        print(f"Results saved to: {results['run_dir']}")
        print("\nKey files:")
        print(f"  - summary.json: Complete results summary")
        print(f"  - methods_summary.csv: Method performance comparison")
        print(f"  - detailed_results.csv: Per-dataset, per-method results")
        print(f"  - config.yaml: Benchmark configuration used")
        print("="*80)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n\nERROR running benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())


