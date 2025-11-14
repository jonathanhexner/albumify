#!/usr/bin/env python
"""
Generate visualizations and report from quality benchmark results.

Usage:
    python visualize_quality_benchmark.py outputs/quality_benchmarks/benchmark_2025-11-12_12-34-56
"""

import argparse
import sys
from pathlib import Path

from sim_bench.quality_assessment.visualization import visualize_benchmark


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations from quality benchmark results"
    )
    
    parser.add_argument(
        'results_dir',
        type=str,
        help='Path to benchmark results directory'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Check for required files
    required_files = ['summary.json', 'detailed_results.csv', 'methods_summary.csv']
    missing = [f for f in required_files if not (results_dir / f).exists()]
    
    if missing:
        print(f"ERROR: Missing required files in results directory:")
        for f in missing:
            print(f"  - {f}")
        sys.exit(1)
    
    try:
        print(f"Generating visualizations for: {results_dir}")
        visualize_benchmark(str(results_dir))
        print("\nDone! Check the results directory for visualizations and report.")
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())


