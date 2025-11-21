#!/usr/bin/env python
"""
Run pairwise quality assessment benchmark.

This evaluates quality methods on pairwise classification:
Given two images, predict which was preferred by users.

This is the CORRECT evaluation for PhotoTriage dataset, which provides
pairwise preferences rather than absolute series rankings.

Usage:
    python run_pairwise_benchmark.py --config configs/pairwise_benchmark.phototriage.yaml
    python run_pairwise_benchmark.py --config configs/pairwise_benchmark.quick_test.yaml
"""

import sys
import argparse
import logging
from pathlib import Path

from sim_bench.quality_assessment.pairwise_benchmark import run_pairwise_benchmark_from_config


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run pairwise quality assessment benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark on PhotoTriage dataset
  python run_pairwise_benchmark.py --config configs/pairwise_benchmark.phototriage.yaml

  # Run quick test with rule-based methods only
  python run_pairwise_benchmark.py --config configs/pairwise_benchmark.quick_test.yaml

  # Specify custom log level
  python run_pairwise_benchmark.py --config configs/pairwise_benchmark.phototriage.yaml --log-level DEBUG

Note:
  This benchmark evaluates methods on PAIRWISE CLASSIFICATION, not series selection.
  For each pair of images, methods predict which was preferred by users.
  This is the correct evaluation for PhotoTriage's pairwise comparison data.
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to benchmark configuration YAML file'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )

    parser.add_argument(
        '--methods',
        type=str,
        default=None,
        help='Comma-separated list of method names to run (e.g., "NIMA-MobileNet,MUSIQ,CLIP-Aesthetic-LAION"). If not specified, runs all methods from config.'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Validate config path
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    logger.info("="*80)
    logger.info("Pairwise Quality Assessment Benchmark")
    logger.info("="*80)
    logger.info(f"Configuration: {config_path}")
    
    # Parse method filter if provided
    method_filter = None
    if args.methods:
        method_filter = [m.strip() for m in args.methods.split(',') if m.strip()]
        logger.info(f"Method filter: {method_filter}")
    logger.info("")

    try:
        # Run benchmark
        results = run_pairwise_benchmark_from_config(config_path, method_filter=method_filter)

        logger.info("\n" + "="*80)
        logger.info("Benchmark completed successfully!")
        logger.info("="*80)

        # Print summary
        num_methods = len(results)
        logger.info(f"Evaluated {num_methods} methods")

        if results:
            # Find best method
            best_method = max(
                results.items(),
                key=lambda x: x[1]['global']['accuracy']
            )
            best_name = best_method[0]
            best_acc = best_method[1]['global']['accuracy']

            logger.info(f"Best method: {best_name} ({best_acc:.4f} accuracy)")

        return 0

    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
