#!/usr/bin/env python3
"""
Run face clustering pipeline using existing sim_bench architecture.

This script uses the generic pipeline framework to:
1. Detect faces in an album
2. Extract embeddings
3. Cluster by identity
4. Export results for manual labeling

Usage:
    python scripts/run_face_clustering_pipeline.py \
        --album /path/to/photos \
        --output results/my_album

    # With custom config:
    python scripts/run_face_clustering_pipeline.py \
        --album D:/Google_Germany \
        --output results/Germany_v1 \
        --config configs/my_custom_pipeline.yaml

Examples:
    # Test data
    python scripts/run_face_clustering_pipeline.py \
        --album test_data/face_clustering \
        --output results/test_export

    # Google Germany
    python scripts/run_face_clustering_pipeline.py \
        --album D:/Google_Germany \
        --output results/Germany_v1

    # Then open labeling app:
    streamlit run app/face_clustering_labeling.py -- --data-dir results/Germany_v1
"""
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.executor import PipelineExecutor
from sim_bench.pipeline.registry import get_registry
from sim_bench.pipeline.config import PipelineConfig
import sim_bench.pipeline.steps.all_steps  # Import to register steps
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load pipeline configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description='Run face clustering pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--album',
        type=Path,
        required=True,
        help='Path to album directory containing photos'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory for clustering results'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/face_clustering_experiment.yaml'),
        help='Pipeline configuration YAML file (default: configs/face_clustering_experiment.yaml)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.album.exists():
        logger.error(f"Album directory not found: {args.album}")
        return 1

    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        return 1

    # Load pipeline configuration
    logger.info(f"Loading config: {args.config}")
    config_data = load_config(args.config)

    # Override output directory in step config
    if 'step_configs' not in config_data:
        config_data['step_configs'] = {}
    if 'export_for_labeling' not in config_data['step_configs']:
        config_data['step_configs']['export_for_labeling'] = {}

    config_data['step_configs']['export_for_labeling']['output_dir'] = str(args.output)

    # Create pipeline config
    pipeline_config = PipelineConfig()
    pipeline_config.step_configs = config_data.get('step_configs', {})

    # Create context
    context = PipelineContext(
        source_directory=args.album
    )
    context.step_configs = config_data.get('step_configs', {})

    # Get step names from config
    step_names = config_data['pipeline']['steps']

    # Print summary
    print()
    print("=" * 70)
    print("FACE CLUSTERING PIPELINE")
    print("=" * 70)
    print(f"Album:   {args.album.absolute()}")
    print(f"Output:  {args.output.absolute()}")
    print(f"Config:  {args.config}")
    print(f"Pipeline: {config_data['pipeline']['name']}")
    print(f"Steps:   {len(step_names)} steps")
    for i, step_name in enumerate(step_names, 1):
        print(f"  {i}. {step_name}")
    print("=" * 70)
    print()

    # Execute pipeline
    try:
        registry = get_registry()
        executor = PipelineExecutor(registry)

        result = executor.execute(
            context,
            step_names=step_names,
            config=pipeline_config
        )

        if result.success:
            print()
            print("=" * 70)
            print("✓ PIPELINE COMPLETED SUCCESSFULLY")
            print("=" * 70)
            print(f"Output directory: {context.export_directory}")
            print()
            print("Files created:")
            print(f"  - {context.faces_csv_path}")
            print(f"  - {context.clusters_csv_path}")
            print(f"  - face_crops/ directory")
            print(f"  - export_summary.json")
            print()
            print("To label clusters:")
            print(f"  streamlit run app/face_clustering_labeling.py -- --data-dir {context.export_directory}")
            print("=" * 70)
            return 0
        else:
            print()
            print("=" * 70)
            print(f"✗ PIPELINE FAILED at step: {result.failed_step}")
            print("=" * 70)
            if result.error_message:
                print(f"Error: {result.error_message}")
            print()
            print("Check the logs above for details.")
            print("=" * 70)
            return 1

    except Exception as e:
        logger.exception("Pipeline execution failed with exception:")
        print()
        print("=" * 70)
        print("✗ PIPELINE FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        print("=" * 70)
        return 1


if __name__ == '__main__':
    sys.exit(main())
