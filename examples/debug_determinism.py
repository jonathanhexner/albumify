"""
Debug script to identify determinism issues between two training configurations.

Usage:
    # Run determinism check on a single config
    python -m examples.debug_determinism --config configs/example_external.yaml

    # Compare two runs (run this twice with different output dirs, then compare)
    python -m examples.debug_determinism --config configs/example_external.yaml --output-dir run1
    python -m examples.debug_determinism --config configs/example_external.yaml --output-dir run2
    python -m examples.debug_determinism --compare run1 run2
"""
import argparse
import yaml
import logging
from pathlib import Path
import sys

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_bench.datasets.phototriage_data import PhotoTriageData
from sim_bench.datasets.dataloader_factory import DataLoaderFactory
from sim_bench.datasets.transform_factory import create_transform
from sim_bench.models.siamese_cnn_ranker import SiameseCNNRanker
from sim_bench.utils.determinism_checker import (
    comprehensive_determinism_check,
    compare_determinism_outputs,
)
from sim_bench.utils.model_inspection import inspect_model_output

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_config(path):
    """Load YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def create_model(config):
    """Create model from config."""
    model_type = config.get('model_type', 'siamese_cnn')
    device = config['device']

    if model_type == 'reference':
        # Load reference model
        reference_path = config.get('reference_model_path', r'D:\Projects\Series-Photo-Selection')
        if reference_path not in sys.path:
            logger.info(f"Adding reference model path: {reference_path}")
            sys.path.insert(0, reference_path)

        from models.ResNet50 import make_network
        model = make_network()
        logger.info("Created reference model")
    else:
        # Our implementation
        model = SiameseCNNRanker(config['model'])
        logger.info(f"Created SiameseCNNRanker with {config['model']['cnn_backbone']}")

    return model.to(device)


def create_dataloaders(config, transform, batch_size):
    """Create dataloaders."""
    use_external = config.get('use_external_dataloader', False)
    factory = DataLoaderFactory(batch_size=batch_size, num_workers=0, seed=config.get('seed'))

    if use_external:
        # External dataloader
        external_path = config['data'].get('external_path', r'D:\Projects\Series-Photo-Selection')
        if external_path not in sys.path:
            logger.info(f"Adding external path: {external_path}")
            sys.path.insert(0, external_path)

        from data.dataloader import MyDataset

        image_root = config['data'].get('image_root') or config['data'].get('root_dir')
        if not image_root:
            raise ValueError("Must specify image_root or root_dir for external dataloader")

        logger.info(f"Using external dataloader: {image_root}, seed={config.get('seed')}")
        train_data = MyDataset(train=True, image_root=image_root, seed=config.get('seed'))
        val_data = MyDataset(train=False, image_root=image_root, seed=config.get('seed'))

        logger.info(f"External data: {len(train_data)} train, {len(val_data)} val")
        return factory.create_from_external(train_data, val_data, None)
    else:
        # PhotoTriage
        data = PhotoTriageData(
            config['data']['root_dir'],
            config['data']['min_agreement'],
            config['data']['min_reviewers']
        )

        train_df, val_df, test_df = data.get_series_based_splits(
            0.8, 0.1, 0.1,
            config['seed'],
            config['data'].get('quick_experiment')
        )

        logger.info(f"PhotoTriage: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
        return factory.create_from_phototriage(data, train_df, val_df, test_df, transform)


def run_determinism_check(config_path, output_dir):
    """Run comprehensive determinism check on a configuration."""
    logger.info("="*70)
    logger.info("DETERMINISM DEBUG SESSION")
    logger.info("="*70)

    # Load config
    config = load_config(config_path)
    logger.info(f"\nConfig: {config_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Seed: {config.get('seed', 'NOT SET!')}")
    logger.info(f"Device: {config.get('device', 'NOT SET!')}")
    logger.info(f"External dataloader: {config.get('use_external_dataloader', False)}")
    logger.info(f"Model type: {config.get('model_type', 'siamese_cnn')}")

    # Set random seeds
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"\n✓ Random seeds set to {seed}")

    # Check if deterministic mode is needed
    logger.info("\nPyTorch determinism settings:")
    logger.info(f"  torch.backends.cudnn.deterministic: {torch.backends.cudnn.deterministic}")
    logger.info(f"  torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    # Create model
    logger.info("\nCreating model...")
    model = create_model(config)

    # Create transform
    logger.info("\nCreating transform...")
    transform = create_transform(config)
    logger.info(f"Transform type: {config.get('transform_type', 'default')}")

    # Create dataloaders
    logger.info("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        config, transform, config['training']['batch_size']
    )

    # Run comprehensive check
    logger.info("\n" + "="*70)
    logger.info("Running comprehensive determinism checks...")
    logger.info("="*70)

    output_path = Path(output_dir)
    results = comprehensive_determinism_check(
        model, train_loader, output_path, transform=transform
    )

    # Run model inspection on first batch
    logger.info("\n" + "="*70)
    logger.info("Running model inspection on first batch...")
    logger.info("="*70)

    df_inspect = inspect_model_output(
        model, train_loader, config['device'],
        save_path=output_path / 'model_inspection.csv'
    )

    logger.info(f"\nFirst batch accuracy: {df_inspect['correct'].mean():.3f}")
    logger.info(f"\nFirst 10 predictions:")
    logger.info(df_inspect[['image1', 'image2', 'winner', 'pred', 'correct']].head(10).to_string())

    # Save summary
    summary = {
        'config': str(config_path),
        'seed': seed,
        'device': config.get('device'),
        'external_dataloader': config.get('use_external_dataloader', False),
        'model_type': config.get('model_type', 'siamese_cnn'),
        'transform_type': config.get('transform_type', 'default'),
        'first_batch_accuracy': float(df_inspect['correct'].mean()),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
    }

    import json
    with open(output_path / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "="*70)
    logger.info("DETERMINISM CHECK COMPLETE")
    logger.info("="*70)
    logger.info(f"\nResults saved to: {output_path}")
    logger.info("\nTo compare with another run:")
    logger.info(f"  python -m examples.debug_determinism --compare {output_dir} <other_dir>")

    return results


def main():
    parser = argparse.ArgumentParser(description='Debug determinism issues')
    parser.add_argument('--config', help='Path to config YAML')
    parser.add_argument('--output-dir', default='determinism_check',
                       help='Output directory for results')
    parser.add_argument('--compare', nargs=2, metavar=('DIR1', 'DIR2'),
                       help='Compare two determinism check outputs')
    args = parser.parse_args()

    if args.compare:
        # Compare mode
        logger.info("Comparing determinism checks...")
        compare_determinism_outputs(args.compare[0], args.compare[1])
    elif args.config:
        # Single run mode
        run_determinism_check(args.config, args.output_dir)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
