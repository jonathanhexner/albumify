"""
Frozen features Siamese training.

Trains a pairwise ranker on pre-extracted features from frozen networks (CLIP, CNN, IQA).
Fast training since features are cached.

Usage:
    python -m sim_bench.training.train_frozen --config configs/frozen/resnet50.yaml
    python -m sim_bench.training.train_frozen --config configs/frozen/multifeature.yaml --quick-experiment 0.1
"""
import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd

from sim_bench.datasets.phototriage_data import PhotoTriageData
from sim_bench.quality_assessment.trained_models.phototriage_multifeature import (
    MultiFeaturePairwiseRanker,
    MultiFeaturePairwiseDataset,
    compute_pairwise_accuracy
)

logger = logging.getLogger(__name__)


def load_config(path):
    """Load YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def create_optimizer(model, config):
    """Create optimizer from config."""
    opt_name = config['training']['optimizer'].lower()
    lr = config['training']['learning_rate']
    wd = config['training']['weight_decay']

    if opt_name == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config['training']['momentum'],
            weight_decay=wd
        )
    else:  # adamw (default)
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def create_model(config, output_dir, cache_dir):
    """Create pairwise ranker model from config dict."""
    # Create model config dict with all necessary parameters
    model_config = {
        **config['model'],  # Copy all model params from YAML
        'device': config['device'],
        'output_dir': str(output_dir),
        'central_cache_dir': cache_dir
    }
    return MultiFeaturePairwiseRanker(model_config).to(config['device'])


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0

    for batch in loader:
        feat1 = batch['feat1'].to(device)
        feat2 = batch['feat2'].to(device)
        winners = batch['winner'].to(device)

        log_probs = model(feat1, feat2)
        loss = F.nll_loss(log_probs, 1 - winners)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += compute_pairwise_accuracy(log_probs, winners)

    return total_loss / len(loader), total_acc / len(loader)


def evaluate(model, loader, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    with torch.no_grad():
        for batch in loader:
            feat1 = batch['feat1'].to(device)
            feat2 = batch['feat2'].to(device)
            winners = batch['winner'].to(device)

            log_probs = model(feat1, feat2)
            loss = F.nll_loss(log_probs, 1 - winners)

            total_loss += loss.item()
            total_acc += compute_pairwise_accuracy(log_probs, winners)

    return total_loss / len(loader), total_acc / len(loader)


def load_data(config):
    """Load and split PhotoTriage data."""
    data = PhotoTriageData(
        config['data']['root_dir'],
        config['data']['min_agreement'],
        config['data']['min_reviewers']
    )

    train_df, val_df, test_df = data.get_series_based_splits(
        config['data']['split_ratios'][0],
        config['data']['split_ratios'][1],
        config['data']['split_ratios'][2],
        config['seed'],
        config['data'].get('quick_experiment')
    )

    logger.info(f"Data loaded: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    return data, train_df, val_df, test_df


def create_dataloaders(train_df, val_df, test_df, data, model, feature_cache, batch_size):
    """Create PyTorch data loaders."""
    train_dataset = MultiFeaturePairwiseDataset(
        train_df, data.train_val_img_dir, model.feature_extractor, feature_cache
    )
    val_dataset = MultiFeaturePairwiseDataset(
        val_df, data.train_val_img_dir, model.feature_extractor, feature_cache
    )
    test_dataset = MultiFeaturePairwiseDataset(
        test_df, data.train_val_img_dir, model.feature_extractor, feature_cache
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, optimizer, config, output_dir):
    """Training loop with early stopping."""
    best_val_acc = 0.0
    patience = 0
    device = config['device']

    for epoch in range(config['training']['max_epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        logger.info(f"Epoch {epoch+1}: train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, output_dir / 'best_model.pt')
            patience = 0
        else:
            patience += 1
            if patience >= config['training']['early_stop_patience']:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    return best_val_acc


def main():
    # Parse arguments first
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to YAML config')
    parser.add_argument('--output-dir', default=None, help='Override output directory')
    parser.add_argument('--data-dir', default=None, help='Override data root directory')
    parser.add_argument('--device', default=None, help='Override device (cpu/cuda)')
    parser.add_argument('--quick-experiment', type=float, default=None, help='Use fraction of data')
    args = parser.parse_args()

    # Load and override config
    config = load_config(args.config)
    if args.output_dir:
        config['output']['output_dir'] = args.output_dir
    if args.data_dir:
        config['data']['root_dir'] = args.data_dir
    if args.device:
        config['device'] = args.device
    if args.quick_experiment:
        config['data']['quick_experiment'] = args.quick_experiment

    # Setup output directory
    output_dir = Path(
        config['output']['output_dir'] or
        f"outputs/frozen/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging (both console and file)
    log_file = output_dir / 'training.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger.info(f"Logging to: {log_file}")

    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    logger.info(f"Training frozen features | Output: {output_dir}")

    # Load data
    data, train_df, val_df, test_df = load_data(config)

    # Create model and cache features
    model = create_model(config, output_dir, config['cache_dir'])

    all_df = pd.concat([train_df, val_df, test_df])
    feature_cache = data.precompute_features(
        all_df,
        model.feature_extractor,
        config['cache_dir'],
        config['model']['use_clip'],
        config['model']['use_cnn'],
        config['model']['use_iqa']
    )

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, data, model, feature_cache,
        config['training']['batch_size']
    )

    # Create optimizer
    optimizer = create_optimizer(model, config)

    # Train
    train_model(model, train_loader, val_loader, optimizer, config, output_dir)

    # Test
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc = evaluate(model, test_loader, config['device'])

    logger.info(f"Test accuracy: {test_acc:.3f}")

    with open(output_dir / 'results.json', 'w') as f:
        json.dump({'test_acc': test_acc, 'test_loss': test_loss}, f, indent=2)


if __name__ == '__main__':
    main()
